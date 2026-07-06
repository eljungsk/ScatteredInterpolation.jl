# Performance review — rbf-pum branch (2026-07-05)

Scope: whole branch (new PUM/LOOCV code and legacy files), measured on the
target-scale configuration used by the branch plans: n = 100,000 data points,
3D, `Gaussian(50.0)`, `PartitionOfUnity` defaults (`pointsperpatch = 80`,
`overlap = 1.5`), 50,000 query points.

Environment: Julia 1.12.6, 12 threads, Linux, LinearSolve v2.39.1,
NearestNeighbors v0.4. Baseline at this scale: **build 0.40 s, evaluate(50k)
0.145 s, 12,167 patches (mean patch size 71)**.

Known and out of scope: LinearSolve.jl per-column matrix-RHS solve (documented
in `rbf.jl` `_solve!` and `rippa.jl`) — do not change.

Secondary scale point, 6D (n = 100k, `Gaussian(2.0)`, defaults): build 8.3 s
with **15,028 MiB allocated** (GC 16 %), evaluate(50k) 3.27 s, 262,144 patches
(mean size 54). Findings 1-3 apply with larger constants there — the
allocation-reduction work pays off most at 6D. The per-evaluate
`[Int[] for _ in 1:P]`/`[Tw[] for _ in 1:P]` setup (Finding 2, fix 3) alone
allocates 2×262k vectors per call at this patch count.

**Threading policy (project rule):** wherever a threaded region writes to
memory (including disjoint slices of a shared array), use OhMyThreads.jl
constructs — `tmap` for per-index results, `@tasks` with `@local` for per-task
reusable buffers, `index_chunks` for chunking — not raw `Threads.@threads`.
This applies to the fixes below *and* to the existing `Threads.@threads` loops
in `src/pum.jl` (build loop `pum.jl:444`, evaluate pass 2 `pum.jl:503`,
`addpoints!` re-solve `pum.jl:619`): migrate them to `tmap` while touching
these paths. OhMyThreads becomes a new direct dependency (registered, SciML-
adjacent, no heavy transitive deps; it pulls ChunkSplitters, StableTasks,
TaskLocalValues).

Each finding lists measured evidence and a prescribed fix. Findings are ordered
by expected impact. Verification for every fix: rerun the reproduction snippet
at the bottom and `Pkg.test()` must stay green.

## Measured aggregate gains (prototyped, not estimated)

All fixes for Findings 1-3 were prototyped end-to-end in a benchmark
environment (OhMyThreads `tmap`/`@tasks`, `alias_A = true`, threaded
`inrange`, count-then-fill pass 1) and verified to produce **identical
outputs** to the current implementation (same patch assignments; weights and
evaluate results bit-identical for the LinearSolve-routed variant):

| metric | current | prototype | gain |
|---|---|---|---|
| 3D build (n = 100k) | 0.399 s | **0.143 s**, 0 % GC, 610 MiB (was ~1.5 GiB total) | **2.8×** |
| 3D evaluate (50k) | 0.145 s | **0.086 s**, 0 % GC | **1.7×** |
| 6D build (n = 100k) | 8.34 s / 15.0 GiB | **4.59 s** / 8.9 GiB (alias-A loop) | **1.8×** |
| 6D build, deeper fix | — | solve loop 4.22 s → **2.25 s** with per-task `@local` A-buffer (Finding 1 step 2) ⇒ build ≈ 2.6 s | **≈3.2×** |
| 6D evaluate (50k) | 3.27 s | **1.90 s** | **1.7×** |

Component-level gains outside these paths (measured individually, mostly
independent): Shepard evaluate 1.4× (Finding 6), `generateMultivariatePolynomial`
2.3× (Finding 5, also feeds generalized-RBF and bordered-LOOCV paths),
`addpoints!` small-insert latency ~1.5× (Finding 4), `tune = :loocv` build
~10-20 % (Finding 7).

Prototype caveats an implementer must handle (details under each finding):
the prototype solve loop ran the `smooth = false`, `linsolve = nothing`,
vector-samples path only — the real `solvelocal` must keep the smoothing and
`linsolve` plumbing; and the per-task-buffer variant bypasses LinearSolve
(plain `lu!` on a buffer view), so it may only serve the
`linsolve === nothing` default path (weights matched to 4.5e-11 there;
the alias-A LinearSolve variant matched exactly).

---

## Finding 1 (high): PUM build solve loop is GC-bound — 1 GiB allocated, threads give only 2.8×

**Where:** `src/pum.jl:441-449` (threaded solve loop) together with the
per-patch path it calls: `solvelocal` (`src/pum.jl:394-400`) →
`interpolate(::RadialBasisFunction, ...)` (`src/rbf.jl:228-254`) →
`_initsolve`/`_solve!` (`src/rbf.jl:294-328`).

**Measured** (n = 100k, 12,167 patches):

| variant | time | GC share | allocated |
|---|---|---|---|
| solve loop, threaded (12) | 0.241 s | 40.9 % | 1047 MiB |
| solve loop, serial | 0.681 s | 8.7 % | 1047 MiB |

Threaded speedup is 2.8× on 12 threads; the single-patch solve itself is fast
(44 µs median, 85 KiB / 22 allocs at patch size 72). The loop is allocation-
and GC-bound, not compute-bound. Per-patch allocations: the `pairwise` distance
matrix (~41 KiB), LinearSolve's internal copy of `A` made by `init` (~41 KiB,
because `alias_A` defaults to false), the `pts[:, idxs]` gather, `vals[idxs]`,
`copy(sol.u)`, and LinearSolve cache internals.

**Fix (in order of simplicity):**

1. In `_initsolve` (`src/rbf.jl:294`), pass `alias_A = true` so LinearSolve
   factorizes the caller's matrix in place instead of copying it:
   `init(LinearProblem(A, b), alg; alias_A = true)`. This is safe for every
   current caller: the plain-RBF path never reuses `A` after the solve except
   to return it via `returnRBFmatrix` — **check that case**: `returnRBFmatrix`
   documents returning the RBF matrix, and an in-place LU destroys it. Guard by
   keeping the copy only when `returnRBFmatrix = true` (pass a flag or copy `A`
   before solving in that branch of `interpolate`). The generalized
   (Schur-complement) path at `src/rbf.jl:337-358` reuses `cacheA` for several
   RHS — that is still fine, aliasing only means the factorization overwrites
   `A`'s storage, which that path never reads again.
   The LOOCV scoring paths in `src/rippa.jl` also call `_initsolve`; they
   likewise never reread `A` after (`opnorm(A, 1)` in `gatedloocvscore` — note:
   `opnorm` **is** a read of `A` after the solve; compute it *before* the first
   `solve!` and pass it along, or keep `alias_A = false` for those two
   call sites).
2. Migrate the loop from `Threads.@threads` to OhMyThreads `tmap` over
   `patchpoints` (returns the `locals` vector directly — no shared writes at
   all). **Measured together with step 1** (3D, n = 100k): solve loop
   0.241 s / 41 % GC / 1047 MiB → **0.199 s / 28 % GC / 564 MiB**, weights
   exactly identical.
3. (Deeper, biggest win at 6D) Per-task reusable kernel-matrix buffer via
   OhMyThreads `@tasks for p in 1:P` + `@local Abuf = Matrix{Float64}(undef,
   maxnp, maxnp)` where `maxnp = maximum(length, patchpoints)`: fill
   `A = view(Abuf, 1:np, 1:np)` with `Distances.pairwise!`, apply the kernel
   in place, factorize with `lu!(A)`, solve. **Measured**: 3D loop
   0.241 s → **0.149 s / 0 % GC / 53 MiB**; 6D loop 4.22 s → **2.25 s /
   3 % GC / 1.2 GiB**. Constraint: this bypasses LinearSolve's algorithm
   selection, so it may only serve the `linsolve === nothing` default path
   (dispatch on it), and must keep `addSmoothing!` for the smoothing path.
   Weights differed from the LinearSolve route by ≤4.5e-11 (different but
   equally valid factorization). If that dual-path complexity is unwanted,
   stop after step 2 and accept 6D staying allocation-bound.

**Measured gain:** steps 1-2 alone: build 0.399 → 0.143 s (3D, with Finding 3
also applied), 8.34 → 4.59 s (6D). Adding step 3 at 6D: build ≈ 2.6 s (≈3.2×
total).

---

## Finding 2 (high): PUM evaluate — sequential pass 1 is half the time; pass 2 GC-bound

**Where:** `src/pum.jl:459-541` (`evaluate(::PartitionOfUnityInterpolant, ...)`).

**Measured** (50k queries, 433k query–patch pairs, total 0.145 s):

| phase | time | notes |
|---|---|---|
| `inrange` batch query (`pum.jl:476`) | 45 ms | sequential inside NearestNeighbors |
| pass 1 weight loop (`pum.jl:479-495`) | 31 ms | sequential, `push!` growth |
| pass 2 local evals (`pum.jl:501-511`) | 66-95 ms | threaded, **271 MiB alloc, 20 % GC** |
| pass 3 scatter (`pum.jl:517-525`) | 5 ms | fine as is |

**Fixes (all prototyped; combined evaluate output bit-identical to current):**

1. Thread the `inrange` step with OhMyThreads: `tmap` over
   `index_chunks(1:nq; n = 8 * Threads.nthreads())`, each task calling
   `inrange(itp.centertree, points[:, rng], grid.radius)`, then
   `reduce(vcat, parts)` — per-chunk results, no shared writes.
   **Measured: 45 → 8.7 ms**, identical output.
2. Replace `Matrix{T}(points)` at `pum.jl:476` with
   `convert(Matrix{T}, points)`: the constructor *always* copies, `convert` is
   a no-op when `points` is already a `Matrix{T}` (the common case). Same at
   `pum.jl:534-535` in the fallback branch.
3. Pass 1: replace the `push!`-grown `patchq`/`patchw` with a count-then-fill
   two-pass build (first pass counts pairs per patch into a `Vector{Int}`,
   allocate exact-size vectors, second pass fills, `resize!` down for pairs
   dropped by the `r < radius`/`ω > 0` guards). **Measured: 31 → 12.3 ms**,
   identical output. The weight computation itself (Wendland kappa ≈ 7
   ns/call, ~3 ms total) is not worth touching.
4. Pass 2: migrate to `tmap(1:P)` (returns `results` directly) and drop the
   `Matrix{Tout}(reshape(v, ...))` conversion copy when `eltype(v) == Tout`
   (typical): `reshape` alone does not copy. Remaining per-patch gathers
   (`points[:, patchq[p]]`, the `pairwise` matrix inside the local
   `evaluate`) stay; they are what feeds BLAS.

**Measured gain (all four combined):** evaluate 0.145 → **0.086 s** (3D),
3.27 → **1.90 s** (6D), outputs bit-identical.

---

## Finding 3 (medium): `assignpatches` is sequential and allocation-heavy — 49 ms of the 0.40 s build

**Where:** `src/pum.jl:78-99`.

**Measured** (12,167 non-empty of 13,824 cells): current 49 ms / 36.6 MiB /
154k allocs, all sequential. Two verified-equivalent variants (results
identical, `==` on both outputs):

- one batched `inrange` over a precomputed center matrix: 39 ms;
- threading the per-cell `inrange` calls: 34 ms median / 22 ms min
  (GC 26 % — again allocation-bound).

**Fix:** combine both with OhMyThreads: build the all-cells center matrix
(cheap, `D × ncells`), `tmap` over `index_chunks(1:ncells; n = 8 *
Threads.nthreads())` with each task running one batched `inrange` on its
column range, `reduce(vcat, parts)`, then filter non-empty and `sort!` in
place as today. Drop the per-cell `collect(c)` (`pum.jl:86`) — with the
center-matrix approach it disappears naturally.

**Measured gain:** as part of the full build prototype (identical patch
assignments verified at 3D and 6D), this phase's cost is no longer separable
but the whole build hit 0.143 s (3D) / 4.59 s (6D); the standalone threaded
variant measured 49 → 22-34 ms before allocation fixes.

---

## Finding 4 (medium): `addpoints!` pays O(n) per call in avoidable work — quadratic for repeated small inserts

**Where:** `src/pum.jl:585-608`.

**Measured** (10-point insert into the n = 100k interpolant, 2.43 ms total):

| component | time |
|---|---|
| bounding-box scan of *all* stored points (`pum.jl:585-586`) | 0.69 ms |
| `hcat(itp.points, newpts)` full copy (`pum.jl:607`) | 0.17 ms |
| `vcat` samples + re-solves | remainder |

The bbox scan alone is ~28 % of a small insert, and every insert copies all
points and samples, so k sequential single-point inserts cost O(k·n).

**Fix:**

1. Store the data bounding box (`lo`, `hi` vectors) as fields in
   `PartitionOfUnityInterpolant` at build time (they are already computed in
   `buildgrid`; recompute once in `interpolate` or return them alongside).
   In `addpoints!`, validate against the stored box and update it — new points
   are inside the box by definition here, so it never widens; just drop the
   scan.
2. (Optional, only if incremental insertion is a real workload) replace
   `points`/`samples` full copies with amortized growth: keep capacity-doubling
   buffers and a live-count, or accept the copy — at 0.17 ms per call it is
   secondary to the bbox scan.

**Expected gain:** small-insert latency 2.4 ms → ~1.6 ms; removes the dominant
O(n) term.

---

## Finding 5 (medium): `generateMultivariatePolynomial` allocates a row copy per term-factor — 2.3× available

**Where:** `src/rbf.jl:398` (`P[:, position] .*= points[var, :]`).

`points[var, :]` materializes a length-n vector for every variable of every
polynomial term. Hits generalized-RBF assembly, *every* generalized `evaluate`,
and the bordered LOOCV tuning path (`src/pum.jl:360`).

**Measured** (3D, degree 3, n = 5000): 358 µs / 4.21 MiB → with
`@views P[:, position] .*= points[var, :]`: 154 µs / 0.79 MiB, identical
output (`==`).

**Fix:** add `@views` (or `@view points[var, :]`) at that line. One-line
change.

---

## Finding 6 (medium, legacy): Shepard `evaluate` allocates per query point — 43 % GC

**Where:** `src/idw.jl:32-67`.

Per query point, `evaluatePoint` allocates `w = 1 ./ d.^P`, `w .* data`, and
the `sum` result — O(n) allocations per point, O(n·m) total.

**Measured** (n = 2000 data, m = 5000 queries, 1 sample column): current
143 ms mean / 230 MiB / **43 % GC**. A whole-matrix formulation:

```julia
dmat = pairwise(itp.metric, itp.points, points; dims = 2)   # n × m
W = dmat .^ (-P)
vals = (W' * itp.data) ./ sum(W, dims = 1)'
# then patch the exact-hit rows (any i with a zero in dmat[:, i])
```

measures 105 ms mean / 153 MiB / 19 % GC and matches the current implementation
to 1e-15 (max abs diff 1.0e-15 on this dataset). The zero-distance handling
must stay: scan each query column for a zero distance (as today) and overwrite
that output row with the exact sample.

Note `Inf` weights: the current per-point code already produces `NaN` if a
distance underflows to 0 without being exactly 0 — the matrix form behaves
identically, so behavior is unchanged.

**Fix:** replace the per-point loop with the matrix formulation above (keep the
zero-hit override loop). Expected ~1.4× plus far lower GC pressure; the
remaining cost is `pairwise` + GEMM, both BLAS-threaded.

---

## Finding 7 (low): LOOCV tuning hot loop churns a fresh LinearSolve cache + identity matrix per candidate

**Where:** `src/rippa.jl:136-159` (`gatedloocvscore`), `_invdiag!`
(`src/rippa.jl:68-74`), called ~9× per patch from `tuneshape`
(`src/pum.jl:315-390`).

**Measured** (np = 72): one gated score costs 61 µs median / 89 KiB / 30
allocs; of that, `_initsolve` + weight solve is 13.5 µs, so the identity
`ldiv!` (n RHS) + `gecon!` dominate — the loop is **compute-bound**, consistent
with the accepted ~150× documented cost of `tune = :loocv`. GC mean across
samples was ~15 %, so there is some headroom, but no order-of-magnitude win.

**Fix (only worth doing after Findings 1-3):** hoist per-patch buffers in
`tuneshape` — reuse one LinearSolve cache across candidates via
`cache.A = Anew` (verified on LinearSolve v2.39.1: assigning `cache.A` marks
the cache stale and the next `solve!` re-factorizes, residual 7e-16; no manual
`isfresh` handling needed) instead of `_initsolve` per candidate,
and preallocate the identity/`X` matrix used by `_invdiag!` (fill with `I`
each round) plus avoid the `diag(X)` copy by summing `abs2(W[i]/X[i,i])`
directly. `tuneshape` runs inside the per-patch parallel loop, so these
buffers are naturally per-task once the loop is an OhMyThreads `@tasks` block
(`@local` buffers) per the threading policy above. Combined with `alias_A`
caveats from Finding 1 (note `opnorm(A, 1)` must be computed before the
factorization overwrites `A`).

**Expected gain:** shaves the ~15 % GC and the 13.5 µs init overhead per
candidate; order 10-20 % off `tune = :loocv` build time, no more.

---

## Finding 8 (low, legacy): trivia

- `src/nearestNeighbor.jl:44-46`: row-by-row gather `values[i, :] =
  itp.data[inds[i], :]` allocates per row. `return itp.data[inds, :]` is
  equivalent (verified `==`) and slightly faster (1.37 → 1.32 ms at 5000
  queries; the KD-tree query dominates). Cosmetic.
- `src/rbf.jl:360-371` (plain RBF `evaluate`): materializes the full
  `nq × n` kernel matrix. Not measured (arithmetic): 10k data × 100k queries
  ×8 B = 8 GB — a memory cliff for large evaluations of *global* RBF
  interpolants. Chunking the query set (e.g. 4k columns per block, reusing the
  block buffer) bounds memory with no asymptotic time cost. Low priority: PUM
  is the intended large-n path.

---

## Measured non-issues (do not "fix")

- **Wendland via KernelFunctions.kappa**: 6.9 ns vs 5.4 ns/call hand-coded —
  ~3 ms of the 145 ms evaluate. Keep the wrapper.
- **`meannndist` brute force** (`src/pum.jl:217-234`): O(np²) at np≈72 is µs
  scale; comment's justification holds.
- **Per-patch solve algorithm**: LinearSolve's small-matrix pick
  (RFLUFactorization) measured *faster* (44 µs) than OpenBLAS `lu!` + `\`
  (52 µs) at patch size 72. Leave algorithm selection alone.
- **Pass 3 scatter** (`src/pum.jl:517-525`): 5 ms sequential, fine.
- **LinearSolve matrix-RHS column loop**: known upstream limitation,
  documented in-source; out of scope per review instructions.

---

## Reproduction

```julia
using ScatteredInterpolation, Random, LinearAlgebra
const SI = ScatteredInterpolation
Random.seed!(42)
n = 100_000; d = 3
pts = rand(d, n)
f(x) = sin(4x[1]) * cos(3x[2]) + x[3]^2
vals = [f(view(pts, :, i)) for i in 1:n]
pum = PartitionOfUnity(Gaussian(50.0))
itp = interpolate(pum, pts, vals)                 # warm up
@time itp = interpolate(pum, pts, vals)           # build: 0.40 s baseline
qpts = rand(d, 50_000)
SI.evaluate(itp, qpts)                            # warm up
@time SI.evaluate(itp, qpts)                      # eval: 0.145 s baseline
```

Run with `julia -t 12` (or the machine's core count); BLAS threads are pinned
inside the PUM paths already. Compare phase timings with the tables above after
each fix; all existing tests (`Pkg.test()`) must pass unchanged.
