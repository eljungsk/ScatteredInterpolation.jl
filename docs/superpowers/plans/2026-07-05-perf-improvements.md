# Performance Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Findings 1-5 of `docs/perf-review-2026-07-05.md`: remove the GC bottleneck from the PUM build, thread the sequential evaluate phases, and fix two smaller allocation hotspots — with zero behavior changes.

**Architecture:** All threaded regions that write memory migrate from `Threads.@threads` to OhMyThreads (`tmap` over per-index/per-chunk work, results returned rather than written into shared arrays). Allocation fixes: `alias_A = true` on the RBF weight-solve LinearSolve cache (opt-in per call site), count-then-fill buffers in the evaluate hot path, a stored bounding box in the PUM interpolant, and `@views` in polynomial assembly. Every change was prototyped with outputs verified identical to the current implementation (see the review doc).

**Tech Stack:** Julia, OhMyThreads.jl (new dependency), LinearSolve.jl, NearestNeighbors.jl. Run all Julia through `mcp__julia__julia_eval` with `env_path = "/home/emil/.julia/dev/ScatteredInterpolation"`; run tests with `Pkg.test()`.

**User decisions (already made):**
- Scope is Findings 1-5 only (no Shepard/LOOCV/trivia work).
- No buffered-`lu!` fast path: single solve path through LinearSolve, `alias_A` + `tmap` only.
- OhMyThreads constructs required wherever a threaded region writes memory.

**Spec:** `docs/superpowers/specs/2026-07-05-perf-improvements-design.md`
**Measured evidence for every fix:** `docs/perf-review-2026-07-05.md`

**Performance targets (from the spec, n = 100,000, 12 threads):**

| metric | current | target |
|---|---|---|
| 3D build | 0.399 s | ≤ 0.16 s |
| 3D evaluate (50k queries) | 0.145 s | ≤ 0.10 s |
| 6D build | 8.34 s | ≤ 5.0 s |
| 6D evaluate (50k queries) | 3.27 s | ≤ 2.1 s |

**Julia session note:** `Distances` also exports `evaluate`, so scripts must call
`ScatteredInterpolation.evaluate` qualified (alias `SI` below). If the session has a
stale module state (world-age `MethodError` mentioning `@world`), restart it with
`mcp__julia__julia_restart` and retry once before investigating.

---

### Task 1: Confirm the performance baseline

**Goal:** Reproduce the review's baseline numbers on this machine so later tasks measure against a trusted reference (`perf-plans-need-scale-benchmark` policy: baseline before hot-path work).

**Files:**
- Modify: `docs/superpowers/plans/2026-07-05-perf-improvements.md` (append results to the "Benchmark results" section at the bottom)

**Acceptance Criteria:**
- [ ] Baseline script runs to completion via `mcp__julia__julia_eval`
- [ ] 3D build within 0.30-0.50 s and 3D evaluate within 0.11-0.19 s (±~25 % of the review's 0.399 s / 0.145 s; if outside this window, STOP and report — the machine or environment differs from the review and the targets need rescaling)
- [ ] Numbers recorded in this plan file under "Benchmark results → Task 1 baseline"

**Verify:** script output shows `build:` and `eval:` lines; values inside the windows above.

**Steps:**

- [ ] **Step 1: Run the baseline benchmark**

Run via `mcp__julia__julia_eval` with `env_path = "/home/emil/.julia/dev/ScatteredInterpolation"`:

```julia
using ScatteredInterpolation, Random, LinearAlgebra
const SI = ScatteredInterpolation
Random.seed!(42)
pts = rand(3, 100_000)
f3(x) = sin(4x[1]) * cos(3x[2]) + x[3]^2
vals = [f3(view(pts, :, i)) for i in 1:100_000]
pum = PartitionOfUnity(Gaussian(50.0))
itp = interpolate(pum, pts, vals)                     # warm up
t_build = @elapsed itp = interpolate(pum, pts, vals)
qpts = rand(3, 50_000)
SI.evaluate(itp, qpts)                                # warm up
t_eval = @elapsed SI.evaluate(itp, qpts)
println("threads=", Threads.nthreads(), " build: ", round(t_build, digits = 3),
        " s | eval: ", round(t_eval, digits = 3), " s | patches: ", length(itp.locals))
```

Expected output shape: `threads=12 build: 0.399 s | eval: 0.145 s | patches: 12167`
(times within the acceptance windows; patch count exactly 12167 for this seed).

- [ ] **Step 2: Record the numbers**

Append under "Benchmark results" at the bottom of this plan file:

```markdown
### Task 1 baseline (this machine)
- 3D build: <measured> s (review: 0.399 s)
- 3D evaluate: <measured> s (review: 0.145 s)
- threads: <N>
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-07-05-perf-improvements.md
git commit -m "Record perf-improvement baseline on implementing machine"
```

```json:metadata
{"files": ["docs/superpowers/plans/2026-07-05-perf-improvements.md"], "verifyCommand": "baseline script via mcp__julia__julia_eval; build in [0.30,0.50] s, eval in [0.11,0.19] s", "acceptanceCriteria": ["script runs to completion", "3D build 0.30-0.50 s and eval 0.11-0.19 s", "numbers recorded in plan"], "modelTier": "mechanical"}
```

---

### Task 2: `@views` in polynomial assembly (Finding 5)

**Goal:** Stop `generateMultivariatePolynomial` from materializing a length-n row copy per polynomial term-factor (measured 2.3×: 358 → 154 µs at 3D degree 3, n = 5000).

**Files:**
- Modify: `src/rbf.jl:398`

**Acceptance Criteria:**
- [ ] `Pkg.test()` passes
- [ ] The only change is the added `@views`

**Verify:** `Pkg.test()` via `mcp__julia__julia_eval` → all tests pass.

**Steps:**

- [ ] **Step 1: Apply the one-line change**

In `src/rbf.jl`, inside `generateMultivariatePolynomial`, change:

```julia
            for var in combination
                P[:, position] .*= points[var, :]
            end
```

to:

```julia
            for var in combination
                @views P[:, position] .*= points[var, :]
            end
```

- [ ] **Step 2: Run the test suite**

Via `mcp__julia__julia_eval` (env_path = repo): `using Pkg; Pkg.test()`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/rbf.jl
git commit -m "Avoid row copies in generateMultivariatePolynomial

@views on the term accumulation stops materializing a length-n vector
per polynomial term-factor: measured 2.3x (358 -> 154 us, 4.21 -> 0.79
MiB at 3D degree 3, n = 5000), identical output. Benefits generalized
RBF assembly, generalized evaluate, and the bordered LOOCV path.
(docs/perf-review-2026-07-05.md, Finding 5)"
```

```json:metadata
{"files": ["src/rbf.jl"], "verifyCommand": "Pkg.test() via mcp__julia__julia_eval", "acceptanceCriteria": ["Pkg.test() passes", "only change is @views"], "modelTier": "mechanical"}
```

---

### Task 3: OhMyThreads dependency, `alias_A` weight solves, `tmap` build loop (Finding 1)

**Goal:** Halve the build solve loop's allocations (LinearSolve's internal copy of every patch's kernel matrix) and migrate the loop to OhMyThreads `tmap`; measured together: 0.241 s / 41 % GC / 1047 MiB → 0.199 s / 28 % GC / 564 MiB with exactly identical weights.

**Files:**
- Modify: `Project.toml` (deps + compat)
- Modify: `src/ScatteredInterpolation.jl:3` (imports)
- Modify: `src/rbf.jl:294` (`_initsolve`), `src/rbf.jl:228-254` (`interpolate`), `src/rbf.jl:330-358` (`solveForWeights`)
- Modify: `src/pum.jl:439-452` (build solve loop)

**Acceptance Criteria:**
- [ ] `Pkg.test()` passes (includes existing `returnRBFmatrix` and smoothing tests)
- [ ] `rippa.jl` call sites still get a non-aliased cache (default `aliasA = false`) — `gatedloocvscore` reads `opnorm(A, 1)` after the solve and must not see LU-overwritten storage
- [ ] `interpolate(...; returnRBFmatrix = true)` still returns the assembled (un-factorized) kernel matrix

**Verify:** `Pkg.test()` via `mcp__julia__julia_eval` → all tests pass; plus the returnRBFmatrix spot-check in Step 6.

**Steps:**

- [ ] **Step 1: Add the dependency**

In `Project.toml`, add to `[deps]` (alphabetical order):

```toml
OhMyThreads = "67456a42-1dca-4109-a031-0a68de7e3ad5"
```

and to `[compat]`:

```toml
OhMyThreads = "0.8"
```

- [ ] **Step 2: Import the constructs**

In `src/ScatteredInterpolation.jl`, change line 3:

```julia
using Distances, NearestNeighbors, Combinatorics, LinearAlgebra, LinearSolve
```

to:

```julia
using Distances, NearestNeighbors, Combinatorics, LinearAlgebra, LinearSolve
using OhMyThreads: tmap, index_chunks
```

- [ ] **Step 3: Make `_initsolve` aliasing opt-in**

In `src/rbf.jl`, replace:

```julia
_initsolve(A, alg) = init(LinearProblem(A, zeros(eltype(A), size(A, 1))), alg)
```

with:

```julia
# `aliasA = true` lets LinearSolve factorize the caller's matrix in place instead
# of copying it first — only for call sites that never read A after the first
# solve. The default stays false: the LOOCV scoring paths (src/rippa.jl) compute
# opnorm(A, 1) after solving and would read LU-overwritten storage otherwise.
_initsolve(A, alg; aliasA = false) =
    init(LinearProblem(A, zeros(eltype(A), size(A, 1))), alg; alias_A = aliasA)
```

- [ ] **Step 4: Opt in the weight-solve paths, guard `returnRBFmatrix`**

In `src/rbf.jl`, in the plain-RBF `solveForWeights`, change:

```julia
    cache = _initsolve(A, linsolve)
```

to:

```julia
    cache = _initsolve(A, linsolve; aliasA = true)
```

In the generalized `solveForWeights`, change:

```julia
    cacheA = _initsolve(A, linsolve)
```

to:

```julia
    cacheA = _initsolve(A, linsolve; aliasA = true)
```

(`cacheB` keeps the default — the Schur matrix `B` is small and not worth the churn.)

In `interpolate` (`src/rbf.jl:228-254`), the assembled `A` is now factorized in
place by the solve, so the branch that returns it must copy first. Replace:

```julia
    A = evaluateRBF!(A, rbf, smooth)

    # Solve for the weights
    itp = solveForWeights(A, points, samples, rbf, metric; linsolve = linsolve)

    # Create and return an interpolation object
    if returnRBFmatrix    # Return matrix A
        return itp, A
    else
        return itp
    end
```

with:

```julia
    A = evaluateRBF!(A, rbf, smooth)

    # The weight solve factorizes A in place (aliasA); keep an untouched copy only
    # when the caller asked for the matrix back.
    Aout = returnRBFmatrix ? copy(A) : nothing

    # Solve for the weights
    itp = solveForWeights(A, points, samples, rbf, metric; linsolve = linsolve)

    # Create and return an interpolation object
    if returnRBFmatrix    # Return matrix A
        return itp, Aout
    else
        return itp
    end
```

- [ ] **Step 5: Migrate the build solve loop to `tmap`**

In `src/pum.jl`, replace:

```julia
    # Local solves are independent — thread across patches. Each per-patch system is
    # small (~pointsperpatch), where single-threaded BLAS per task is appropriate.
    P = length(patchpoints)
    locals = Vector{RadialBasisInterpolant}(undef, P)
    withpinnedblas() do
        Threads.@threads for p in 1:P
            idxs = patchpoints[p]
            locals[p] = solvelocal(pum, pts[:, idxs], patchsamples(samples, idxs),
                                   patchsmooth(smooth, idxs), metric, linsolve)
        end
    end
    # Narrow to the concrete local-interpolant type (homogeneous in practice) so the
    # evaluation hot path dispatches statically.
    locals = [l for l in locals]
```

with:

```julia
    # Local solves are independent — thread across patches with tmap (each task
    # returns its result; no shared writes). Each per-patch system is small
    # (~pointsperpatch), where single-threaded BLAS per task is appropriate.
    solved = withpinnedblas() do
        tmap(patchpoints) do idxs
            solvelocal(pum, pts[:, idxs], patchsamples(samples, idxs),
                       patchsmooth(smooth, idxs), metric, linsolve)
        end
    end
    # Narrow to the concrete local-interpolant type (homogeneous in practice) so the
    # evaluation hot path dispatches statically.
    locals = [l for l in solved]
```

- [ ] **Step 6: Test**

Via `mcp__julia__julia_eval` (env_path = repo):

```julia
using Pkg; Pkg.test()
```

Expected: all tests pass. Then spot-check that `returnRBFmatrix` returns a valid
(symmetric, un-factorized) kernel matrix:

```julia
using ScatteredInterpolation
p = rand(2, 20); s = rand(20)
itp, A = interpolate(Multiquadratic(), p, s; returnRBFmatrix = true)
println("symmetric: ", A ≈ A', " | diag ones: ", all(A[i, i] ≈ 1 for i in 1:20))
```

Expected: `symmetric: true | diag ones: true` (an LU-overwritten matrix is not
symmetric; Multiquadratic has ϕ(0) = 1 on the diagonal).

- [ ] **Step 7: Commit**

```bash
git add Project.toml src/ScatteredInterpolation.jl src/rbf.jl src/pum.jl
git commit -m "Alias weight-solve matrices and migrate build loop to tmap

_initsolve gains an opt-in aliasA flag: the RBF weight-solve paths let
LinearSolve factorize the assembled kernel matrix in place instead of
copying it (~41 KiB per patch). returnRBFmatrix now copies explicitly;
the LOOCV scoring paths keep non-aliased caches since gatedloocvscore
reads opnorm(A, 1) after solving. The PUM build loop moves from
Threads.@threads writes into a shared vector to OhMyThreads tmap.

Measured (3D, n = 100k): solve loop 0.241 s / 41 % GC / 1047 MiB ->
0.199 s / 28 % GC / 564 MiB, weights exactly identical.
(docs/perf-review-2026-07-05.md, Finding 1 steps 1-2)"
```

```json:metadata
{"files": ["Project.toml", "src/ScatteredInterpolation.jl", "src/rbf.jl", "src/pum.jl"], "verifyCommand": "Pkg.test() via mcp__julia__julia_eval; returnRBFmatrix spot-check returns symmetric matrix", "acceptanceCriteria": ["Pkg.test() passes", "rippa call sites keep aliasA=false", "returnRBFmatrix returns un-factorized matrix"], "modelTier": "standard"}
```

---

### Task 4: Threaded patch assignment (Finding 3)

**Goal:** Rewrite `assignpatches` to batch cell centers into one matrix and chunk the range queries across threads (standalone measured 49 → 22-34 ms; verified identical output at 3D and 6D).

**Files:**
- Modify: `src/pum.jl:74-99` (`assignpatches`)

**Acceptance Criteria:**
- [ ] `Pkg.test()` passes
- [ ] No `Threads.@threads` in the new code; `tmap` over `index_chunks`, each task returning its own chunk's results

**Verify:** `Pkg.test()` via `mcp__julia__julia_eval` → all tests pass.

**Steps:**

- [ ] **Step 1: Replace the function**

In `src/pum.jl`, replace the whole `assignpatches` function (keep the doc comment
above it, update its body) with:

```julia
# Assign data points to patches. Returns:
# - patchpoints: per non-empty patch, sorted indices of the data points it contains
# - centers:     d × P matrix of the non-empty patch centers
# (Patch lookup during evaluation/insertion uses a KDTree over these centers.)
function assignpatches(points::Matrix{T}, grid::PatchGrid{T, D},
                       tree = KDTree(points)) where {T, D}
    cells = CartesianIndices(grid.ncells)
    ncell = length(cells)

    # All cell centers in one D × ncell matrix so the range queries can be batched
    # and chunked across threads (tmap: each task returns its own chunk's lists, no
    # shared writes).
    allcenters = Matrix{T}(undef, D, ncell)
    for (k, ci) in enumerate(cells)
        c = centerof(grid, ci)
        for i in 1:D
            allcenters[i, k] = c[i]
        end
    end
    chunks = index_chunks(1:ncell; n = 8 * Threads.nthreads())
    parts = tmap(chunks) do rng
        inrange(tree, allcenters[:, rng], grid.radius)
    end
    idxlists = reduce(vcat, parts)

    keep = findall(!isempty, idxlists)   # empty patches are dropped
    patchpoints = [sort!(idxlists[k]) for k in keep]
    centers = allcenters[:, keep]

    patchpoints, centers
end
```

- [ ] **Step 2: Test**

Via `mcp__julia__julia_eval` (env_path = repo): `using Pkg; Pkg.test()`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/pum.jl
git commit -m "Thread patch assignment with chunked batched range queries

assignpatches batches all cell centers into one matrix and runs the
KDTree range queries chunked across tasks via tmap, dropping the
per-cell collect and sequential query loop. Output verified identical
(same patchpoints and centers) at 3D and 6D target scale; standalone
49 -> 22-34 ms at n = 100k, 3D.
(docs/perf-review-2026-07-05.md, Finding 3)"
```

```json:metadata
{"files": ["src/pum.jl"], "verifyCommand": "Pkg.test() via mcp__julia__julia_eval", "acceptanceCriteria": ["Pkg.test() passes", "tmap over index_chunks, no Threads.@threads"], "modelTier": "mechanical"}
```

---

### Task 5: Evaluate restructure (Finding 2)

**Goal:** Thread the candidate query, build exact-size per-patch buffers (count-then-fill), migrate pass 2 to `tmap`, and drop two forced copies — measured together: evaluate 0.145 → 0.086 s (3D), 3.27 → 1.90 s (6D), outputs bit-identical.

**Files:**
- Modify: `src/pum.jl:459-541` (`evaluate(::PartitionOfUnityInterpolant, ...)`)

**Acceptance Criteria:**
- [ ] `Pkg.test()` passes
- [ ] Pass 1 candidate query threaded via `tmap` over `index_chunks`; per-patch buffers count-then-fill with `resize!` trim; no `push!` growth
- [ ] `convert(Matrix{T}, ...)` used instead of the `Matrix{T}(...)` constructor for the query matrix and the fallback branch
- [ ] Pass 3 scatter and normalization logic unchanged

**Verify:** `Pkg.test()` via `mcp__julia__julia_eval` → all tests pass.

**Steps:**

- [ ] **Step 1: Replace the evaluate method**

In `src/pum.jl`, replace the whole `evaluate(itp::PartitionOfUnityInterpolant{T, D}, ...)`
method with:

```julia
function evaluate(itp::PartitionOfUnityInterpolant{T, D},
                  points::AbstractArray{<:Real, 2}) where {T, D}

    size(points, 1) == D || throw(DimensionMismatch(
        "the interpolant was built in $D dimensions, but the evaluation points " *
        "have dimension $(size(points, 1))"))

    grid = itp.grid
    P = length(itp.locals)
    nq = size(points, 2)
    m = size(itp.samples, 2)
    Tw = float(promote_type(eltype(points), T))
    Tout = promote_type(Tw, eltype(itp.samples))

    # Pass 1: per query, find the covering patches with range queries against the
    # patch-center tree — chunked across threads with tmap (each task returns its
    # own chunk's lists) — then compute raw PU weights, grouped by patch so pass 2
    # can evaluate each local interpolant on one batched block. convert, not the
    # Matrix{T} constructor: the constructor always copies, convert is a no-op for
    # the common already-Matrix{T} input.
    qmat = convert(Matrix{T}, points)
    chunks = index_chunks(1:nq; n = 8 * Threads.nthreads())
    parts = tmap(chunks) do rng
        inrange(itp.centertree, qmat[:, rng], grid.radius)
    end
    candidates = reduce(vcat, parts)

    # Count-then-fill: exact-size per-patch buffers instead of push!-grown vectors.
    # The candidate counts are an upper bound; entries dropped by the radius and
    # weight guards below are trimmed off with resize! afterwards.
    counts = zeros(Int, P)
    for q in 1:nq, p in candidates[q]
        counts[p] += 1
    end
    patchq = [Vector{Int}(undef, counts[p]) for p in 1:P]
    patchw = [Vector{Tw}(undef, counts[p]) for p in 1:P]
    fill!(counts, 0)
    wsum = zeros(Tw, nq)
    for q in 1:nq
        x = view(points, :, q)
        for p in candidates[q]
            r2 = zero(Tw)
            for i in 1:D
                r2 += abs2(Tw(x[i]) - itp.centers[i, p])
            end
            r = sqrt(r2)
            r < grid.radius || continue
            ω = Tw(itp.weight(r / grid.radius))
            ω > 0 || continue
            k = (counts[p] += 1)
            patchq[p][k] = q
            patchw[p][k] = ω
            wsum[q] += ω
        end
    end
    for p in 1:P
        resize!(patchq[p], counts[p])
        resize!(patchw[p], counts[p])
    end

    # Pass 2 (threaded via tmap, BLAS pinned like the build): one batched evaluation
    # per patch — GEMM-shaped work, each task returning its own block. Local
    # evaluate returns a Vector for vector samples; normalize to a matrix so the
    # scatter below is shape-agnostic (reshape does not copy; the eltype conversion
    # only pays a copy in the mixed-eltype case).
    results = withpinnedblas() do
        tmap(1:P) do p
            if isempty(patchq[p])
                Matrix{Tout}(undef, 0, m)
            else
                v = evaluate(itp.locals[p], points[:, patchq[p]])
                R = reshape(v, length(patchq[p]), m)
                eltype(R) === Tout ? R : Matrix{Tout}(R)
            end
        end
    end

    # Pass 3 (sequential): scatter-accumulate weighted patch results. Queries covered
    # by several patches receive several contributions — keeping this phase serial
    # avoids write races without per-thread output copies, and its cost is only
    # O(query–patch pairs).
    out = zeros(Tout, nq, m)
    for p in 1:P
        qs = patchq[p]
        ws = patchw[p]
        R = results[p]
        for (i, q) in enumerate(qs)
            @views out[q, :] .+= ws[i] .* R[i, :]
        end
    end

    # Normalize the partition of unity; queries not covered by any patch (outside the
    # bounding box, or in a region that held no data) fall back to the nearest
    # patch's local interpolant. This is extrapolation and documented as such.
    for q in 1:nq
        if wsum[q] > 0
            @views out[q, :] ./= wsum[q]
        else
            p, _ = nn(itp.centertree, Vector{T}(view(points, :, q)))
            v = evaluate(itp.locals[p], convert(Matrix{T}, reshape(points[:, q], D, 1)))
            @views out[q, :] .= vec(v)
        end
    end

    itp.samples isa AbstractVector ? vec(out) : out
end
```

- [ ] **Step 2: Test**

Via `mcp__julia__julia_eval` (env_path = repo): `using Pkg; Pkg.test()`
Expected: all tests pass (the PUM tests cover exactness at nodes, matrix samples,
mixed eltypes, uncovered-query fallback, and batched-vs-single-query agreement).

- [ ] **Step 3: Commit**

```bash
git add src/pum.jl
git commit -m "Thread PUM evaluate pass 1 and remove its allocation churn

The candidate range query is chunked across tasks with tmap (45 ->
8.7 ms at 50k queries), per-patch buffers are built count-then-fill
instead of push!-grown (31 -> 12.3 ms), pass 2 moves to tmap and drops
its conversion copy in the common eltype case, and the forced query
matrix copy becomes a convert no-op.

Measured (n = 100k, 50k queries): evaluate 0.145 -> 0.086 s (3D) and
3.27 -> 1.90 s (6D), outputs bit-identical.
(docs/perf-review-2026-07-05.md, Finding 2)"
```

```json:metadata
{"files": ["src/pum.jl"], "verifyCommand": "Pkg.test() via mcp__julia__julia_eval", "acceptanceCriteria": ["Pkg.test() passes", "pass 1 threaded + count-then-fill", "convert instead of Matrix{T} constructor", "pass 3 unchanged"], "modelTier": "standard"}
```

---

### Task 6: Stored bounding box in `addpoints!`, `tmap` re-solve loop (Finding 4)

**Goal:** Stop `addpoints!` from scanning all stored points for the bounding box on every call (0.69 ms of a 2.43 ms 10-point insert at n = 100k), and migrate its re-solve loop to `tmap` per the threading policy.

**Files:**
- Modify: `src/pum.jl:166-182` (`PartitionOfUnityInterpolant` struct), `src/pum.jl:454-456` (constructor call in `interpolate`), `src/pum.jl:577-628` (`addpoints!`)

**Acceptance Criteria:**
- [ ] `Pkg.test()` passes (existing tests cover out-of-bbox rejection and post-insert exactness)
- [ ] `addpoints!` no longer computes `minimum`/`maximum` over `itp.points`
- [ ] Re-solve loop uses `tmap`; results written back by index sequentially

**Verify:** `Pkg.test()` via `mcp__julia__julia_eval` → all tests pass.

**Steps:**

- [ ] **Step 1: Add the bounding-box fields**

In `src/pum.jl`, in the `PartitionOfUnityInterpolant` struct, insert two fields
after `samples::S`:

```julia
    samples::S
    lo::Vector{T}               # data bounding box, fixed at build time —
    hi::Vector{T}               # addpoints! validates against it without a scan
    method::PU
```

- [ ] **Step 2: Fill them at build time**

In `interpolate(pum::PartitionOfUnity, ...)`, change the constructor call:

```julia
    PartitionOfUnityInterpolant(grid, patchpoints, locals, centers, KDTree(centers),
                                weight, pts, collect(samples), pum,
                                smooth, linsolve, metric)
```

to:

```julia
    PartitionOfUnityInterpolant(grid, patchpoints, locals, centers, KDTree(centers),
                                weight, pts, collect(samples),
                                vec(minimum(pts, dims = 2)), vec(maximum(pts, dims = 2)),
                                pum, smooth, linsolve, metric)
```

- [ ] **Step 3: Use the stored box in `addpoints!`**

In `addpoints!`, replace:

```julia
    # The patch grid is fixed at build time: reject points outside the bounding box
    # of the data. The box is computed from the stored points rather than from the
    # grid, whose flat dimensions carry an artificial unit spacing that would
    # otherwise admit off-plane points.
    lo = vec(minimum(itp.points, dims = 2))
    hi = vec(maximum(itp.points, dims = 2))
```

with:

```julia
    # The patch grid is fixed at build time: reject points outside the bounding box
    # of the original data, stored on the interpolant at build time (computed from
    # the points rather than from the grid, whose flat dimensions carry an
    # artificial unit spacing that would otherwise admit off-plane points). The box
    # never widens: accepted points lie inside it by definition.
    lo = itp.lo
    hi = itp.hi
```

- [ ] **Step 4: Migrate the re-solve loop to `tmap`**

Still in `addpoints!`, replace:

```julia
    # Re-solve only the affected local systems (threaded and BLAS-pinned, like the
    # build).
    aff = collect(affected)
    withpinnedblas() do
        Threads.@threads for k in eachindex(aff)
            p = aff[k]
            idxs = itp.patchpoints[p]
            itp.locals[p] = solvelocal(itp.method, itp.points[:, idxs],
                                       patchsamples(itp.samples, idxs),
                                       itp.smooth, itp.metric, itp.linsolve)
        end
    end
```

with:

```julia
    # Re-solve only the affected local systems (tmap, BLAS-pinned, like the build);
    # each task returns its solve, written back by index afterwards.
    aff = collect(affected)
    solved = withpinnedblas() do
        tmap(aff) do p
            idxs = itp.patchpoints[p]
            solvelocal(itp.method, itp.points[:, idxs],
                       patchsamples(itp.samples, idxs),
                       itp.smooth, itp.metric, itp.linsolve)
        end
    end
    for (k, p) in enumerate(aff)
        itp.locals[p] = solved[k]
    end
```

- [ ] **Step 5: Test**

Via `mcp__julia__julia_eval` (env_path = repo): `using Pkg; Pkg.test()`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pum.jl
git commit -m "Store the PUM data bounding box for addpoints! validation

addpoints! validated new points against a bounding box recomputed by
scanning all stored points on every call (0.69 ms of a 2.43 ms
10-point insert at n = 100k). The box is now computed once at build
time and stored on the interpolant; it never widens since accepted
points lie inside it. The re-solve loop also moves to tmap per the
project threading policy.
(docs/perf-review-2026-07-05.md, Finding 4 step 1)"
```

```json:metadata
{"files": ["src/pum.jl"], "verifyCommand": "Pkg.test() via mcp__julia__julia_eval", "acceptanceCriteria": ["Pkg.test() passes", "no min/max scan in addpoints!", "re-solve loop via tmap"], "modelTier": "mechanical"}
```

---

### Task 7: Target-scale benchmark validation

**Goal:** Measure the finished branch against the spec's performance targets at 3D and 6D target scale and record the results.

**Files:**
- Modify: `docs/superpowers/plans/2026-07-05-perf-improvements.md` (append results under "Benchmark results")

**Acceptance Criteria:**
- [ ] 3D build ≤ 0.16 s, 3D evaluate ≤ 0.10 s, 6D build ≤ 5.0 s, 6D evaluate ≤ 2.1 s (scale windows proportionally if Task 1 found the machine off the review baseline)
- [ ] Results recorded in this plan file
- [ ] If any budget is missed: do NOT tune ad hoc — record the measured numbers, identify which finding's expected gain did not materialize (compare against the phase timings in `docs/perf-review-2026-07-05.md`), and report back for a decision

**Verify:** benchmark script output; all four numbers within budget.

**Steps:**

- [ ] **Step 1: Run the 3D benchmark**

Same script as Task 1, verbatim. Expected: `build:` ≤ 0.16 s, `eval:` ≤ 0.10 s,
`patches: 12167`.

- [ ] **Step 2: Run the 6D benchmark**

Via `mcp__julia__julia_eval` (env_path = repo):

```julia
using ScatteredInterpolation, Random, LinearAlgebra
const SI = ScatteredInterpolation
Random.seed!(7)
pts6 = rand(6, 100_000)
f6(x) = sin(4x[1]) * cos(3x[2]) + x[3]^2 + x[4] - x[5] * x[6]
vals6 = [f6(view(pts6, :, i)) for i in 1:100_000]
pum6 = PartitionOfUnity(Gaussian(2.0))
itp6 = interpolate(pum6, pts6, vals6)                    # warm up
t_build = @elapsed itp6 = interpolate(pum6, pts6, vals6)
q6 = rand(6, 50_000)
SI.evaluate(itp6, q6)                                    # warm up
t_eval = @elapsed SI.evaluate(itp6, q6)
println("6D build: ", round(t_build, digits = 3), " s | eval: ", round(t_eval, digits = 3),
        " s | patches: ", length(itp6.locals))
```

Expected: `6D build:` ≤ 5.0 s, `eval:` ≤ 2.1 s, `patches: 262144`.

- [ ] **Step 3: Record and commit**

Append under "Benchmark results":

```markdown
### Task 7 final (all fixes applied)
| metric | baseline (Task 1 / review) | measured | budget | met |
|---|---|---|---|---|
| 3D build | ... / 0.399 s | ... | ≤ 0.16 s | ... |
| 3D evaluate | ... / 0.145 s | ... | ≤ 0.10 s | ... |
| 6D build | — / 8.34 s | ... | ≤ 5.0 s | ... |
| 6D evaluate | — / 3.27 s | ... | ≤ 2.1 s | ... |
```

```bash
git add docs/superpowers/plans/2026-07-05-perf-improvements.md
git commit -m "Record target-scale benchmark results for perf improvements"
```

```json:metadata
{"files": ["docs/superpowers/plans/2026-07-05-perf-improvements.md"], "verifyCommand": "3D+6D benchmark scripts via mcp__julia__julia_eval; budgets 0.16/0.10/5.0/2.1 s", "acceptanceCriteria": ["all four budgets met or miss documented with attribution", "results recorded in plan"], "modelTier": "standard"}
```

---

## Benchmark results

(Appended by Tasks 1 and 7.)
