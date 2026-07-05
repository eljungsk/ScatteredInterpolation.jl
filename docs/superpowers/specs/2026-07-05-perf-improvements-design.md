# Performance Improvements: PUM Hot Paths and Polynomial Assembly

**Date:** 2026-07-05
**Branch:** continues on `rbf-pum`
**Status:** Approved design, pending implementation plan
**Source:** `docs/perf-review-2026-07-05.md` (measured findings; all fixes below were
prototyped end-to-end in a benchmark environment with outputs verified identical)

## Goal

Implement Findings 1-5 of the performance review: remove the GC bottleneck from the
PUM build, thread the sequential evaluate phases, and fix two smaller allocation
hotspots. Measured prototype targets at n = 100,000:

| metric | current | target (prototype + 10 % slack) |
|---|---|---|
| 3D build | 0.399 s | ≤ 0.16 s |
| 3D evaluate (50k queries) | 0.145 s | ≤ 0.10 s |
| 6D build | 8.34 s | ≤ 5.0 s |
| 6D evaluate (50k queries) | 3.27 s | ≤ 2.1 s |

Out of scope (decided during design): the buffered-`lu!` fast path that bypasses
LinearSolve (Finding 1 step 3 — rejected to keep a single solve path), Shepard/legacy
findings 6-8, LOOCV buffer reuse, and amortized growth buffers in `addpoints!`.

## Decisions

- **OhMyThreads.jl becomes a direct dependency** (compat `0.8`). Project threading
  rule: any threaded region that writes memory — including disjoint slices of a
  shared array — uses OhMyThreads constructs (`tmap`, `@tasks`/`@local`,
  `index_chunks`), not raw `Threads.@threads`. All three existing `@threads` loops
  in `src/pum.jl` migrate as part of this work.
- **Single solve path through LinearSolve stays.** The allocation fix is
  `alias_A = true` on the weight-solve cache, opt-in per call site, so weights remain
  exactly identical to today's (verified: max abs diff 0.0 at 100k-point build).
- **No behavior changes.** Every fix was verified output-identical against the
  current implementation (same patch assignments, bit-identical evaluate results) at
  both 3D and 6D target scale. Existing tests must pass unchanged.

## Design

### 1. Dependency and threading migration

Add OhMyThreads to `Project.toml` deps and compat. Migrate the three
`Threads.@threads` loops in `src/pum.jl`:

- build solve loop (`pum.jl:444`) → `tmap` over `patchpoints` returning `locals`
  directly. This also removes the abstract `Vector{RadialBasisInterpolant}(undef, P)`
  preallocation and the `[l for l in locals]` narrowing step: `tmap`'s return type is
  already concrete for a homogeneous local-interpolant type.
- evaluate pass 2 (`pum.jl:503`) → `tmap(1:P)` returning `results` directly.
- `addpoints!` re-solve loop (`pum.jl:619`) → `tmap` over the affected patch list;
  write results back by index after the map (small list, sequential scatter is fine).

`withpinnedblas` wraps each region unchanged.

### 2. Solve-loop allocation fix (Finding 1)

`_initsolve` (`src/rbf.jl:294`) gains an opt-in aliasing flag:

```julia
_initsolve(A, alg; aliasA = false) =
    init(LinearProblem(A, zeros(eltype(A), size(A, 1))), alg; alias_A = aliasA)
```

- The plain-RBF and generalized-RBF weight-solve paths (`solveForWeights`) pass
  `aliasA = true`: LinearSolve factorizes the caller's kernel matrix in place instead
  of copying it (~41 KiB per patch saved). Neither path reads `A` after the first
  `solve!`; the generalized Schur-complement path only reuses the *factorization*.
- `interpolate(...; returnRBFmatrix = true)` must keep returning the assembled kernel
  matrix. Guard: in that branch, copy `A` before solving (the copy the old code paid
  unconditionally is now paid only when the caller asks for the matrix).
- `src/rippa.jl` call sites keep the default `aliasA = false`: `gatedloocvscore`
  computes `opnorm(A, 1)` *after* the solve, which reads `A` — aliasing would feed it
  LU garbage. Out of scope to restructure (Finding 7 was descoped).

Measured (steps 1-2 together, 3D): solve loop 0.241 s / 41 % GC / 1047 MiB →
0.199 s / 28 % GC / 564 MiB, weights exactly identical.

### 3. Threaded patch assignment (Finding 3)

Rewrite `assignpatches` (`src/pum.jl:78-99`):

1. Materialize all cell centers into a `D × ncells` matrix (drops the per-cell
   `collect(c)`).
2. `tmap` over `index_chunks(1:ncells; n = 8 * Threads.nthreads())`, each task
   running one batched `inrange(tree, allcenters[:, rng], radius)` on its column
   range; `reduce(vcat, parts)`.
3. Filter non-empty cells, `sort!` index lists in place, gather kept centers —
   as today.

Verified: identical `patchpoints` and `centers` at 3D and 6D. Standalone measurement
49 → 22-34 ms; folded into the whole-build target above.

### 4. Evaluate restructure (Finding 2)

`evaluate(::PartitionOfUnityInterpolant, ...)` (`src/pum.jl:459-541`), four changes:

1. **Threaded candidate query:** replace the single batched `inrange` call with
   `tmap` over `index_chunks(1:nq; n = 8 * Threads.nthreads())` +
   `reduce(vcat, parts)`. Measured 45 → 8.7 ms.
2. **No forced copy:** `convert(Matrix{T}, points)` instead of `Matrix{T}(points)`
   (constructor always copies; convert is a no-op for the common `Matrix{T}` input).
   Same in the uncovered-query fallback (`pum.jl:534-535`).
3. **Count-then-fill pass 1:** first pass counts candidate pairs per patch, allocate
   exact-size `patchq[p]`/`patchw[p]`, second pass fills; `resize!` down for pairs
   dropped by the `r < radius` / `ω > 0` guards. Replaces `push!` growth and the 2×P
   empty-vector setup (262k patches at 6D). Measured 31 → 12.3 ms.
4. **Pass 2 via `tmap`** (see §1) and drop the `Matrix{Tout}(reshape(v, ...))`
   conversion copy when `eltype(v) == Tout` (the typical case) — `reshape` alone does
   not copy. Keep the conversion for the mixed-eltype case.

Pass 3 (sequential scatter) and normalization stay unchanged.

Measured (all four combined): evaluate 0.145 → 0.086 s (3D), 3.27 → 1.90 s (6D),
outputs bit-identical.

### 5. `addpoints!` bounding box (Finding 4, step 1 only)

Store the data bounding box as fields on `PartitionOfUnityInterpolant`
(`lo::Vector{T}`, `hi::Vector{T}`, computed once in `interpolate` where the points
matrix is already materialized). `addpoints!` validates new points against the stored
box instead of scanning all stored points per call (0.69 ms of a 2.43 ms 10-point
insert at n = 100k). The box never widens: `addpoints!` only accepts points inside
it by definition. No growth-buffer change for `points`/`samples` (0.17 ms per call,
YAGNI).

This is a field addition to a `mutable struct`; the inner layout change is invisible
to users (the type is not exported API beyond what `interpolate` returns).

### 6. Polynomial assembly views (Finding 5)

`generateMultivariatePolynomial` (`src/rbf.jl:398`): add `@views` to
`P[:, position] .*= points[var, :]` so the row extraction stops materializing a
length-n vector per term-factor. Measured 2.3× (358 → 154 µs, 4.21 → 0.79 MiB at 3D
degree 3, n = 5000), identical output. Benefits generalized-RBF assembly, every
generalized `evaluate`, and the bordered LOOCV tuning path.

## Error handling

Unchanged. All argument validation in `interpolate`, `evaluate`, and `addpoints!`
runs before any mutation or threaded region, exactly as today. `tmap` propagates
task exceptions like `@threads` does.

## Testing and verification

- Existing test suite (`Pkg.test()`) must pass unchanged after every task — no
  behavior changes anywhere in this design.
- No new unit tests: every change is output-identical by construction, and the
  existing PUM tests cover `addpoints!` bounding-box rejection (the only behavior
  with restructured internals).
- Benchmark verification (per the `perf-plans-need-scale-benchmark` policy):
  - **Before hot-path work:** re-run the reproduction snippet from
    `docs/perf-review-2026-07-05.md` to confirm the baseline reproduces on the
    implementing machine (0.399 s / 0.145 s at 3D within noise).
  - **After all tasks:** re-measure 3D and 6D build + evaluate against the target
    table above; record results in the plan file.
