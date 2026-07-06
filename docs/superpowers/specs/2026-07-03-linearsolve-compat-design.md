# LinearSolve compatibility for all linear solves

**Date:** 2026-07-03
**Branch:** `linearsolve-v2` (from `master` @ `5dfc856`)
**Status:** Approved design, pending implementation plan

## Goal

Route every linear solve in ScatteredInterpolation through
[LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) so callers can select the
solver algorithm, while keeping current behavior as the default. Fix the matrix-RHS
handling that broke the previous attempt (multi-column data and the polynomial block
were passed as matrix right-hand sides to a solver that only accepted vectors).

## Decisions

- **Always LinearSolve.** Every solve goes through LinearSolve. `linsolve = nothing`
  (the default) means LinearSolve auto-selects its default algorithm. No native
  `\` / `factorize` fallback path is kept.
- **Matrix RHS handled by column-loop** over a reused solver cache — algorithm-agnostic,
  works for every solver family. This is the structural fix for the prior breakage.
- **Reusable cache** per matrix: `init(LinearProblem(A, b), alg)`, then `solve!` per RHS
  column. Preserves the current "factorize once, reuse" performance in the generalized path.
- **Tiny Schur system uses the default algorithm**, not the user's. The user's chosen
  algorithm drives the large N×N RBF system (the expensive one); the small `npoly×npoly`
  Schur complement `B` is always solved with LinearSolve's default (robust for the small,
  possibly indefinite block).
- **Minimum Julia 1.10 (LTS)** — modern LinearSolve forces the bump from the current
  `julia = "1"`. CI matrix `lts` + `1` (already master's shape).
- **Tested algorithm contract** (CI guards against version drift): `nothing`,
  `LUFactorization()`, `IterativeSolversJL_GMRES()`, `KrylovJL_GMRES()`.

## Solve sites (current `src/rbf.jl`)

1. **Plain RBF** — `solveForWeights(...RadialBasisFunction...)`: `w = A\samples` (line ~274).
2. **Generalized RBF** — `solveForWeights(...Generalized...)`: blocked system (lines ~285–289):
   ```julia
   Af = factorize(A)
   B  = -P'*(Af\P)
   E  = B\(P'*(Af\samples))
   w  = Af\(samples + P*E)
   ```
   `A` is reused across `Af\P`, `Af\samples`, and the final solve. `samples` and `P` are
   frequently matrices (vector-valued data; polynomial block).

## Design

### 1. Public API

Add a `linsolve` keyword to `interpolate`, alongside `metric`, `returnRBFmatrix`, `smooth`:

```julia
interpolate(rbf, points, samples; ..., linsolve = nothing)
```

`nothing` → LinearSolve default algorithm. The value is forwarded to `solveForWeights`.

### 2. Solve helper (new internal, `src/rbf.jl`)

Reusable cache plus algorithm-agnostic matrix-RHS handling via a column-loop:

```julia
# Reusable solver cache for matrix A and algorithm alg (nothing = LinearSolve default).
_initsolve(A, alg) = init(LinearProblem(A, zeros(eltype(A), size(A, 1))), alg)

# Solve A*X = B reusing the cache. B a vector or matrix; returns X with matching shape.
_solve!(cache, b::AbstractVector) = copy(solve!(_setb!(cache, b)).u)
_solve!(cache, B::AbstractMatrix) = reduce(hcat, (_solve!(cache, B[:, j]) for j in axes(B, 2)))
```

`_setb!` sets the cache's `b` (via LinearSolve's documented cache-mutation API) without
invalidating the cached factorization, so repeated `_solve!` calls on one cache re-use the
factorization. Exact mutation call to be confirmed against the pinned LinearSolve version
during implementation.

### 3. Plain RBF path

```julia
cache = _initsolve(A, linsolve)
w = _solve!(cache, samples)
RBFInterpolant(w, points, rbf, metric)
```

### 4. Generalized RBF path

```julia
P      = getPolynomial(rbf, points)
cacheA = _initsolve(A, linsolve)          # user alg; factorization cached on 1st solve!, reused
AinvP  = _solve!(cacheA, P)
Ainvs  = _solve!(cacheA, samples)
B      = -P' * AinvP
cacheB = _initsolve(B, nothing)           # tiny Schur system: always LinearSolve default
E      = _solve!(cacheB, P' * Ainvs)
w      = _solve!(cacheA, samples + P * E)   # reuse cacheA (no re-factorization)
λ      = -E
GeneralizedRBFInterpolant(w, λ, points, rbf, metric)
```

Reusing `cacheA` across `P`, `samples`, and the final RHS reproduces the current
"factorize once" behavior.

### 5. Dependencies / compat (`Project.toml`)

- `[deps]` + `[compat]`: add `LinearSolve` (modern major — exact bound pinned during
  implementation), bump `julia = "1.10"`.
- Test extras: add `LinearSolve` and `IterativeSolvers` (the latter is required for the
  `IterativeSolversJL_GMRES` algorithm extension; `KrylovJL_GMRES` and `LUFactorization`
  ship with LinearSolve).
- Version: `0.3.6 → 0.4.0` (breaking — Julia floor bump and default solve-path change).

### 6. Tests

Parametrize a representative subset over the four algorithms (`nothing`,
`LUFactorization()`, `IterativeSolversJL_GMRES()`, `KrylovJL_GMRES()`):

- plain RBF (`Gaussian`) reproduces data at sample points,
- generalized RBF (`GeneralizedPolyharmonic`) reproduces data and a linear field,
- **multi-column samples** (matrix RHS) — explicit regression guard for the old bug.

Iterative solvers (GMRES) will not reach direct-solver precision. Acceptance uses a
per-algorithm tolerance: direct solvers checked with `≈`; iterative solvers with a looser
`atol`. Keep parametrized tests on a small point set for speed.

### 7. CI

Matrix `version: [lts, 1]` (already master's shape). The `julia = "1.10"` compat bump
enforces the floor.

## Out of scope

- No changes to RBF definitions, metrics, or the `evaluate` path.
- No new solver algorithms beyond exposing LinearSolve's.
- No unrelated refactoring of `src/rbf.jl`.
