# LinearSolve Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every linear solve in ScatteredInterpolation through LinearSolve.jl with a user-selectable `linsolve` algorithm, keeping current behavior as the default.

**Architecture:** Add a `linsolve` keyword to `interpolate`, forwarded to `solveForWeights`. Introduce a small internal helper (`_initsolve`/`_setb!`/`_solve!`) that wraps a reusable LinearSolve cache and handles matrix right-hand sides by looping over columns (the structural fix for the previous attempt's breakage). The generalized-RBF path reuses one cache for the large N×N system and solves the tiny Schur complement with LinearSolve's default algorithm.

**Tech Stack:** Julia, LinearSolve.jl 2.x, IterativeSolvers.jl 0.9 (test-only, for the `IterativeSolversJL_GMRES` extension), Krylov (transitive via LinearSolve).

**User decisions (already made):**
- "Always LinearSolve" — every solve routes through LinearSolve; `linsolve = nothing` → LinearSolve default algorithm. No native `\`/`factorize` fallback kept.
- Tested algorithm contract: `nothing`, `LUFactorization()`, `IterativeSolversJL_GMRES()`, `KrylovJL_GMRES()`.
- Minimum Julia 1.10 (LTS); CI matrix `lts` + `1`.
- Architecture C: user algorithm drives the big N×N RBF system; the tiny `npoly×npoly` Schur system always uses LinearSolve's default.
- Iterative solvers checked with a looser tolerance than direct solvers (they don't reach machine precision).

**Verified facts (checked against LinearSolve 2.16.2 in a scratch project):**
- `init(LinearProblem(A, b), nothing)` selects the default algorithm and solves correctly.
- `cache.b = newb; solve!(cache)` reuses the cached factorization (no re-factorization).
- Column-loop matrix RHS reconstructs `A \ B` correctly.
- All four algorithms solve a small SPD system. **`IterativeSolversJL_GMRES()` only exists once `using IterativeSolvers` is executed** — the constructor lives in the `LinearSolveIterativeSolversExt` package extension. Declaring the dependency is not enough; the test file must `using IterativeSolvers`.

---

## File Structure

- `Project.toml` — add `LinearSolve` (and test-only `IterativeSolvers`) deps/compat, bump `julia` floor, bump package version.
- `src/ScatteredInterpolation.jl` — add `LinearSolve` to the module `using` line; update the `interpolate` docstring.
- `src/rbf.jl` — add the solve helpers; add `linsolve` kwarg to `interpolate`; rewrite both `solveForWeights` methods.
- `test/rbf.jl` — algorithm-parametrized tests + multi-column matrix-RHS regression guard.
- `test/runtests.jl` — `using IterativeSolvers` so the GMRES extension loads.

---

### Task 1: Add LinearSolve dependency and bump compat

**Goal:** Declare LinearSolve (and test-only IterativeSolvers), raise the Julia floor to 1.10, and bump the package version.

**Files:**
- Modify: `Project.toml`

**Acceptance Criteria:**
- [ ] `LinearSolve` in `[deps]` and `[compat]`
- [ ] `IterativeSolvers` in `[extras]`, `[compat]`, and the test target
- [ ] `julia = "1.10"` in `[compat]`
- [ ] `version = "0.4.0"`
- [ ] `julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'` succeeds

**Verify:** `julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); using ScatteredInterpolation, LinearSolve'` → exits 0

**Steps:**

- [ ] **Step 1: Rewrite `Project.toml`**

```toml
name = "ScatteredInterpolation"
uuid = "3f865c0f-6dca-5f4d-999b-29fe1e7e3c92"
version = "0.4.0"

[deps]
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LinearSolve = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"

[compat]
Combinatorics = "1"
Distances = "0.9,0.10"
LinearSolve = "2"
NearestNeighbors = "0.4"
julia = "1.10"

[extras]
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat.IterativeSolvers]
version = "0.9"

[targets]
test = ["Test", "Distances", "IterativeSolvers"]
```

Note: `IterativeSolvers` compat is written as a `[compat.IterativeSolvers]` section because a bare `IterativeSolvers = "0.9"` line under `[compat]` is equally valid — either form is acceptable; keep it consistent with the rest of the file if the reviewer prefers the inline form.

- [ ] **Step 2: Resolve and instantiate**

Run: `julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'`
Expected: completes without error; a `Manifest.toml` is written (gitignored).

- [ ] **Step 3: Smoke-load**

Run: `julia --project=. -e 'using ScatteredInterpolation, LinearSolve; println("ok")'`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add Project.toml
git commit -m "Add LinearSolve dependency, bump Julia floor to 1.10"
```

---

### Task 2: Add the LinearSolve solve helpers

**Goal:** Add internal `_initsolve`, `_setb!`, and `_solve!` helpers that wrap a reusable LinearSolve cache and solve vector or matrix right-hand sides.

**Files:**
- Modify: `src/ScatteredInterpolation.jl:3` (add `LinearSolve` to the `using` line)
- Modify: `src/rbf.jl` (add helpers directly above the `solveForWeights` definitions, ~line 270)
- Test: `test/rbf.jl` (new `@testset "Solve helpers"`)

**Acceptance Criteria:**
- [ ] `_initsolve(A, nothing)` builds a cache whose first `_solve!` solves `A x = b`
- [ ] `_solve!(cache, b::Vector)` returns `x` with `A*x ≈ b`
- [ ] `_solve!(cache, B::Matrix)` returns `X` with `A*X ≈ B` (column-loop)
- [ ] A second `_solve!` on the same cache with a different RHS still solves correctly (factorization reuse)

**Verify:** `julia --project=. -e 'include("test/runtests.jl")'` → the `Solve helpers` testset passes

**Steps:**

- [ ] **Step 1: Add `LinearSolve` to the module `using` line**

In `src/ScatteredInterpolation.jl`, change line 3 from:

```julia
using Distances, NearestNeighbors, Combinatorics, LinearAlgebra
```

to:

```julia
using Distances, NearestNeighbors, Combinatorics, LinearAlgebra, LinearSolve
```

- [ ] **Step 2: Write the failing test**

Add to `test/rbf.jl`, inside the top-level `@testset "RBF" begin ... end` block (e.g. right after the `Constructors` testsets):

```julia
    @testset "Solve helpers" begin
        A = [4.0 1.0; 1.0 3.0]
        cache = ScatteredInterpolation._initsolve(A, nothing)

        b = [1.0, 2.0]
        x = ScatteredInterpolation._solve!(cache, b)
        @test A * x ≈ b

        # Reuse the same cache (and its factorization) for a new RHS
        b2 = [3.0, 4.0]
        x2 = ScatteredInterpolation._solve!(cache, b2)
        @test A * x2 ≈ b2

        # Matrix RHS solved column by column
        B = [1.0 3.0; 2.0 4.0]
        X = ScatteredInterpolation._solve!(cache, B)
        @test A * X ≈ B
    end
```

- [ ] **Step 2b: Run test to verify it fails**

Run: `julia --project=. -e 'include("test/runtests.jl")'`
Expected: FAIL — `UndefVarError: _initsolve not defined` (helpers not written yet).

- [ ] **Step 3: Write the helpers**

In `src/rbf.jl`, immediately above the first `solveForWeights` definition (currently ~line 271), add:

```julia
# --- LinearSolve helpers -------------------------------------------------------
# Build a reusable LinearSolve cache for matrix `A` and algorithm `alg`
# (`nothing` selects LinearSolve's default algorithm). The factorization is
# computed on the first solve and reused for subsequent right-hand sides.
_initsolve(A, alg) = init(LinearProblem(A, zeros(eltype(A), size(A, 1))), alg)

# Point the cache at a new right-hand side without invalidating the cached
# factorization, then return the cache for chaining.
function _setb!(cache, b)
    cache.b = b
    return cache
end

# Solve `A * x = b` for a vector RHS, reusing the cache's factorization.
_solve!(cache, b::AbstractVector) = copy(solve!(_setb!(cache, b)).u)

# Solve `A * X = B` for a matrix RHS, column by column, reusing the factorization.
_solve!(cache, B::AbstractMatrix) =
    reduce(hcat, (_solve!(cache, B[:, j]) for j in axes(B, 2)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'include("test/runtests.jl")'`
Expected: `Solve helpers` testset PASSES.

- [ ] **Step 5: Commit**

```bash
git add src/ScatteredInterpolation.jl src/rbf.jl test/rbf.jl
git commit -m "Add reusable LinearSolve helpers with matrix-RHS support"
```

---

### Task 3: Route interpolate and solveForWeights through LinearSolve

**Goal:** Add the `linsolve` kwarg to `interpolate` and rewrite both `solveForWeights` methods to use the helpers, keeping default behavior identical.

**Files:**
- Modify: `src/rbf.jl:210-214` (interpolate signature), `src/rbf.jl:226` (call site), `src/rbf.jl:271-293` (both `solveForWeights` methods)
- Modify: `src/ScatteredInterpolation.jl:26` (interpolate docstring — mention `linsolve`)
- Test: `test/rbf.jl` (existing evaluation tests act as the default-path regression check)

**Acceptance Criteria:**
- [ ] `interpolate(...; linsolve = nothing)` accepted and forwarded
- [ ] Plain-RBF `solveForWeights` uses `_initsolve`/`_solve!`
- [ ] Generalized-RBF `solveForWeights` reuses one cache for `A` and solves the Schur system with the default algorithm
- [ ] The full existing test suite passes unchanged with the default `linsolve`

**Verify:** `julia --project=. -e 'include("test/runtests.jl")'` → all pre-existing testsets pass

**Steps:**

- [ ] **Step 1: Add the `linsolve` kwarg to `interpolate`**

In `src/rbf.jl`, change the signature (lines ~210-214) from:

```julia
function interpolate(rbf::Union{T, AbstractVector{T}} where T <: AbstractRadialBasisFunction,
                     points::AbstractArray{<:Real,2},
                     samples::AbstractArray{<:Number,N};
                     metric = Euclidean(), returnRBFmatrix::Bool = false,
                     smooth::Union{S, AbstractVector{S}} = false) where {N} where {S<:Number}
```

to:

```julia
function interpolate(rbf::Union{T, AbstractVector{T}} where T <: AbstractRadialBasisFunction,
                     points::AbstractArray{<:Real,2},
                     samples::AbstractArray{<:Number,N};
                     metric = Euclidean(), returnRBFmatrix::Bool = false,
                     smooth::Union{S, AbstractVector{S}} = false,
                     linsolve = nothing) where {N} where {S<:Number}
```

- [ ] **Step 2: Forward `linsolve` at the call site**

In `src/rbf.jl` line ~226, change:

```julia
    itp = solveForWeights(A, points, samples, rbf, metric)
```

to:

```julia
    itp = solveForWeights(A, points, samples, rbf, metric; linsolve = linsolve)
```

- [ ] **Step 3: Rewrite the plain-RBF `solveForWeights`**

Replace (lines ~271-276):

```julia
@inline function solveForWeights(A, points, samples,
                                    rbf::Union{T, AbstractVector{T}} where T <: RadialBasisFunction,
                                    metric)
    w = A\samples
    RBFInterpolant(w, points, rbf, metric)
end
```

with:

```julia
@inline function solveForWeights(A, points, samples,
                                    rbf::Union{T, AbstractVector{T}} where T <: RadialBasisFunction,
                                    metric; linsolve = nothing)
    cache = _initsolve(A, linsolve)
    w = _solve!(cache, samples)
    RBFInterpolant(w, points, rbf, metric)
end
```

- [ ] **Step 4: Rewrite the generalized-RBF `solveForWeights`**

Replace (lines ~277-293):

```julia
@inline function solveForWeights(A, points, samples,
                                    rbf::Union{T, AbstractVector{T}} where T <: Union{GeneralizedRadialBasisFunction, RadialBasisFunction},
                                    metric)
    # Use the maximum degree among the generalized RBF:s
    P = getPolynomial(rbf, points)

    # Solve for the weights and polynomial coefficients
    # We end up with a blocked system, so we don't have to form the full matrix
    Af = factorize(A)
    B = -P'*(Af\P)
    E = B\(P'*(Af\samples))

    w = Af\(samples + P*E)
    λ = -E

    GeneralizedRBFInterpolant(w, λ, points, rbf, metric)
end
```

with:

```julia
@inline function solveForWeights(A, points, samples,
                                    rbf::Union{T, AbstractVector{T}} where T <: Union{GeneralizedRadialBasisFunction, RadialBasisFunction},
                                    metric; linsolve = nothing)
    # Use the maximum degree among the generalized RBF:s
    P = getPolynomial(rbf, points)

    # Blocked (Schur-complement) system. One reusable cache factorizes A once and
    # is reused for every RHS (P, samples, and the final combined RHS); the small
    # npoly×npoly Schur system uses LinearSolve's default algorithm.
    cacheA = _initsolve(A, linsolve)
    AinvP  = _solve!(cacheA, P)
    Ainvs  = _solve!(cacheA, samples)

    B = -P' * AinvP
    cacheB = _initsolve(B, nothing)
    E = _solve!(cacheB, P' * Ainvs)

    w = _solve!(cacheA, samples + P * E)
    λ = -E

    GeneralizedRBFInterpolant(w, λ, points, rbf, metric)
end
```

- [ ] **Step 5: Update the `interpolate` docstring**

In `src/ScatteredInterpolation.jl`, update the signature line (line ~26) of the docstring and add a `linsolve` sentence. Change:

```julia
    interpolate(method, points, samples; metric = Euclidean(), returnRBFmatrix = false, smooth = false)
```

to:

```julia
    interpolate(method, points, samples; metric = Euclidean(), returnRBFmatrix = false, smooth = false, linsolve = nothing)
```

and add this sentence to the docstring body (after the `returnRBFmatrix` sentence):

```
`linsolve` selects the linear-solver algorithm from `LinearSolve.jl` used to solve for the
weights; `nothing` uses LinearSolve's default algorithm.
```

- [ ] **Step 6: Run the full suite (default path regression)**

Run: `julia --project=. -e 'include("test/runtests.jl")'`
Expected: all pre-existing testsets pass (RBF evaluation, generalized RBF, multi-column samples, metric, mixed RBF, Shepard, NearestNeighbor).

- [ ] **Step 7: Commit**

```bash
git add src/rbf.jl src/ScatteredInterpolation.jl
git commit -m "Route all RBF linear solves through LinearSolve"
```

---

### Task 4: Algorithm-contract tests

**Goal:** Prove `interpolate` works across all four supported algorithms, including the multi-column matrix-RHS case that broke the previous attempt.

**Files:**
- Modify: `test/runtests.jl:1` (add `using IterativeSolvers`)
- Modify: `test/rbf.jl` (new `@testset "Linear solver algorithms"`)

**Acceptance Criteria:**
- [ ] Tests parametrize over `nothing`, `LUFactorization()`, `IterativeSolversJL_GMRES()`, `KrylovJL_GMRES()`
- [ ] Plain RBF (`Gaussian`) reproduces the data at sample points for every algorithm
- [ ] Generalized RBF (`GeneralizedPolyharmonic`) reproduces a linear field for every algorithm
- [ ] Multi-column samples (matrix RHS) reproduce each column for every algorithm
- [ ] Direct solvers use `≈`; iterative solvers use `atol = 1e-6`

**Verify:** `julia --project=. -e 'include("test/runtests.jl")'` → the `Linear solver algorithms` testset passes

**Steps:**

- [ ] **Step 1: Load IterativeSolvers so the GMRES extension is available**

In `test/runtests.jl`, change line 1 from:

```julia
using ScatteredInterpolation, Test, LinearAlgebra
```

to:

```julia
using ScatteredInterpolation, Test, LinearAlgebra, LinearSolve, IterativeSolvers
```

(`using IterativeSolvers` is required: `IterativeSolversJL_GMRES` is defined only in LinearSolve's IterativeSolvers extension, which loads when IterativeSolvers is imported.)

- [ ] **Step 2: Write the algorithm-contract test**

Add to `test/rbf.jl`, inside the top-level `@testset "RBF" begin ... end` block (e.g. after the `Multi-column samples` testset). It reuses the file-level `arrayPoints`/`data` and the deterministic `perturbation` helper from `runtests.jl`:

```julia
    @testset "Linear solver algorithms" begin
        # `nothing` and direct factorizations reach machine precision; iterative
        # solvers only converge to a tolerance, so check them with a looser atol.
        directAlgs   = (nothing, LUFactorization())
        iterativeAlgs = (IterativeSolversJL_GMRES(), KrylovJL_GMRES())

        # A linear field for the generalized-RBF reproduction check
        linear = [2 + 3 * arrayPoints[1, i] - arrayPoints[2, i] for i in 1:size(arrayPoints, 2)]
        query  = [0.3; 0.7]
        truth  = 2 + 3 * 0.3 - 0.7

        # Multi-column (matrix RHS) data — the regression guard for the old breakage
        multiData = hcat(data, 2 .* data, -data)

        @testset "algorithm = $(alg === nothing ? "default" : nameof(typeof(alg)))" for
                (alg, tol) in (((a, nothing) for a in directAlgs)...,
                               ((a, 1e-6) for a in iterativeAlgs)...)

            approxeq(x, y) = tol === nothing ? isapprox(x, y) : isapprox(x, y; atol = tol)

            # Plain RBF reproduces the data at the sample points
            itp = interpolate(Gaussian(2), arrayPoints, data; linsolve = alg)
            @test approxeq(evaluate(itp, arrayPoints), data)

            # Generalized RBF reproduces a linear field
            itpGen = interpolate(GeneralizedPolyharmonic(3, 2), arrayPoints, linear; linsolve = alg)
            @test approxeq(evaluate(itpGen, query)[1], truth)

            # Multi-column samples: each column recovered independently (matrix RHS)
            itpMulti = interpolate(Gaussian(2), arrayPoints, multiData; linsolve = alg)
            ev = evaluate(itpMulti, arrayPoints)
            @test size(ev) == size(multiData)
            @test approxeq(ev, multiData)
        end
    end
```

- [ ] **Step 3: Run the suite**

Run: `julia --project=. -e 'include("test/runtests.jl")'`
Expected: the `Linear solver algorithms` testset passes for all four algorithms (12 checks: 3 per algorithm × 4 algorithms), and the full suite is green.

- [ ] **Step 4: Commit**

```bash
git add test/runtests.jl test/rbf.jl
git commit -m "Test all four LinearSolve algorithms incl. matrix RHS"
```

---

## Self-Review

**Spec coverage:**
- API `linsolve` kwarg → Task 3. ✓
- Solve helper (reusable cache + column-loop matrix RHS) → Task 2. ✓
- Plain RBF path → Task 3 Step 3. ✓
- Generalized RBF path (reuse cacheA, default-alg Schur) → Task 3 Step 4. ✓
- Deps/compat/version/test extras → Task 1. ✓
- Tests over 4 algorithms + multi-column guard + per-alg tolerance → Task 4. ✓
- CI matrix `lts` + `1` → already master's `.github/workflows/ci.yml`; the `julia = "1.10"` compat bump enforces the floor, no workflow edit required. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type/name consistency:** `_initsolve`, `_setb!`, `_solve!` defined in Task 2 and used identically in Task 3. `linsolve` kwarg name consistent across `interpolate` and both `solveForWeights` methods. ✓
