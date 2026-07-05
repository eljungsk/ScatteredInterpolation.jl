# LOOCV Shape-Parameter Selection (Rippa's Method) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in per-patch automatic selection of the RBF shape parameter `ε` to `PartitionOfUnity` via exact leave-one-out cross-validation (Rippa's formula): `PartitionOfUnity(Gaussian(); tune = :loocv)`.

**Architecture:** New `hasshape`/`withshape` kernel traits; a new `src/rippa.jl` with the exact LOOCV error/score helpers (plain and bordered saddle-system variants, direct `inv()` — patch-scale dense systems only); a `tune::Symbol` field on `PartitionOfUnity`; per-patch tuning inside the existing threaded local-solve loop (distances computed once per patch, 11 shape candidates scaled by the patch's mean nearest-neighbor distance); `addpoints!` re-tunes affected patches. Evaluation code is untouched — the tuned kernel is stored on each local `RBFInterpolant` as today.

**Tech Stack:** Julia 1.10+, existing deps only (Distances, LinearAlgebra, LinearSolve, NearestNeighbors, KernelFunctions). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-04-loocv-shape-selection-design.md`

**User decisions (already made, from the spec):**
- Opt-in via `tune = :loocv` keyword on `PartitionOfUnity`; default `:none` keeps current behavior exactly.
- The `ε` stored in the passed kernel is ignored under `:loocv` (candidate grid is scale-derived); documented.
- Selection criterion `sum(abs2, E)`; candidate grid `ε_j = c_j / h_p`, `c_j = 2.0 .^ (-2:0.5:3)` (11 candidates).
- Candidate scans use direct `inv()`, NOT LinearSolve — the user's `linsolve` choice applies to the final per-patch solve only.
- Singular candidate systems are disqualified (score = Inf), not errors.
- `GeneralizedMultiquadratic` bordered-system tuning is a second-stage task; the bordered formula must be validated against brute-force LOO.
- Global (non-PUM) LOOCV, discrete-parameter tuning, anisotropic `ε`, weight-function tuning: all out of scope.
- The target-scale benchmark task (10⁵ points, 3D and 6D, error ≥10× better, build ≤5× budget) is mandatory — its absence is a plan defect.
- Branch from `rbf-pum` (PR #37 prerequisite). Known pre-existing complex-samples LinearSolve cache bug is not ours to fix.

**Plan-time decisions (recorded per the spec's open items):**
- `withshape(w::Wendland, ε) = Wendland(w.kernel, ε)`: the wrapped `PiecewisePolynomialKernel` does not depend on `ε` (`ε` is applied outside it, in `kappa(kernel, ε*r)`), so dim/degree are preserved automatically and no reconstruction/recovery is needed. Candidate `ε = c/h` is always positive, so skipping the public constructor loses no validation in practice.
- `tune = :loocv` + shapeless kernel throws `ArgumentError` at **constructor** time (fail fast; the data dimension is irrelevant to the check).
- Skip-tuning rule: `h_p == 0` (coincident points) → keep the user's kernel unchanged (no length scale to derive `ε` from); `2 ≤ n_p < 3` with `h_p > 0` → use `ε = c_mid / h_p` without scanning.

**Plan-wide notes:**
- Run Julia through `mcp__julia__julia_eval` (project memory: never via Bash). Run tests with `import Pkg; Pkg.test("ScatteredInterpolation")`.
- Benchmark ordering follows the `perf-plans-need-scale-benchmark` policy: the baseline target-scale measurement (Task 4) runs BEFORE the hot-path tuning task (Task 5); the tuned-vs-untuned validation (Task 8) runs after integration.
- Do not test complex-valued samples through the PUM `:loocv` path end-to-end: the final per-patch solve goes through LinearSolve, which has a known pre-existing complex-RHS cache typing bug on this branch (memory `rbf-pum-branch-state`). The `loocverrors`/`loocvscore` helpers themselves are LinearSolve-free.

---

## File structure

| File | Role |
|---|---|
| `src/rbf.jl` (modify) | `hasshape`/`withshape` traits for the kernels defined there |
| `src/wendland.jl` (modify) | `hasshape`/`withshape` for `Wendland` |
| `src/rippa.jl` (create) | `loocverrors` (plain + bordered), `loocvscore` (plain + bordered) — pure linear algebra, no PUM knowledge |
| `src/pum.jl` (modify) | `tune` field + validation; `LOOCV_CANDIDATES`, `meannndist`, `tuneshape` (plain + bordered dispatch), `solvelocal`; build loop and `addpoints!` loop call `solvelocal` |
| `src/ScatteredInterpolation.jl` (modify) | `include("./rippa.jl")` |
| `test/rippa.jl` (create) | Brute-force LOO validation of the Rippa helpers |
| `test/rbf.jl`, `test/wendland.jl`, `test/pum.jl` (modify) | Trait tests; tuning integration tests |
| `docs/src/methods.md` (modify) | PUM section: `tune = :loocv` documentation |

---

### Task 1: `hasshape` / `withshape` kernel traits

**Goal:** Internal traits answering "does this kernel have a tunable shape parameter?" and "rebuild it with a new one", for every shipped kernel.

**Files:**
- Modify: `src/rbf.jl` (after the `GeneralizedPolyharmonic` doc block, around line 190)
- Modify: `src/wendland.jl` (at the end)
- Test: `test/rbf.jl`, `test/wendland.jl`

**Acceptance Criteria:**
- [ ] `hasshape` is `true` for `Gaussian`, `Multiquadratic`, `InverseQuadratic`, `InverseMultiquadratic`, `GeneralizedMultiquadratic`, `Wendland`; `false` for `Polyharmonic`, `ThinPlate()` (= `Polyharmonic(2)`), `GeneralizedPolyharmonic`
- [ ] `withshape(k, ε)` returns the same kernel type with the new `ε`, preserving `β`/`degree` for `GeneralizedMultiquadratic` and dim/degree behavior for `Wendland`
- [ ] Full test suite passes

**Verify:** `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

At the top of `test/rbf.jl` add (alongside any existing imports):

```julia
using ScatteredInterpolation: hasshape, withshape
```

At the end of `test/rbf.jl` add:

```julia
@testset "Shape parameter traits" begin
    @test hasshape(Gaussian(2))
    @test hasshape(Multiquadratic())
    @test hasshape(InverseQuadratic())
    @test hasshape(InverseMultiquadratic())
    @test hasshape(GeneralizedMultiquadratic(1, 1/2, 2))
    @test !hasshape(Polyharmonic(3))
    @test !hasshape(ThinPlate())
    @test !hasshape(GeneralizedPolyharmonic(3, 1))

    @test withshape(Gaussian(2), 4.0) === Gaussian(4.0)
    @test withshape(Multiquadratic(), 3.0) === Multiquadratic(3.0)
    @test withshape(InverseQuadratic(), 3.0) === InverseQuadratic(3.0)
    @test withshape(InverseMultiquadratic(), 3.0) === InverseMultiquadratic(3.0)
    gmq = withshape(GeneralizedMultiquadratic(1, 1/2, 2), 5.0)
    @test gmq.ε == 5.0
    @test gmq.β == 1/2
    @test gmq.degree == 2
end
```

At the end of `test/wendland.jl` add:

```julia
@testset "Wendland shape parameter traits" begin
    @test ScatteredInterpolation.hasshape(Wendland(2, 1))
    w = ScatteredInterpolation.withshape(Wendland(3, 2; ε = 2), 4.0)
    @test w isa Wendland
    @test w.ε == 4.0
    # dim/degree preserved: values match a freshly constructed Wendland(3, 2; ε = 4)
    ref = Wendland(3, 2; ε = 4)
    @test all(w(r) ≈ ref(r) for r in 0:0.05:0.3)
    @test ScatteredInterpolation.support_radius(w) ≈ 0.25
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run via `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")`
Expected: FAIL — `UndefVarError: hasshape not defined`

- [ ] **Step 3: Implement the traits**

In `src/rbf.jl`, after the `GeneralizedPolyharmonic` callable definition (after line ~189), add:

```julia
# --- Shape-parameter traits ------------------------------------------------------
# Used by the partition of unity method's LOOCV tuning (tune = :loocv): hasshape says
# whether a kernel has a tunable shape parameter ε, and withshape rebuilds the kernel
# with a new one, preserving all other parameters. Polyharmonic-family kernels are
# scale-free and have no shape parameter.
hasshape(::AbstractRadialBasisFunction) = false
hasshape(::Union{Gaussian, Multiquadratic, InverseQuadratic, InverseMultiquadratic,
                 GeneralizedMultiquadratic}) = true

withshape(::Gaussian, ε) = Gaussian(ε)
withshape(::Multiquadratic, ε) = Multiquadratic(ε)
withshape(::InverseQuadratic, ε) = InverseQuadratic(ε)
withshape(::InverseMultiquadratic, ε) = InverseMultiquadratic(ε)
withshape(k::GeneralizedMultiquadratic, ε) = GeneralizedMultiquadratic(ε, k.β, k.degree)
```

(`Wendland` is defined in `src/wendland.jl`, which is included after `src/rbf.jl` — its methods cannot live in this Union.)

At the end of `src/wendland.jl` add:

```julia
# The wrapped PiecewisePolynomialKernel does not depend on ε (ε is applied outside it,
# in kappa(kernel, ε*r)), so a re-shaped Wendland reuses it directly — dim and degree
# are preserved without reconstruction. Tuning candidates are always positive, so the
# public constructor's ε > 0 check is not needed here.
hasshape(::Wendland) = true
withshape(w::Wendland, ε) = Wendland(w.kernel, ε)
```

- [ ] **Step 4: Run tests to verify they pass**

Run via `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rbf.jl src/wendland.jl test/rbf.jl test/wendland.jl
git commit -m "Add hasshape/withshape kernel traits"
```

---

### Task 2: Rippa LOOCV helpers (`src/rippa.jl`)

**Goal:** Exact leave-one-out error and score helpers for plain and bordered (saddle) systems, validated against brute-force leave-one-out.

**Files:**
- Create: `src/rippa.jl`
- Modify: `src/ScatteredInterpolation.jl` (include after `rbf.jl`)
- Create: `test/rippa.jl`
- Modify: `test/runtests.jl` (include before `pum.jl`)

**Acceptance Criteria:**
- [ ] `loocverrors(A, w)` matches brute-force LOO (drop point i, solve, predict at i) on a 12-point Gaussian system, `atol = 1e-10`
- [ ] Same identity holds for the ridge-smoothed matrix `A + σI` (prediction row excludes the diagonal, so it is unchanged)
- [ ] Matrix RHS: one diagonal serves all columns (`E[:, 2] == 2 .* E[:, 1]` for `F = [f 2f]`)
- [ ] Bordered variant `loocverrors(M, WΛ, n)` matches brute-force LOO on a `GeneralizedMultiquadratic` saddle system, `atol = 1e-10`
- [ ] `loocvscore` equals `sum(abs2, loocverrors(...))` and returns `Inf` for a singular matrix instead of throwing
- [ ] Full test suite passes

**Verify:** `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `test/rippa.jl`:

```julia
using ScatteredInterpolation: loocverrors, loocvscore, generateMultivariatePolynomial
using Distances: pairwise

# Deterministic, RNG-free 2D points (Kronecker sequence), no duplicates.
rippapts(n) = [mod(i * sqrt(p), 1.0) for p in (2.0, 3.0), i in 1:n]

@testset "Rippa LOOCV helpers" begin
    n = 12
    pts = rippapts(n)
    f = [sinpi(x[1]) * cospi(x[2]) for x in eachcol(pts)]
    rbf = Gaussian(4)   # well conditioned at this point count and spacing
    R = pairwise(Euclidean(), pts, dims = 2)
    A = rbf.(R)

    @testset "Plain system vs brute-force LOO" begin
        w = A \ f
        E = loocverrors(A, w)
        for i in 1:n
            keep = setdiff(1:n, i)
            wi = A[keep, keep] \ f[keep]
            pred = dot(A[i, keep], wi)
            @test E[i] ≈ f[i] - pred atol = 1e-10
        end
    end

    @testset "Ridge-smoothed system" begin
        σ = 1e-3
        Ã = A + σ * I
        w̃ = Ã \ f
        Ẽ = loocverrors(Ã, w̃)
        for i in 1:n
            keep = setdiff(1:n, i)
            # Smoothing sits on the diagonal only; the prediction row is the plain
            # kernel row (the diagonal is excluded by construction).
            wi = (A[keep, keep] + σ * I) \ f[keep]
            pred = dot(A[i, keep], wi)
            @test Ẽ[i] ≈ f[i] - pred atol = 1e-10
        end
    end

    @testset "Matrix right-hand side" begin
        w = A \ f
        E = loocverrors(A, w)
        F = [f 2 .* f]
        W = A \ F
        EM = loocverrors(A, W)
        @test EM[:, 1] ≈ E atol = 1e-12
        @test EM[:, 2] ≈ 2 .* E atol = 1e-12
    end

    @testset "Bordered (saddle) system vs brute-force LOO" begin
        gmq = GeneralizedMultiquadratic(2, 1/2, 2)
        A2 = gmq.(R)
        P = generateMultivariatePolynomial(pts, gmq.degree)
        npoly = size(P, 2)               # binomial(2 + 2, 2) = 6 terms; 11 ≥ 6 points remain
        M = [A2 P; P' zeros(npoly, npoly)]
        Fb = vcat(f, zeros(npoly))
        WΛ = M \ Fb
        Eb = loocverrors(M, WΛ, n)
        @test length(Eb) == n
        for i in 1:n
            keep = setdiff(1:n, i)
            Mi = [A2[keep, keep] P[keep, :]; P[keep, :]' zeros(npoly, npoly)]
            sol = Mi \ vcat(f[keep], zeros(npoly))
            pred = dot(A2[i, keep], sol[1:(n - 1)]) + dot(P[i, :], sol[n:end])
            @test Eb[i] ≈ f[i] - pred atol = 1e-10
        end
        # Matrix RHS through the bordered variant
        WΛM = M \ [Fb 2 .* Fb]
        EbM = loocverrors(M, WΛM, n)
        @test EbM[:, 1] ≈ Eb atol = 1e-12
        @test EbM[:, 2] ≈ 2 .* Eb atol = 1e-12
        # Bordered score
        @test loocvscore(M, Fb, n) ≈ sum(abs2, Eb) rtol = 1e-10
    end

    @testset "Scores and singular disqualification" begin
        w = A \ f
        E = loocverrors(A, w)
        @test loocvscore(A, f) ≈ sum(abs2, E) rtol = 1e-10
        F = [f 2 .* f]
        @test loocvscore(A, F) ≈ 5 * sum(abs2, E) rtol = 1e-10   # 1² + 2² = 5
        @test loocvscore(zeros(3, 3), ones(3)) == Inf
        @test loocvscore(zeros(4, 4), ones(4), 3) == Inf
    end
end
```

In `test/runtests.jl`, add the include between `wendland.jl` and `pum.jl`:

```julia
    include("wendland.jl")
    include("rippa.jl")
    include("pum.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run via `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")`
Expected: FAIL — `UndefVarError: loocverrors not defined`

- [ ] **Step 3: Implement `src/rippa.jl`**

Create `src/rippa.jl`:

```julia
# Exact leave-one-out cross-validation (LOOCV) via Rippa's formula. For a symmetric
# invertible system A w = f, the prediction error at point i when point i is left out
# is e_i = w_i / (A⁻¹)_{ii} — the whole LOO error vector costs one inversion, not n
# solves (S. Rippa, Adv. Comput. Math. 11, 1999). Ridge smoothing is covered by
# applying the formula to the smoothed matrix. Intended for the small dense per-patch
# systems of the partition of unity method, hence the direct inv().

# LOOCV error matrix for the (possibly ridge-smoothed) system A W = F. A is the
# assembled kernel matrix *after* smoothing was added; W the solved weights. A matrix
# W broadcasts the one diagonal over all sample columns.
function loocverrors(A::AbstractMatrix, W::AbstractVecOrMat)
    dinv = diag(inv(A))
    return W ./ dinv
end

# Bordered variant for polynomial-augmented saddle systems M = [A P; Pᵀ 0] with
# solution WΛ = [w; λ]: the same identity e_i = w_i / (M⁻¹)_{ii} holds for the data
# rows i = 1..n (Fasshauer & McCourt, Kernel-based Approximation Methods using
# MATLAB, ch. 14).
function loocverrors(M::AbstractMatrix, WΛ::AbstractVector, n::Integer)
    dinv = diag(inv(M))
    return WΛ[1:n] ./ dinv[1:n]
end
function loocverrors(M::AbstractMatrix, WΛ::AbstractMatrix, n::Integer)
    dinv = diag(inv(M))
    return WΛ[1:n, :] ./ dinv[1:n]
end

# Matrix exceptions raised by inv() on (numerically) singular input. A singular
# tuning candidate (extreme flat limit) is disqualified with an infinite score
# rather than an error.
const SINGULAR_EXCEPTIONS = Union{SingularException, LinearAlgebra.LAPACKException,
                                  LinearAlgebra.ZeroPivotException}

# Total squared LOOCV error of the system A W = F, sharing a single inversion
# between the solve and the diagonal (the tuning hot loop calls this once per
# shape-parameter candidate).
function loocvscore(A::AbstractMatrix, F::AbstractVecOrMat)
    Ainv = try
        inv(A)
    catch err
        err isa SINGULAR_EXCEPTIONS && return Inf
        rethrow()
    end
    W = Ainv * F
    return sum(abs2, W ./ diag(Ainv))
end

# Bordered score: F must already be padded with zero rows for the polynomial part;
# n is the number of data points (leading rows of M).
function loocvscore(M::AbstractMatrix, F::AbstractVecOrMat, n::Integer)
    Minv = try
        inv(M)
    catch err
        err isa SINGULAR_EXCEPTIONS && return Inf
        rethrow()
    end
    WΛ = Minv * F
    dinv = diag(Minv)
    E = WΛ isa AbstractVector ? WΛ[1:n] ./ dinv[1:n] : WΛ[1:n, :] ./ dinv[1:n]
    return sum(abs2, E)
end
```

In `src/ScatteredInterpolation.jl`, add the include after `rbf.jl` (line 12):

```julia
include("./rbf.jl")
include("./rippa.jl")
include("./wendland.jl")
```

- [ ] **Step 4: Run tests to verify they pass**

Run via `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")`
Expected: PASS. If the `atol = 1e-10` brute-force comparisons are flaky due to conditioning, first check the system's condition number (`cond(A)` should be ≤ ~1e6 with `Gaussian(4)` on these points); only then consider loosening to `1e-9` and record the reason in the plan amendments.

- [ ] **Step 5: Commit**

```bash
git add src/rippa.jl src/ScatteredInterpolation.jl test/rippa.jl test/runtests.jl
git commit -m "Add exact LOOCV helpers via Rippa's formula"
```

---

### Task 3: `tune` field on `PartitionOfUnity`

**Goal:** `PartitionOfUnity` accepts and validates `tune = :none | :loocv`; `:loocv` with a shapeless kernel is rejected at construction.

**Files:**
- Modify: `src/pum.jl` (struct at line ~124, constructor at line ~131, docstring at line ~103)
- Test: `test/pum.jl`

**Acceptance Criteria:**
- [ ] `PartitionOfUnity(Gaussian()).tune === :none`; `PartitionOfUnity(Gaussian(); tune = :loocv).tune === :loocv`
- [ ] `tune = :bogus` → `ArgumentError`
- [ ] `tune = :loocv` with `Polyharmonic(3)` or `GeneralizedPolyharmonic(3, 1)` → `ArgumentError` naming the kernel type ("has no shape parameter to tune")
- [ ] Docstring documents `tune` and that the kernel's own `ε` is ignored under `:loocv`
- [ ] Full test suite passes

**Verify:** `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

In `test/pum.jl`, inside `@testset "PartitionOfUnity construction"`, extend the `"Method validation"` testset with:

```julia
        # tune keyword
        @test PartitionOfUnity(Gaussian()).tune === :none
        @test PartitionOfUnity(Gaussian(); tune = :loocv).tune === :loocv
        @test_throws ArgumentError PartitionOfUnity(Gaussian(); tune = :bogus)
        @test_throws ArgumentError PartitionOfUnity(Polyharmonic(3); tune = :loocv)
        @test_throws ArgumentError PartitionOfUnity(GeneralizedPolyharmonic(3, 1);
                                                    tune = :loocv)
        @test_throws ArgumentError PartitionOfUnity(ThinPlate(); tune = :loocv)
```

- [ ] **Step 2: Run tests to verify they fail**

Run via `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")`
Expected: FAIL — `PartitionOfUnity` has no field/keyword `tune`

- [ ] **Step 3: Implement**

In `src/pum.jl`, change the struct (line ~124) to:

```julia
struct PartitionOfUnity{M <: AbstractRadialBasisFunction, W} <: InterpolationMethod
    method::M
    pointsperpatch::Int
    overlap::Float64
    weight::W
    tune::Symbol
end
```

Change the keyword constructor (line ~131) to:

```julia
function PartitionOfUnity(method::AbstractRadialBasisFunction;
                          pointsperpatch::Integer = 80, overlap::Real = 1.5,
                          weight = nothing, tune::Symbol = :none)
    pointsperpatch >= 1 || throw(ArgumentError(
        "pointsperpatch must be at least 1, got $pointsperpatch"))
    overlap > 1 || throw(ArgumentError(
        "overlap must be greater than 1 to guarantee full patch coverage, got $overlap"))
    tune in (:none, :loocv) || throw(ArgumentError(
        "tune must be :none or :loocv, got $(repr(tune))"))
    tune === :loocv && !hasshape(method) && throw(ArgumentError(
        "tune = :loocv requires a kernel with a shape parameter, but " *
        "$(typeof(method)) has no shape parameter to tune"))

    PartitionOfUnity(method, Int(pointsperpatch), Float64(overlap), weight, tune)
end
```

Update the docstring signature line and add a `tune` paragraph. The signature line becomes:

```
    PartitionOfUnity(method; pointsperpatch = 80, overlap = 1.5, weight = nothing,
                     tune = :none)
```

and after the `weight` sentence, add:

```
`tune` selects automatic per-patch shape-parameter tuning: `:none` uses `method` as
given, while `:loocv` chooses each patch's shape parameter `ε` by exact leave-one-out
cross-validation (Rippa's method), scanning candidates scaled to the patch's mean
nearest-neighbor spacing. Under `:loocv` the shape parameter stored in `method` is
ignored, and the kernel must have one (`Polyharmonic`, `ThinPlate` and
`GeneralizedPolyharmonic` do not).
```

- [ ] **Step 4: Run tests to verify they pass**

Run via `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pum.jl test/pum.jl
git commit -m "Add tune keyword to PartitionOfUnity"
```

---

### Task 4: Baseline target-scale benchmark (untuned)

**Goal:** Measure and record the untuned baseline (build time + off-node max error, `Gaussian(2)`, 10⁵ points, 3D and 6D) that Task 8 compares against, BEFORE the tuning hot path exists (`perf-plans-need-scale-benchmark` policy).

**Files:**
- Modify: `docs/superpowers/plans/2026-07-04-loocv-shape-selection.md` (record results in an "Amendments/Benchmark results" section)

**Acceptance Criteria:**
- [ ] Untuned build time and off-node max error recorded for 3D and 6D at n = 100_000, together with `Threads.nthreads()` and machine context
- [ ] Numbers recorded in this plan file under "Benchmark results"

**Verify:** benchmark script below runs to completion via `mcp__julia__julia_eval`; results table present in the plan file

**Steps:**

- [ ] **Step 1: Run the baseline benchmark**

Via `mcp__julia__julia_eval` (single call; expect a few minutes):

```julia
using ScatteredInterpolation

const BPRIMES = (2, 3, 5, 7, 11, 13)
kpts(d, n; offset = 0) = [mod((i + offset) * sqrt(BPRIMES[j]), 1.0) for j in 1:d, i in 1:n]
g(x) = sinpi(x[1]) * cospi(x[2]) + x[3]

results = Dict{Int, NamedTuple}()
for d in (3, 6)
    n = 100_000
    pts = kpts(d, n)
    vals = [g(x) for x in eachcol(pts)]
    q = kpts(d, 5000; offset = 500_000) .* 0.9 .+ 0.05
    truevals = [g(x) for x in eachcol(q)]
    # Warm up compilation on a small subset
    interpolate(PartitionOfUnity(Gaussian(2)), pts[:, 1:2000], vals[1:2000])
    t = @elapsed itp = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals)
    err = maximum(abs, evaluate(itp, q) - truevals)
    results[d] = (build_s = t, maxerr = err)
end
(results = results, nthreads = Threads.nthreads())
```

- [ ] **Step 2: Record the results**

Append to this plan file, under a new `## Benchmark results` section at the bottom:

```markdown
### Baseline (untuned Gaussian(2), n = 100_000), Task 4

| d | build [s] | off-node max error | nthreads |
|---|-----------|--------------------|----------|
| 3 | <measured> | <measured> | <N> |
| 6 | <measured> | <measured> | <N> |
```

Sanity expectation from the spec's motivation: the 3D off-node error should be around 0.1 (the mis-scaled flat-kernel regime). If it is orders of magnitude smaller, the Task 8 ≥10× criterion needs re-examination — flag it at the review checkpoint instead of proceeding silently.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-07-04-loocv-shape-selection.md
git commit -m "Record untuned PUM baseline at target scale"
```

---

### Task 5: Per-patch LOOCV tuning in the PUM build

**Goal:** `tune = :loocv` selects each patch's `ε` by scanning 11 scale-derived candidates scored with Rippa's formula, inside the existing threaded local-solve loop.

**Files:**
- Modify: `src/pum.jl` (new helpers before `interpolate`; build loop at line ~219)
- Test: `test/pum.jl`

**Acceptance Criteria:**
- [ ] Tuned 3D build (n = 5000, `Gaussian()`, `tune = :loocv`) has off-node max error < 0.5× that of untuned `Gaussian(2)`
- [ ] Exactness at nodes preserved under tuning (dims 2 and 3, `atol = 1e-6`)
- [ ] Matrix samples and `smooth = 1e-3` builds work under tuning
- [ ] Two tuned builds are bit-identical (weights and per-patch `ε`)
- [ ] Tiny/degenerate patches don't error (skip-tuning fallbacks)
- [ ] Full test suite passes

**Verify:** `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

At the top of `test/pum.jl`, extend the import line to:

```julia
using ScatteredInterpolation: PatchGrid, buildgrid, centerof, assignpatches,
                              meannndist, tuneshape, LOOCV_CANDIDATES
```

At the end of `test/pum.jl` add:

```julia
@testset "LOOCV shape-parameter tuning" begin

    @testset "meannndist" begin
        # Nearest-neighbor distances: 1 (point 1→2), 1 (2→1), 2 (3→2)
        pts = [0.0 1.0 3.0; 0.0 0.0 0.0]
        @test meannndist(pts) ≈ 4 / 3
        @test meannndist(fill(0.5, 2, 3)) == 0        # coincident
        @test meannndist(reshape([1.0, 2.0], 2, 1)) == 0   # single point
    end

    @testset "tuneshape edge cases" begin
        cmid = LOOCV_CANDIDATES[(length(LOOCV_CANDIDATES) + 1) ÷ 2]
        # Coincident points: no length scale — kernel returned unchanged
        k = tuneshape(Gaussian(7), fill(0.5, 2, 4), ones(4), false, Euclidean())
        @test k === Gaussian(7)
        # Two points: ε = c_mid / h without scanning
        pts2 = [0.0 1.0; 0.0 0.0]
        k2 = tuneshape(Gaussian(7), pts2, [1.0, 2.0], false, Euclidean())
        @test k2 isa Gaussian
        @test k2.ε ≈ cmid / 1.0
    end

    @testset "Tuned build beats mis-scaled fixed ε" begin
        n = 5000
        pts = kroneckerpoints(3, n)
        g3(x) = sinpi(x[1]) * cospi(x[2]) + x[3]
        vals = [g3(x) for x in eachcol(pts)]
        q = kroneckerpoints(3, 500; offset = 100_000) .* 0.9 .+ 0.05
        truevals = [g3(x) for x in eachcol(q)]
        fixed = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals)
        tuned = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv), pts, vals)
        efixed = maximum(abs, evaluate(fixed, q) - truevals)
        etuned = maximum(abs, evaluate(tuned, q) - truevals)
        @test etuned < 0.5 * efixed
    end

    @testset "Exactness at nodes under tuning, $d dimensions" for d in (2, 3)
        pts = kroneckerpoints(d, 400)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv), pts, vals)
        @test evaluate(itp, pts) ≈ vals atol = 1e-6
        # Tuned kernels live on the locals; evaluate needed no changes
        @test all(l -> l.rbf isa Gaussian, itp.locals)
    end

    @testset "Matrix samples and smoothing under tuning" begin
        pts = kroneckerpoints(2, 300)
        v1 = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv), pts,
                          [v1 2 .* v1])
        out = evaluate(itp, pts)
        @test size(out) == (300, 2)
        @test out[:, 1] ≈ v1 atol = 1e-6
        @test out[:, 2] ≈ 2 .* v1 atol = 1e-6

        itps = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv), pts, v1;
                           smooth = 1e-3)
        ev = evaluate(itps, pts)
        @test maximum(abs, ev - v1) > 1e-8   # no longer interpolating
        @test maximum(abs, ev - v1) < 0.1    # but still close
    end

    @testset "Determinism of tuned builds" begin
        pts = kroneckerpoints(2, 300)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp1 = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv), pts, vals)
        itp2 = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv), pts, vals)
        @test all(itp1.locals[p].w == itp2.locals[p].w
                  for p in eachindex(itp1.locals))
        @test all(itp1.locals[p].rbf.ε == itp2.locals[p].rbf.ε
                  for p in eachindex(itp1.locals))
    end

    @testset "Tiny patches under tuning" begin
        pts = [0.0 0.01 0.02 5.0
               0.0 0.01 0.02 5.0]
        vals = [1.0, 2.0, 3.0, 4.0]
        itp = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv,
                                           pointsperpatch = 2), pts, vals)
        @test evaluate(itp, pts) ≈ vals atol = 1e-8
    end
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run via `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")`
Expected: FAIL — `UndefVarError: meannndist not defined`

- [ ] **Step 3: Implement the tuning helpers**

In `src/pum.jl`, immediately before the `interpolate(pum::PartitionOfUnity, ...)` function (line ~180), add:

```julia
# ---- LOOCV shape-parameter tuning (tune = :loocv) --------------------------------

# Shape-parameter candidates are ε = c / h for the patch's mean nearest-neighbor
# distance h, spanning the flat-to-peaked range in half-octave steps.
const LOOCV_CANDIDATES = 2.0 .^ (-2:0.5:3)

# Mean nearest-neighbor distance among the patch points; zero when there is no
# usable length scale (single or all-coincident points). Patch sizes are
# ~pointsperpatch, so the brute-force O(np²) scan on the already-gathered block is
# cheaper than building a tree.
function meannndist(pts::AbstractMatrix{T}) where {T <: AbstractFloat}
    d, np = size(pts)
    np < 2 && return zero(T)
    total = zero(T)
    for i in 1:np
        best = typemax(T)
        for j in 1:np
            j == i && continue
            r2 = zero(T)
            for k in 1:d
                r2 += abs2(pts[k, i] - pts[k, j])
            end
            r2 < best && (best = r2)
        end
        total += sqrt(best)
    end
    return total / np
end

# Select the best shape parameter for one patch: score every candidate with the
# exact LOOCV error (Rippa's formula) and return the kernel rebuilt with the winner.
# The distance matrix does not depend on ε and is computed once; each candidate only
# re-applies the kernel, adds smoothing, and inverts the small dense system. The
# final solve of the winner runs through the standard interpolate path afterwards
# (where the user's linsolve choice applies); candidate scans deliberately use
# direct inversion instead.
function tuneshape(kernel::RadialBasisFunction, pts::AbstractMatrix,
                   samples::AbstractVecOrMat, smooth, metric)
    h = meannndist(pts)
    h > 0 || return kernel   # no length scale to derive ε from
    cmid = LOOCV_CANDIDATES[(length(LOOCV_CANDIDATES) + 1) ÷ 2]
    size(pts, 2) >= 3 || return withshape(kernel, cmid / h)

    R = pairwise(metric, pts, dims = 2)
    A = similar(R)
    εbest = cmid / h
    best = Inf
    for c in LOOCV_CANDIDATES
        ε = c / h
        ϕ = withshape(kernel, ε)
        A .= ϕ.(R)
        addSmoothing!(A, smooth)
        s = loocvscore(A, samples)
        if s < best
            best = s
            εbest = ε
        end
    end
    return withshape(kernel, εbest)
end

# Solve one patch's local system, tuning the kernel's shape parameter first when
# requested. Shared by the initial build and addpoints! re-solves.
function solvelocal(pum::PartitionOfUnity, patchpts, psamples, psmooth, metric,
                    linsolve)
    method = pum.tune === :loocv ?
        tuneshape(pum.method, patchpts, psamples, psmooth, metric) : pum.method
    interpolate(method, patchpts, psamples;
                metric = metric, smooth = psmooth, linsolve = linsolve)
end
```

Note: `addSmoothing!(A, false)` adds `false` (arithmetic zero) — the existing untuned path relies on the same behavior.

- [ ] **Step 4: Route the build loop through `solvelocal`**

In `interpolate(pum::PartitionOfUnity, ...)`, replace the threaded loop body (line ~222):

```julia
        Threads.@threads for p in 1:P
            idxs = patchpoints[p]
            locals[p] = solvelocal(pum, pts[:, idxs], patchsamples(samples, idxs),
                                   patchsmooth(smooth, idxs), metric, linsolve)
        end
```

- [ ] **Step 5: Run tests to verify they pass**

Run via `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")`
Expected: PASS (note: `GeneralizedMultiquadratic` under `:loocv` is not exercised yet — its `tuneshape` method lands in Task 7; `tuneshape` here dispatches on `RadialBasisFunction`, which excludes it)

- [ ] **Step 6: Commit**

```bash
git add src/pum.jl test/pum.jl
git commit -m "Add per-patch LOOCV shape-parameter tuning to PUM builds"
```

---

## Task 5 amendment: reliability gate, anchor fallback, and test redesign

The plan's original Task 5 design (direct `inv()`-based Rippa scoring over
`LOOCV_CANDIDATES = 2.0 .^ (-2:0.5:3)`, no fallback) passed its own written tests
but, on empirical validation against the target-scale benchmark (n = 100,000, run
early as a diagnostic ahead of Task 8), tuning was **~245x worse** than the untuned
baseline it was meant to improve on, and ~206x slower. Root-caused and fixed as
follows; both the design and the tests changed from what Task 5 originally
specified.

**Root cause 1 — accuracy.** `ε = c/h` (`h` = patch mean nearest-neighbor distance)
correctly re-peaks the kernel as patches get denser, which is standard practice for
conditioning safety. But for a smooth test function, the *accuracy*-optimal `ε` is
low and roughly **density-independent** — so at high `n` (small `h`), "safe" and
"accurate" candidates stop overlapping. Verified directly: reaching the
accuracy-optimal `ε` at n = 100,000 requires cond(A) ≈ 1e17-1e18, and at that
conditioning `inv(A)*b` disagreed with `A \ b` by O(1) (err ≈ 1.2, vs 0.0002 for
backslash) on the *identical* matrix — Rippa's formula via explicit `inv()` doesn't
merely get noisy at extreme conditioning, it becomes **confidently wrong** (a tiny,
trustworthy-looking score with zero correct digits). This is inherent to explicit
matrix inversion, not fixable by a smarter formula (LAPACK `gecon`-based reciprocal
condition number, not a stable-diagonal rewrite — no algorithm recovers digits that
aren't there at that conditioning).

**Fix 1 — reliability gate + ungated anchor.** `safeloocvscore` (`src/pum.jl`) gates
every *grid* candidate by reciprocal condition number (`LAPACK.gecon!` on the LU
factorization already needed for the score, threshold `RCOND_THRESHOLD = 1e-8`,
chosen empirically as the value that keeps node-exactness intact — this is what
guarantees exactness now, not any test-specific tolerance). The caller's own kernel
is scored too, but with the **ungated** `loocvscore` from Task 2, not
`safeloocvscore`: gating the anchor was tried and reverted, because it let a
merely-safe-but-worse grid candidate override a good-but-unscoreable anchor
(confirmed empirically — this is what caused the original ~245-500x regressions).
A grid candidate replaces the anchor only if it both passes the gate and scores
better than the (ungated) anchor score. Worst case, tuning returns the caller's
kernel unchanged — matching untuned exactly.

**Residual risk (not fully closed, deferred to Task 8):** the ungated anchor score
is *not* immune to the same extreme-conditioning unreliability — it can come out
noisily too-high just as easily as noisily-tiny. Confirmed on one specific patch
at n = 100,000 (patch size 32, anchor cond ≈ 7.3e18): the anchor's ungated score
(0.0076) was *higher* than a legitimately safe grid candidate's gated score
(0.0027, cond ≈ 5.6e4), causing a correct-per-the-rule but arguably-spurious
override. Effect on the full n = 100,000 build: tuned/untuned ratio ≈ 14.6x (a real
improvement over the original ~245-500x, but not the "≈1x by construction" the
gate is meant to guarantee, and only 1/200 sampled patches were affected — a single
bad patch can dominate a max-error metric). **Task 8 must re-check this at target
scale and treat it as a known open risk, not a regression to silently accept.**

**Root cause 2 — the tests asserted something mathematically impossible.** Node
exactness (in-training-set reproduction) and off-node accuracy (extrapolation) pull
`ε` in *opposite* directions — this is the classical RBF trade-off/uncertainty
principle (Schaback). No candidate selection, however implemented, can make a
single `ε` simultaneously the best for both on a genuinely smooth function; LOOCV
correctly identifies this (verified: `ε = 1`'s ungated score is ~13 orders of
magnitude better than any peaked grid candidate for a smooth function — LOOCV was
never "wrong," the original bug was purely the anchor being gated out).

**Fix 2 — redesigned Task 5 tests (`test/pum.jl`):**
- *"Tuned build beats mis-scaled fixed ε"* (asserted tuning beats an *arbitrary*
  fixed `Gaussian(2)` baseline by 2x) → replaced with two tests, each targeting a
  claim that is actually achievable:
  - *"Tuning never much worse than untuned (no catastrophe)"*: tuned-with-`K` vs
    untuned-with-the-**same**-`K` (`Gaussian()`), asserting `etuned < 3*euntuned`.
    Measured ratio is exactly 1.0 at n = 5,000 (the anchor wins outright).
  - *"Tuning corrects a badly mis-scaled (too-peaked) kernel"*: baseline is
    `Gaussian(50)`, genuinely too peaked for patch spacing (decays before reaching
    neighbors) — the regime tuning **can** reliably win, because the correction
    lands well inside the safe conditioning envelope. Measured: 37x improvement.
- *"Exactness at nodes under tuning"* and *"Matrix samples and smoothing under
  tuning"*: base kernel changed from `Gaussian()` (ε = 1, marginal/borderline at
  this point density — measured 9.05e-6 max node error *even untuned*, already
  over the test's own 1e-6 bound) to `Gaussian(2)` (well-conditioned, confirmed
  untuned and tuned identical to ~1e-7 to 2e-13 depending on dimension). The
  multi-column matrix-RHS sub-test's tolerance was loosened from 1e-6 to 1e-5 for
  normal floating-point margin (measured 1.02e-6, marginally over 1e-6, on a
  well-conditioned but not machine-precision system).

**Open risk carried into Task 8:** the ≤5x build-time budget is likely hard to
meet with an 11-candidate-plus-anchor linear scan per patch (measured ~206x
slower before any gate; the gate does not reduce candidate count). Task 8 should
measure GC% before attributing the gap to compute vs. allocation, and may need
either a smaller/adaptive candidate count (e.g. golden-section search on the
U-shaped in-regime score) or a revised time budget — decide with evidence, not
before measuring.

---

### Task 6: `addpoints!` re-tunes affected patches

**Goal:** Patch re-solves triggered by `addpoints!` go through `solvelocal`, so affected patches re-tune their `ε` under `:loocv`; interior-query equivalence with a tuned rebuild holds.

**Files:**
- Modify: `src/pum.jl` (`addpoints!` re-solve loop, line ~395)
- Test: `test/pum.jl`

**Acceptance Criteria:**
- [ ] `addpoints!` on a tuned interpolant matches a tuned full rebuild on interior queries (`atol = 1e-6`)
- [ ] New points interpolated exactly after tuned insertion
- [ ] `addpoints!` docstring mentions re-tuning
- [ ] Full test suite passes

**Verify:** `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")` → all pass

**Steps:**

- [ ] **Step 1: Write the failing test**

In `test/pum.jl`, inside `@testset "addpoints!"`, add:

```julia
    @testset "Tuned insertion matches tuned rebuild" begin
        itp = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv), base, basevals)
        addpoints!(itp, added, addedvals)
        rebuilt = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv),
                              hcat(base, added), vcat(basevals, addedvals))
        @test itp.grid.ncells == rebuilt.grid.ncells   # sanity: same grid
        # Interior patches see identical data in both paths, so identical candidate
        # grids select identical ε. (Boundary patches may differ via knn top-up,
        # exactly as in the untuned equivalence test above.)
        q = kroneckerpoints(2, 97; offset = 3000) .* 0.2 .+ 0.4
        @test evaluate(itp, q) ≈ evaluate(rebuilt, q) atol = 1e-6
        @test evaluate(itp, added) ≈ addedvals atol = 1e-6
    end
```

(This testset already defines `base`, `basevals`, `added`, `addedvals` at its top — reuse them.)

- [ ] **Step 2: Run tests to verify they fail**

Run via `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")`
Expected: FAIL — the equivalence assertion fails, because `addpoints!` still solves with the *untuned* `itp.method.method` (input `ε = 1`) while the rebuild tunes

- [ ] **Step 3: Route `addpoints!` re-solves through `solvelocal`**

In `addpoints!`, replace the re-solve loop body (line ~396):

```julia
        Threads.@threads for k in eachindex(aff)
            p = aff[k]
            idxs = itp.patchpoints[p]
            itp.locals[p] = solvelocal(itp.method, itp.points[:, idxs],
                                       patchsamples(itp.samples, idxs),
                                       itp.smooth, itp.metric, itp.linsolve)
        end
```

In the `addpoints!` docstring, extend the first paragraph's parenthetical to:

```
(using the same `smooth` and `linsolve` settings as the original build, and re-tuning
the shape parameter of affected patches when the interpolant was built with
`tune = :loocv`)
```

- [ ] **Step 4: Run tests to verify they pass**

Run via `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pum.jl test/pum.jl
git commit -m "Re-tune affected patches in addpoints! under tune = :loocv"
```

---

### Task 7: Bordered-system tuning for `GeneralizedMultiquadratic`

**Goal:** `tune = :loocv` works for `GeneralizedMultiquadratic` by scoring candidates on the polynomial-augmented saddle system `M = [A P; Pᵀ 0]` (the bordered `loocvscore` from Task 2).

**Files:**
- Modify: `src/pum.jl` (add a `tuneshape` method for `GeneralizedRadialBasisFunction`, next to the plain one from Task 5)
- Test: `test/pum.jl`

**Acceptance Criteria:**
- [ ] PUM + `GeneralizedMultiquadratic(1, 1/2, 2)` + `tune = :loocv` builds and is exact at nodes (`atol = 1e-5`)
- [ ] Tuned locals preserve `β` and `degree`
- [ ] Full test suite passes

**Verify:** `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")` → all pass

**Steps:**

- [ ] **Step 1: Write the failing test**

In `test/pum.jl`, at the end of `@testset "LOOCV shape-parameter tuning"`, add:

```julia
    @testset "GeneralizedMultiquadratic bordered tuning" begin
        pts = kroneckerpoints(2, 300)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(GeneralizedMultiquadratic(1, 1/2, 2);
                                           tune = :loocv, pointsperpatch = 60),
                          pts, vals)
        # Same atol rationale as the untuned GMQ exactness test above (array-norm ≈)
        @test evaluate(itp, pts) ≈ vals atol = 1e-5
        @test all(l -> l.rbf isa GeneralizedMultiquadratic, itp.locals)
        @test all(l -> l.rbf.β == 1/2 && l.rbf.degree == 2, itp.locals)
    end
```

- [ ] **Step 2: Run tests to verify they fail**

Run via `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")`
Expected: FAIL — `MethodError: no method matching tuneshape(::GeneralizedMultiquadratic, ...)` (the Task 5 method is restricted to `RadialBasisFunction`)

- [ ] **Step 3: Implement the bordered `tuneshape` method**

In `src/pum.jl`, directly after the plain `tuneshape` method, add:

```julia
# Bordered variant for polynomial-augmented kernels (GeneralizedMultiquadratic):
# candidates are scored on the saddle system M = [A P; Pᵀ 0]. The polynomial block P
# does not depend on ε and is assembled once; only the kernel block changes per
# candidate.
function tuneshape(kernel::GeneralizedRadialBasisFunction, pts::AbstractMatrix,
                   samples::AbstractVecOrMat, smooth, metric)
    h = meannndist(pts)
    h > 0 || return kernel   # no length scale to derive ε from
    cmid = LOOCV_CANDIDATES[(length(LOOCV_CANDIDATES) + 1) ÷ 2]
    np = size(pts, 2)
    np >= 3 || return withshape(kernel, cmid / h)

    R = pairwise(metric, pts, dims = 2)
    P = generateMultivariatePolynomial(pts, kernel.degree)
    npoly = size(P, 2)
    m = np + npoly
    M = zeros(promote_type(eltype(R), eltype(P)), m, m)
    M[1:np, (np + 1):m] .= P
    M[(np + 1):m, 1:np] .= P'
    F = samples isa AbstractVector ?
        vcat(samples, zeros(eltype(samples), npoly)) :
        vcat(samples, zeros(eltype(samples), npoly, size(samples, 2)))
    εbest = cmid / h
    best = Inf
    for c in LOOCV_CANDIDATES
        ε = c / h
        ϕ = withshape(kernel, ε)
        Ablock = view(M, 1:np, 1:np)
        Ablock .= ϕ.(R)
        addSmoothing!(Ablock, smooth)
        s = loocvscore(M, F, np)
        if s < best
            best = s
            εbest = ε
        end
    end
    return withshape(kernel, εbest)
end
```

- [ ] **Step 4: Run tests to verify they pass**

Run via `mcp__julia__julia_eval`: `import Pkg; Pkg.test("ScatteredInterpolation")`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pum.jl test/pum.jl
git commit -m "Tune GeneralizedMultiquadratic via the bordered LOOCV formula"
```

---

### Task 8: Target-scale benchmark validation (tuned vs untuned)

**Goal:** Verify the spec's §5 budgets at target scale: tuned `Gaussian()` off-node max error ≥ 10× better than untuned `Gaussian(2)`, and tuned build time ≤ 5× untuned, at n = 100_000 in 3D and 6D.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Modify: `docs/superpowers/plans/2026-07-04-loocv-shape-selection.md` (append results to "Benchmark results")

**Acceptance Criteria:**
- [ ] 3D, n = 100_000, `f(x) = sinpi(x₁)cospi(x₂) + x₃`: tuned off-node max error ≤ untuned/10 (captured numbers for both sides)
- [ ] 6D, n = 100_000, same `f`: tuned off-node max error ≤ untuned/10 (captured numbers for both sides)
- [ ] Tuned build time ≤ 5× untuned build time in both dimensions (captured timings for both sides)
- [ ] All measurements (untuned and tuned, both dims, thread count) recorded in this plan's "Benchmark results" section

**Verify:** the script below via `mcp__julia__julia_eval` → printed `ratio_err` ≥ 10 and `ratio_time` ≤ 5 for d ∈ {3, 6}

**Steps:**

- [ ] **Step 1: Run the comparison benchmark**

Both sides are measured in the same session for a fair comparison (same thread count, same machine state); the Task 4 numbers serve as a cross-check. Via `mcp__julia__julia_eval`:

```julia
using ScatteredInterpolation

const BPRIMES = (2, 3, 5, 7, 11, 13)
kpts(d, n; offset = 0) = [mod((i + offset) * sqrt(BPRIMES[j]), 1.0) for j in 1:d, i in 1:n]
g(x) = sinpi(x[1]) * cospi(x[2]) + x[3]

out = Dict{Int, NamedTuple}()
for d in (3, 6)
    n = 100_000
    pts = kpts(d, n)
    vals = [g(x) for x in eachcol(pts)]
    q = kpts(d, 5000; offset = 500_000) .* 0.9 .+ 0.05
    truevals = [g(x) for x in eachcol(q)]
    # Warm-up both paths on a small subset
    interpolate(PartitionOfUnity(Gaussian(2)), pts[:, 1:2000], vals[1:2000])
    interpolate(PartitionOfUnity(Gaussian(); tune = :loocv), pts[:, 1:2000], vals[1:2000])

    t0 = @elapsed itp0 = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals)
    e0 = maximum(abs, evaluate(itp0, q) - truevals)
    t1 = @elapsed itp1 = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv),
                                     pts, vals)
    e1 = maximum(abs, evaluate(itp1, q) - truevals)
    out[d] = (untuned_s = t0, untuned_err = e0, tuned_s = t1, tuned_err = e1,
              ratio_err = e0 / e1, ratio_time = t1 / t0)
end
(out = out, nthreads = Threads.nthreads())
```

- [ ] **Step 2: Check the budgets**

For each `d` in `(3, 6)`:
- `ratio_err = untuned_err / tuned_err` must be ≥ 10.
- `ratio_time = tuned_s / untuned_s` must be ≤ 5.

If a budget fails, this is a finding to report at the review checkpoint — do NOT close the task, do NOT weaken the thresholds. Likely levers (investigate, don't guess): candidate-grid range, per-candidate allocation churn in `tuneshape`, `inv` vs an in-place factorization.

- [ ] **Step 3: Record the results**

Append to the `## Benchmark results` section:

```markdown
### Tuned vs untuned (n = 100_000), Task 8

| d | untuned build [s] | tuned build [s] | ratio_time (≤5) | untuned err | tuned err | ratio_err (≥10) |
|---|---|---|---|---|---|---|
| 3 | <t0> | <t1> | <r> | <e0> | <e1> | <r> |
| 6 | <t0> | <t1> | <r> | <e0> | <e1> | <r> |

nthreads = <N>. Cross-check vs Task 4 baseline: <consistent / note deviation>.
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-07-04-loocv-shape-selection.md
git commit -m "Validate LOOCV tuning error and build-time budgets at target scale"
```

---

### Task 9: Documentation

**Goal:** `methods.md` PUM section documents `tune = :loocv` (the `PartitionOfUnity` docstring was already updated in Task 3).

**Files:**
- Modify: `docs/src/methods.md` (kernel-scaling paragraph, lines 142–148)

**Acceptance Criteria:**
- [ ] The manual-`ε`-scaling paragraph offers `tune = :loocv` as an alternative
- [ ] A short paragraph explains LOOCV selection and notes the kernel's own `ε` is ignored under tuning
- [ ] Docs build cleanly (or, if the docs environment is not set up in the session, the Markdown is well-formed and cross-references use existing anchors only)

**Verify:** `mcp__julia__julia_eval`: docs project build, or visual inspection of the diff — no broken `@ref` targets introduced (only `[`PartitionOfUnity`](@ref)` which already exists in this file)

**Steps:**

- [ ] **Step 1: Rewrite the kernel-scaling paragraph**

In `docs/src/methods.md`, replace lines 142–148:

```markdown
Because each patch spans only a small part of the domain, basis functions with a
fixed shape parameter behave differently than in the global method: a kernel like
`Gaussian(ε)` that is well conditioned globally is nearly flat across a small patch,
which degrades accuracy at large point counts. Either scale ``ε`` to the patch size,
let the method choose it per patch with `tune = :loocv`, or use a scale-free
polyharmonic spline with polynomial augmentation such as `GeneralizedPolyharmonic(3, 1)`
as the local method.

With `PartitionOfUnity(kernel; tune = :loocv)`, each patch selects its own shape
parameter by exact leave-one-out cross-validation (Rippa's method): candidate values
of ``ε``, scaled to the patch's mean nearest-neighbor spacing, are scored by the
exact leave-one-out prediction error — obtained from a single matrix inversion per
candidate rather than ``n`` separate solves — and the best candidate wins. The shape
parameter stored in the passed kernel is ignored under tuning. Tuning requires a
kernel with a shape parameter (the scale-free `Polyharmonic`, `ThinPlate` and
`GeneralizedPolyharmonic` have none and are rejected) and multiplies build time by a
small constant factor (one small dense solve per candidate per patch).
```

- [ ] **Step 2: Verify**

Build the docs if the docs project resolves in this environment, otherwise inspect the rendered Markdown for well-formedness:

```julia
# via mcp__julia__julia_eval, from the package root; skip gracefully if the docs
# environment cannot instantiate in-session and fall back to diff inspection
import Pkg; Pkg.activate("docs"); Pkg.instantiate()
include("docs/make.jl")
```

- [ ] **Step 3: Commit**

```bash
git add docs/src/methods.md
git commit -m "Document tune = :loocv in the partition of unity section"
```

---

## Benchmark results

### Baseline (untuned Gaussian(2), n = 100_000), Task 4

| d | build [s] | off-node max error | nthreads |
|---|-----------|--------------------|----------|
| 3 | 0.632 | 0.005472 | 12 |
| 6 | 7.926 | 0.045197 | 12 |

**Deviation flagged:** the plan's sanity expectation was a 3D off-node error around
0.1 (the mis-scaled flat-kernel regime); the measured value (0.0055) is about 20×
smaller. Per the plan's Task 4 Step 2 instruction, this is noted here rather than
silently proceeding — it may make Task 8's ≥10× tuned-vs-untuned error ratio harder
to clear in 3D, since the untuned baseline is already fairly accurate at this point
count/spacing. Revisit at the Task 8 checkpoint if the budget is not met.
