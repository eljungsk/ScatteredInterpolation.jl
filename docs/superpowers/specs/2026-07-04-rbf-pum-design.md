# Radial Basis Function Partition of Unity Method (RBF-PUM)

**Date:** 2026-07-04
**Branch:** to be created from `linearsolve-v2` (depends on its LinearSolve helpers)
**Status:** Approved design, pending implementation plan

## Goal

Add an interpolation method that scales to large scattered datasets (10⁵–10⁷ points) in
low dimension (≤ 6): the RBF partition of unity method. The domain is covered with
overlapping patches, a small dense RBF interpolant is solved per patch, and patches are
blended with smooth partition-of-unity weights. Requirements settled during design:

- Exact interpolation by default, with the existing per-patch `smooth` knob for
  approximation.
- C¹⁺ smoothness everywhere (achieved: C² from Wendland weight functions).
- Optimized for build-once / evaluate-many usage.
- Incremental point insertion without full rebuild (bonus, supported).
- Structure the code so a global compactly-supported-RBF sparse method ("option B") is
  easy to add later; document that path (see Future Work).

## Decisions

- **Wendland kernels come from KernelFunctions.jl** (new hard dependency), not a local
  re-implementation. `PiecewisePolynomialKernel{degree}(dim = d)` is exactly the Wendland
  family (degrees 0–3, dimension-aware, C²ᵛ). We wrap it in a thin adapter; coefficients
  and dimension logic stay upstream. `KernelFunctions.kappa(k, r)` is the documented
  scalar-evaluation API and matches this package's `ϕ(r)` convention.
- **Local solves reuse the existing RBF machinery.** Each patch is a plain
  `RBFInterpolant` built by the existing `interpolate` internals, inheriting LinearSolve
  algorithm selection (`linsolve`), ridge smoothing (`smooth`), and generalized-RBF
  polynomial augmentation.
- **Regular grid of spherical patches** over the data bounding box (standard RBF-PUM
  construction, cf. Cavoretto/Fasshauer). Grid cell lookup makes patch location O(1)
  per query.
- **Threaded build and evaluation** using base `Threads` only. Patch solves are
  independent; evaluation threads the per-patch GEMM phase and keeps the
  scatter-accumulate phase sequential to avoid write races.
- **Single kernel for all patches.** The existing vector-of-methods-per-point RBF
  feature is out of scope for PUM.
- **`support_radius` trait** added to all RBF types now (`Inf` for existing kernels,
  finite for Wendland) as the dispatch hook for option B later.

## Design

### 1. Public API

```julia
struct PartitionOfUnity{M <: AbstractRadialBasisFunction, W} <: InterpolationMethod
    method::M              # local RBF kernel, any existing kernel incl. generalized
    pointsperpatch::Int    # target average points per patch, default 80
    overlap::Float64       # patch radius inflation factor, default 1.5, must be > 1
    weight::W              # PU weight kernel, default Wendland C² for the data dimension
end

PartitionOfUnity(method; pointsperpatch = 80, overlap = 1.5, weight = nothing)
```

`weight = nothing` means "Wendland C² of the data dimension", resolved at `interpolate`
time when the dimension is known.

```julia
itp = interpolate(pum, points, samples; smooth = false, linsolve = nothing, metric = Euclidean())
evaluate(itp, querypoints)
addpoints!(itp, newpoints, newsamples)   # new export, PUM interpolants only
```

`smooth` and `linsolve` are forwarded to every per-patch solve. `smooth = false` gives
exact interpolation: every patch covering a data point interpolates it, and the PU
weights sum to 1, so the blend reproduces the data exactly.

New kernel adapter, exported and usable in the plain dense RBF path immediately:

```julia
struct Wendland{K, T <: Real} <: RadialBasisFunction
    kernel::K   # KernelFunctions.PiecewisePolynomialKernel{degree}(dim = dim)
    ε::T        # inverse support radius; support_radius(w) = 1 / ε
end

Wendland(dim, degree; ε = 1)
(w::Wendland)(r) = KernelFunctions.kappa(w.kernel, w.ε * r)
```

Constructor validation: `degree ∈ 0:3` and `dim ≥ 1` (enforced upstream, surfaced with a
clear local error message).

New trait, internal for now:

```julia
support_radius(::AbstractRadialBasisFunction) = Inf
support_radius(w::Wendland) = 1 / w.ε
```

### 2. Patch construction (at `interpolate`)

1. Compute the data bounding box.
2. Choose the number of grid cells ≈ `n / pointsperpatch`; per-side count
   `ceil((n / pointsperpatch)^(1/d))` (clamped to ≥ 1).
3. Patch centers on the regular grid; every patch is a ball of radius
   `overlap × (half cell diagonal)`. `overlap > 1` guarantees the bounding box is fully
   covered.
4. Assign points to patches with one `inrange` query per patch center against a KDTree
   of the data (NearestNeighbors.jl, existing dependency). A point may belong to several
   patches.
5. Drop empty patches. Patches with very few points (even one) are kept — a small RBF
   system is still solvable.
6. Solve one local RBF system per patch via the existing machinery on column views of
   `points` / `samples`, `Threads.@threads` over patches. Per-patch systems are ~80×80,
   where single-threaded BLAS per task is appropriate; the plan should pin down BLAS
   thread interaction (e.g. document `BLAS.set_num_threads` guidance) rather than leave
   it implicit.

Resulting interpolant:

```julia
struct PartitionOfUnityInterpolant{...} <: ScatteredInterpolant
    grid        # origin, spacing, per-side counts, patch radius
    patchpoints::Vector{Vector{Int}}   # data-point indices per (non-empty) patch
    locals::Vector{<:RBFInterpolant}
    weight      # PU weight kernel (Wendland instance)
    points      # reference to data, needed by addpoints!
    samples
    method      # the PartitionOfUnity method object (for addpoints! re-solves)
    smooth, linsolve, metric   # stored so addpoints! re-solves match the build
end
```

### 3. Evaluation (performance-critical)

Per query point:

1. Locate the grid cell by floor division — O(1).
2. Candidate patches from a fixed stencil of neighboring cells — those whose centers
   can lie within the patch radius. The stencil extent is a small constant computed once
   from `overlap` at build time.
3. Compute distances to candidate centers; patches with `r < radius` contribute.
4. PU weights `w_j(x) = ψ(r_j / ρ) / Σ_k ψ(r_k / ρ)` with ψ the Wendland weight kernel.

Batching for BLAS3 and threading:

- Group query indices by contributing patch (one pass over queries).
- For each patch (threaded), evaluate its local interpolant on its whole query block via
  the existing matrix `evaluate` — one GEMM-shaped operation per patch — into a
  per-patch buffer.
- Sequential scatter-accumulate of weighted per-patch results into the output. This
  phase is O(total query–patch covering pairs) and avoids write races without
  per-thread output copies.

Uncovered queries (outside the bounding box or in an interior coverage hole): evaluate
the nearest non-empty patch's local interpolant. This is extrapolation and is documented
as such.

### 4. Incremental insertion — `addpoints!`

```julia
addpoints!(itp::PartitionOfUnityInterpolant, newpoints, newsamples)
```

1. Validate dimensions (matching the existing sharpened dimension-check style).
2. Any new point outside the original bounding box → `ArgumentError` explaining that the
   patch grid is fixed at build time and suggesting a rebuild.
3. Locate covering patches by grid index, append the new point indices, and re-solve
   only those local systems (typically 1–4 small dense solves) with the stored
   `smooth` / `linsolve` / `metric`.
4. `points` / `samples` are replaced by concatenated copies (simple; insertion is rare
   relative to evaluation, so copy cost is acceptable).

Other interpolant types get an informative error (explicit `addpoints!` method that
throws, not a bare `MethodError`).

### 5. Error handling

- `overlap ≤ 1` → `ArgumentError` (coverage guarantee broken).
- `pointsperpatch < 1` → `ArgumentError`.
- Dimension mismatches → same style as existing checks.
- `Wendland` constructor: invalid `degree` / `dim` → clear error.
- Degenerate patches (1–2 points): no special case, small systems solve fine.

### 6. Testing

- Exactness at data points (`smooth = false`) for representative kernels (Gaussian,
  Polyharmonic, a generalized kernel, Wendland) in dims 1–6.
- Accuracy sanity: PUM vs dense global RBF on the same smooth-function dataset, tight
  tolerance.
- Smoothness: finite-difference gradient continuity across a patch boundary.
- `smooth > 0`: per-patch smoothing qualitatively matches the global smoothing behavior.
- `addpoints!`: result matches a full rebuild to tight tolerance; out-of-bbox insertion
  errors; non-PUM interpolants error informatively.
- Uncovered-query fallback (query outside bbox) returns the nearest patch's value.
- Single-point patches work.
- Threaded vs serial evaluation equivalence.
- `Wendland` works in the plain dense RBF path; `support_radius` returns the expected
  values for all kernel types.
- Invalid-argument errors (`overlap`, `pointsperpatch`, `Wendland` args).

### 7. Documentation

- `docs/src/methods.md`: new "Partition of Unity" section in the existing style — patch
  construction, the PU blend formula, Wendland kernel definitions (with math), parameter
  guidance (`pointsperpatch`, `overlap`), exactness-vs-smoothing note, extrapolation
  behavior, `addpoints!` usage and its bounding-box limitation.
- `docs/src/api.md`: new exports `PartitionOfUnity`, `Wendland`, `addpoints!`.

## Future Work: option B — global compactly supported RBF (sparse)

Documented here so it can be implemented later without design archaeology.

Goal: exact global interpolation with a compactly supported kernel and a sparse solve,
as an alternative to PUM when a single global system is preferred.

Recipe:

1. **Dispatch point already in place:** `support_radius(rbf)` is finite for `Wendland`.
   In the plain RBF `interpolate` path, branch on `isfinite(support_radius(method))`.
2. **Sparse assembly:** find all point pairs within the support radius via
   `NearestNeighbors.inrange` (KDTree already built for PUM; here built on the data),
   evaluate ϕ on those distances only, and assemble a `SparseMatrixCSC`. The matrix is
   symmetric positive definite for Wendland kernels within their valid dimension.
3. **Solve through the existing LinearSolve helpers** with a sparse-appropriate
   algorithm (sparse Cholesky, e.g. `CHOLMODFactorization`, or CG). The helpers from the
   LinearSolve migration already accept algorithm objects; only the matrix type changes.
4. **Evaluation:** for each query, `inrange` against the data KDTree with the support
   radius, sum `w_i ϕ(r_i)` over neighbors only — O(neighbors) per query instead of O(n).
5. **Tuning caveat to document:** the support radius (via `ε`) trades sparsity against
   accuracy/conditioning; small support = sparse but inaccurate, large = accurate but
   dense. This is the main reason PUM is the primary large-n method.
6. **Insertion** is not natural in this path (global refactorization); document as
   unsupported there, pointing at PUM.

## Out of scope

- Vector-of-kernels-per-point inside PUM patches.
- Generic KernelFunctions.jl `SimpleKernel` adapter (Matérn etc.) — the `Wendland`
  adapter pattern makes this a natural later addition.
- `returnRBFmatrix` for PUM (meaningless across many local systems).
