# Automatic shape-parameter selection for PUM via LOOCV (Rippa's method)

**Date:** 2026-07-04
**Branch:** to be created from `rbf-pum` (requires the PUM feature, PR #37)
**Status:** Proposed design, handoff document — run the writing-plans skill against
this spec in a fresh session before implementing.

## Goal

Add opt-in automatic selection of the RBF shape parameter `ε`, per patch, to the
`PartitionOfUnity` method: `PartitionOfUnity(Gaussian(); tune = :loocv)`. The user no
longer guesses `ε`; each patch gets an `ε` matched to its local point density.

## Motivation

A fixed `ε` that is well conditioned globally is nearly flat across a small patch:
`Gaussian(2)` on 10⁵ points in 3D produced patch systems with condition number ~9e17
and off-node errors of ~0.1 (measured 2026-07-04 during the PUM build; see
`docs/superpowers/plans/2026-07-04-rbf-pum.md` amendments and the kernel-scaling
paragraph in `docs/src/methods.md`). Today the manual tells users to scale `ε` by hand
or use scale-free polyharmonic locals. Leave-one-out cross-validation via Rippa's
formula is exact and nearly free on small systems, and the PUM architecture makes it
affordable precisely because patches are small (~`pointsperpatch` × `pointsperpatch`):
the O(n³)-per-candidate cost that rules LOOCV out for global RBF systems is
milliseconds per patch, and the per-patch loop is already threaded. No mainstream
competitor (SciPy `RBFInterpolator`, MATLAB `scatteredInterpolant`) offers this.

## Mathematical basis

**Rippa's formula.** For an interpolation system `A w = f` with `A` symmetric and
invertible, the exact leave-one-out errors are

```
e_i = w_i / (A⁻¹)_{ii},   i = 1..n
```

i.e. the vector of prediction errors "at point i when point i is excluded" requires
only the solution `w` and the diagonal of `A⁻¹` — no n separate solves.
Reference: S. Rippa, *An algorithm for selecting a good value for the parameter c in
radial basis function interpolation*, Adv. Comput. Math. 11 (1999). Ridge smoothing is
covered by the same formula applied to `Ã = A + diag(smooth)` (the matrix the existing
`evaluateRBF!`/`addSmoothing!` path already assembles).

**Matrix-valued samples.** For `A W = F` with m columns, `E[i, k] = W[i, k] / (A⁻¹)_{ii}`
— one diagonal serves all columns. Selection criterion: `sum(abs2, E)`.

**Polynomial-augmented kernels (bordered systems).** For the generalized path with
saddle matrix `M = [A P; P' 0]`, the analogous formula `e_i = w_i / (M⁻¹)_{ii}` for
`i ≤ n` holds (Fasshauer & McCourt, *Kernel-based Approximation Methods using MATLAB*,
ch. 14). Among shipped kernels only `GeneralizedMultiquadratic` both has an `ε` and
uses the bordered system; treat it as an optional second-stage task and validate the
bordered formula against brute-force LOO in its tests. `GeneralizedPolyharmonic`,
`Polyharmonic`, `ThinPlate` have no shape parameter — tuning does not apply.

## API

```julia
PartitionOfUnity(method; pointsperpatch = 80, overlap = 1.5, weight = nothing,
                 tune = :none)
```

- New field `tune::Symbol` on the `PartitionOfUnity` struct, validated in the keyword
  constructor: `:none` (default, current behavior) or `:loocv`.
- `tune = :loocv` with a kernel that has no shape parameter → `ArgumentError` naming
  the kernel type ("has no shape parameter to tune").
- The `ε` stored in the passed kernel is ignored under `:loocv` (the candidate grid is
  scale-derived); the docstring must say so.
- Vector-of-kernels methods remain out of scope for PUM (unchanged).

## Design

### 1. New internal traits and helpers (`src/rbf.jl` or `src/wendland.jl`)

```julia
# Does this kernel have a tunable shape parameter, and how to rebuild it with a new one?
hasshape(::AbstractRadialBasisFunction) = false
hasshape(::Union{Gaussian, Multiquadratic, InverseQuadratic, InverseMultiquadratic,
                 GeneralizedMultiquadratic, Wendland}) = true

withshape(k::Gaussian, ε) = Gaussian(ε)
withshape(k::Multiquadratic, ε) = Multiquadratic(ε)
withshape(k::InverseQuadratic, ε) = InverseQuadratic(ε)
withshape(k::InverseMultiquadratic, ε) = InverseMultiquadratic(ε)
withshape(k::GeneralizedMultiquadratic, ε) = GeneralizedMultiquadratic(ε, k.β, k.degree)
# Wendland must preserve dim/degree; reconstruct through the public constructor so
# validation runs. Store dim/degree on construction or recover them from the wrapped
# PiecewisePolynomialKernel — decide at plan time and record it.
```

### 2. Rippa helper (new `src/rippa.jl`)

```julia
# Exact LOOCV error matrix for the (possibly ridge-smoothed) system A W = F.
# A is the assembled kernel matrix *after* smoothing was added; W the solved weights.
# Small dense systems only (patch scale): uses inv().
function loocverrors(A::AbstractMatrix, W::AbstractVecOrMat)
    dinv = diag(inv(A))              # (A⁻¹)_{ii}
    return W ./ dinv                 # broadcasts over columns for matrix W
end
```

Notes for the implementer:
- `inv` on an 80×80 matrix is microseconds; do NOT route candidate scans through
  LinearSolve — the user's `linsolve` choice applies to the *final* solve only.
- The bordered variant (optional task) takes `M` and returns rows `1..n` of
  `W ./ diag(inv(M))[1:n]`.

### 3. Per-patch tuning (in `src/pum.jl`, inside the build)

Integration point: the threaded local-solve loop in `interpolate(pum::PartitionOfUnity, ...)`
(currently `Threads.@threads for p in 1:P` inside `withpinnedblas`). Under `:loocv`,
replace the direct `interpolate(pum.method, ...)` call per patch with:

```julia
# 1. Patch geometry scale: mean nearest-neighbor distance among the patch's points.
#    Patch sizes are ~pointsperpatch, so the O(np²) brute-force pairwise minimum is
#    fine; do it on the already-gathered pts[:, idxs] block.
# 2. Candidate grid: ε_j = c_j / h_p with c_j = 2.0 .^ (-2:0.5:3)  (11 candidates).
# 3. Distances do not depend on ε: compute R = pairwise(metric, patchpts; dims = 2)
#    ONCE per patch; per candidate only A = ϕ_ε.(R) (+ smoothing), solve, score with
#    loocverrors. Track the best score.
# 4. Final: build the local interpolant with withshape(kernel, ε_best) through the
#    existing per-patch interpolate call (user's linsolve applies here).
```

Selection score: `sum(abs2, loocverrors(A, W))`. Skip tuning (use `ε = c_mid / h_p`)
when the patch has fewer than 3 points or `h_p == 0` (coincident points).

The chosen kernel is stored where it already lives: each `RBFInterpolant` carries its
own `rbf` field, and PUM evaluation reads `itp.locals[p].rbf` — **no change to
`evaluate` at all**.

### 4. `addpoints!`

Affected patches are re-solved with re-tuning (they received new data, so their
optimal `ε` may shift). The insertion-vs-rebuild equivalence test must therefore keep
its existing structure (compare on interior queries; boundary patches may differ) —
under `:loocv` interior patches see identical data in both paths, so identical
candidate grids select identical `ε`.

### 5. Performance guardrails (per the scale-benchmark policy)

- Tuning cost bound: 11 candidates × O(np³) per patch. At `pointsperpatch = 80` this
  is ~11 × 0.1 ms; budget: **tuned build ≤ 5× untuned build** at 10⁵ points in 3D.
- The plan MUST include a target-scale benchmark task (10⁵ points, 3D and 6D):
  tuned `Gaussian()` off-node max error must beat untuned `Gaussian(2)` by ≥ 10× on
  the standard `f(x) = sinpi(x₁)cospi(x₂) + x₃` setup, and build time must stay within
  the 5× budget. See memory `perf-plans-need-scale-benchmark` and the RBF-PUM plan's
  amendments for why toy-size tests are not sufficient.

## Error handling

- `tune ∉ (:none, :loocv)` → `ArgumentError` in the `PartitionOfUnity` constructor.
- `tune = :loocv` + `hasshape(kernel) == false` → `ArgumentError` at `interpolate`
  time (the data dimension is irrelevant, so constructor time is also acceptable —
  pick one and test it).
- Singular candidate systems (extreme flat limit): guard the per-candidate solve with
  a try/catch or a `cond` estimate is NOT needed — `inv`/`lu` throwing `SingularException`
  for a candidate should simply disqualify that candidate (treat score = Inf).

## Testing

- `loocverrors` against brute-force leave-one-out (drop point i, solve, predict at i)
  on a ~12-point Gaussian system, tolerance 1e-10. Same for the bordered variant if
  implemented.
- Tuned build beats fixed mis-scaled `ε`: 3D, n = 5000, `tune = :loocv` with
  `Gaussian()` vs `Gaussian(2)` — off-node max error ratio < 0.5.
- Exactness at nodes preserved under tuning (dims 2 and 3).
- `tune = :loocv` + `Polyharmonic(3)` → `ArgumentError`.
- Matrix samples tuned build works; smoothing (`smooth = 1e-3`) tuned build works.
- Determinism: two tuned builds bit-identical.
- `addpoints!` on a tuned interpolant: interior-query equivalence with tuned rebuild.
- Target-scale benchmark task per §5.

## Documentation

- `methods.md`: extend the Partition of Unity section — the paragraph currently
  advising manual `ε` scaling gains "or let the method choose per patch with
  `tune = :loocv`", plus a short LOOCV explanation and the note that the kernel's own
  `ε` is ignored under tuning.
- `PartitionOfUnity` docstring: document `tune`.

## Out of scope

- LOOCV for the global (non-PUM) RBF path — O(n³) per candidate; revisit only if a
  user asks.
- Tuning `Polyharmonic` order `k`, Wendland `degree`, or `GeneralizedMultiquadratic.β`
  (discrete parameters; grid search possible later, YAGNI now).
- Per-patch *anisotropic* shape parameters.
- Auto-tuning the PU weight function (it has no accuracy-critical parameter).

## Handoff checklist for the implementing session

1. Branch from `rbf-pum` (or from `linearsolve-v2`/master after PR #37 merges).
2. Run the writing-plans skill against this spec; keep tasks TDD with complete code.
3. Include the §5 benchmark task in the plan — its absence is a plan defect here.
4. Known pre-existing issue you will hit if you test complex samples: the LinearSolve
   cache RHS typing bug (see memory `rbf-pum-branch-state`); it is not yours to fix.
