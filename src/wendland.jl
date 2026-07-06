# Compactly supported Wendland radial basis functions, provided by wrapping
# KernelFunctions.PiecewisePolynomialKernel (which implements the Wendland family).

export Wendland

"""
    Wendland(dim, degree; ε = nothing)

Define a compactly supported Wendland Radial Basis Function. The resulting kernel is
positive definite for data of dimension up to `dim`, is `2*degree` times continuously
differentiable, and vanishes identically for `r ≥ 1/ε`.

```math
ϕ(r) = \\max(1 - εr, 0)^{α(v, m)} \\, f_{v,m}(εr)
```

where ``v`` = `degree` ∈ {0, 1, 2, 3}, ``m`` = `dim`, and ``f_{v,m}`` is a polynomial of
degree ``v``. The kernel evaluation is provided by
`KernelFunctions.PiecewisePolynomialKernel`; see its documentation for the exact
coefficients. For `dim` ∈ {2, 3} and `degree = 1` this is the classic C² function
``ϕ(r) = (1 - εr)_+^4 (4εr + 1)``.

`ε` may be omitted (or passed explicitly as `nothing`), leaving the shape parameter
unset. An unset `ε` cannot be evaluated or queried for `support_radius` directly; it
must first be resolved from data via `resolveshape` (used internally by `interpolate`
with the `neighbors` keyword), which sets `ε` to the reciprocal of the median distance
to each point's `neighbors`-th nearest neighbor.
"""
struct Wendland{K <: KernelFunctions.PiecewisePolynomialKernel,
                T <: Union{Real, Nothing}} <: RadialBasisFunction
    kernel::K
    ε::T
end

function Wendland(dim::Integer, degree::Integer; ε::Union{Real, Nothing} = nothing)
    0 <= degree <= 3 || throw(ArgumentError(
        "Wendland degree must be 0, 1, 2 or 3, got $degree"))
    dim >= 1 || throw(ArgumentError(
        "Wendland dim must be a positive integer, got $dim"))
    ε === nothing || ε > 0 || throw(ArgumentError(
        "Wendland shape parameter ε must be positive, got $ε"))

    kernel = KernelFunctions.PiecewisePolynomialKernel(; degree = Int(degree), dim = Int(dim))
    Wendland(kernel, ε)
end

const _unsetShapeMessage = "this Wendland kernel has no shape parameter yet; pass ε " *
    "to the constructor or let interpolate resolve it from the data via the " *
    "neighbors keyword"

(w::Wendland{<:Any, <:Real})(r) = KernelFunctions.kappa(w.kernel, w.ε * r)
(w::Wendland{<:Any, Nothing})(r) = throw(ArgumentError(_unsetShapeMessage))

"""
    support_radius(rbf)

Radius beyond which `rbf(r)` is identically zero, or `Inf` for globally supported basis
functions. A finite support radius is the dispatch hook for a future sparse-assembly
interpolation path (see the RBF-PUM design spec, Future Work).
"""
support_radius(::AbstractRadialBasisFunction) = Inf
support_radius(w::Wendland{<:Any, <:Real}) = 1 / w.ε
support_radius(::Wendland{<:Any, Nothing}) = throw(ArgumentError(_unsetShapeMessage))

# The wrapped PiecewisePolynomialKernel does not depend on ε (ε is applied outside it,
# in kappa(kernel, ε*r)), so a re-shaped Wendland reuses it directly — dim and degree
# are preserved without reconstruction. Tuning candidates are always positive, so the
# public constructor's ε > 0 check is not needed here.
hasshape(::Wendland) = true
withshape(w::Wendland, ε) = Wendland(w.kernel, ε)

"""
    resolveshape(rbf, points, metric, neighbors)

Resolve an unset shape parameter from the data, returning `(rbf, tree)`. `tree` is the
`KDTree` built to answer the nearest-neighbor query (reused by callers that need it
afterwards, e.g. for a sparse assembly path), or `nothing` when no query was needed.

The fallback method (any `rbf` other than a `Wendland` with unset `ε`) returns the
`rbf` unchanged and `tree = nothing`.
"""
resolveshape(rbf, points, metric, neighbors) = (rbf, nothing)

function resolveshape(w::Wendland{<:Any, Nothing}, points, metric, neighbors)
    metric isa Distances.MinkowskiMetric || throw(ArgumentError(
        "resolving Wendland ε from the data requires a Minkowski-family metric " *
        "(Euclidean, Cityblock, Chebyshev or Minkowski); pass ε explicitly instead"))
    neighbors >= 1 || throw(ArgumentError(
        "neighbors must be at least 1, got $neighbors"))

    n = size(points, 2)
    tree = KDTree(points, metric)
    k = min(neighbors + 1, n)
    sample = round.(Int, range(1, n; length = min(n, 100)))
    _, dists = knn(tree, points[:, sample], k)
    r = median(maximum.(dists))
    r > 0 || throw(ArgumentError(
        "cannot resolve a Wendland support radius: sampled neighbor distances are " *
        "all zero (duplicate points?); pass ε explicitly"))

    (withshape(w, 1 / r), tree)
end
