# Sparse assembly and solve for compactly supported radial basis functions (currently
# Wendland; any RBF with a finite `support_radius` could use this path). Distances
# beyond the support radius are never computed: a KDTree range query restricts the
# assembly to the pairs that actually contribute, giving a genuinely sparse matrix
# instead of a dense one filled mostly with zeros.

export CompactSupportRBFInterpolant

# Routing thresholds for the `sparse = :auto` decision in `interpolate` (rbf.jl).
# Calibrated on d = 2 golden-ratio lattice, 2026-07-06:
#   n-crossover: sparse wins at all n ≥ 500 with low fill (0.005s vs 0.008s at n=500,
#                43× at n=10000). SPARSE_MIN_POINTS = 500 confirmed.
#   fill-crossover at n = 10000: sparse wins up to ~6% fill (ε=7, 4.2s vs 9.6s),
#                loses at ~11% (ε=5, 41s vs 9.7s). SPARSE_MAX_FILL set to 0.07.
const SPARSE_MIN_POINTS = 500
const SPARSE_MAX_FILL = 0.07
const FILL_SAMPLE_SIZE = 32

"""
    CompactSupportRBFInterpolant

Interpolant produced by [`interpolate`](@ref) for a compactly supported radial basis
function (e.g. [`Wendland`](@ref)). Holds the solved weights `w`, the training
`points`, the `rbf` kernel, the `metric` used to compute distances, and the `tree`
(a `NearestNeighbors.KDTree`) used for the sparse assembly and reused by `evaluate`.
"""
struct CompactSupportRBFInterpolant{T1 <: AbstractArray, T2 <: AbstractMatrix{<:Real},
                                    F, M, KT} <: RadialBasisInterpolant
    w::T1
    points::T2
    rbf::F
    metric::M
    tree::KT
end

# Smoothing (ridge regression) value for point i: a scalar applies uniformly, a vector
# gives one value per point. Mirrors addSmoothing!'s dispatch in rbf.jl.
smoothvalue(smooth::Number, i) = smooth
smoothvalue(smooth::AbstractVector, i) = smooth[i]

"""
    decidesparse(sparse, rbf, points, metric, tree)

Resolve the `sparse` kwarg of `interpolate` (`:auto`, `true` or `false`) into a
concrete `(usesparse::Bool, tree)` decision. `tree` is the `KDTree` built by
`resolveshape` when a Wendland ε needed resolving, or `nothing`; it is built here
(and returned for reuse by the caller) only when actually needed for the decision or
the subsequent sparse assembly.

`sparse = true` throws an `ArgumentError` (via [`sparseineligibility`](@ref)) when the
kernel or metric makes a sparse solve impossible. `sparse = :auto` falls back to the
dense path instead of throwing, and additionally requires at least
`SPARSE_MIN_POINTS` points and an estimated matrix fill (via
[`estimatefill`](@ref)) below `SPARSE_MAX_FILL`.
"""
function decidesparse(sparse, rbf, points, metric, tree)
    sparse === false && return (false, tree)
    sparse === true || sparse === :auto || throw(ArgumentError(
        "sparse must be :auto, true or false, got $(repr(sparse))"))

    reason = sparseineligibility(rbf, metric)
    if sparse === true
        reason === nothing || throw(ArgumentError(reason))
    else
        reason === nothing || return (false, tree)
        size(points, 2) >= SPARSE_MIN_POINTS || return (false, tree)
    end

    tree = tree === nothing ? KDTree(points, metric) : tree
    if sparse === :auto &&
       estimatefill(tree, points, support_radius(rbf)) > SPARSE_MAX_FILL
        return (false, tree)
    end
    (true, tree)
end

"""
    sparseineligibility(rbf, metric)

Return a human-readable reason `String` why `rbf`/`metric` cannot go through the
sparse interpolation path, or `nothing` if they can. Used by
[`decidesparse`](@ref).
"""
function sparseineligibility(rbf, metric)
    if rbf isa AbstractVector
        return "sparse interpolation does not support a vector of per-point " *
               "kernels: there is no single support radius"
    elseif rbf isa GeneralizedRadialBasisFunction
        return "sparse interpolation does not support generalized " *
               "(polynomial-augmented) RBFs: the augmented system is not sparse " *
               "positive definite"
    elseif !isfinite(support_radius(rbf))
        return "$(nameof(typeof(rbf))) is globally supported (nonzero at every " *
               "distance), so its interpolation matrix has no zero entries and " *
               "cannot be sparse. Use a compactly supported kernel (Wendland) for " *
               "a sparse global solve, or PartitionOfUnity for locality with " *
               "this kernel"
    elseif !(metric isa Distances.MinkowskiMetric)
        return "sparse interpolation requires a Minkowski-family metric " *
               "(Euclidean, Cityblock, Chebyshev or Minkowski) for the KDTree " *
               "range search, got $(nameof(typeof(metric)))"
    end
    nothing
end

"""
    estimatefill(tree, points, r)

Estimate the fraction of nonzero entries the sparse RBF matrix would have, by
sampling up to `FILL_SAMPLE_SIZE` points and averaging the fraction of `points` that
fall within radius `r` of each (via a KDTree range query over `tree`). Used by
[`decidesparse`](@ref) to fall back to the dense path when the support radius is wide
enough that the matrix would be nearly dense anyway.
"""
function estimatefill(tree, points, r)
    n = size(points, 2)
    sample = round.(Int, range(1, n; length = min(n, FILL_SAMPLE_SIZE)))
    total = 0
    for i in sample
        total += length(inrange(tree, view(points, :, i), r))
    end
    total / (length(sample) * n)
end

"""
    assemblesparse(rbf, points, tree, metric, smooth)

Assemble the sparse compactly-supported RBF interpolation matrix. For each point `i`,
a KDTree range query over `tree` (radius `support_radius(rbf)`) finds the columns `j`
for which `rbf` is potentially nonzero, and only those entries are evaluated and
stored — pairs outside the support radius are never computed. `smooth` is added to the
diagonal (scalar applies to every point, a vector gives one value per point; `false`
adds nothing).
"""
function assemblesparse(rbf, points::AbstractMatrix{<:Real}, tree, metric, smooth)
    n = size(points, 2)
    r = support_radius(rbf)
    Tv = typeof(rbf(zero(float(eltype(points)))))

    Is = Int[]
    Js = Int[]
    Vs = Tv[]

    for i in 1:n
        xi = view(points, :, i)
        for j in inrange(tree, xi, r)
            v = rbf(metric(xi, view(points, :, j)))
            if j == i && smooth !== false
                v += smoothvalue(smooth, i)
            end
            push!(Is, i)
            push!(Js, j)
            push!(Vs, v)
        end
    end

    sparse(Is, Js, Vs, n, n)
end

"""
    interpolatecsrbf(rbf, points, samples, tree, metric, smooth, linsolve, returnRBFmatrix)

Assemble the sparse compactly-supported RBF matrix and solve for the interpolation
weights via a sparse Cholesky factorization (`LinearSolve.CHOLMODFactorization`, unless
`linsolve` overrides it). Returns a [`CompactSupportRBFInterpolant`](@ref), or a
`(itp, A)` tuple when `returnRBFmatrix` is `true`.
"""
function interpolatecsrbf(rbf, points, samples, tree, metric, smooth, linsolve,
                          returnRBFmatrix)
    A = assemblesparse(rbf, points, tree, metric, smooth)

    alg = linsolve === nothing ? CHOLMODFactorization() : linsolve
    cache = _initsolve(A, alg)
    w = _solve!(cache, samples)

    itp = CompactSupportRBFInterpolant(w, points, rbf, metric, tree)

    returnRBFmatrix ? (itp, A) : itp
end

"""
    evaluate(itp::CompactSupportRBFInterpolant, points)

Evaluate the interpolant at `points` (dimension × number of query points). For each
query point, a KDTree range query over `itp.tree` (radius `support_radius(itp.rbf)`)
finds the training points that can contribute; only those pairs are evaluated and
assembled into a sparse Φ, so points with no in-range neighbors evaluate to exactly
zero rather than requiring a dense pairwise distance matrix.
"""
function evaluate(itp::CompactSupportRBFInterpolant, points::AbstractArray{<:Real, 2})
    size(points, 1) == size(itp.points, 1) || throw(DimensionMismatch(
        "query points have dimension $(size(points, 1)) but the interpolant was " *
        "built for dimension $(size(itp.points, 1))"))

    n = size(itp.points, 2)
    m = size(points, 2)
    r = support_radius(itp.rbf)
    Tv = typeof(itp.rbf(zero(float(eltype(points)))))
    Is = Int[]; Js = Int[]; Vs = Tv[]
    for q in 1:m
        xq = view(points, :, q)
        for j in inrange(itp.tree, xq, r)
            push!(Is, q); push!(Js, j)
            push!(Vs, itp.rbf(itp.metric(xq, view(itp.points, :, j))))
        end
    end
    Φ = sparse(Is, Js, Vs, m, n)
    Φ * itp.w
end
