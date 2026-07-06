# Sparse assembly and solve for compactly supported radial basis functions (currently
# Wendland; any RBF with a finite `support_radius` could use this path). Distances
# beyond the support radius are never computed: a KDTree range query restricts the
# assembly to the pairs that actually contribute, giving a genuinely sparse matrix
# instead of a dense one filled mostly with zeros.

export CompactSupportRBFInterpolant

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
