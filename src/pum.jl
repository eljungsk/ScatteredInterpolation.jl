# Radial basis function partition of unity method (RBF-PUM).
#
# The data bounding box is covered with a regular grid of overlapping spherical
# patches. A small dense RBF interpolant is solved per patch (reusing the standard
# RBF machinery) and patches are blended with smooth, compactly supported
# partition-of-unity weights.

# Regular grid of overlapping spherical patches covering the data bounding box.
# The tuple fields make grid traversal type-stable for any dimension D.
struct PatchGrid{T <: AbstractFloat, D}
    origin::NTuple{D, T}    # lower corner of the bounding box
    spacing::NTuple{D, T}   # cell size per dimension (positive also for flat dimensions)
    ncells::NTuple{D, Int}  # number of cells per dimension
    radius::T               # patch radius
end

function buildgrid(points::AbstractMatrix{T}, pointsperpatch::Integer,
                   overlap::Real) where {T <: AbstractFloat}
    d, n = size(points)
    lo = vec(minimum(points, dims = 2))
    hi = vec(maximum(points, dims = 2))

    nside = max(1, ceil(Int, (n / pointsperpatch)^(1 / d)))
    ncells = Vector{Int}(undef, d)
    spacing = Vector{T}(undef, d)
    for i in 1:d
        extent = hi[i] - lo[i]
        if extent > 0
            ncells[i] = nside
            spacing[i] = extent / nside
        else
            # Flat dimension: a single cell of arbitrary positive size, so index
            # arithmetic stays finite.
            ncells[i] = 1
            spacing[i] = one(T)
        end
    end

    # Patch radius: overlap × half cell diagonal over all dimensions. Flat dimensions
    # contribute their artificial unit spacing, which exactly compensates the
    # half-cell offset centerof gives patch centers in those dimensions (a data point
    # there sits 0.5·spacing from the center); excluding them would leave flat data
    # uncovered by every patch. diag2 > 0 always, since every dimension contributes
    # either a real spacing² or 1.
    diag2 = zero(T)
    for i in 1:d
        diag2 += spacing[i]^2
    end
    radius = T(overlap) * sqrt(diag2) / 2

    PatchGrid{T, d}(Tuple(lo), Tuple(spacing), Tuple(ncells), radius)
end

# Grid cell containing x, clamped to the grid (queries outside the bounding box map
# to the nearest boundary cell).
@inline function cellof(grid::PatchGrid{T, D}, x) where {T, D}
    CartesianIndex(ntuple(
        i -> clamp(floor(Int, (x[i] - grid.origin[i]) / grid.spacing[i]) + 1,
                   1, grid.ncells[i]),
        Val(D)))
end

# Center of grid cell ci.
@inline function centerof(grid::PatchGrid{T, D}, ci::CartesianIndex{D}) where {T, D}
    ntuple(i -> grid.origin[i] + (ci[i] - T(0.5)) * grid.spacing[i], Val(D))
end

# Assign data points to patches. Returns:
# - patchpoints: per non-empty patch, sorted indices of the data points it contains
# - centers:     d × P matrix of the non-empty patch centers
# - cellpatches: per grid cell, the patches whose ball intersects that cell — the
#                candidate list used for O(1) patch lookup during evaluation
function assignpatches(points::Matrix{T}, grid::PatchGrid{T, D}) where {T, D}
    tree = KDTree(points)
    cells = CartesianIndices(grid.ncells)

    patchpoints = Vector{Vector{Int}}()
    centerlist = Vector{NTuple{D, T}}()
    patchcell = Vector{CartesianIndex{D}}()
    for ci in cells
        c = centerof(grid, ci)
        idxs = inrange(tree, collect(c), grid.radius)
        isempty(idxs) && continue   # empty patches are dropped
        push!(patchpoints, sort!(idxs))
        push!(centerlist, c)
        push!(patchcell, ci)
    end

    P = length(patchpoints)
    centers = Matrix{T}(undef, D, P)
    for p in 1:P, i in 1:D
        centers[i, p] = centerlist[p][i]
    end

    # For each patch, register it in every cell its ball can reach. A ball of radius
    # R centered in a cell reaches at most ceil(R / spacing) cells away per dimension
    # (conservative). Together with center-in-cell this over-covers, which is safe:
    # evaluation re-checks the exact distance.
    stencil = ntuple(i -> ceil(Int, grid.radius / grid.spacing[i]), Val(D))
    cellpatches = [Int[] for _ in 1:length(cells)]
    lin = LinearIndices(cells)
    for p in 1:P
        ci = patchcell[p]
        ranges = ntuple(
            i -> max(1, ci[i] - stencil[i]):min(grid.ncells[i], ci[i] + stencil[i]),
            Val(D))
        for cj in CartesianIndices(ranges)
            push!(cellpatches[lin[cj]], p)
        end
    end

    patchpoints, centers, cellpatches
end

export PartitionOfUnity, addpoints!

"""
    PartitionOfUnity(method; pointsperpatch = 80, overlap = 1.5, weight = nothing)

Radial basis function partition of unity method (RBF-PUM) for large datasets. The data
bounding box is covered with a regular grid of overlapping spherical patches; a small
dense RBF interpolant using `method` (any radial basis function) is solved per patch,
and patches are blended with smooth partition-of-unity weights, preserving exact
interpolation at the data points.

`pointsperpatch` sets the targeted average number of points per patch (the trade-off
between many small solves and few large ones). `overlap` inflates the patch radius
relative to the grid cell half-diagonal and must be greater than 1 so the patches cover
the whole bounding box; larger values increase smoothness of the blend regions at a
higher evaluation cost. `weight` is the partition-of-unity weight function `ψ(r)` with
support on `[0, 1]`; the default `nothing` selects the C² Wendland function of the data
dimension at `interpolate` time.

Only the `Euclidean` metric is supported. The `smooth` and `linsolve` keywords of
`interpolate` are forwarded to every per-patch solve. Additional points can be added to
the returned interpolant with [`addpoints!`](@ref).
"""
struct PartitionOfUnity{M <: AbstractRadialBasisFunction, W} <: InterpolationMethod
    method::M
    pointsperpatch::Int
    overlap::Float64
    weight::W
end

function PartitionOfUnity(method::AbstractRadialBasisFunction;
                          pointsperpatch::Integer = 80, overlap::Real = 1.5,
                          weight = nothing)
    pointsperpatch >= 1 || throw(ArgumentError(
        "pointsperpatch must be at least 1, got $pointsperpatch"))
    overlap > 1 || throw(ArgumentError(
        "overlap must be greater than 1 to guarantee full patch coverage, got $overlap"))

    PartitionOfUnity(method, Int(pointsperpatch), Float64(overlap), weight)
end

mutable struct PartitionOfUnityInterpolant{T <: AbstractFloat, D, S <: AbstractArray,
                                           W, PU <: PartitionOfUnity, SM, LS, M,
                                           KT} <: ScatteredInterpolant
    grid::PatchGrid{T, D}
    patchpoints::Vector{Vector{Int}}
    locals::Vector{RadialBasisInterpolant}
    centers::Matrix{T}
    centertree::KT              # KDTree over patch centers, for uncovered-query fallback
    cellpatches::Vector{Vector{Int}}
    weight::W
    points::Matrix{T}
    samples::S
    method::PU
    smooth::SM                  # stored so addpoints! re-solves match the build
    linsolve::LS
    metric::M
end

# Per-patch views of the sample data and smoothing parameter
patchsamples(samples::AbstractVector, idxs) = samples[idxs]
patchsamples(samples::AbstractMatrix, idxs) = samples[idxs, :]
patchsmooth(smooth::Number, idxs) = smooth
patchsmooth(smooth::AbstractVector, idxs) = smooth[idxs]

function interpolate(pum::PartitionOfUnity, points::AbstractArray{<:Real, 2},
                     samples::AbstractArray{<:Number, N};
                     metric = Euclidean(),
                     smooth::Union{S, AbstractVector{S}} = false,
                     linsolve = nothing) where {N} where {S <: Number}

    metric isa Euclidean || throw(ArgumentError(
        "PartitionOfUnity only supports the Euclidean metric, since the patch " *
        "geometry is Euclidean; got $(typeof(metric))"))
    @assert smooth != true "set the smoothing value as a number or vector of numbers"
    size(points, 2) == size(samples, 1) || throw(DimensionMismatch(
        "got $(size(points, 2)) points but $(size(samples, 1)) sample rows"))

    T = float(eltype(points))
    pts = Matrix{T}(points)
    d = size(pts, 1)

    grid = buildgrid(pts, pum.pointsperpatch, pum.overlap)
    patchpoints, centers, cellpatches = assignpatches(pts, grid)
    weight = pum.weight === nothing ? Wendland(d, 1) : pum.weight

    # Local solves are independent — thread across patches. Each per-patch system is
    # small (~pointsperpatch), where single-threaded BLAS per task is appropriate.
    P = length(patchpoints)
    locals = Vector{RadialBasisInterpolant}(undef, P)
    Threads.@threads for p in 1:P
        idxs = patchpoints[p]
        locals[p] = interpolate(pum.method, pts[:, idxs], patchsamples(samples, idxs);
                                metric = metric, smooth = patchsmooth(smooth, idxs),
                                linsolve = linsolve)
    end

    PartitionOfUnityInterpolant(grid, patchpoints, locals, centers, KDTree(centers),
                                cellpatches, weight, pts, collect(samples), pum,
                                smooth, linsolve, metric)
end
