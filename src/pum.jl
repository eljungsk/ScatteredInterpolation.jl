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
