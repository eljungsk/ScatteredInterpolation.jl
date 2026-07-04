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

    # Subdivide only the non-flat dimensions, and calibrate the per-side cell count so
    # the expected number of points in a patch ball matches pointsperpatch: a patch is
    # a ball of radius overlap·(half cell diagonal), whose volume exceeds a cell's by
    # kd = V(1) · (overlap·√deff/2)^deff with V(1) the unit-ball volume in deff
    # dimensions. Without this correction patch occupancy grows geometrically with
    # the dimension (≈22× the target at d = 6).
    deff = count(i -> hi[i] > lo[i], 1:d)
    nside = 1
    if deff > 0
        ballvol = 1.0   # unit-ball volume via the recursion V_k = V_{k-2} · 2π/k
        for k in (iseven(deff) ? 2 : 1):2:deff
            ballvol *= k == 1 ? 2.0 : 2π / k
        end
        kd = ballvol * (Float64(overlap) * sqrt(deff) / 2)^deff
        nside = max(1, ceil(Int, (n * kd / pointsperpatch)^(1 / deff)))
    end
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

# Center of grid cell ci.
@inline function centerof(grid::PatchGrid{T, D}, ci::CartesianIndex{D}) where {T, D}
    ntuple(i -> grid.origin[i] + (ci[i] - T(0.5)) * grid.spacing[i], Val(D))
end

# Assign data points to patches. Returns:
# - patchpoints: per non-empty patch, sorted indices of the data points it contains
# - centers:     d × P matrix of the non-empty patch centers
# (Patch lookup during evaluation/insertion uses a KDTree over these centers.)
function assignpatches(points::Matrix{T}, grid::PatchGrid{T, D},
                       tree = KDTree(points)) where {T, D}
    cells = CartesianIndices(grid.ncells)

    patchpoints = Vector{Vector{Int}}()
    centerlist = Vector{NTuple{D, T}}()
    for ci in cells
        c = centerof(grid, ci)
        idxs = inrange(tree, collect(c), grid.radius)
        isempty(idxs) && continue   # empty patches are dropped
        push!(patchpoints, sort!(idxs))
        push!(centerlist, c)
    end

    P = length(patchpoints)
    centers = Matrix{T}(undef, D, P)
    for p in 1:P, i in 1:D
        centers[i, p] = centerlist[p][i]
    end

    patchpoints, centers
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
                                           L <: RadialBasisInterpolant, W,
                                           PU <: PartitionOfUnity, SM, LS, M,
                                           KT} <: ScatteredInterpolant
    grid::PatchGrid{T, D}
    patchpoints::Vector{Vector{Int}}
    locals::Vector{L}
    centers::Matrix{T}
    centertree::KT              # KDTree over patch centers, for uncovered-query fallback
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

# Run f with BLAS pinned to one thread while our own threads are active: the
# per-patch systems are small, and nthreads() × BLAS-threads oversubscription only
# slows them down. Restored afterwards.
function withpinnedblas(f)
    Threads.nthreads() == 1 && return f()
    old = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        return f()
    finally
        BLAS.set_num_threads(old)
    end
end

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
    tree = KDTree(pts)
    patchpoints, centers = assignpatches(pts, grid, tree)
    weight = pum.weight === nothing ? Wendland(d, 1) : pum.weight

    # Patches near the boundary of the data can end up with few, one-sided points
    # (their ball sticks out of the data region), and a starved local interpolant
    # produces wild values in the data-free part of its ball — exactly where its PU
    # weight is still nonzero. Top such patches up with the nearest data points so
    # every local system is well fed. Extra members beyond the ball are harmless:
    # the weights are unchanged, and exactness only requires that every patch whose
    # ball covers a data point interpolates it.
    minpts = min(max(2 * (d + 1), pum.pointsperpatch ÷ 2), size(pts, 2))
    for p in eachindex(patchpoints)
        if length(patchpoints[p]) < minpts
            idxs, _ = knn(tree, view(centers, :, p), minpts)
            patchpoints[p] = sort!(union(patchpoints[p], idxs))
        end
    end

    # Local solves are independent — thread across patches. Each per-patch system is
    # small (~pointsperpatch), where single-threaded BLAS per task is appropriate.
    P = length(patchpoints)
    locals = Vector{RadialBasisInterpolant}(undef, P)
    withpinnedblas() do
        Threads.@threads for p in 1:P
            idxs = patchpoints[p]
            locals[p] = interpolate(pum.method, pts[:, idxs], patchsamples(samples, idxs);
                                    metric = metric, smooth = patchsmooth(smooth, idxs),
                                    linsolve = linsolve)
        end
    end
    # Narrow to the concrete local-interpolant type (homogeneous in practice) so the
    # evaluation hot path dispatches statically.
    locals = [l for l in locals]

    PartitionOfUnityInterpolant(grid, patchpoints, locals, centers, KDTree(centers),
                                weight, pts, collect(samples), pum,
                                smooth, linsolve, metric)
end

function evaluate(itp::PartitionOfUnityInterpolant{T, D},
                  points::AbstractArray{<:Real, 2}) where {T, D}

    size(points, 1) == D || throw(DimensionMismatch(
        "the interpolant was built in $D dimensions, but the evaluation points " *
        "have dimension $(size(points, 1))"))

    grid = itp.grid
    P = length(itp.locals)
    nq = size(points, 2)
    m = size(itp.samples, 2)
    Tw = float(promote_type(eltype(points), T))
    Tout = promote_type(Tw, eltype(itp.samples))

    # Pass 1: per query, find the covering patches with one batched range query
    # against the patch-center tree, then compute raw PU weights, grouped by patch so
    # pass 2 can evaluate each local interpolant on one batched block.
    candidates = inrange(itp.centertree, Matrix{T}(points), grid.radius)
    patchq = [Int[] for _ in 1:P]
    patchw = [Tw[] for _ in 1:P]
    wsum = zeros(Tw, nq)
    for q in 1:nq
        x = view(points, :, q)
        for p in candidates[q]
            r2 = zero(Tw)
            for i in 1:D
                r2 += abs2(Tw(x[i]) - itp.centers[i, p])
            end
            r = sqrt(r2)
            r < grid.radius || continue
            ω = Tw(itp.weight(r / grid.radius))
            ω > 0 || continue
            push!(patchq[p], q)
            push!(patchw[p], ω)
            wsum[q] += ω
        end
    end

    # Pass 2 (threaded, BLAS pinned like the build): one batched evaluation per
    # patch — GEMM-shaped work with no shared writes. Local evaluate returns a
    # Vector for vector samples; normalize to a matrix so the scatter below is
    # shape-agnostic.
    results = Vector{Matrix{Tout}}(undef, P)
    withpinnedblas() do
        Threads.@threads for p in 1:P
            if isempty(patchq[p])
                results[p] = Matrix{Tout}(undef, 0, m)
            else
                v = evaluate(itp.locals[p], points[:, patchq[p]])
                results[p] = Matrix{Tout}(reshape(v, length(patchq[p]), m))
            end
        end
    end

    # Pass 3 (sequential): scatter-accumulate weighted patch results. Queries covered
    # by several patches receive several contributions — keeping this phase serial
    # avoids write races without per-thread output copies, and its cost is only
    # O(query–patch pairs).
    out = zeros(Tout, nq, m)
    for p in 1:P
        qs = patchq[p]
        ws = patchw[p]
        R = results[p]
        for (i, q) in enumerate(qs)
            @views out[q, :] .+= ws[i] .* R[i, :]
        end
    end

    # Normalize the partition of unity; queries not covered by any patch (outside the
    # bounding box, or in a region that held no data) fall back to the nearest
    # patch's local interpolant. This is extrapolation and documented as such.
    for q in 1:nq
        if wsum[q] > 0
            @views out[q, :] ./= wsum[q]
        else
            p, _ = nn(itp.centertree, Vector{T}(view(points, :, q)))
            v = evaluate(itp.locals[p], Matrix{T}(reshape(points[:, q], D, 1)))
            @views out[q, :] .= vec(v)
        end
    end

    itp.samples isa AbstractVector ? vec(out) : out
end
