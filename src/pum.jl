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
    PartitionOfUnity(method; pointsperpatch = 80, overlap = 1.5, weight = nothing,
                     tune = :none)

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

`tune` selects automatic per-patch shape-parameter tuning: `:none` uses `method` as
given, while `:loocv` chooses each patch's shape parameter `ε` by exact leave-one-out
cross-validation (Rippa's method), scanning candidates scaled to the patch's mean
nearest-neighbor spacing plus the shape parameter stored in `method` itself (used as
a fallback reference, not ignored); the kernel must have a shape parameter to tune
(`Polyharmonic`, `ThinPlate` and `GeneralizedPolyharmonic` do not).

Only the `Euclidean` metric is supported. The `smooth` and `linsolve` keywords of
`interpolate` are forwarded to every per-patch solve. Additional points can be added to
the returned interpolant with [`addpoints!`](@ref).
"""
struct PartitionOfUnity{M <: AbstractRadialBasisFunction, W} <: InterpolationMethod
    method::M
    pointsperpatch::Int
    overlap::Float64
    weight::W
    tune::Symbol
end

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

# ---- LOOCV shape-parameter tuning (tune = :loocv) --------------------------------

# Bounds of the shape-parameter search: ε = c / h for the patch's mean
# nearest-neighbor distance h, c ranging flat-to-peaked. Historically an 11-point
# half-octave grid scanned in full (superseded below by a golden-section search over
# the same [min, max] range); still used directly for the too-few-points fallback
# (`cmid`, its geometric middle) and to fix the golden-section search's bracket.
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

# Reciprocal-condition-number gate below which a candidate's LOOCV score is not
# merely noisy but can be *confidently wrong*: explicit inv() computes diag(A⁻¹) with
# relative error ~ eps/rcond, so once rcond drops under ~1e-12 the "error" Rippa's
# formula reports has no correct digits, yet reads as a small, trustworthy-looking
# number — verified empirically (docs/superpowers/plans/
# 2026-07-04-loocv-shape-selection.md, Task 5 amendment): at cond(A) ≈ 1e18,
# inv(A)*b disagreed with A\b by O(1) while both LOOCV scores looked fine. This is
# not a "somewhat less accurate" regime, it's `A⁻¹` having zero correct digits — no
# amount of scoring cleverness recovers information that isn't there, so gated-out
# candidates are simply not scored (Inf), never selected. `rcond` is obtained from
# the LU factorization via LAPACK gecon! (not Cholesky: Multiquadratic-family
# kernels are indefinite), reusing the same factorization `gatedloocvscore`
# (src/rippa.jl) needs for the score itself — see the Task 8 amendment there for why
# this is no longer a second, independent factorization.
const RCOND_THRESHOLD = 1e-8

# Golden-section search replaces an earlier fixed 11-point linear scan over
# LOOCV_CANDIDATES (docs/superpowers/plans/2026-07-04-loocv-shape-selection.md, Task
# 8 amendment): the LOO score is empirically a clean single-trough (unimodal)
# function of log(ε) in the safe conditioning region — the RBF trade-off principle
# predicts exactly this shape (too flat ⟹ ill-conditioned/uninformative, too peaked
# ⟹ no cross-point information, one interior optimum between) — and where a fixed
# kernel is genuinely mis-scaled for the local patch, that interior optimum lines up
# with the true off-node error minimum, not just the LOO proxy (verified directly:
# scored a synthetic localized-peak patch across 41 log-spaced ε and compared each
# LOO score against true holdout error — both minimized at the same ε). Golden
# section is comparison-based (only ever asks "is f(c) < f(d)?"), so Inf-scored
# (gated-out) candidates compare correctly without special-casing, unlike
# derivative- or interpolation-based methods (e.g. Brent's parabolic step) that
# assume finite values throughout the bracket.
function goldensectionmin(f, a, b; iters::Integer)
    invphi = (sqrt(5) - 1) / 2
    invphi2 = (3 - sqrt(5)) / 2
    h = b - a
    c = a + invphi2 * h
    d = a + invphi * h
    fc = f(c)
    fd = f(d)
    for _ in 1:iters
        if fc < fd
            b, d, fd = d, c, fc
            h *= invphi
            c = a + invphi2 * h
            fc = f(c)
        else
            a, c, fc = c, d, fd
            h *= invphi
            d = a + invphi * h
            fd = f(d)
        end
    end
    return fc < fd ? (c, fc) : (d, fd)
end

# 6 golden-section iterations (8 score evaluations total, including the two initial
# points) shrink the log2(c) bracket to ~5.6% of its original width — finer than the
# ~41%-per-step (2^0.5) resolution of the old 11-point grid — while evaluating fewer
# candidates than that grid did.
const LOOCV_GOLDEN_ITERS = 6

# Select the best shape parameter for one patch: score the caller's own kernel with
# the plain (ungated) Rippa score as the baseline, then let a golden-section search
# over the scale-derived candidate range challenge it — but only through the gated
# `gatedloocvscore`. The search result replaces the anchor only if it both passes
# the reliability gate AND scores better than the anchor. This asymmetry is
# deliberate: the anchor is the kernel the caller (or the untuned path) would use
# anyway, so it is never rejected for being "unscoreable" — a flat, unscoreable
# anchor on smooth data typically has an extremely good (tiny) LOO score in exact
# arithmetic even though it cannot be certified, and demanding gate-passing on both
# sides only lets tuning move away from the caller's kernel when a *trustworthy*
# alternative is genuinely better (see docs/superpowers/plans/
# 2026-07-04-loocv-shape-selection.md, Task 5 amendment: an earlier design that
# gated the anchor too let a merely-safe-but-worse candidate override a
# good-but-unscoreable anchor, causing ~250-500x regressions). The distance matrix
# does not depend on ε and is computed once; each candidate only re-applies the
# kernel, adds smoothing, and factorizes the small dense system. The final solve of
# the winner runs through the standard interpolate path afterwards (where the
# user's linsolve choice applies); the search itself deliberately uses direct
# factorization instead.
function tuneshape(kernel::RadialBasisFunction, pts::AbstractMatrix,
                   samples::AbstractVecOrMat, smooth, metric)
    h = meannndist(pts)
    h > 0 || return kernel   # no length scale to derive ε from
    cmid = LOOCV_CANDIDATES[(length(LOOCV_CANDIDATES) + 1) ÷ 2]
    size(pts, 2) >= 3 || return withshape(kernel, cmid / h)

    R = pairwise(metric, pts, dims = 2)
    A = similar(R)

    A .= kernel.(R)
    addSmoothing!(A, smooth)
    best = loocvscore(A, samples)   # ungated: the anchor is never rejected as unscoreable
    εbest = kernel.ε

    lo, hi = extrema(log2, LOOCV_CANDIDATES)
    function f(t)
        ε = exp2(t) / h
        ϕ = withshape(kernel, ε)
        A .= ϕ.(R)
        addSmoothing!(A, smooth)
        return gatedloocvscore(A, samples, RCOND_THRESHOLD)   # gated: only a trustworthy score may win
    end
    tbest, sbest = goldensectionmin(f, lo, hi; iters = LOOCV_GOLDEN_ITERS)
    if sbest < best
        best = sbest
        εbest = exp2(tbest) / h
    end
    return withshape(kernel, εbest)
end

# Bordered variant of `tuneshape` for polynomial-augmented kernels
# (GeneralizedMultiquadratic): candidates (and the caller's own kernel, as the
# anchor) are scored on the saddle system M = [A P; Pᵀ 0] via the bordered
# `gatedloocvscore`/`loocvscore`. The polynomial block P is ε-independent and
# assembled once; each candidate only refills the A-block view of M in place.
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

    Ablock = view(M, 1:np, 1:np)
    Ablock .= kernel.(R)
    addSmoothing!(Ablock, smooth)
    best = loocvscore(M, F, np)   # ungated: the anchor is never rejected as unscoreable
    εbest = kernel.ε

    lo, hi = extrema(log2, LOOCV_CANDIDATES)
    function f(t)
        ε = exp2(t) / h
        ϕ = withshape(kernel, ε)
        Ablock .= ϕ.(R)
        addSmoothing!(Ablock, smooth)
        return gatedloocvscore(M, F, np, RCOND_THRESHOLD)   # gated: only a trustworthy score may win
    end
    tbest, sbest = goldensectionmin(f, lo, hi; iters = LOOCV_GOLDEN_ITERS)
    if sbest < best
        best = sbest
        εbest = exp2(tbest) / h
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
            locals[p] = solvelocal(pum, pts[:, idxs], patchsamples(samples, idxs),
                                   patchsmooth(smooth, idxs), metric, linsolve)
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

"""
    addpoints!(itp, points, samples)

Add new data points to an existing [`PartitionOfUnity`](@ref) interpolant without a
full rebuild: only the local systems of the patches covering the new points are
re-solved (using the same `smooth` and `linsolve` settings as the original build,
and re-tuning the shape parameter of affected patches when the interpolant was
built with `tune = :loocv`).

The patch grid is fixed at construction, so every new point must lie inside the
bounding box of the original data and inside at least one existing patch; otherwise an
`ArgumentError` suggests rebuilding with `interpolate`. Interpolants built with a
per-point smoothing *vector* are not supported (the smoothing values of the new points
would be ambiguous) — rebuild instead.

`points` and `samples` follow the same layout as in [`interpolate`](@ref); `samples`
must have the same second dimension as the original sample data. Returns `itp`.
"""
function addpoints!(itp::PartitionOfUnityInterpolant{T, D},
                    points::AbstractArray{<:Real, 2},
                    samples::AbstractArray{<:Number}) where {T, D}

    size(points, 1) == D || throw(DimensionMismatch(
        "the interpolant was built in $D dimensions, but the new points have " *
        "dimension $(size(points, 1))"))
    size(points, 2) == size(samples, 1) || throw(DimensionMismatch(
        "got $(size(points, 2)) new points but $(size(samples, 1)) new sample rows"))
    size(samples, 2) == size(itp.samples, 2) || throw(DimensionMismatch(
        "new samples have $(size(samples, 2)) columns but the original samples " *
        "have $(size(itp.samples, 2))"))
    itp.smooth isa AbstractVector && throw(ArgumentError(
        "addpoints! does not support interpolants built with a per-point smoothing " *
        "vector; rebuild with interpolate instead"))

    grid = itp.grid
    newpts = Matrix{T}(points)
    nnew = size(newpts, 2)

    # The patch grid is fixed at build time: reject points outside the bounding box
    # of the data. The box is computed from the stored points rather than from the
    # grid, whose flat dimensions carry an artificial unit spacing that would
    # otherwise admit off-plane points.
    lo = vec(minimum(itp.points, dims = 2))
    hi = vec(maximum(itp.points, dims = 2))
    for q in 1:nnew, i in 1:D
        lo[i] <= newpts[i, q] <= hi[i] || throw(ArgumentError(
            "new point $q lies outside the bounding box of the original data; the " *
            "patch grid is fixed at construction — rebuild with interpolate"))
    end

    # Find the covering patches of every new point before mutating any state, so a
    # coverage error cannot leave the interpolant half-updated. The range query
    # against the patch-center tree returns exactly the patches whose ball contains
    # the point.
    covering = inrange(itp.centertree, newpts, grid.radius)
    for q in 1:nnew
        isempty(covering[q]) && throw(ArgumentError(
            "new point $q is not covered by any existing patch (that region held no " *
            "data when the interpolant was built) — rebuild with interpolate"))
    end

    # Append the data and update patch membership. Adding the point to *every*
    # covering patch preserves exact interpolation at the new points.
    n0 = size(itp.points, 2)
    itp.points = hcat(itp.points, newpts)
    itp.samples = vcat(itp.samples, collect(samples))
    affected = Set{Int}()
    for q in 1:nnew, p in covering[q]
        push!(itp.patchpoints[p], n0 + q)
        push!(affected, p)
    end

    # Re-solve only the affected local systems (threaded and BLAS-pinned, like the
    # build).
    aff = collect(affected)
    withpinnedblas() do
        Threads.@threads for k in eachindex(aff)
            p = aff[k]
            idxs = itp.patchpoints[p]
            itp.locals[p] = solvelocal(itp.method, itp.points[:, idxs],
                                       patchsamples(itp.samples, idxs),
                                       itp.smooth, itp.metric, itp.linsolve)
        end
    end

    itp
end

addpoints!(itp::ScatteredInterpolant, points, samples) = throw(ArgumentError(
    "addpoints! is only supported for PartitionOfUnity interpolants; rebuild with " *
    "interpolate instead"))
