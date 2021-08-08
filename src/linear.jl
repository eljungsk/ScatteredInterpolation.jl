
export Linear

"""
    Linear()

Linear interpolation based on the Delaunay triangulation of the sample points.

"""
struct Linear <: InterpolationMethod end

struct LinearInterpolant{S, BT} <: ScatteredInterpolant where {S <: AbstractArray{<:Number, N}, BT} where {N}
    samples::S
    barycentric_setup::BT
end

function interpolate(m::Linear,
            points::AbstractArray{<:Real, 2},
            samples::AbstractArray{<:Number, N}) where {N}

    # Delaunay.jl requires Float64 for the points at the moment
    @assert eltype(points) == Float64 "Linear interpolation is only supported for Float64 point coordinates"

    dim = size(points, 1)
    spoints = copy_svec(Float64, points, Val(dim))

    # Build the Delaunay triangulation
    # Delaunay.jl expects the points array to be nPoints x nDims, we take the opposite as input
    points = permutedims(points)
    mesh = delaunay(points)

    # Extract each simplex as a coordinate matrix with one point per row, and prepare for computing the Barycentric coordinates
    nSimplices = size(mesh.simplices, 1)
    simplices = [SMatrix{dim+1, dim, eltype(points)}(points[mesh.simplices[i, :], :]) for i in 1:nSimplices]
    barycentric_setup = cartesian2barycentric_setup.(simplices)

    # Group the samples for each simplex
    grouped_samples = [SMatrix{dim+1, size(samples, 2), eltype(samples)}(samples[mesh.simplices[i, :], :]) for i in 1:nSimplices]

    return LinearInterpolant(grouped_samples, barycentric_setup)
end

function evaluate(itp::LinearInterpolant, points::AbstractArray{<:Real, 2})

    # Compute a reasonable type for the output data
    T = StaticArrays.arithmetic_closure(eltype(itp.samples[1]))

    dim = size(itp.samples[1], 1) - 1
    spoints = copy_svec(eltype(points), points, Val(dim))

    m = length(spoints)
    n = size(itp.samples[1], 2)
    values = zeros(T, m, n)

    for (i, point) in enumerate(spoints)
        (ind, bCoord) = find_simplex(itp, point)
        values[i,:] = bCoord'*itp.samples[ind]
    end

    return values
end

# Find the simplex containing the point to interpolate, and return the barycentric coordinates
function find_simplex(itp, point)

    for (i, bs) in enumerate(itp.barycentric_setup)
        coord = cartesian2barycentric(bs, point)

        if all(coord .≥ 0) && all(coord .≤ 1)
            return (i, coord)
        end
    end

    error("Data out of range at $point. Extrapolation is not supported.")
end

# Helper function to copy the points array into a vector of SVector.
# Borrowed from NearestNeighbors.jl
@inline function copy_svec(::Type{T}, data, ::Val{dim}) where {T, dim} 
    [SVector{dim, T}(ntuple(i -> data[n+i], Val(dim))) for n in 0:dim:(length(data)-1)]
end