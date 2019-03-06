
export NearestNeighbor

"""
    NearestNeigbor()

Nearest neighbor interpolation.

"""
struct NearestNeighbor <: InterpolationMethod end

struct NearestNeighborInterpolant{T, TT} <: ScatteredInterpolant
    data::Array{T}
    tree::TT
end

function interpolate(nn::NearestNeighbor,
                     points::AbstractArray{<:Real,2},
                     samples::AbstractArray{<:Number,N};
                     metric = Euclidean()) where {N}

    # Build a kd-tree of the points
    # If we get an adjoint, make a copy (or else KDTree will be sad)
    if points isa LinearAlgebra.Adjoint
        points = copy(points)
    end
    if samples isa LinearAlgebra.Adjoint
        samples = copy(samples)
    end
    tree = KDTree(points, metric)

    return NearestNeighborInterpolant(samples, tree)
end

function evaluate(itp::NearestNeighborInterpolant, points::AbstractArray{<:Real,2})

    # Get the indices for each points closest neighbor
    inds, _ = knn(itp.tree, points, 1)

    m = size(points, 2)
    n = size(itp.data, 2)
    values = zeros(eltype(itp.data), m, n)

    # knn returns a vector of vectors, so we need a loop
    for i in 1:m
        values[i, :] = itp.data[inds[i][1], :]
    end
    
    return values
end