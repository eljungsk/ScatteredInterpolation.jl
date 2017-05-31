
export NearestNeighbor

"""
    NearestNeigbor()

Nearest neighbor "interpolation".

"""
struct NearestNeighbor end

struct NearestNeighborInterpolant{T, TT, N} <: ScatteredInterpolant

    data::Array{T,N}
    tree::TT
end

function interpolate(nn::NearestNeighbor, points::Array{T,2}, samples::Array{T,N}; metric = Euclidean()) where T <: Number where N

    # Build a kd-tree of the points
    tree = KDTree(points, metric)

    return NearestNeighborInterpolant(samples, tree)
end

function evaluate(itp::NearestNeighborInterpolant, points::Array{T,2}) where T <: Real

    # Get the indices for each points closest neighbor
    inds, _ = knn(itp.tree, points, 1)

    m = size(points, 2)
    n = size(itp.data, 2)
    values = zeros(m, n)

    # knn returns a vector of vectors, so we need a loop
    for i in 1:m
        values[i, :] = itp.data[inds[i][1], :]
    end
    
    return values
end
