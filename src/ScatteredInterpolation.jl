module ScatteredInterpolation

using Distances

export interpolate,
    evaluate

abstract type ScatteredInterpolant end

include("./rbf.jl")


# Fallback method for the case of just one point
function evaluate(itp::ScatteredInterpolant, points::Array{T, 1}) where T <: Number

    # pairwise requires the points array to be 2-d.
    n = length(points)
    points = reshape(points, n, 1)

    evaluate(itp, points)
end

end
