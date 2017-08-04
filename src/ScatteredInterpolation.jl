module ScatteredInterpolation

using Distances, 
    NearestNeighbors,
    Compat

export interpolate,
    evaluate

@compat abstract type ScatteredInterpolant end
@compat abstract type InterpolationMethod end

include("./rbf.jl")
include("./idw.jl")
include("./nearestNeighbor.jl")


# Fallback method for the case of just one point
@compat function evaluate(itp::ScatteredInterpolant, points::Array{<:Real, 1})

    # pairwise requires the points array to be 2-d.
    n = length(points)
    points = reshape(points, n, 1)

    evaluate(itp, points)
end

"""
    interpolate(method, points, samples[; metric])

Create an interpolation of the data in `samples` sampled at the locations defined in 
`points` based on the interpolation method `method`. `metric` is any of the metrics defined 
by the `Distances` package.

`points` should be an ``n×k`` matrix, where ``n`` is dimension of the sampled space and 
``k`` is the number of points. This means that each column in the matrix defines one point.

`samples` is an ``k×m`` array, where ``k`` is the number of sampled points (same as for
`points`) and ``m`` is the dimension of the sampled data.

The returned `ScatteredInterpolant` object can be passed to `evaluate` to interpolate the
data to new points.
"""
function interpolate end


"""
    evaluate(itp, points)

Evaluate an interpolation object `itp` at the locations defined in `points`.

`points` should be an ``n×k`` matrix, where ``n`` is dimension of the sampled space and 
``k`` is the number of points. This means that each column in the matrix defines one point.
"""
function evaluate end
end
