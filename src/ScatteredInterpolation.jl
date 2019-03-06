module ScatteredInterpolation

using Distances, NearestNeighbors, Combinatorics, LinearAlgebra

export interpolate,
    evaluate

abstract type ScatteredInterpolant end
abstract type InterpolationMethod end

include("./rbf.jl")
include("./idw.jl")
include("./nearestNeighbor.jl")


# Fallback method for the case of just one point
function evaluate(itp::ScatteredInterpolant, points::AbstractArray{<:Real, 1})

    # pairwise requires the points array to be 2-d.
    n = length(points)
    points = reshape(points, n, 1)

    evaluate(itp, points)
end

"""
    interpolate(method, points, samples; metric = Euclidean(), returnRBFmatrix = false; smooth = false)

Create an interpolation of the data in `samples` sampled at the locations defined in
`points` based on the interpolation method `method`. `metric` is any of the metrics defined
by the `Distances` package. The RBF matrix used for solving the weights can be returned with
the boolean `returnRBFmatrix`. Note that this option is only valid for RadialBasisFunction
interpolations.

`points` should be an ``n×k`` matrix, where ``n`` is dimension of the sampled space and
``k`` is the number of points. This means that each column in the matrix defines one point.

`samples` is an ``k×m`` array, where ``k`` is the number of sampled points (same as for
`points`) and ``m`` is the dimension of the sampled data.

The RadialBasisFunction interpolation supports the use of unique RBF functions and widths
for each sampled point by supplying `method` with a vector of interpolation methods
of length ``k``.

The RadialBasisFunction interpolation also supports smoothing of the data points using
ridge regression. All points can be smoothed equally supplying a scalar value,
alternatively each point can be smoothed independently by supplying a vector of smoothing
values. Note that it is no longer interpolating when using smoothing. 

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