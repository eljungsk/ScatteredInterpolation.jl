using ScatteredInterpolation, Random

Random.seed!(2)

if VERSION < v"0.7-"
    using Base.Test
else
    using Test
end

include("rbf.jl")
include("idw.jl")
include("nearestNeighbor.jl")
