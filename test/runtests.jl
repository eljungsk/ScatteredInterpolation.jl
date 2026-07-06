using ScatteredInterpolation, Test, LinearAlgebra, LinearSolve, IterativeSolvers

# Import metrics explicitly: `using Distances` would also bring in `Distances.evaluate`,
# which clashes with `ScatteredInterpolation.evaluate` and shadows it out of scope.
using Distances: Cityblock, Euclidean

# Deterministic, RNG-free perturbation used by the interpolation tests. Using a fixed
# pattern instead of `randn` keeps the "evaluate near a sample point" checks reproducible
# across Julia versions (whose RNG streams differ), avoiding spurious CI failures.
# The (2i+1)/7 phase is never an integer, so no component is ever exactly zero (which
# would silently turn a "near a sample point" check into an "at a sample point" check).
perturbation(sz; scale = 5e-4) = reshape([scale * sinpi((2i + 1) / 7) for i in 1:prod(sz)], sz)

@testset "ScatteredInterpolation" begin
    include("rbf.jl")
    include("idw.jl")
    include("nearestNeighbor.jl")
    include("wendland.jl")
    include("rippa.jl")
    include("pum.jl")
end
