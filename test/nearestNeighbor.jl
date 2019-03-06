
# Define some points and data in 2D
points = permutedims([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0], (2,1))
points_adjoint = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]'
data = [0.0; 0.5; 0.5; 1.0]

@testset "NearestNeighbor" begin
    
    @testset "Evaluation" for p in (points, points_adjoint)

        itp = interpolate(NearestNeighbor(), p, data)

        # Check that we get back the original data at the sample points
        ev = evaluate(itp, points)
        @test ev ≈ data

        # Check for a different point, should be closest to the last point
        ev = evaluate(itp, [2.0; 2.0])
        @test ev[1] ≈ data[end]
    end
end
