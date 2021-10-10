# Define some points and data in 2D
points = permutedims([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0], (2,1))
points_adjoint = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]'
data = [0.0; 0.5; 0.5; 1.0]

@testset "Linear" begin
    
    @testset "Evaluation" for p in (points, points_adjoint)

        itp = interpolate(Linear(), p, data)

        # Check that we get back the original data at the sample points
        ev = evaluate(itp, points)
        @test ev ≈ data

        # Make sure that we cannot do extrapolation
        @test_throws DomainError evaluate(itp, [2.0; 2.0]) 
        @test_throws DomainError evaluate(itp, [-1.0; -1.0]) 

        # Test for linearity
        tp = [0.2 0.2; 0.4 0.4; 0.6 0.6; 0.8 0.8]'
        ev = evaluate(itp, tp)
        @test ev ≈ tp[1,:]
    end
end
