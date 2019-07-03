
# Define some points and data in 2D
arrayPoints = permutedims([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0], (2,1))
adjointPoints = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]'
data = [0.0; 0.5; 0.5; 1.0]

@testset "Shepard" begin
    
    @testset "Constructors" for idw in (:Shepard, )
        @eval @test $idw(2) == $idw()
    end

    @testset "Evaluation" for points in (arrayPoints, adjointPoints)

        itp = interpolate(Shepard(), points, data)

        # Check that we get back the original data at the sample points and that we get close
        # when evaluating near the sampling points
        ev = evaluate(itp, points)
        @test ev ≈ data
        ev = evaluate(itp, points .+ 0.001*randn(size(points)))
        @test all(isapprox.(data, ev, atol = 1e-2))
    end
end
