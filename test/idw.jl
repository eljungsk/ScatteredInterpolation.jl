
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

        # Check that we get back the original data at the sample points
        ev = evaluate(itp, points)
        @test ev ≈ data
    end
end
