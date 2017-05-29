
# Define some points and data in 2D
points = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]'
data = [0.0; 0.5; 0.5; 1.0]

radialBasisFunctions = (Gaussian{2}(), Multiquadratic{2}(), InverseQuadratic{2}(), InverseMultiquadratic{2}(), Polyharmonic{2}(), ThinPlate())

@testset "RBF" begin
    
    @testset "Constructors" for rbf in (:Gaussian, :Multiquadratic, :InverseQuadratic, :InverseMultiquadratic, :Polyharmonic)
        @eval @test $rbf(1) == $rbf{1}() == $rbf()
    end

    @testset "Evaluation" for r in radialBasisFunctions

        f(x) = if r isa Gaussian
            exp(-(2*x)^2)
        elseif r isa Multiquadratic
            sqrt(1 + (2x)^2)
        elseif r isa InverseQuadratic
            1/(1 + (2x)^2)
        elseif r isa InverseMultiquadratic
            1/sqrt(1 + (2x)^2)
        elseif r isa Polyharmonic
            x > 0.0 ? x^2*log(x) : 0.0
        end

        @test r.(data) ≈ f.(data)

        itp = interpolate(r, points, data)

        # Check that we get back the original data at the sample points
        ev = evaluate(itp, points)
        @test ev ≈ data
    end
end
