
# Define some points and data in 2D
points = permutedims([0.0 0.0; 0.0 1.0; 0.5 0.5; 1.0 0.0; 1.0 1.0], (2,1))
data = [0.0; 0.5; 0.5; 0.5; 1.0]

radialBasisFunctions = (Gaussian(2),
                        Multiquadratic(2),
                        InverseQuadratic(2),
                        InverseMultiquadratic(2),
                        Polyharmonic(2),
                        ThinPlate(),
                        GeneralizedMultiquadratic(1, 1/2, 2),
                        GeneralizedPolyharmonic(3, 2))

@testset "RBF" begin

    @testset "Constructors" for rbf in (:Gaussian, :Multiquadratic, :InverseQuadratic, :InverseMultiquadratic, :Polyharmonic)
        @eval @test $rbf(1) == $rbf()
    end

    @testset "Evaluation" for r in radialBasisFunctions

        f(x) = if isa(r, Gaussian)
            exp(-(2*x)^2)
        elseif isa(r, Multiquadratic)
            sqrt(1 + (2x)^2)
        elseif isa(r, InverseQuadratic)
            1/(1 + (2x)^2)
        elseif isa(r, InverseMultiquadratic)
            1/sqrt(1 + (2x)^2)
        elseif isa(r, Polyharmonic)
            x > 0.0 ? x^2*log(x) : 0.0
        elseif isa(r, GeneralizedMultiquadratic)
            (1 + (1x)^2)^(1/2)
        elseif isa(r, GeneralizedPolyharmonic)
            x^3
        end

        @test r.(data) ≈ f.(data)

        itp = interpolate(r, points, data)

        # Check that we get back the original data at the sample points
        ev = evaluate(itp, points)
        @test ev ≈ data
    end

    @testset "Mixed RBF Evaluation" begin

        RBFs = [Gaussian(2), Multiquadratic(2), InverseQuadratic(2), InverseMultiquadratic(2)]
        itp = interpolate(RBFs, points, data)

        # Check that we get back the original data at the sample points
        ev = evaluate(itp, points)
        @test ev ≈ data

        @testset "Mixed RBF method equality" begin

            itpConstant = interpolate(Gaussian(), points, data)
            itpMixed = interpolate(repeat([Gaussian()], outer = size(points,2)), points, data)

            # Check that the result is the same when dispatching on multiple,
            # but equal RBFs for each interpolation point
            @test evaluate(itpConstant, points) == evaluate(itpMixed, points)
        end
    end

    @testset "returnRBFmatrix" begin
        r = radialBasisFunctions[1]
        @test typeof(interpolate(r, points, data; returnRBFmatrix = true)) <: Tuple
        @test_throws TypeError interpolate(r, points, data; returnRBFmatrix = "true")
    end

    @testset "Smooth RBF" begin
        r = radialBasisFunctions[1]
        @test interpolate(r, points, data; smooth = false).w ≈ interpolate(r, points, data).w


        itp = interpolate(r, points, data; smooth = 1000.0)
        @test itp.w ≈ interpolate(r, points, data; smooth = [1000.0 for i = 1:size(points,2)]).w

        # Can not call the method with true causing true to be silently interpreted as 1 
        @test_throws AssertionError interpolate(r, points, data; smooth = true)

        # The method is no longer interpolating using smoothing
        ev = evaluate(itp, points)
        @test !(ev ≈ data)
    end

    @testset "Generalized RBF:s" begin
        points = [1. 2; 3 4]
        @test ScatteredInterpolation.generateMultivariatePolynomial(points, 2) ≈ 
                                                                            [1 1 3 1 3 9;
                                                                             1 2 4 4 8 16]
    end

end
