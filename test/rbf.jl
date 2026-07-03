
# Define some points and data in 2D
arrayPoints = permutedims([0.0 0.0; 0.0 1.0; 0.5 0.5; 1.0 0.0; 1.0 1.0], (2,1))
adjointPoints = [0.0 0.0; 0.0 1.0; 0.5 0.5; 1.0 0.0; 1.0 1.0]'
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

    @testset "Generalized constructors" begin
        # Generalized RBF:s have no default constructor; check the fields are stored
        g = GeneralizedMultiquadratic(1, 1/2, 2)
        @test (g.ε, g.β, g.degree) == (1, 1/2, 2)

        p = GeneralizedPolyharmonic(3, 2)
        @test (p.k, p.degree) == (3, 2)

        @test ThinPlate() == Polyharmonic(2)
    end

    @testset "Order validation" begin
        # Polyharmonic order must be positive, and it is now checked at construction time
        # (not per evaluation).
        @test Polyharmonic(3).k == 3
        @test_throws AssertionError Polyharmonic(0)
        @test_throws AssertionError Polyharmonic(-2)

        @test GeneralizedPolyharmonic(2, 3).k == 2
        @test_throws AssertionError GeneralizedPolyharmonic(0, 2)
        @test_throws AssertionError GeneralizedPolyharmonic(-1, 2)
    end

    @testset "Polyharmonic polynomial reproduction" begin
        # A plain Polyharmonic/ThinPlate system is only conditionally positive definite:
        # its RBF matrix has a zero diagonal and is indefinite, and it does NOT reproduce
        # linear trends. GeneralizedPolyharmonic augments the system with a polynomial term
        # and reproduces a linear field exactly. This documents the recommended path.
        pts = arrayPoints
        linear = [2 + 3 * pts[1, i] - pts[2, i] for i in 1:size(pts, 2)]
        query = [0.3; 0.7]
        truth = 2 + 3 * 0.3 - 0.7

        itpGen = interpolate(GeneralizedPolyharmonic(2, 2), pts, linear)
        @test evaluate(itpGen, query)[1] ≈ truth

        itpPlain = interpolate(Polyharmonic(2), pts, linear)
        @test !(evaluate(itpPlain, query)[1] ≈ truth)

        # The underlying reason: the plain RBF matrix is not positive definite
        np = size(pts, 2)
        Aφ = [Polyharmonic(2)(norm(pts[:, i] - pts[:, j])) for i in 1:np, j in 1:np]
        @test !isposdef(Aφ)
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

        for points in (arrayPoints, adjointPoints)
            itp = interpolate(r, points, data)

            # Check that we get back the original data at the sample points and that we get
            # close when evaluating near the sampling points
            ev = evaluate(itp, points)
            @test ev ≈ data
            ev = evaluate(itp, points .+ perturbation(size(points)))
            @test all(isapprox.(data, ev, atol = 1e-2))
        end
    end

    @testset "Single point evaluation" for r in radialBasisFunctions
        itp = interpolate(r, arrayPoints, data)

        # The 1-D fallback method reshapes a length-n vector into an n×1 matrix. It must
        # agree with the matrix path at a genuine off-sample point, not just reproduce a
        # coincident sample value.
        q = arrayPoints[:, end] .+ [0.02, 0.02]
        ev = evaluate(itp, q)
        @test length(ev) == 1
        @test ev ≈ evaluate(itp, reshape(q, :, 1))
    end

    @testset "One-dimensional points" begin
        # A 1×k point set exercises the n=1 branch of pairwise / polynomial generation
        points1d = reshape([0.0, 1.0, 2.0, 3.0], 1, 4)
        data1d = [0.0, 1.0, 4.0, 9.0]
        itp = interpolate(Gaussian(1), points1d, data1d)
        @test evaluate(itp, points1d) ≈ data1d
        @test evaluate(itp, [1.0])[1] ≈ data1d[2]
    end

    @testset "Multi-column samples" for points in (arrayPoints, adjointPoints)
        r = Gaussian(2)

        # Interpolating vector-valued data should recover each column independently
        multiData = hcat(data, 2 .* data, -data)
        itp = interpolate(r, points, multiData)
        ev = evaluate(itp, points)
        @test size(ev) == size(multiData)
        @test ev ≈ multiData
    end

    @testset "Metric" for points in (arrayPoints, adjointPoints)
        r = Gaussian(2)

        # A non-default metric should still reproduce the data at the sample points
        itp = interpolate(r, points, data; metric = Cityblock())
        @test evaluate(itp, points) ≈ data

        # Different metrics generally give different interpolants
        itpEuclidean = interpolate(r, points, data)
        offPoints = points .+ 0.25
        @test !(evaluate(itp, offPoints) ≈ evaluate(itpEuclidean, offPoints))
    end

    @testset "Mixed RBF Evaluation" for points in (arrayPoints, adjointPoints)

        # One RBF per sample point (must match size(points, 2) == 5, otherwise the last
        # column of the RBF matrix is left as raw distance and the feature is not exercised)
        RBFs = [Gaussian(2), Multiquadratic(2), InverseQuadratic(2), InverseMultiquadratic(2), Polyharmonic(2)]
        itp = interpolate(RBFs, points, data)

        # Check that we get back the original data at the sample points
        ev = evaluate(itp, points)
        @test ev ≈ data

        # The RBFs are assigned per point, so permuting them must change the off-sample
        # result. This guards against every column being evaluated with the same RBF.
        itpPermuted = interpolate(circshift(RBFs, 1), points, data)
        @test !(evaluate(itp, [0.3; 0.7]) ≈ evaluate(itpPermuted, [0.3; 0.7]))

        @testset "Mixed RBF method equality" begin

            itpConstant = interpolate(Gaussian(), points, data)
            itpMixed = interpolate(repeat([Gaussian()], outer = size(points,2)), points, data)

            # Check that the result is the same when dispatching on multiple,
            # but equal RBFs for each interpolation point
            @test evaluate(itpConstant, points) ≈ evaluate(itpMixed, points)
        end
    end

    @testset "returnRBFmatrix" for points in (arrayPoints, adjointPoints)
        r = radialBasisFunctions[1]
        itp, A = interpolate(r, points, data; returnRBFmatrix = true)

        @test A isa AbstractMatrix
        @test size(A) == (size(points, 2), size(points, 2))
        # The returned matrix is the RBF matrix used to solve for the weights
        @test A * itp.w ≈ data

        @test typeof(interpolate(r, points, data; returnRBFmatrix = true)) <: Tuple
        @test_throws TypeError interpolate(r, points, data; returnRBFmatrix = "true")
    end

    @testset "Smooth RBF" for points in (arrayPoints, adjointPoints)
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

    @testset "Invalid input" begin
        # Number of samples must match the number of points (5 columns here)
        @test_throws DimensionMismatch interpolate(Gaussian(1), arrayPoints, data[1:3])
    end

end
