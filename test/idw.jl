
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
        ev = evaluate(itp, points .+ perturbation(size(points)))
        @test all(isapprox.(data, ev, atol = 1e-2))
    end

    @testset "Power parameter" for P in (1, 2, 4)
        itp = interpolate(Shepard(P), arrayPoints, data)

        # Regardless of the power, sample points reproduce the data exactly
        @test evaluate(itp, arrayPoints) ≈ data
    end

    @testset "Power parameter affects result" begin
        # Different powers should give different interpolated values off the sample points
        itp1 = interpolate(Shepard(1), arrayPoints, data)
        itp4 = interpolate(Shepard(4), arrayPoints, data)
        # Asymmetric query (nearer the [0,0] corner) so the power actually changes the
        # weighted average; a point equidistant from all corners would not.
        query = [0.1; 0.2]
        @test !(evaluate(itp1, query) ≈ evaluate(itp4, query))
    end

    @testset "Coincident and non-coincident points" begin
        itp = interpolate(Shepard(), arrayPoints, data)

        # Coincident with a sample point -> the original data is returned exactly
        @test evaluate(itp, arrayPoints[:, 2])[1] == data[2]

        # The centroid is equidistant from all four corners, so the weighted average is
        # simply the mean of the data
        @test evaluate(itp, [0.5; 0.5])[1] ≈ sum(data) / length(data)
    end

    @testset "Single point evaluation" begin
        itp = interpolate(Shepard(), arrayPoints, data)

        # 1-D fallback must agree with the matrix path at a genuine off-sample point
        q = [0.3; 0.4]
        ev = evaluate(itp, q)
        @test size(ev) == (1, 1)
        @test ev ≈ evaluate(itp, reshape(q, :, 1))
    end

    @testset "One-dimensional points" begin
        points1d = reshape([0.0, 1.0, 2.0, 3.0], 1, 4)
        data1d = [0.0, 1.0, 4.0, 9.0]
        itp = interpolate(Shepard(), points1d, data1d)
        @test evaluate(itp, points1d) ≈ data1d
        @test evaluate(itp, [1.0])[1] ≈ data1d[2]
    end

    @testset "Multi-column samples" for points in (arrayPoints, adjointPoints)
        multiData = hcat(data, 2 .* data, -data)
        itp = interpolate(Shepard(), points, multiData)
        ev = evaluate(itp, points)
        @test size(ev) == size(multiData)
        @test ev ≈ multiData
    end

    @testset "Metric" for points in (arrayPoints, adjointPoints)
        itp = interpolate(Shepard(), points, data; metric = Cityblock())
        @test evaluate(itp, points) ≈ data

        # The metric must actually be used: it changes the distance weighting off-sample.
        # (Sample-point recovery alone can't detect an ignored metric, since coincident
        # points short-circuit before any distance is computed.)
        itpEuclidean = interpolate(Shepard(), points, data)
        @test !(evaluate(itp, [0.3; 0.6]) ≈ evaluate(itpEuclidean, [0.3; 0.6]))
    end
end
