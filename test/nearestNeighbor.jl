
# Define some points and data in 2D
points = permutedims([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0], (2,1))
points_adjoint = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]'
data = [0.0; 0.5; 0.5; 1.0]

@testset "NearestNeighbor" begin

    # `p` varies the *construction* input (plain array vs adjoint); queries use a plain
    # matrix, which is what `evaluate` expects.
    @testset "Evaluation" for p in (points, points_adjoint)

        itp = interpolate(NearestNeighbor(), p, data)

        # Check that we get back the original data at the sample points
        ev = evaluate(itp, points)
        @test ev ≈ data

        # Check for a different point, should be closest to the last point
        ev = evaluate(itp, [2.0; 2.0])
        @test ev[1] ≈ data[end]
    end

    @testset "Off-sample nearest neighbor" begin
        itp = interpolate(NearestNeighbor(), points, data)

        # A point just outside a corner should snap to that corner's value
        @test evaluate(itp, [-0.1; -0.1])[1] == data[1]   # nearest [0,0]
        @test evaluate(itp, [0.9; 0.1])[1] == data[3]     # nearest [1,0]
    end

    @testset "Integer data eltype preservation" begin
        intData = [1, 2, 3, 4]
        itp = interpolate(NearestNeighbor(), points, intData)
        ev = evaluate(itp, points)
        @test eltype(ev) == eltype(intData)
        @test ev == reshape(intData, :, 1)
    end

    @testset "Multi-column samples" for p in (points, points_adjoint)
        multiData = hcat(data, 2 .* data, -data)
        itp = interpolate(NearestNeighbor(), p, multiData)
        ev = evaluate(itp, points)
        @test size(ev) == size(multiData)
        @test ev ≈ multiData
    end

    @testset "Metric" for p in (points, points_adjoint)
        itp = interpolate(NearestNeighbor(), p, data; metric = Cityblock())
        @test evaluate(itp, points) ≈ data
    end

    @testset "Metric changes nearest neighbor" begin
        # Point A is diagonal, B is on the axis. From the origin, Euclidean is nearest to
        # A (dist √2 ≈ 1.41) while Cityblock is nearest to B (dist 1.5), so an ignored
        # metric would give the wrong value here.
        mp = [1.0 0.0; 1.0 1.5]
        md = [100.0, 200.0]
        itpE = interpolate(NearestNeighbor(), mp, md; metric = Euclidean())
        itpC = interpolate(NearestNeighbor(), mp, md; metric = Cityblock())
        @test evaluate(itpE, [0.0; 0.0])[1] == 100.0
        @test evaluate(itpC, [0.0; 0.0])[1] == 200.0
    end

    @testset "Adjoint query points" begin
        # `evaluate` should accept an adjoint (2×k) query matrix, not only a plain Matrix
        itp = interpolate(NearestNeighbor(), points, data)
        @test evaluate(itp, points_adjoint) ≈ data
    end

    @testset "One-dimensional points" begin
        points1d = reshape([0.0, 1.0, 2.0, 3.0], 1, 4)
        data1d = [0.0, 10.0, 20.0, 30.0]
        itp = interpolate(NearestNeighbor(), points1d, data1d)
        @test evaluate(itp, points1d) ≈ data1d
        @test evaluate(itp, [2.4])[1] == data1d[3]   # nearest to 2.0
    end
end
