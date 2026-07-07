# Self-contained test data (do not reuse variables leaked from other test files).
wendlandPoints = permutedims([0.0  0.0
                              1.0  0.0
                              0.0  1.0
                              1.0  1.0
                              0.5  0.5
                              0.2  0.7])
wendlandData = [0.0, 0.5, 0.5, 1.0, 0.5, 0.3]

@testset "Wendland" begin

    @testset "Constructor validation" begin
        @test_throws ArgumentError Wendland(2, 4)
        @test_throws ArgumentError Wendland(2, -1)
        @test_throws ArgumentError Wendland(0, 1)
        @test_throws ArgumentError Wendland(2, 1; ε = 0)
        @test_throws ArgumentError Wendland(2, 1; ε = -1.0)
    end

    @testset "Known values (dims 2 and 3, degree 1)" begin
        # For dim ∈ {2, 3}, degree 1 is the classic C² Wendland function
        # ϕ(r) = (1 - r)₊⁴ (4r + 1).
        for dim in (2, 3)
            w = Wendland(dim, 1; ε = 1)
            @test w(0.0) ≈ 1.0
            @test w(0.3) ≈ (1 - 0.3)^4 * (4 * 0.3 + 1)
            @test w(0.5) ≈ 0.5^4 * 3.0
            @test w(1.0) == 0.0
            @test w(1.5) == 0.0
        end
    end

    @testset "Support radius and ε scaling" begin
        @test ScatteredInterpolation.support_radius(Wendland(2, 1; ε = 2)) ≈ 0.5
        @test ScatteredInterpolation.support_radius(Wendland(2, 1; ε = 1)) ≈ 1.0
        for rbf in (Gaussian(2), Multiquadratic(2), InverseQuadratic(2),
                    InverseMultiquadratic(2), Polyharmonic(3), ThinPlate(),
                    GeneralizedMultiquadratic(1, 1/2, 2), GeneralizedPolyharmonic(3, 2))
            @test ScatteredInterpolation.support_radius(rbf) == Inf
        end
        # ε rescales the argument: ϕ_ε(r) = ϕ(εr)
        @test Wendland(3, 1; ε = 2)(0.25) ≈ Wendland(3, 1; ε = 1)(0.5)
        @test Wendland(3, 1; ε = 2)(0.6) == 0.0
    end

    @testset "Dense RBF interpolation path" begin
        # Support radius 2 covers the whole unit square, so the dense system is SPD
        # and interpolation is exact at the nodes.
        itp = interpolate(Wendland(2, 1; ε = 0.5), wendlandPoints, wendlandData)
        @test evaluate(itp, wendlandPoints) ≈ wendlandData atol = 1e-10

        # Small support (zeros in the matrix) must still interpolate exactly.
        itpSmall = interpolate(Wendland(2, 1; ε = 2.0), wendlandPoints, wendlandData)
        @test evaluate(itpSmall, wendlandPoints) ≈ wendlandData atol = 1e-10
    end

    @testset "Wendland shape parameter traits" begin
        @test ScatteredInterpolation.hasshape(Wendland(2, 1; ε = 1))
        w = ScatteredInterpolation.withshape(Wendland(3, 2; ε = 2), 4.0)
        @test w isa Wendland
        @test w.ε == 4.0
        # dim/degree preserved: values match a freshly constructed Wendland(3, 2; ε = 4)
        ref = Wendland(3, 2; ε = 4)
        @test all(w(r) ≈ ref(r) for r in 0:0.05:0.3)
        @test ScatteredInterpolation.support_radius(w) ≈ 0.25
    end

    @testset "Deferred shape parameter" begin
        w = Wendland(2, 1)
        @test w.ε === nothing
        @test_throws ArgumentError w(0.3)
        @test_throws ArgumentError ScatteredInterpolation.support_radius(w)

        # 1D grid, spacing 0.1: with neighbors = 2 the k-NN query takes k = 3 points
        # (the point itself plus its two neighbors at ±0.1), so the k-th-neighbor
        # distance is 0.1 for every interior point and the median support radius is
        # r = 0.1, i.e. ε = 10.
        gridPoints = collect(0.0:0.1:10.0)'
        resolved, tree = ScatteredInterpolation.resolveshape(
            Wendland(1, 1), gridPoints, Euclidean(), 2)
        @test resolved isa Wendland
        @test resolved.ε ≈ 10.0 rtol = 0.01
        @test tree isa ScatteredInterpolation.NearestNeighbors.KDTree

        # Explicit ε passes through untouched, no tree built.
        passthrough, notree = ScatteredInterpolation.resolveshape(
            Wendland(1, 1; ε = 3), gridPoints, Euclidean(), 2)
        @test passthrough.ε == 3
        @test notree === nothing

        # Non-Minkowski metric cannot drive the KDTree.
        @test_throws ArgumentError ScatteredInterpolation.resolveshape(
            Wendland(2, 1), wendlandPoints, Haversine(), 2)

        # All-duplicate points give a zero radius.
        dupPoints = zeros(2, 20)
        @test_throws ArgumentError ScatteredInterpolation.resolveshape(
            Wendland(2, 1), dupPoints, Euclidean(), 2)

        @test_throws ArgumentError ScatteredInterpolation.resolveshape(
            Wendland(2, 1), gridPoints, Euclidean(), 0)
    end
end
