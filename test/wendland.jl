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
            w = Wendland(dim, 1)
            @test w(0.0) ≈ 1.0
            @test w(0.3) ≈ (1 - 0.3)^4 * (4 * 0.3 + 1)
            @test w(0.5) ≈ 0.5^4 * 3.0
            @test w(1.0) == 0.0
            @test w(1.5) == 0.0
        end
    end

    @testset "Support radius and ε scaling" begin
        @test ScatteredInterpolation.support_radius(Wendland(2, 1; ε = 2)) ≈ 0.5
        @test ScatteredInterpolation.support_radius(Wendland(2, 1)) ≈ 1.0
        for rbf in (Gaussian(2), Multiquadratic(2), InverseQuadratic(2),
                    InverseMultiquadratic(2), Polyharmonic(3), ThinPlate(),
                    GeneralizedMultiquadratic(1, 1/2, 2), GeneralizedPolyharmonic(3, 2))
            @test ScatteredInterpolation.support_radius(rbf) == Inf
        end
        # ε rescales the argument: ϕ_ε(r) = ϕ(εr)
        @test Wendland(3, 1; ε = 2)(0.25) ≈ Wendland(3, 1)(0.5)
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
end
