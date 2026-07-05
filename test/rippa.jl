using ScatteredInterpolation: loocverrors, loocvscore, generateMultivariatePolynomial
using Distances: pairwise

# Deterministic, RNG-free 2D points (Kronecker sequence), no duplicates.
rippapts(n) = [mod(i * sqrt(p), 1.0) for p in (2.0, 3.0), i in 1:n]

@testset "Rippa LOOCV helpers" begin
    n = 12
    pts = rippapts(n)
    f = [sinpi(x[1]) * cospi(x[2]) for x in eachcol(pts)]
    rbf = Gaussian(4)   # well conditioned at this point count and spacing
    R = pairwise(Euclidean(), pts, dims = 2)
    A = rbf.(R)

    @testset "Plain system vs brute-force LOO" begin
        w = A \ f
        E = loocverrors(A, w)
        for i in 1:n
            keep = setdiff(1:n, i)
            wi = A[keep, keep] \ f[keep]
            pred = dot(A[i, keep], wi)
            @test E[i] ≈ f[i] - pred atol = 1e-10
        end
    end

    @testset "Ridge-smoothed system" begin
        σ = 1e-3
        Ã = A + σ * I
        w̃ = Ã \ f
        Ẽ = loocverrors(Ã, w̃)
        for i in 1:n
            keep = setdiff(1:n, i)
            # Smoothing sits on the diagonal only; the prediction row is the plain
            # kernel row (the diagonal is excluded by construction).
            wi = (A[keep, keep] + σ * I) \ f[keep]
            pred = dot(A[i, keep], wi)
            @test Ẽ[i] ≈ f[i] - pred atol = 1e-10
        end
    end

    @testset "Matrix right-hand side" begin
        w = A \ f
        E = loocverrors(A, w)
        F = [f 2 .* f]
        W = A \ F
        EM = loocverrors(A, W)
        @test EM[:, 1] ≈ E atol = 1e-12
        @test EM[:, 2] ≈ 2 .* E atol = 1e-12
    end

    @testset "Bordered (saddle) system vs brute-force LOO" begin
        gmq = GeneralizedMultiquadratic(2, 1/2, 2)
        A2 = gmq.(R)
        P = generateMultivariatePolynomial(pts, gmq.degree)
        npoly = size(P, 2)               # binomial(2 + 2, 2) = 6 terms; 11 ≥ 6 points remain
        M = [A2 P; P' zeros(npoly, npoly)]
        Fb = vcat(f, zeros(npoly))
        WΛ = M \ Fb
        Eb = loocverrors(M, WΛ, n)
        @test length(Eb) == n
        for i in 1:n
            keep = setdiff(1:n, i)
            Mi = [A2[keep, keep] P[keep, :]; P[keep, :]' zeros(npoly, npoly)]
            sol = Mi \ vcat(f[keep], zeros(npoly))
            pred = dot(A2[i, keep], sol[1:(n - 1)]) + dot(P[i, :], sol[n:end])
            @test Eb[i] ≈ f[i] - pred atol = 1e-10
        end
        # Matrix RHS through the bordered variant
        WΛM = M \ [Fb 2 .* Fb]
        EbM = loocverrors(M, WΛM, n)
        @test EbM[:, 1] ≈ Eb atol = 1e-12
        @test EbM[:, 2] ≈ 2 .* Eb atol = 1e-12
        # Bordered score
        @test loocvscore(M, Fb, n) ≈ sum(abs2, Eb) rtol = 1e-10
    end

    @testset "Scores and singular disqualification" begin
        w = A \ f
        E = loocverrors(A, w)
        @test loocvscore(A, f) ≈ sum(abs2, E) rtol = 1e-10
        F = [f 2 .* f]
        @test loocvscore(A, F) ≈ 5 * sum(abs2, E) rtol = 1e-10   # 1² + 2² = 5
        @test loocvscore(zeros(3, 3), ones(3)) == Inf
        @test loocvscore(zeros(4, 4), ones(4), 3) == Inf
    end
end
