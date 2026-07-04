using ScatteredInterpolation: PatchGrid, buildgrid, centerof, assignpatches

# Deterministic low-discrepancy points in [0,1]^d (Kronecker sequence). RNG-free so the
# tests are reproducible across Julia versions, and duplicate-free unlike periodic
# sinpi patterns.
const PUM_PRIMES = (2, 3, 5, 7, 11, 13)
kroneckerpoints(d, n; offset = 0) =
    [mod((i + offset) * sqrt(PUM_PRIMES[j]), 1.0) for j in 1:d, i in 1:n]

@testset "PUM patch grid" begin

    @testset "buildgrid basic properties" begin
        pts = kroneckerpoints(2, 400)
        grid = buildgrid(pts, 80, 1.5)
        # Volume-calibrated cell count: kd = π·(1.5·√2/2)² ≈ 3.534 → ceil(√(400·kd/80)) = 5
        @test grid.ncells == (5, 5)
        @test all(s -> s > 0, grid.spacing)
        @test collect(grid.origin) ≈ vec(minimum(pts, dims = 2))
        # radius = overlap × half cell diagonal
        @test grid.radius ≈ 1.5 * sqrt(sum(abs2, grid.spacing)) / 2
    end

    @testset "center helper" begin
        pts = kroneckerpoints(2, 400)
        grid = buildgrid(pts, 80, 1.5)
        c = centerof(grid, CartesianIndex(1, 1))
        @test collect(c) ≈ collect(grid.origin) .+ 0.5 .* collect(grid.spacing)
    end

    @testset "assignpatches covers all points, $d dimensions" for d in (1, 2, 3, 6)
        n = 400
        pts = kroneckerpoints(d, n)
        grid = buildgrid(pts, 80, 1.5)
        patchpoints, centers = assignpatches(pts, grid)
        P = length(patchpoints)
        @test size(centers) == (d, P)
        # Every point is in at least one patch
        covered = falses(n)
        for idxs in patchpoints, i in idxs
            covered[i] = true
        end
        @test all(covered)
        # Patch point lists are exactly the points within radius of the center.
        # Aggregated into one @test per property: a per-pair @test would record
        # millions of results on the finer calibrated grids and dominate suite time.
        membershipok = true
        for p in 1:P
            member = falses(n)
            member[patchpoints[p]] .= true
            for i in 1:n
                dist = sqrt(sum(abs2, pts[:, i] .- centers[:, p]))
                membershipok &= member[i] == (dist <= grid.radius)
            end
        end
        @test membershipok
    end

    @testset "degenerate geometry" begin
        # Flat second dimension
        flat = vcat(kroneckerpoints(1, 50), fill(0.5, 1, 50))
        gridF = buildgrid(flat, 10, 1.5)
        @test all(s -> s > 0, gridF.spacing)
        @test gridF.ncells[2] == 1
        # Radius includes the flat dimension's artificial unit spacing, which
        # compensates the half-cell center offset in that dimension.
        @test gridF.radius ≈ 1.5 * sqrt(gridF.spacing[1]^2 + 1) / 2
        patchpointsF, _ = assignpatches(flat, gridF)
        coveredF = falses(50)
        for idxs in patchpointsF, i in idxs
            coveredF[i] = true
        end
        @test all(coveredF)

        # All points coincident, low and high dimension
        for dd in (2, 6)
            same = fill(0.3, dd, 5)
            gridS = buildgrid(same, 10, 1.5)
            @test gridS.radius > 0
            patchpointsS, _ = assignpatches(same, gridS)
            @test sort(reduce(vcat, patchpointsS)) ⊇ 1:5
        end
    end

    @testset "patch occupancy stays near target" begin
        for d in (2, 3, 6)
            n = 4000
            pts = kroneckerpoints(d, n)
            grid = buildgrid(pts, 80, 1.5)
            patchpoints, _ = assignpatches(pts, grid)
            meansize = sum(length, patchpoints) / length(patchpoints)
            # Mean patch size within a modest factor of the target, independent of d
            # (boundary patches are clipped, so the mean sits below the interior value)
            @test 10 <= meansize <= 400
        end
    end
end

@testset "PartitionOfUnity construction" begin

    @testset "Method validation" begin
        @test_throws ArgumentError PartitionOfUnity(Gaussian(); overlap = 1.0)
        @test_throws ArgumentError PartitionOfUnity(Gaussian(); overlap = 0.5)
        @test_throws ArgumentError PartitionOfUnity(Gaussian(); pointsperpatch = 0)
        pum = PartitionOfUnity(Gaussian(2))
        @test pum.pointsperpatch == 80
        @test pum.overlap == 1.5
        @test pum.weight === nothing
    end

    @testset "interpolate builds local interpolants" begin
        pts = kroneckerpoints(2, 400)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals)
        @test itp isa ScatteredInterpolation.PartitionOfUnityInterpolant
        P = length(itp.patchpoints)
        @test P >= 1
        @test length(itp.locals) == P
        @test all(l -> l isa ScatteredInterpolation.RBFInterpolant, itp.locals)
        # Default weight resolved to the C² Wendland of the data dimension
        @test itp.weight isa Wendland
        # Integer points are accepted
        ipts = [0 1 0 1 2; 0 0 1 1 2]
        itpI = interpolate(PartitionOfUnity(Gaussian()), ipts, [1.0, 2.0, 3.0, 4.0, 5.0])
        @test itpI isa ScatteredInterpolation.PartitionOfUnityInterpolant
    end

    @testset "Vector smoothing and threaded build determinism" begin
        pts = kroneckerpoints(2, 300)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        sv = fill(1e-3, 300)
        itp1 = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals; smooth = sv)
        itp2 = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals; smooth = sv)
        # Concrete local-interpolant storage and deterministic threaded builds
        @test isconcretetype(eltype(itp1.locals))
        @test all(itp1.locals[p].w == itp2.locals[p].w for p in 1:length(itp1.locals))
        # A uniform smoothing vector matches the equivalent scalar smoothing
        itp3 = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals; smooth = 1e-3)
        @test all(itp1.locals[p].w ≈ itp3.locals[p].w for p in 1:length(itp1.locals))
    end

    @testset "interpolate argument errors" begin
        pts = kroneckerpoints(2, 50)
        vals = ones(50)
        @test_throws ArgumentError interpolate(PartitionOfUnity(Gaussian()), pts, vals;
                                               metric = Cityblock())
        @test_throws AssertionError interpolate(PartitionOfUnity(Gaussian()), pts, vals;
                                                smooth = true)
        @test_throws DimensionMismatch interpolate(PartitionOfUnity(Gaussian()), pts,
                                                   ones(49))
    end
end

@testset "PartitionOfUnity evaluation" begin

    @testset "Exactness at nodes, $d dimensions" for d in 1:6
        n = 200
        pts = kroneckerpoints(d, n)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(Gaussian(2); pointsperpatch = 40), pts, vals)
        @test evaluate(itp, pts) ≈ vals atol = 1e-6
    end

    @testset "Exactness with kernel $(typeof(kernel))" for kernel in (
            Gaussian(2), InverseMultiquadratic(2),
            GeneralizedMultiquadratic(1, 1/2, 2), Wendland(2, 1))
        pts = kroneckerpoints(2, 300)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(kernel; pointsperpatch = 60), pts, vals)
        # atol = 1e-5, not 1e-6: `≈` on arrays compares the 2-norm of the residual
        # over all 300 points, not elementwise. GeneralizedMultiquadratic's per-patch
        # solve has ~1e-7 pointwise error (same order as the dense global RBF fit with
        # this kernel — verified independent of PUM), which a 300-point 2-norm inflates
        # past 1e-6; the other three kernels here are 1-6 orders of magnitude tighter.
        @test evaluate(itp, pts) ≈ vals atol = 1e-5
    end

    @testset "Accuracy vs dense global RBF" begin
        pts = kroneckerpoints(2, 400)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        qpts = kroneckerpoints(2, 137; offset = 1000) .* 0.9 .+ 0.05
        truevals = [prod(sinpi, x) for x in eachcol(qpts)]

        dense = interpolate(Gaussian(2), pts, vals)
        pum = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals)
        pumvals = evaluate(pum, qpts)
        @test maximum(abs, pumvals - truevals) < 1e-2
        @test maximum(abs, pumvals - evaluate(dense, qpts)) < 1e-2
    end

    @testset "Matrix samples and vector samples" begin
        pts = kroneckerpoints(2, 300)
        v1 = [prod(sinpi, x) for x in eachcol(pts)]
        vals = [v1 2 .* v1]
        itp = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals)
        out = evaluate(itp, pts)
        @test out isa AbstractMatrix
        @test size(out) == (300, 2)
        @test out[:, 2] ≈ 2 .* out[:, 1] atol = 1e-8
        @test out[:, 1] ≈ v1 atol = 1e-6

        itpv = interpolate(PartitionOfUnity(Gaussian(2)), pts, v1)
        @test evaluate(itpv, pts) isa AbstractVector
    end

    @testset "Smoothing" begin
        pts = kroneckerpoints(2, 300)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals; smooth = 1e-2)
        ev = evaluate(itp, pts)
        @test maximum(abs, ev - vals) > 1e-8   # no longer interpolating
        @test maximum(abs, ev - vals) < 0.1    # but still close
    end

    @testset "C1 continuity across patch boundaries" begin
        pts = kroneckerpoints(2, 400)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals)
        xs = range(0.05, 0.95; length = 401)
        h = step(xs)
        line = vcat(collect(xs)', fill(0.5, length(xs))')
        v = evaluate(itp, line)
        g = diff(v) ./ h
        # A C¹ function has successive derivative estimates differing by O(h);
        # a kink at a patch boundary would appear as an O(1) jump.
        @test maximum(abs, diff(g)) < 0.05
    end

    @testset "Uncovered queries fall back to nearest patch" begin
        pts = kroneckerpoints(2, 300)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals)
        out = evaluate(itp, reshape([2.0, 2.0], 2, 1))
        @test all(isfinite, out)
        # Single-point vector form (dispatches through the generic reshape fallback)
        @test evaluate(itp, [0.5, 0.5])[1] ≈ evaluate(itp, reshape([0.5, 0.5], 2, 1))[1]
    end

    @testset "Deterministic under repeated evaluation" begin
        pts = kroneckerpoints(2, 300)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals)
        q = kroneckerpoints(2, 100; offset = 500)
        @test evaluate(itp, q) == evaluate(itp, q)
    end

    @testset "Tiny patches (isolated points)" begin
        pts = [0.0 0.01 0.02 5.0
               0.0 0.01 0.02 5.0]
        vals = [1.0, 2.0, 3.0, 4.0]
        itp = interpolate(PartitionOfUnity(Gaussian(); pointsperpatch = 2), pts, vals)
        @test evaluate(itp, pts) ≈ vals atol = 1e-8
    end

    @testset "Flat dimension" begin
        pts = vcat(kroneckerpoints(1, 60), fill(0.5, 1, 60))
        vals = [sinpi(x[1]) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(Gaussian(2); pointsperpatch = 15), pts, vals)
        @test evaluate(itp, pts) ≈ vals atol = 1e-6
    end

    @testset "Evaluation dimension mismatch" begin
        pts = kroneckerpoints(2, 50)
        itp = interpolate(PartitionOfUnity(Gaussian()), pts, ones(50))
        @test_throws DimensionMismatch evaluate(itp, kroneckerpoints(3, 5))
    end
end
