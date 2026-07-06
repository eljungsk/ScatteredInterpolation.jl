using ScatteredInterpolation: PatchGrid, buildgrid, centerof, assignpatches,
                              meannndist, tuneshape, LOOCV_CANDIDATES

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

        # tune keyword
        @test PartitionOfUnity(Gaussian()).tune === :none
        @test PartitionOfUnity(Gaussian(); tune = :loocv).tune === :loocv
        @test_throws ArgumentError PartitionOfUnity(Gaussian(); tune = :bogus)
        @test_throws ArgumentError PartitionOfUnity(Polyharmonic(3); tune = :loocv)
        @test_throws ArgumentError PartitionOfUnity(GeneralizedPolyharmonic(3, 1);
                                                    tune = :loocv)
        @test_throws ArgumentError PartitionOfUnity(ThinPlate(); tune = :loocv)
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
            GeneralizedMultiquadratic(1, 1/2, 2), Wendland(2, 1; ε = 1))
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
        # The fallback value is exactly the nearest patch's local extrapolation
        dists = vec(sqrt.(sum(abs2, itp.centers .- [2.0, 2.0]; dims = 1)))
        pnear = argmin(dists)
        @test out[1] ≈ evaluate(itp.locals[pnear], reshape([2.0, 2.0], 2, 1))[1]
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

    @testset "Sparse patches are topped up" begin
        pts = kroneckerpoints(3, 2000)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(Gaussian(2)), pts, vals)
        # minpts = min(max(2(d+1), pointsperpatch ÷ 2), n) = 40 here
        @test all(idx -> length(idx) >= 40, itp.patchpoints)
        @test evaluate(itp, pts[:, 1:100]) ≈ vals[1:100] atol = 1e-6
    end

    @testset "Boundary accuracy with scale-free locals" begin
        # Regression test: boundary patches used to be starved and one-sided, making
        # local interpolants blow up in the data-free part of their ball (errors of
        # O(10³) with polyharmonic locals before the knn top-up).
        n = 50_000
        pts = kroneckerpoints(3, n)
        g(x) = x[1] + sinpi(x[2]) * x[3]
        vals = [g(x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(GeneralizedPolyharmonic(3, 1)), pts, vals)
        # Queries spanning the whole box, including near faces and corners
        qs = kroneckerpoints(3, 5000; offset = 42_000)
        truevals = [g(x) for x in eachcol(qs)]
        @test maximum(abs, evaluate(itp, qs) - truevals) < 0.05
    end

    @testset "Evaluation dimension mismatch" begin
        pts = kroneckerpoints(2, 50)
        itp = interpolate(PartitionOfUnity(Gaussian()), pts, ones(50))
        @test_throws DimensionMismatch evaluate(itp, kroneckerpoints(3, 5))
    end
end

@testset "addpoints!" begin
    # Base set: 280 points; added set: 20 points strictly inside the base bounding
    # box. Sizes are chosen so the volume-calibrated per-side cell count is 4 for
    # both 280 and 300 points — the rebuild then uses the identical patch grid.
    base = kroneckerpoints(2, 280)
    basevals = [prod(sinpi, x) for x in eachcol(base)]
    added = kroneckerpoints(2, 20; offset = 2000) .* 0.8 .+ 0.1
    addedvals = [prod(sinpi, x) for x in eachcol(added)]

    @testset "Insertion matches full rebuild" begin
        itp = interpolate(PartitionOfUnity(Gaussian(2)), base, basevals)
        ret = addpoints!(itp, added, addedvals)
        @test ret === itp
        rebuilt = interpolate(PartitionOfUnity(Gaussian(2)),
                              hcat(base, added), vcat(basevals, addedvals))
        @test itp.grid.ncells == rebuilt.grid.ncells   # sanity: same grid
        # Compare away from the boundary: interior patches hold identical point sets
        # in both interpolants. (Sparse boundary patches are topped up by knn over
        # 280 vs 300 candidate points respectively and may legitimately differ.)
        q = kroneckerpoints(2, 97; offset = 3000) .* 0.2 .+ 0.4
        @test evaluate(itp, q) ≈ evaluate(rebuilt, q) atol = 1e-6
        # New points are interpolated exactly
        @test evaluate(itp, added) ≈ addedvals atol = 1e-6
        @test size(itp.points, 2) == 300
    end

    @testset "Matrix samples" begin
        itp = interpolate(PartitionOfUnity(Gaussian(2)), base, [basevals 2 .* basevals])
        addpoints!(itp, added, [addedvals 2 .* addedvals])
        out = evaluate(itp, added)
        @test out[:, 1] ≈ addedvals atol = 1e-6
        @test out[:, 2] ≈ 2 .* addedvals atol = 1e-6
    end

    @testset "Tuned insertion matches tuned rebuild" begin
        # Gaussian(2), not the default Gaussian() (ε = 1): see the Task 5 amendment
        # in the plan — ε = 1 is marginal/borderline for node-exactness at this
        # point density even untuned, unrelated to addpoints! itself.
        itp = interpolate(PartitionOfUnity(Gaussian(2); tune = :loocv), base, basevals)
        addpoints!(itp, added, addedvals)
        rebuilt = interpolate(PartitionOfUnity(Gaussian(2); tune = :loocv),
                              hcat(base, added), vcat(basevals, addedvals))
        @test itp.grid.ncells == rebuilt.grid.ncells   # sanity: same grid
        # Interior patches see identical data in both paths, so identical candidate
        # grids select identical ε. (Boundary patches may differ via knn top-up,
        # exactly as in the untuned equivalence test above.)
        q = kroneckerpoints(2, 97; offset = 3000) .* 0.2 .+ 0.4
        @test evaluate(itp, q) ≈ evaluate(rebuilt, q) atol = 1e-6
        @test evaluate(itp, added) ≈ addedvals atol = 1e-6
    end

    @testset "Errors" begin
        itp = interpolate(PartitionOfUnity(Gaussian(2)), base, basevals)
        # Outside the original bounding box
        @test_throws ArgumentError addpoints!(itp, reshape([5.0, 5.0], 2, 1), [1.0])
        # Wrong point dimension
        @test_throws DimensionMismatch addpoints!(itp, reshape([0.5, 0.5, 0.5], 3, 1),
                                                  [1.0])
        # Point/sample count mismatch
        @test_throws DimensionMismatch addpoints!(itp, reshape([0.5, 0.5], 2, 1),
                                                  [1.0, 2.0])
        # Sample shape mismatch (matrix onto vector-sample interpolant)
        @test_throws DimensionMismatch addpoints!(itp, reshape([0.5, 0.5], 2, 1),
                                                  reshape([1.0, 2.0], 1, 2))
        # Smoothing vector stored at build time
        itps = interpolate(PartitionOfUnity(Gaussian(2)), base, basevals;
                           smooth = fill(1e-3, 280))
        @test_throws ArgumentError addpoints!(itps, reshape([0.5, 0.5], 2, 1), [1.0])
        # Non-PUM interpolant
        nnitp = interpolate(NearestNeighbor(), base, basevals)
        @test_throws ArgumentError addpoints!(nnitp, reshape([0.5, 0.5], 2, 1), [1.0])
        # Flat dimension: off-plane points are outside the (degenerate) bounding box
        flatpts = vcat(kroneckerpoints(1, 60), fill(0.5, 1, 60))
        flatvals = [sinpi(x[1]) for x in eachcol(flatpts)]
        itpf = interpolate(PartitionOfUnity(Gaussian(2); pointsperpatch = 15),
                           flatpts, flatvals)
        @test_throws ArgumentError addpoints!(itpf, reshape([0.5, 0.9], 2, 1), [1.0])
    end
end

@testset "LOOCV shape-parameter tuning" begin

    @testset "meannndist" begin
        # Nearest-neighbor distances: 1 (point 1→2), 1 (2→1), 2 (3→2)
        pts = [0.0 1.0 3.0; 0.0 0.0 0.0]
        @test meannndist(pts) ≈ 4 / 3
        @test meannndist(fill(0.5, 2, 3)) == 0        # coincident
        @test meannndist(reshape([1.0, 2.0], 2, 1)) == 0   # single point
    end

    @testset "tuneshape edge cases" begin
        cmid = LOOCV_CANDIDATES[(length(LOOCV_CANDIDATES) + 1) ÷ 2]
        # Coincident points: no length scale — kernel returned unchanged
        k = tuneshape(Gaussian(7), fill(0.5, 2, 4), ones(4), false, Euclidean())
        @test k === Gaussian(7)
        # Two points: ε = c_mid / h without scanning
        pts2 = [0.0 1.0; 0.0 0.0]
        k2 = tuneshape(Gaussian(7), pts2, [1.0, 2.0], false, Euclidean())
        @test k2 isa Gaussian
        @test k2.ε ≈ cmid / 1.0
    end

    @testset "Tuning never much worse than untuned (no catastrophe)" begin
        # Rippa's LOOCV score is only reliable within a safe conditioning envelope
        # (see safeloocvscore); for a smooth function at high point density, the
        # scale-derived grid can only ever match — not beat — a well-scaled fixed
        # kernel (the RBF trade-off principle: you cannot out-tune an optimum).
        # What tuning must never do is make things *much* worse, since a grid
        # candidate can only override the caller's own kernel by scoring better on
        # a gated (trustworthy) comparison.
        n = 5000
        pts = kroneckerpoints(3, n)
        g3(x) = sinpi(x[1]) * cospi(x[2]) + x[3]
        vals = [g3(x) for x in eachcol(pts)]
        q = kroneckerpoints(3, 500; offset = 100_000) .* 0.9 .+ 0.05
        truevals = [g3(x) for x in eachcol(q)]
        untuned = interpolate(PartitionOfUnity(Gaussian()), pts, vals)
        tuned = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv), pts, vals)
        euntuned = maximum(abs, evaluate(untuned, q) - truevals)
        etuned = maximum(abs, evaluate(tuned, q) - truevals)
        @test etuned < 3 * euntuned
    end

    @testset "Tuning corrects a badly mis-scaled (too-peaked) kernel" begin
        # The regime tuning *can* reliably win in: a fixed kernel so peaked it has
        # decayed to near zero between patch points is easy to detect and safely
        # correct — the winning candidate is comfortably within the safe
        # conditioning envelope, unlike the smooth-function/high-density case above.
        n = 5000
        pts = kroneckerpoints(3, n)
        g3(x) = sinpi(x[1]) * cospi(x[2]) + x[3]
        vals = [g3(x) for x in eachcol(pts)]
        q = kroneckerpoints(3, 500; offset = 100_000) .* 0.9 .+ 0.05
        truevals = [g3(x) for x in eachcol(q)]
        toopeaked = interpolate(PartitionOfUnity(Gaussian(50)), pts, vals)
        tuned = interpolate(PartitionOfUnity(Gaussian(50); tune = :loocv), pts, vals)
        etoopeaked = maximum(abs, evaluate(toopeaked, q) - truevals)
        etuned = maximum(abs, evaluate(tuned, q) - truevals)
        @test etuned < 0.1 * etoopeaked
    end

    @testset "Exactness at nodes under tuning, $d dimensions" for d in (2, 3)
        # Gaussian(2), not the default Gaussian() (ε = 1): node-exactness and
        # off-node accuracy pull the shape parameter in opposite directions (the
        # RBF trade-off principle), so this checks that tuning *preserves* node
        # exactness for an already well-conditioned kernel, not that it can improve
        # node exactness for a kernel whose good candidates are gated out (that
        # scenario is covered by the no-catastrophe test above instead).
        pts = kroneckerpoints(d, 400)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(Gaussian(2); tune = :loocv), pts, vals)
        @test evaluate(itp, pts) ≈ vals atol = 1e-6
        # Tuned kernels live on the locals; evaluate needed no changes
        @test all(l -> l.rbf isa Gaussian, itp.locals)
    end

    @testset "Matrix samples and smoothing under tuning" begin
        pts = kroneckerpoints(2, 300)
        v1 = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(Gaussian(2); tune = :loocv), pts,
                          [v1 2 .* v1])
        out = evaluate(itp, pts)
        @test size(out) == (300, 2)
        # atol = 1e-5, not 1e-6: Gaussian(2) is well-conditioned but not at machine
        # precision, and the shared per-patch factorization serving both columns
        # leaves a small, normal margin above the single-column exactness bound.
        @test out[:, 1] ≈ v1 atol = 1e-5
        @test out[:, 2] ≈ 2 .* v1 atol = 1e-5

        itps = interpolate(PartitionOfUnity(Gaussian(2); tune = :loocv), pts, v1;
                           smooth = 1e-3)
        ev = evaluate(itps, pts)
        @test maximum(abs, ev - v1) > 1e-8   # no longer interpolating
        @test maximum(abs, ev - v1) < 0.1    # but still close
    end

    @testset "Determinism of tuned builds" begin
        pts = kroneckerpoints(2, 300)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp1 = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv), pts, vals)
        itp2 = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv), pts, vals)
        @test all(itp1.locals[p].w == itp2.locals[p].w
                  for p in eachindex(itp1.locals))
        @test all(itp1.locals[p].rbf.ε == itp2.locals[p].rbf.ε
                  for p in eachindex(itp1.locals))
    end

    @testset "Tiny patches under tuning" begin
        pts = [0.0 0.01 0.02 5.0
               0.0 0.01 0.02 5.0]
        vals = [1.0, 2.0, 3.0, 4.0]
        itp = interpolate(PartitionOfUnity(Gaussian(); tune = :loocv,
                                           pointsperpatch = 2), pts, vals)
        @test evaluate(itp, pts) ≈ vals atol = 1e-8
    end

    @testset "GeneralizedMultiquadratic bordered tuning" begin
        pts = kroneckerpoints(2, 300)
        vals = [prod(sinpi, x) for x in eachcol(pts)]
        itp = interpolate(PartitionOfUnity(GeneralizedMultiquadratic(1, 1/2, 2);
                                           tune = :loocv, pointsperpatch = 60),
                          pts, vals)
        # Same atol rationale as the untuned GMQ exactness test above (array-norm ≈)
        @test evaluate(itp, pts) ≈ vals atol = 1e-5
        @test all(l -> l.rbf isa GeneralizedMultiquadratic, itp.locals)
        @test all(l -> l.rbf.β == 1/2 && l.rbf.degree == 2, itp.locals)
    end
end
