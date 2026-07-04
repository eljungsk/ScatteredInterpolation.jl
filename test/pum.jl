using ScatteredInterpolation: PatchGrid, buildgrid, cellof, centerof, assignpatches

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
        # 400 / 80 = 5 cells wanted → ceil(5^(1/2)) = 3 per side
        @test grid.ncells == (3, 3)
        @test all(s -> s > 0, grid.spacing)
        @test collect(grid.origin) ≈ vec(minimum(pts, dims = 2))
        # radius = overlap × half cell diagonal
        @test grid.radius ≈ 1.5 * sqrt(sum(abs2, grid.spacing)) / 2
    end

    @testset "cell and center helpers" begin
        pts = kroneckerpoints(2, 400)
        grid = buildgrid(pts, 80, 1.5)
        # A point inside cell (1,1): just above the origin
        x = collect(grid.origin) .+ 0.25 .* collect(grid.spacing)
        @test cellof(grid, x) == CartesianIndex(1, 1)
        # Points outside the bounding box clamp to boundary cells
        @test cellof(grid, [-10.0, -10.0]) == CartesianIndex(1, 1)
        @test cellof(grid, [10.0, 10.0]) == CartesianIndex(3, 3)
        c = centerof(grid, CartesianIndex(1, 1))
        @test collect(c) ≈ collect(grid.origin) .+ 0.5 .* collect(grid.spacing)
    end

    @testset "assignpatches covers all points, $d dimensions" for d in (1, 2, 3, 6)
        n = 400
        pts = kroneckerpoints(d, n)
        grid = buildgrid(pts, 80, 1.5)
        patchpoints, centers, cellpatches = assignpatches(pts, grid)
        P = length(patchpoints)
        @test size(centers) == (d, P)
        @test length(cellpatches) == prod(grid.ncells)
        # Every point is in at least one patch
        covered = falses(n)
        for idxs in patchpoints, i in idxs
            covered[i] = true
        end
        @test all(covered)
        # Patch point lists are exactly the points within radius of the center
        for p in 1:P
            member = falses(n)
            member[patchpoints[p]] .= true
            for i in 1:n
                dist = sqrt(sum(abs2, pts[:, i] .- centers[:, p]))
                @test member[i] == (dist <= grid.radius)
            end
        end
        # cellpatches is conservative: every patch whose ball intersects a cell's box
        # is listed for that cell (brute-force check).
        for ci in CartesianIndices(grid.ncells)
            li = LinearIndices(CartesianIndices(grid.ncells))[ci]
            for p in 1:P
                # Distance from the patch center to the cell box
                dist2 = 0.0
                for i in 1:d
                    lo = grid.origin[i] + (ci[i] - 1) * grid.spacing[i]
                    hi = grid.origin[i] + ci[i] * grid.spacing[i]
                    dist2 += max(lo - centers[i, p], 0.0, centers[i, p] - hi)^2
                end
                if sqrt(dist2) <= grid.radius
                    @test p in cellpatches[li]
                end
            end
        end
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
        patchpointsF, _, _ = assignpatches(flat, gridF)
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
            patchpointsS, _, _ = assignpatches(same, gridS)
            @test sort(reduce(vcat, patchpointsS)) ⊇ 1:5
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
