# Self-contained test data (do not reuse variables leaked from other test files).
# 200 deterministic points in the unit square: enough for real sparsity with a
# small support radius, small enough for exact dense comparison.
csrbfPoints = permutedims(reduce(vcat,
    [mod(0.618034 * i, 1.0)  mod(0.414214 * i, 1.0)] for i in 1:200))
csrbfData = vec(sin.(2π .* csrbfPoints[1, :]) .* cos.(π .* csrbfPoints[2, :]))
csrbfKernel = Wendland(2, 1; ε = 4.0)   # support radius 0.25 ⇒ genuinely sparse

@testset "Compact-support sparse RBF" begin

    @testset "Assembly matches dense matrix" begin
        tree = ScatteredInterpolation.NearestNeighbors.KDTree(csrbfPoints, Euclidean())
        A = ScatteredInterpolation.assemblesparse(
            csrbfKernel, csrbfPoints, tree, Euclidean(), false)
        @test A isa ScatteredInterpolation.SparseArrays.SparseMatrixCSC
        Adense = csrbfKernel.(Distances.pairwise(Euclidean(), csrbfPoints, dims = 2))
        @test Matrix(A) ≈ Adense atol = 1e-12
        # Actually sparse: with support radius 0.25 most pairs are beyond range.
        @test ScatteredInterpolation.SparseArrays.nnz(A) < 0.5 * length(Adense)

        # Scalar and vector smoothing land on the diagonal only.
        As = ScatteredInterpolation.assemblesparse(
            csrbfKernel, csrbfPoints, tree, Euclidean(), 0.1)
        @test Matrix(As) ≈ Adense + 0.1 * I atol = 1e-12
        sv = collect(range(0.01, 0.2; length = size(csrbfPoints, 2)))
        Av = ScatteredInterpolation.assemblesparse(
            csrbfKernel, csrbfPoints, tree, Euclidean(), sv)
        @test Matrix(Av) ≈ Adense + Diagonal(sv) atol = 1e-12
    end

    @testset "Sparse weights match dense weights" begin
        tree = ScatteredInterpolation.NearestNeighbors.KDTree(csrbfPoints, Euclidean())
        itp = ScatteredInterpolation.interpolatecsrbf(
            csrbfKernel, csrbfPoints, csrbfData, tree, Euclidean(), false, nothing, false)
        @test itp isa ScatteredInterpolation.CompactSupportRBFInterpolant
        dense = interpolate(csrbfKernel, csrbfPoints, csrbfData)
        @test itp.w ≈ dense.w atol = 1e-8

        # returnRBFmatrix hands back the assembled sparse matrix.
        itp2, A = ScatteredInterpolation.interpolatecsrbf(
            csrbfKernel, csrbfPoints, csrbfData, tree, Euclidean(), false, nothing, true)
        @test A isa ScatteredInterpolation.SparseArrays.SparseMatrixCSC
        @test itp2.w ≈ itp.w

        # Matrix-valued samples: one solve per column.
        multi = [csrbfData 2 .* csrbfData]
        itpm = ScatteredInterpolation.interpolatecsrbf(
            csrbfKernel, csrbfPoints, multi, tree, Euclidean(), false, nothing, false)
        @test itpm.w[:, 2] ≈ 2 .* itpm.w[:, 1] atol = 1e-8
    end

    @testset "Neighbor-only evaluation" begin
        tree = ScatteredInterpolation.NearestNeighbors.KDTree(csrbfPoints, Euclidean())
        itp = ScatteredInterpolation.interpolatecsrbf(
            csrbfKernel, csrbfPoints, csrbfData, tree, Euclidean(), false, nothing, false)

        # Exact at the data points.
        @test evaluate(itp, csrbfPoints) ≈ csrbfData atol = 1e-8

        # Matches the dense path on an off-node query grid.
        dense = interpolate(csrbfKernel, csrbfPoints, csrbfData)
        grid = permutedims(reduce(vcat, [x y] for x in 0.05:0.1:0.95, y in 0.05:0.1:0.95))
        @test evaluate(itp, grid) ≈ evaluate(dense, grid) atol = 1e-8

        # Identically zero outside every kernel's support (radius 0.25, data in the
        # unit square, query at distance > 1 from the square).
        @test all(iszero, evaluate(itp, reshape([5.0, 5.0], 2, 1)))

        # Vector query goes through the generic reshape fallback.
        @test evaluate(itp, [0.5, 0.5]) ≈ evaluate(dense, [0.5, 0.5]) atol = 1e-8

        # Wrong dimension errors.
        @test_throws DimensionMismatch evaluate(itp, reshape([0.5], 1, 1))
    end
end
