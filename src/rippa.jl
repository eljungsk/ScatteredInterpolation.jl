# Exact leave-one-out cross-validation (LOOCV) via Rippa's formula. For a symmetric
# invertible system A w = f, the prediction error at point i when point i is left out
# is e_i = w_i / (A⁻¹)_{ii} — the whole LOO error vector costs one inversion, not n
# solves (S. Rippa, Adv. Comput. Math. 11, 1999). Ridge smoothing is covered by
# applying the formula to the smoothed matrix. Solves route through LinearSolve.jl
# (the same `_initsolve` cache the main RBF solve path in rbf.jl uses), so its
# algorithm-selection heuristics apply to these small, patch-scale dense systems too,
# instead of a fixed direct inv() — but never by calling `inv()` or by looping
# LinearSolve's per-vector `solve!` over n right-hand sides (measured ~600x slower
# than the direct-inv() baseline for even a few-hundred-point build: LinearSolve's
# small-system algorithms, e.g. RFLUFactorization, don't accept a matrix
# right-hand side at all, so every one of those n solves pays full per-call
# overhead). Instead, the factorization LinearSolve's heuristic already built for
# the weight-vector solve is extracted and reused for one batched multi-RHS
# `ldiv!` against the identity — a proper solve, not `inv()`, and not a per-column
# loop. See docs/superpowers/plans/2026-07-04-loocv-shape-selection.md, Task 6
# amendment.
#
# TODO: revisit once LinearSolve.jl ships native multiple-RHS batching (already
# called out in rbf.jl's `_solve!` for the same underlying limitation) — the
# `_factorization`/`ldiv!` extraction below could then be replaced by routing the
# identity solve through LinearSolve directly, matching `_trysolve!`'s pattern.
#
# `gatedloocvscore` (used by pum.jl's tuning search) folds the reciprocal-
# condition-number gate into this same factorization instead of factorizing a
# second time — see its docstring below.

# Like `_solve!` (rbf.jl), but returns `nothing` instead of trusting a result when
# LinearSolve reports failure. Needed here (and not in the main RBF solve path)
# because LOOCV candidate scans deliberately explore near- and fully-singular
# systems (extreme flat-limit shape parameters); a singular candidate must be
# disqualified with an infinite score, not returned as a meaningless answer.
function _trysolve!(cache, b::AbstractVector)
    sol = solve!(_setb!(cache, b))
    sol.retcode == LinearSolve.ReturnCode.Failure ? nothing : copy(sol.u)
end
# Matrix RHS, column by column (samples typically have very few columns — unlike
# the identity solve below, batching this one is not worth the complexity).
function _trysolve!(cache, B::AbstractMatrix)
    x1 = _trysolve!(cache, B[:, 1])
    x1 === nothing && return nothing
    X = Matrix{eltype(x1)}(undef, length(x1), size(B, 2))
    X[:, 1] = x1
    for j in 2:size(B, 2)
        xj = _trysolve!(cache, B[:, j])
        xj === nothing && return nothing
        X[:, j] = xj
    end
    return X
end

# The concrete Base.LinearAlgebra factorization LinearSolve's default
# algorithm-selection heuristic chose and built while solving `cache` (only valid
# to call after at least one successful solve on this cache). Some algorithm
# choices store it directly (`cacheval.LUFactorization`), others pair it with
# auxiliary state (`RFLUFactorization` stores `(LU, pivot buffer)`); the
# factorization itself is always the first element in that case.
function _factorization(cache)
    val = getproperty(cache.cacheval, Symbol(cache.alg.alg))
    return val isa Tuple ? val[1] : val
end

# Diagonal of A⁻¹: reuses the factorization already built by an earlier solve on
# `cache` for one batched multi-right-hand-side `ldiv!` against the identity — the
# same asymptotic cost as inv(A), and a genuine solve (not `inv()`), but without
# looping LinearSolve's per-vector API n times (see the file-level note above).
# Only valid to call after at least one successful solve on `cache`.
function _invdiag!(cache)
    n = size(cache.A, 1)
    F = _factorization(cache)
    X = Matrix{eltype(cache.A)}(I, n, n)
    ldiv!(F, X)
    return diag(X)
end

# LOOCV error matrix for the (possibly ridge-smoothed) system A W = F. A is the
# assembled kernel matrix *after* smoothing was added; W the solved weights. A matrix
# W broadcasts the one diagonal over all sample columns.
function loocverrors(A::AbstractMatrix, W::AbstractVecOrMat)
    cache = _initsolve(A, nothing)
    _trysolve!(cache, zeros(eltype(A), size(A, 1)))   # build the factorization
    dinv = _invdiag!(cache)
    return W ./ dinv
end

# Bordered variant for polynomial-augmented saddle systems M = [A P; Pᵀ 0] with
# solution WΛ = [w; λ]: the same identity e_i = w_i / (M⁻¹)_{ii} holds for the data
# rows i = 1..n (Fasshauer & McCourt, Kernel-based Approximation Methods using
# MATLAB, ch. 14).
function loocverrors(M::AbstractMatrix, WΛ::AbstractVector, n::Integer)
    cache = _initsolve(M, nothing)
    _trysolve!(cache, zeros(eltype(M), size(M, 1)))
    dinv = _invdiag!(cache)
    return WΛ[1:n] ./ dinv[1:n]
end
function loocverrors(M::AbstractMatrix, WΛ::AbstractMatrix, n::Integer)
    cache = _initsolve(M, nothing)
    _trysolve!(cache, zeros(eltype(M), size(M, 1)))
    dinv = _invdiag!(cache)
    return WΛ[1:n, :] ./ dinv[1:n]
end

# Total squared LOOCV error of the system A W = F, sharing one factorization
# between the solve and the diagonal (the tuning hot loop calls this once per
# shape-parameter candidate). Returns Inf when LinearSolve reports failure (near or
# exactly singular A, e.g. an extreme flat-limit tuning candidate) rather than
# trusting a meaningless result.
function loocvscore(A::AbstractMatrix, F::AbstractVecOrMat)
    cache = _initsolve(A, nothing)
    W = _trysolve!(cache, F)
    W === nothing && return Inf
    dinv = _invdiag!(cache)
    return sum(abs2, W ./ dinv)
end

# Bordered score: F must already be padded with zero rows for the polynomial part;
# n is the number of data points (leading rows of M).
function loocvscore(M::AbstractMatrix, F::AbstractVecOrMat, n::Integer)
    cache = _initsolve(M, nothing)
    WΛ = _trysolve!(cache, F)
    WΛ === nothing && return Inf
    dinv = _invdiag!(cache)
    E = WΛ isa AbstractVector ? WΛ[1:n] ./ dinv[1:n] : WΛ[1:n, :] ./ dinv[1:n]
    return sum(abs2, E)
end

# Gated LOOCV score for the tuning hot loop (pum.jl): reuses the *same*
# factorization LinearSolve already built for the weight solve, both for the
# reciprocal-condition-number gate (gecon!) and for the score's diag(A⁻¹) — one
# factorization total per candidate, not two (an earlier design factorized
# independently via a raw `lu!` just for the gate, on top of loocvscore's own
# LinearSolve factorization; see docs/superpowers/plans/
# 2026-07-04-loocv-shape-selection.md, Task 8 amendment). Returns Inf when
# LinearSolve reports failure (near/exactly singular A) or when rcond falls below
# `rcondthreshold`, exactly as safeloocvscore used to.
function gatedloocvscore(A::AbstractMatrix, F::AbstractVecOrMat, rcondthreshold)
    cache = _initsolve(A, nothing)
    W = _trysolve!(cache, F)
    W === nothing && return Inf
    Fact = _factorization(cache)
    rcond = LinearAlgebra.LAPACK.gecon!('1', Fact.factors, opnorm(A, 1))
    rcond < rcondthreshold && return Inf
    dinv = _invdiag!(cache)
    return sum(abs2, W ./ dinv)
end

# Bordered variant of the gated score above.
function gatedloocvscore(M::AbstractMatrix, F::AbstractVecOrMat, n::Integer,
                         rcondthreshold)
    cache = _initsolve(M, nothing)
    WΛ = _trysolve!(cache, F)
    WΛ === nothing && return Inf
    Fact = _factorization(cache)
    rcond = LinearAlgebra.LAPACK.gecon!('1', Fact.factors, opnorm(M, 1))
    rcond < rcondthreshold && return Inf
    dinv = _invdiag!(cache)
    E = WΛ isa AbstractVector ? WΛ[1:n] ./ dinv[1:n] : WΛ[1:n, :] ./ dinv[1:n]
    return sum(abs2, E)
end
