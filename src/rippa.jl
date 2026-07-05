# Exact leave-one-out cross-validation (LOOCV) via Rippa's formula. For a symmetric
# invertible system A w = f, the prediction error at point i when point i is left out
# is e_i = w_i / (A⁻¹)_{ii} — the whole LOO error vector costs one inversion, not n
# solves (S. Rippa, Adv. Comput. Math. 11, 1999). Ridge smoothing is covered by
# applying the formula to the smoothed matrix. Intended for the small dense per-patch
# systems of the partition of unity method, hence the direct inv().

# LOOCV error matrix for the (possibly ridge-smoothed) system A W = F. A is the
# assembled kernel matrix *after* smoothing was added; W the solved weights. A matrix
# W broadcasts the one diagonal over all sample columns.
function loocverrors(A::AbstractMatrix, W::AbstractVecOrMat)
    dinv = diag(inv(A))
    return W ./ dinv
end

# Bordered variant for polynomial-augmented saddle systems M = [A P; Pᵀ 0] with
# solution WΛ = [w; λ]: the same identity e_i = w_i / (M⁻¹)_{ii} holds for the data
# rows i = 1..n (Fasshauer & McCourt, Kernel-based Approximation Methods using
# MATLAB, ch. 14).
function loocverrors(M::AbstractMatrix, WΛ::AbstractVector, n::Integer)
    dinv = diag(inv(M))
    return WΛ[1:n] ./ dinv[1:n]
end
function loocverrors(M::AbstractMatrix, WΛ::AbstractMatrix, n::Integer)
    dinv = diag(inv(M))
    return WΛ[1:n, :] ./ dinv[1:n]
end

# Matrix exceptions raised by inv() on (numerically) singular input. A singular
# tuning candidate (extreme flat limit) is disqualified with an infinite score
# rather than an error.
const SINGULAR_EXCEPTIONS = Union{SingularException, LinearAlgebra.LAPACKException,
                                  LinearAlgebra.ZeroPivotException}

# Total squared LOOCV error of the system A W = F, sharing a single inversion
# between the solve and the diagonal (the tuning hot loop calls this once per
# shape-parameter candidate).
function loocvscore(A::AbstractMatrix, F::AbstractVecOrMat)
    Ainv = try
        inv(A)
    catch err
        err isa SINGULAR_EXCEPTIONS && return Inf
        rethrow()
    end
    W = Ainv * F
    return sum(abs2, W ./ diag(Ainv))
end

# Bordered score: F must already be padded with zero rows for the polynomial part;
# n is the number of data points (leading rows of M).
function loocvscore(M::AbstractMatrix, F::AbstractVecOrMat, n::Integer)
    Minv = try
        inv(M)
    catch err
        err isa SINGULAR_EXCEPTIONS && return Inf
        rethrow()
    end
    WΛ = Minv * F
    dinv = diag(Minv)
    E = WΛ isa AbstractVector ? WΛ[1:n] ./ dinv[1:n] : WΛ[1:n, :] ./ dinv[1:n]
    return sum(abs2, E)
end
