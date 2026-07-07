abstract type AbstractRadialBasisFunction <: InterpolationMethod end
abstract type RadialBasisFunction <: AbstractRadialBasisFunction end
abstract type GeneralizedRadialBasisFunction <: AbstractRadialBasisFunction end

Base.iterate(x::AbstractRadialBasisFunction) = (x, nothing)
Base.iterate(x::AbstractRadialBasisFunction, ::Any) = nothing

export  Gaussian,
        Multiquadratic,
        InverseQuadratic,
        InverseMultiquadratic,
        Polyharmonic,
        ThinPlate,
        GeneralizedMultiquadratic,
        GeneralizedPolyharmonic

# Define types for the different kinds of radial basis functions
for rbf in (:Gaussian,
            :Multiquadratic,
            :InverseQuadratic,
            :InverseMultiquadratic)
    @eval begin
        struct $rbf{T <: Real} <: RadialBasisFunction
            ε::T
        end

        # Define default constructors
        $rbf() = $rbf(1)
    end
end

# Polyharmonic is defined outside the loop so its constructor can validate the order once,
# at construction time, rather than on every scalar evaluation.
struct Polyharmonic{T <: Integer} <: RadialBasisFunction
    k::T
    function Polyharmonic{T}(k) where {T <: Integer}
        @assert k > 0 "Polyharmonic order must be positive"
        new{T}(convert(T, k))
    end
end
Polyharmonic(k::T) where {T <: Integer} = Polyharmonic{T}(k)
Polyharmonic() = Polyharmonic(1)

# Generalized RBF:s
struct GeneralizedMultiquadratic{T<:Real, S<:Real, U<:Integer} <: GeneralizedRadialBasisFunction
    ε::T
    β::S
    degree::U
end

struct GeneralizedPolyharmonic{S<:Integer, U<:Integer} <: GeneralizedRadialBasisFunction
    k::S
    degree::U
    function GeneralizedPolyharmonic{S, U}(k, degree) where {S <: Integer, U <: Integer}
        @assert k > 0 "Polyharmonic order k must be positive"
        new{S, U}(convert(S, k), convert(U, degree))
    end
end
GeneralizedPolyharmonic(k::S, degree::U) where {S <: Integer, U <: Integer} =
    GeneralizedPolyharmonic{S, U}(k, degree)

@doc "
    Gaussian(ε = 1)

Define a Gaussian Radial Basis Function

```math
ϕ(r) = e^{-(εr)^2}
```
" Gaussian
(rbf::Gaussian)(r) = exp(-(rbf.ε*r)^2) 

@doc "
    Multiquadratic(ε = 1)

Define a Multiquadratic Radial Basis Function

```math
ϕ(r) = \\sqrt{1 + (εr)^2}
```
" Multiquadratic
(rbf::Multiquadratic)(r) = (sqrt(1 + (rbf.ε*r)^2))

@doc "
    InverseQuadratic(ε = 1)

Define an Inverse Quadratic Radial Basis Function

```math
ϕ(r) = \\frac{1}{1 + (εr)^2}
```
" InverseQuadratic
(rbf::InverseQuadratic)(r) = (1/(1 + (rbf.ε*r)^2))

@doc "
    InverseMultiquadratic(ε = 1)

Define an Inverse Multiquadratic Radial Basis Function

```math
ϕ(r) = \\frac{1}{\\sqrt{1 + (εr)^2}}
```
" InverseMultiquadratic
(rbf::InverseMultiquadratic)(r) = (1/sqrt(1 + (rbf.ε*r)^2))

@doc "
    Polyharmonic(k = 1)

Define a Polyharmonic Spline Radial Basis Function

```math
ϕ(r) = r^k, k = 1, 3, 5, ...
\\\\
ϕ(r) = r^k ln(r), k = 2, 4, 6, ...
```

Polyharmonic splines are only *conditionally* positive definite. The plain interpolation
matrix has a zero diagonal (``ϕ(0) = 0``) and is indefinite, so the system is not
guaranteed to be solvable and the interpolant does not reproduce polynomial trends. For a
well-posed system that also reproduces low-order polynomials, use
[`GeneralizedPolyharmonic`](@ref), which augments the system with a polynomial term.
" Polyharmonic
function (rbf::Polyharmonic)(r)
    # Order positivity and integrality are enforced by the constructor

    # Distinguish odd and even cases
    expr = if rbf.k % 2 == 0
        (r > 0 ? r^rbf.k*log(r) : zero(float(r)))
    else
        (r^rbf.k)
    end

    expr
end

@doc "
    ThinPlate()

Define a Thin Plate Spline Radial Basis Function

```math
ϕ(r) = r^2 ln(r)
```

This is a shorthand for `Polyharmonic(2)`. As with [`Polyharmonic`](@ref), the plain
thin plate system is only conditionally positive definite; use
[`GeneralizedPolyharmonic`](@ref) for a well-posed system with polynomial reproduction.
" ThinPlate
ThinPlate() = Polyharmonic(2)

@doc "
    GeneralizedMultiquadratic(ε, β, degree)

Define a generalized Multiquadratic Radial Basis Function

```math
ϕ(r) = (1 + (ε*r)^2)^β
```
Results in a positive definite system for a 'degree' of ⌈β⌉ or higher.

" GeneralizedMultiquadratic
(rbf::GeneralizedMultiquadratic)(r) = (1 + (rbf.ε*r)^2)^rbf.β


@doc "
    GeneralizedPolyharmonic(k, degree)

Define a generalized Polyharmonic Radial Basis Function

```math
ϕ(r) = r^k, k = 1, 3, 5, ...
\\\\
ϕ(r) = r^k ln(r), k = 2, 4, 6, ...
```
Results in a positive definite system for a 'degree' of ⌈k/2⌉ or higher for k = 1, 3, 5, ...
and of exactly k + 1 for k = 2, 4, 6, ...
" GeneralizedPolyharmonic
function (rbf::GeneralizedPolyharmonic)(r)
    # Order positivity is enforced by the constructor

    # Distinguish odd and even cases
    expr = if rbf.k % 2 == 0
        (r > 0 ? r^rbf.k*log(r) : zero(float(r)))
    else
        (r^rbf.k)
    end

    expr
end

# --- Shape-parameter traits ------------------------------------------------------
# Used by the partition of unity method's LOOCV tuning (tune = :loocv): hasshape says
# whether a kernel has a tunable shape parameter ε, and withshape rebuilds the kernel
# with a new one, preserving all other parameters. Polyharmonic-family kernels are
# scale-free and have no shape parameter.
hasshape(::AbstractRadialBasisFunction) = false
hasshape(::Union{Gaussian, Multiquadratic, InverseQuadratic, InverseMultiquadratic,
                 GeneralizedMultiquadratic}) = true

withshape(::Gaussian, ε) = Gaussian(ε)
withshape(::Multiquadratic, ε) = Multiquadratic(ε)
withshape(::InverseQuadratic, ε) = InverseQuadratic(ε)
withshape(::InverseMultiquadratic, ε) = InverseMultiquadratic(ε)
withshape(k::GeneralizedMultiquadratic, ε) = GeneralizedMultiquadratic(ε, k.β, k.degree)

# (Wendland is defined in src/wendland.jl, which is included after src/rbf.jl — its
# methods cannot live in this Union.)

abstract type RadialBasisInterpolant <: ScatteredInterpolant end

struct RBFInterpolant{T1 <: AbstractArray, T2 <: AbstractMatrix{<:Real}, F, M} <: RadialBasisInterpolant

    w::T1
    points::T2
    rbf::F
    metric::M
end

struct GeneralizedRBFInterpolant{T1 <: AbstractArray, T2 <: AbstractMatrix{<:Real}, F, M} <: RadialBasisInterpolant

    w::T1
    λ::T1
    points::T2
    rbf::F
    metric::M
end

function interpolate(rbf::Union{T, AbstractVector{T}} where T <: AbstractRadialBasisFunction,
                     points::AbstractArray{<:Real,2},
                     samples::AbstractArray{<:Number,N};
                     metric = Euclidean(), returnRBFmatrix::Bool = false,
                     smooth::Union{S, AbstractVector{S}} = false,
                     linsolve = nothing,
                     sparse = :auto,
                     neighbors::Integer = 50) where {N} where {S<:Number}

    #hinder smooth from being set to true and interpreted as the value 1
    @assert smooth != true "set the smoothing value as a number or vector of numbers"

    rbf, tree = resolveshape(rbf, points, metric, neighbors)

    usesparse, tree = decidesparse(sparse, rbf, points, metric, tree)
    if usesparse
        return interpolatecsrbf(rbf, points, samples, tree, metric, smooth,
                                linsolve, returnRBFmatrix)
    end

    # Compute pairwise distances, apply the Radial Basis Function
    # and optional smoothing (ridge regression)
    A = pairwise(metric, points;dims=2)
    
    A = evaluateRBF!(A, rbf, smooth)

    # The weight solve factorizes A in place (aliasA); keep an untouched copy only
    # when the caller asked for the matrix back.
    Aout = returnRBFmatrix ? copy(A) : nothing

    # Solve for the weights
    itp = solveForWeights(A, points, samples, rbf, metric; linsolve = linsolve)

    # Create and return an interpolation object
    if returnRBFmatrix    # Return matrix A
        return itp, Aout
    else
        return itp
    end

end

@inline function evaluateRBF!(A, rbf, symmetric::Bool = false)
    A .= rbf.(A)

    if symmetric
        A = Symmetric(A)
    end

    return A
end
@inline function evaluateRBF!(A, rbfs::AbstractVector{<:AbstractRadialBasisFunction}, symmetric::Bool = false)
    for (j, rbf) in enumerate(rbfs)
        A[:,j] .= rbf.(@view A[:,j])
    end
    return A
end
@inline function evaluateRBF!(A, rbf, smooth)
    A = evaluateRBF!(A, rbf)
    A = addSmoothing!(A, smooth)
    return A
end

@inline function addSmoothing!(A, smooth::Vector{T}) where T <: Number
    for i = 1:size(A, 1)
        A[i,i] += smooth[i]
    end
    return A
end
@inline function addSmoothing!(A, smooth::T) where T <: Number
    for i = 1:size(A, 1)
        A[i,i] += smooth
    end
    return A
end

# --- LinearSolve helpers -------------------------------------------------------
# Build a reusable LinearSolve cache for matrix `A` and algorithm `alg`
# (`nothing` selects LinearSolve's default algorithm). The factorization is
# computed on the first solve and reused for subsequent right-hand sides.
# `aliasA = true` lets LinearSolve factorize the caller's matrix in place instead
# of copying it first — only for call sites that never read A after the first
# solve. The default stays false: the LOOCV scoring paths (src/rippa.jl) compute
# opnorm(A, 1) after solving and would read LU-overwritten storage otherwise.
_initsolve(A, alg; aliasA = false) =
    init(LinearProblem(A, zeros(eltype(A), size(A, 1))), alg; alias_A = aliasA)

# Point the cache at a new right-hand side without invalidating the cached
# factorization, then return the cache for chaining. LinearSolve's cache does not
# itself validate the RHS length against A (it can silently return a wrong-length
# result), so check explicitly and throw the same `DimensionMismatch` type that
# `A \ b` throws for a mismatched RHS.
function _setb!(cache, b)
    size(b, 1) == size(cache.A, 1) || throw(DimensionMismatch(
        "A has $(size(cache.A, 1)) rows, but the right-hand side has $(size(b, 1))"))
    cache.b = b
    return cache
end

# Solve `A * x = b` for a vector RHS, reusing the cache's factorization.
_solve!(cache, b::AbstractVector) = copy(solve!(_setb!(cache, b)).u)

# Solve `A * X = B` for a matrix RHS, column by column, reusing the factorization.
# The result is preallocated and each solved column is written in place, avoiding
# the repeated reallocation that `reduce(hcat, ...)` would incur.
#
# NOTE: LinearSolve's `solve!` only accepts a vector RHS, so we solve one column at
# a time. This forfeits the batched BLAS3 (`trsm`) solve that a plain `A \ B` would
# use, making wide right-hand sides (multi-column samples, or `A \ P` in the
# generalized path at large `npoly`) slower than a direct factorization. Revisit once
# LinearSolve.jl 4.0 ships native multiple-RHS batching.
function _solve!(cache, B::AbstractMatrix)
    x1 = solve!(_setb!(cache, B[:, 1])).u
    X = Matrix{eltype(x1)}(undef, length(x1), size(B, 2))
    X[:, 1] = x1
    for j in 2:size(B, 2)
        X[:, j] = solve!(_setb!(cache, B[:, j])).u
    end
    return X
end

@inline function solveForWeights(A, points, samples,
                                    rbf::Union{T, AbstractVector{T}} where T <: RadialBasisFunction,
                                    metric; linsolve = nothing)
    cache = _initsolve(A, linsolve; aliasA = true)
    w = _solve!(cache, samples)
    RBFInterpolant(w, points, rbf, metric)
end
@inline function solveForWeights(A, points, samples,
                                    rbf::Union{T, AbstractVector{T}} where T <: Union{GeneralizedRadialBasisFunction, RadialBasisFunction},
                                    metric; linsolve = nothing)
    # Use the maximum degree among the generalized RBF:s
    P = getPolynomial(rbf, points)

    # Blocked (Schur-complement) system. One reusable cache factorizes A once and
    # is reused for every RHS (P, samples, and the final combined RHS); the small
    # npoly×npoly Schur system uses LinearSolve's default algorithm.
    cacheA = _initsolve(A, linsolve; aliasA = true)
    AinvP  = _solve!(cacheA, P)
    Ainvs  = _solve!(cacheA, samples)

    B = -P' * AinvP
    cacheB = _initsolve(B, nothing)
    E = _solve!(cacheB, P' * Ainvs)

    w = _solve!(cacheA, samples + P * E)
    λ = -E

    GeneralizedRBFInterpolant(w, λ, points, rbf, metric)
end

function evaluate(itp::RadialBasisInterpolant, points::AbstractArray{<:Real, 2})

    # Compute distance matrix and evaluate the RBF
    A = pairwise(itp.metric, points, itp.points; dims=2)
    A = evaluateRBF!(A, itp.rbf, false)

    # Compute polynomial matrix for generalized RBF:s
    P = getPolynomial(itp.rbf, points)

    # Compute the interpolated values
    return computeInterpolatedValues(A, P, itp)
end

@inline getPolynomial(rbf::Union{T, AbstractVector{T}} where T <: RadialBasisFunction, points) = nothing
@inline function getPolynomial(rbf::Union{T, AbstractVector{T}} where T <: Union{GeneralizedRadialBasisFunction, RadialBasisFunction}, points)

    # Use the maximum degree among the generalized RBF:s
    degree = maximum(x isa GeneralizedRadialBasisFunction ? x.degree : 0 for x in rbf)
    P = generateMultivariatePolynomial(points, degree)
end

@inline computeInterpolatedValues(A, P, itp::RBFInterpolant) = A*itp.w
@inline computeInterpolatedValues(A, P, itp::GeneralizedRBFInterpolant) = A*itp.w + P*itp.λ

# Helper function to generate matrices defining complete homogenic symmetric polynomials
function generateMultivariatePolynomial(points::AbstractArray{<:Real, 2}, degree::Integer)

    # How big should the matrix be?
    nDimensions, nPoints = size(points)
    nTerms = binomial(degree + nDimensions, degree)

    P = ones(nPoints, nTerms)

    # Start with the lowest orders and work upwards
    position = 2
    for order = 1:degree
        for combination in with_replacement_combinations(1:nDimensions, order)
            for var in combination
                @views P[:, position] .*= points[var, :]
            end
            position += 1
        end
    end

    P
end