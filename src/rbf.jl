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
                     linsolve = nothing) where {N} where {S<:Number}

    #hinder smooth from being set to true and interpreted as the value 1 
    @assert smooth != true "set the smoothing value as a number or vector of numbers"

    # Compute pairwise distances, apply the Radial Basis Function
    # and optional smoothing (ridge regression)
    A = pairwise(metric, points;dims=2)
    
    A = evaluateRBF!(A, rbf, smooth)

    # Solve for the weights
    itp = solveForWeights(A, points, samples, rbf, metric; linsolve = linsolve)

    # Create and return an interpolation object
    if returnRBFmatrix    # Return matrix A
        return itp, A
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
_initsolve(A, alg) = init(LinearProblem(A, zeros(eltype(A), size(A, 1))), alg)

# Point the cache at a new right-hand side without invalidating the cached
# factorization, then return the cache for chaining. LinearSolve's cache does not
# itself validate the RHS length against A, so check explicitly here (matching the
# DimensionMismatch that `A \ b` would throw for a mismatched RHS).
function _setb!(cache, b)
    size(b, 1) == size(cache.A, 1) || throw(DimensionMismatch(
        "second dimension of A, $(size(cache.A, 1)), does not match length of b, $(size(b, 1))"))
    cache.b = b
    return cache
end

# Solve `A * x = b` for a vector RHS, reusing the cache's factorization.
_solve!(cache, b::AbstractVector) = copy(solve!(_setb!(cache, b)).u)

# Solve `A * X = B` for a matrix RHS, column by column, reusing the factorization.
_solve!(cache, B::AbstractMatrix) =
    reduce(hcat, (_solve!(cache, B[:, j]) for j in axes(B, 2)))

@inline function solveForWeights(A, points, samples,
                                    rbf::Union{T, AbstractVector{T}} where T <: RadialBasisFunction,
                                    metric; linsolve = nothing)
    cache = _initsolve(A, linsolve)
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
    cacheA = _initsolve(A, linsolve)
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
                P[:, position] .*= points[var, :]
            end
            position += 1
        end
    end

    P
end