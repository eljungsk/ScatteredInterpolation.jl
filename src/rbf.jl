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
            :InverseMultiquadratic,
            :Polyharmonic)
    @eval begin
        struct $rbf{T} <: RadialBasisFunction where T <: Real
            ɛ::T    
        end

        # Define default constructors
        $rbf() = $rbf(1)
    end
end

# Generalized RBF:s
struct GeneralizedMultiquadratic{T<:Real, S<:Real, U<:Integer} <: GeneralizedRadialBasisFunction
    ε::T
    β::S
    degree::U
end

struct GeneralizedPolyharmonic{S<:Integer, U<:Integer} <: GeneralizedRadialBasisFunction
    k::S
    degree::U
end

@doc "
    Gaussian(ɛ = 1)

Define a Gaussian Radial Basis Function

```math
ϕ(r) = e^{-(ɛr)^2}
```
" Gaussian
(rbf::Gaussian)(r) = exp(-(rbf.ɛ*r)^2) 

@doc "
    Multiquadratic(ɛ = 1)

Define a Multiquadratic Radial Basis Function

```math
ϕ(r) = \\sqrt{1 + (ɛr)^2}
```
" Multiquadratic
(rbf::Multiquadratic)(r) = (sqrt(1 + (rbf.ɛ*r)^2))

@doc "
    InverseQuadratic(ɛ = 1)

Define an Inverse Quadratic Radial Basis Function

```math
ϕ(r) = \\frac{1}{1 + (ɛr)^2}
```
" InverseQuadratic
(rbf::InverseQuadratic)(r) = (1/(1 + (rbf.ɛ*r)^2))

@doc "
    InverseMultiquadratic(ɛ = 1)

Define an Inverse Multiquadratic Radial Basis Function

```math
ϕ(r) = \\frac{1}{\\sqrt{1 + (ɛr)^2}}
```
" InverseMultiquadratic
(rbf::InverseMultiquadratic)(r) = (1/sqrt(1 + (rbf.ɛ*r)^2))

@doc "
    Polyharmonic(k = 1)

Define a Polyharmonic Spline Radial Basis Function

```math
ϕ(r) = r^k, k = 1, 3, 5, ...
\\\\
ϕ(r) = r^k ln(r), k = 2, 4, 6, ...
```
" Polyharmonic
function (rbf::Polyharmonic{T})(r) where T <: Integer
    @assert rbf.ɛ > 0

    # Distinguish odd and even cases
    expr = if rbf.ɛ % 2 == 0
        (r > 0 ? r^rbf.ɛ*log(r) : 0.0)
    else
        (r^rbf.ɛ)
    end

    expr
end

@doc "
    ThinPlate()

Define a Thin Plate Spline Radial Basis Function

```math
ϕ(r) = r^2 ln(r)
``` 

This is a shorthand for `Polyharmonic(2)`.
" ThinPlate
ThinPlate() = Polyharmonic(2)

@doc "
    GeneralizedMultiquadratic(ɛ, β, degree)

Define a generalized Multiquadratic Radial Basis Function

```math
ϕ(r) = (1 + (ɛ*r)^2)^β
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
    @assert rbf.k > 0

    # Distinguish odd and even cases
    expr = if rbf.k % 2 == 0
        (r > 0 ? r^rbf.k*log(r) : 0.0)
    else
        (r^rbf.k)
    end

    expr
end

abstract type RadialBasisInterpolant <: ScatteredInterpolant end

struct RBFInterpolant{F, T, A, N, M} <: RadialBasisInterpolant where A <: AbstractArray{<:Real, 2}

    w::Array{T,N}
    points::A
    rbf::F
    metric::M
end

struct GeneralizedRBFInterpolant{F, T, A, N, M} <: RadialBasisInterpolant where A <: AbstractArray{<:Real, 2}

    w::Array{T,N}
    λ::Array{T,N}
    points::A
    rbf::F
    metric::M
end

function interpolate(rbf::Union{T, AbstractVector{T}} where T <: AbstractRadialBasisFunction,
                     points::AbstractArray{<:Real,2},
                     samples::AbstractArray{<:Number,N};
                     metric = Euclidean(), returnRBFmatrix::Bool = false,
                     smooth::Union{S, AbstractVector{S}} = false) where {N} where {S<:Number}

    #hinder smooth from being set to true and interpreted as the value 1 
    @assert smooth != true "set the smoothing value as a number or vector of numbers"

    # Compute pairwise distances, apply the Radial Basis Function
    # and optional smoothing (ridge regression)
    A = pairwise(metric, points;dims=2)
    
    evaluateRBF!(A, rbf, smooth)

    # Solve for the weights
    itp = solveForWeights(A, points, samples, rbf, metric)

    # Create and return an interpolation object
    if returnRBFmatrix    # Return matrix A
        return itp, A
    else
        return itp
    end

end

@inline function evaluateRBF!(A, rbf)
    A .= rbf.(A)
end
@inline function evaluateRBF!(A, rbfs::AbstractVector{<:AbstractRadialBasisFunction})
    for (j, rbf) in enumerate(rbfs)
        A[:,j] .= rbf.(@view A[:,j])
    end
end
@inline function evaluateRBF!(A, rbf, smooth::Bool)
    A .= rbf.(A)
end
@inline function evaluateRBF!(A, rbfs::AbstractVector{<:AbstractRadialBasisFunction}, smooth::Bool)
    for (j, rbf) in enumerate(rbfs)
        A[:,j] .= rbf.(@view A[:,j])
    end
end
@inline function evaluateRBF!(A, rbf, smooth::Vector{T}) where T <: Number
    A .= rbf.(A)
    for i = 1:size(A,1)
        A[i,i] += smooth[i]
    end
end
@inline function evaluateRBF!(A, rbfs::AbstractVector{<:AbstractRadialBasisFunction}, smooth::Vector{T}) where T <: Number
    for (j, rbf) in enumerate(rbfs)
        A[:,j] .= rbf.(@view A[:,j])
    end
    for i = 1:size(A,1)
        A[i,i] += smooth[i]
    end
end
@inline function evaluateRBF!(A, rbf, smooth::T) where T <: Number
    A .= rbf.(A)
    for i = 1:size(A,1)
        A[i,i] += smooth
    end
end
@inline function evaluateRBF!(A, rbfs::AbstractVector{<:AbstractRadialBasisFunction}, smooth::T) where T <: Number
    for (j, rbf) in enumerate(rbfs)
        A[:,j] .= rbf.(@view A[:,j])
    end
    for i = 1:size(A,1)
        A[i,i] += smooth
    end
end

@inline function solveForWeights(A, points, samples,
                                    rbf::Union{T, AbstractVector{T}} where T <: RadialBasisFunction,
                                    metric)
    w = A\samples
    RBFInterpolant(w, points, rbf, metric)
end
@inline function solveForWeights(A, points, samples,
                                    rbf::Union{T, AbstractVector{T}} where T <: Union{GeneralizedRadialBasisFunction, RadialBasisFunction},
                                    metric)
    # Use the maximum degree among the generalized RBF:s
    P = getPolynomial(rbf, points)

    # Solve for the weights and polynomial coefficients
    # We end up with a blocked system, so we don't have to form the full matrix
    Af = factorize(A)
    B = -P'*(Af\P)
    E = B\(P'*(Af\samples))

    w = A\(I*samples + P*E)
    λ = -E

    GeneralizedRBFInterpolant(w, λ, points, rbf, metric)
end

function evaluate(itp::RadialBasisInterpolant, points::AbstractArray{<:Real, 2})

    # Compute distance matrix and evaluate the RBF
    A = pairwise(itp.metric, points, itp.points;dims=2)
    evaluateRBF!(A, itp.rbf)

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
    while position <= nTerms
        for order = 1:degree
            for combination in with_replacement_combinations(1:nDimensions, order)
                for var in combination
                    P[:, position] .*= points[var, :]
                end
                position += 1
            end
        end
    end

    P
end