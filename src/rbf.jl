abstract type RadialBasisFunction <: InterpolationMethod end

export  Gaussian,
        Multiquadratic,
        InverseQuadratic,
        InverseMultiquadratic,
        Polyharmonic,
        ThinPlate

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


struct RBFInterpolant{F, T, N, M} <: ScatteredInterpolant

    w::Array{T,N}
    points::Array{T,2}
    rbf::F
    metric::M
end

function interpolate(rbf::RadialBasisFunction,
                     points::AbstractArray{<:Real,2},
                     samples::AbstractArray{<:Number,N};
                     metric = Euclidean(), returnRBFmatrix::Bool = false) where {N}

    # Compute pairwise distances and apply the Radial Basis Function
    A = pairwise(metric, points)
    @dotcompat A = rbf(A)

    # Solve for the weights
    w = A\samples

    # Create and return an interpolation object
    if returnRBFmatrix    # Return matrix A
        return RBFInterpolant(w, points, rbf, metric), A
    else
        return RBFInterpolant(w, points, rbf, metric)
    end

end

function interpolate(rbfs::Vector{T} where T <: ScatteredInterpolation.RadialBasisFunction,
                     points::AbstractArray{<:Real,2},
                     samples::AbstractArray{<:Number,N};
                     metric = Euclidean(), returnRBFmatrix::Bool = false) where {N}

    # Compute pairwise distances and apply the Radial Basis Function
    A = pairwise(metric, points)

    n = size(points,2)
    for (j, rbf) in enumerate(rbfs)
        for i = 1:n
            A[i,j] = rbf(A[i,j])
        end
    end

    # Solve for the weights
    w = A\samples

    # Create and return an interpolation object
    if returnRBFmatrix    # Return matrix A
        return RBFInterpolant(w, points, rbfs, metric), A
    else
        return RBFInterpolant(w, points, rbfs, metric)
    end

end

function evaluate(itp::RBFInterpolant, points::AbstractArray{<:Real, 2})

    # Compute distance matrix
    A = pairwise(itp.metric, points, itp.points)
    @dotcompat A = itp.rbf(A)

    # Compute the interpolated values
    return A*itp.w
end

function evaluate(itp::RBFInterpolant{S,T,U,V}, points::AbstractArray{<:Real, 2}) where {S <: Vector, T, U, V}

    # Compute distance matrix
    A = pairwise(itp.metric, points, itp.points)
    for (j, rbf) in enumerate(itp.rbf)
        @. A[:,j] = rbf(A[:,j])
    end

    # Compute the interpolated values
    return A*itp.w
end
