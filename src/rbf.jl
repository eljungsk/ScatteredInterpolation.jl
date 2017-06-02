abstract type RadialBasisFunction end

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
        struct $rbf{T} <: RadialBasisFunction
        end

        # Define constructors
        function $rbf(c::Real = 1)
            $rbf{c}()
        end
    end
end

"""
    ThinPlate()

Define a Thin Plate Spline Radial Basis Function

```math
ϕ(r) = r^2 ln(r)
```

This is a shorthand for `Polyharmonic(2)`.
"""
const ThinPlate = Polyharmonic{2}

"""
    Gaussian(ɛ = 1)

Define a Gaussian Radial Basis Function

```math
ϕ(r) = e^{-(ɛr)^2}
```
"""
@generated function (::Gaussian{C})(r::Real) where C
    :(exp(-($C*r)^2))
end

"""
    Multiquadratic(ɛ = 1)

Define a Multiquadratic Radial Basis Function

```math
ϕ(r) = \\sqrt{1 + (ɛr)^2}
```
"""
@generated function (::Multiquadratic{C})(r::Real) where C
    :(sqrt(1 + ($C*r)^2))
end

"""
    InverseQuadratic(ɛ = 1)

Define an Inverse Quadratic Radial Basis Function

```math
ϕ(r) = \\frac{1}{1 + (ɛr)^2}
```
"""
@generated function (::InverseQuadratic{C})(r::Real) where C
    :(1/(1 + ($C*r)^2))
end

"""
    InverseMultiquadratic(ɛ = 1)

Define an Inverse Multiquadratic Radial Basis Function

```math
ϕ(r) = \\frac{1}{\\sqrt{1 + (ɛr)^2}}
```
"""
@generated function (::InverseMultiquadratic{C})(r::Real) where C
    :(1/sqrt(1 + ($C*r)^2))
end

"""
    Polyharmonic(k = 1)

Define a Polyharmonic Spline Radial Basis Function

```math
ϕ(r) = r^k, k = 1, 3, 5, ...
\\\\
ϕ(r) = r^k ln(r), k = 2, 4, 6, ...
```
"""
@generated function (::Polyharmonic{C})(r::Real) where C

    @assert typeof(C) <: Integer && C > 0

    # Distinguish odd and even cases
    expr = if C % 2 == 0
        :(r > 0 ? r^$C*log(r) : 0.0)
    else
        :(r^$C)
    end

    expr
end

struct RBFInterpolant{F, T, N, M} <: ScatteredInterpolant

    w::Array{T,N}
    points::Array{T,2}
    rbf::F
    metric::M
end

function interpolate(rbf::RadialBasisFunction,
                     points::Array{<:Real,2},
                     samples::Array{<:Number,N};
                     metric = Euclidean()) where N

    # Compute pairwise distances and apply the Radial Basis Function
    A = pairwise(metric, points)
    @. A = rbf(A)

    # Solve for the weights
    w = A\samples

    # Create and return an interpolation object
    return RBFInterpolant(w, points, rbf, metric)

end

function evaluate(itp::RBFInterpolant, points::Array{<:Real, 2})

    # Compute distance matrix
    A = pairwise(itp.metric, points, itp.points)
    @. A = itp.rbf(A)

    # Compute the interpolated values
    return A*itp.w
end
