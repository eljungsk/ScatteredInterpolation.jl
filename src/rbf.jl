abstract type RadialBasisFunction end

export  Gaussian,
        Multiquadratic,
        InverseQuadratic,
        InverseMultiquadratic,
        Polyharmonic,
        ThinPlate

# Define types for the different kinds of radial basis functions
for rbf in (:Gaussian, :Multiquadratic, :InverseQuadratic, :InverseMultiquadratic, :Polyharmonic)
    @eval begin
        struct $rbf{T} <: RadialBasisFunction
        end

        # Define constructors
        function $rbf(c::Real)
            $rbf{c}()
        end

        function $rbf()
            $rbf{1}()
        end
    end
end

"""
    ThinPlate()

Define a Thin Plate Radial Basis Function

``
ϕ(r) = r^2 ln(r)
``

This is a shorthand for `Polyharmonic{2}()`.
"""
const ThinPlate = Polyharmonic{2}

"""
    Gaussian(ɛ::Real)

Define a Gaussian Radial Basis Function

``
ϕ(r) = e^{-(ɛr)^2}
``
"""
@generated function (::Gaussian{C})(r::Real) where C
    :(exp(-($C*r)^2))
end

"""
    Multiquadratic(ɛ::Real)

Define a Multiquadratic Radial Basis Function

``
ϕ(r) = \sqrt{1 + (ɛr)^2}
``
"""
@generated function (::Multiquadratic{C})(r::Real) where C
    :(sqrt(1 + ($C*r)^2))
end

"""
    InverseQuadratic(ɛ::Real)

Define an Inverse Quadratic Radial Basis Function

``
ϕ(r) = \frac{1}{1 + (ɛr)^2}
``
"""
@generated function (::InverseQuadratic{C})(r::Real) where C
    :(1/(1 + ($C*r)^2))
end

"""
    InverseMultiquadratic(ɛ::Real)

Define an Inverse Multiquadratic Radial Basis Function

``
ϕ(r) = \frac{1}{\sqrt{1 + (ɛr)^2}}
``
"""
@generated function (::InverseMultiquadratic{C})(r::Real) where C
    :(1/sqrt(1 + ($C*r)^2))
end

"""
    Polyharmonic(ɛ::Real)

Define a Polyharmonic Radial Basis Function

``
ϕ(r) = r^k, k = 1, 3, 5, ...
``

``
ϕ(r) = r^k ln(r), k = 2, 4, 6, ...
``
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

"""
    interpolate(rbf, points, samples [; metric]) -> RBFInterpolant

Create an interpolation of the data in `samples` sampled at the locations defined in `points` based on the Radial Basis Function `rbf`.
`metric` is any of the metrics defined by the `Distances` package.
"""
function interpolate(rbf::RadialBasisFunction, points::Array{T,2}, samples::Array{T,N}; metric = Euclidean()) where T <: Number where N

    # Compute pairwise distances and apply the Radial Basis Function
    A = pairwise(metric, points)
    @. A = rbf(A)

    # Solve for the weights
    w = A\samples

    # Create and return an interpolation object
    return RBFInterpolant(w, points, rbf, metric)

end

"""
    evaluate(itp, points)

Evaluate an interpolation object `itp` at the locations defined in `points`.
"""
function evaluate(itp::RBFInterpolant, points::Array{T, 2}) where T <: Number

    # Compute distance matrix
    A = pairwise(itp.metric, points, itp.points)
    @. A = itp.rbf(A)

    # Compute the interpolated values
    return A*itp.w
end
