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
        function $rbf(c::T) where T <: Real
            $rbf{c}()
        end

        function $rbf()
            $rbf{1}()
        end
    end
end

# Add shorthand for thin plate splines
const ThinPlate = Polyharmonic{2}

# Define fuctions to return the RBF functions
@generated function (::Gaussian{C})(r::T) where C where T <: Real
    :(exp(-($C*r)^2))
end

@generated function (::Multiquadratic{C})(r::T) where C where T <: Real
    :(sqrt(1 + ($C*r)^2))
end

@generated function (::InverseQuadratic{C})(r::T) where C where T <: Real
    :(1/(1 + ($C*r)^2))
end

@generated function (::InverseMultiquadratic{C})(r::T) where C where T <: Real
    :(1/sqrt(1 + ($C*r)^2))
end

@generated function (::Polyharmonic{C})(r::T) where C where T <: Real

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

# Create the interpolation
function interpolate(rbf::F, points::Array{T,2}, samples::Array{T,N}; metric = Euclidean()) where F <: RadialBasisFunction where T <: Number where N

    # Compute pairwise distances and apply the Radial Basis Function
    A = pairwise(metric, points)
    @. A = rbf(A)

    # Solve for the weights
    w = A\samples

    # Create and return an interpolation object
    return RBFInterpolant(w, points, rbf, metric)

end

# Evaluate the interpolation at given locations
function evaluate(itp::RBFInterpolant, points::Array{T, 2}) where T <: Number

    # Compute distance matrix
    A = pairwise(itp.metric, points, itp.points)
    @. A = itp.rbf(A)

    # Compute the interpolated values
    return A*itp.w
end

# Fallback method for the case of just one point
function evaluate(itp::RBFInterpolant, points::Array{T, 1}) where T <: Number

    # pairwise requires the points array to be 2-d.
    n = length(points)
    points = reshape(points, n, 1)

    evaluate(itp, points)
end
