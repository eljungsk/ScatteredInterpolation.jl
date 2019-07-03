abstract type ShepardType <: InterpolationMethod end

export Shepard

"""
    Shepard(P = 2)

Standard Shepard interpolation with power parameter `P`.
"""
struct Shepard{T} <: ShepardType where T <: Real
    P::T
end
Shepard() = Shepard(2)

struct ShepardInterpolant{T1, T2, F, M} <: ScatteredInterpolant where {T1 <: AbstractArray, T2 <: AbstractMatrix{<:Real}}

    data::T1
    points::T2
    idw::F
    metric::M
end

# No need to compute anything here, everything is done in the evaluation step.
function interpolate(idw::ShepardType,
                     points::AbstractArray{<:Real,2},
                     samples::AbstractArray{<:Number,N};
                     metric = Euclidean()) where {N}

    return ShepardInterpolant(samples, points, idw, metric)
end

function evaluate(itp::ShepardInterpolant, points::AbstractArray{<:Real,2})

    # Compute distances between sample points and interpolation points
    d = pairwise(itp.metric, itp.points, points;dims=2)

    # Evaluate point by point
    m = size(points, 2)
    n = size(itp.data, 2)
    values = zeros(eltype(itp.data), m, n)
    for i = 1:m

        d_col = d[:,i]

        # If an interpolation point coincide with a sampling point, just return the 
        # original data. Otherwise, compute distance-weighted sum
        if !all(r > 0 for r in d_col)
            ind = findfirst(x -> x ≈ 0.0, d_col)
            values[i,:] = itp.data[ind, :]
        else
            values[i,:] = evaluatePoint(itp.idw, itp.points, itp.data, d_col)
        end
    end

    return values
end

# Original Shepard
function evaluatePoint(idw::Shepard,
                       dataPoints::AbstractArray{<:Real,2},
                       data::AbstractArray{<:Number,N},
                       d::AbstractVector) where {N}

    # Compute weigths and return the weighted sum
    w = 1.0./(d.^idw.P)
    value = sum(w.*data, dims = 1)./sum(w)
end