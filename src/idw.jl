@compat abstract type ShepardType end

export  Shepard

"""
    Shepard(P = 2)

Standard Shepard interpolation with power parameter `P`.
"""
immutable Shepard{P} <: ShepardType end
Shepard(P::Real = 2) = Shepard{P}()

immutable ShepardInterpolant{F, T1, T2, N, M} <: ScatteredInterpolant

    data::Array{T1,N}
    points::Array{T2,2}
    idw::F
    metric::M
end

# No need to compute anything here, everything is done in the evaluation step.
@compat function interpolate{N}(idw::ShepardType,
                     points::Array{<:Real,2},
                     samples::Array{<:Number,N};
                     metric = Euclidean())

    return ShepardInterpolant(samples, points, idw, metric)
end

@compat function evaluate(itp::ShepardInterpolant, points::Array{<:Real,2})

    # Compute distances between sample points and interpolation points
    d = pairwise(itp.metric, itp.points, points)

    # Evaluate point by point
    m = size(points, 2)
    n = size(itp.data, 2)
    values = zeros(m, n)
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
@compat function evaluatePoint{N, P}(::Shepard{P},
                       dataPoints::Array{<:Real,2},
                       data::Array{<:Number,N},
                       d::Vector)

    # Compute weigths and return the weighted sum
    w = d.^P
    value = sum(w.*data, 1)./sum(w)
end
