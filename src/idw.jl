abstract type ShepardType end

export  Shepard

"""
    Shepard(P::Real)

Standard Shepard interpolation with power parameter `P`.
Default is `P = 2`.

```math
\begin{cases} 
    \frac{\displaystyle \sum_{i = 1}^{N}{ w_i(\mathbf{x}) u_i } } 
        { \displaystyle \sum_{i = 1}^{N}{ w_i(\mathbf{x}) } }, 
         & \text{if } d(\mathbf{x},\mathbf{x}_i) \neq 0 \text{ for all } i \\ 
    u_i, & \text{if } d(\mathbf{x},\mathbf{x}_i) = 0 \text{ for some } i
\end{cases}
```
"""
struct Shepard{P} <: ShepardType end
Shepard(P::Real = 2) = Shepard{P}()

struct ShepardInterpolant{F, T1, T2, N, M} <: ScatteredInterpolant

    data::Array{T1,N}
    points::Array{T2,2}
    idw::F
    metric::M
end

# No need to compute anything here, everything is done in the evaluation step.
function interpolate(idw::ShepardType, points::Array{T,2}, samples::Array{T,N}; metric = Euclidean()) where T <: Number where N

    return ShepardInterpolant(samples, points, idw, metric)
end

function evaluate(itp::ShepardInterpolant, points::Array{T,2}) where T <: Real

    # Compute distances between sample points and interpolation points
    d = pairwise(itp.metric, itp.points, points)

    # Evaluate point by point
    m = size(points, 2)
    n = size(itp.data, 2)
    values = zeros(m, n)
    for i = 1:m

        d_col = d[:,i]

        # If an interpolation point coincide with a sampling point, just return the original data
        # Otherwise, compute distance-weighted sum
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
function evaluatePoint(::Shepard{P}, dataPoints::Array{T,2}, data::Array{T,N}, d::Vector) where T <: Number where N where P

    # Compute weigths and return the weighted sum
    w = d.^P
    value = sum(w.*data, 1)./sum(w)
end
