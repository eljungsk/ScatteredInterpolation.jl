module ScatteredInterpolation

using Distances

export interpolate,
    evaluate

abstract type ScatteredInterpolant end

include("./rbf.jl")

end
