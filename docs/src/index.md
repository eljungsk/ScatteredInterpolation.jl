# ScatteredInterpolation.jl

*Interpolation of scattered data in Julia*

`ScatteredInterpolation.jl` interpolates scattered data in arbitrary dimensions. 

```@contents
Pages = [
    "index.md",
    "methods.md",
    "api.md"
]
Depth = 2
```

## Example Usage

```@example 1
```

Interpolate data defined on the vertices and in the center of a square using a
multiquadratic radial basis function:

```@meta
DocTestSetup = quote
    using Printf
    Base.show(io::IO, f::Float64) = @printf(io, "%.4f", f)
end
```

```jldoctest intro
julia> using ScatteredInterpolation

julia> samples = [0.0; 0.5; 0.5; 0.5; 1.0];

julia> points = [0.0 0.0; 0.0 1.0; 1.0 0.0; 0.5 0.5; 1.0 1.0]';

julia> itp = interpolate(Multiquadratic(), points, samples);

julia> interpolated = evaluate(itp, [0.6; 0.6])
1-element Vector{Float64}:
 0.6105
```

If we instead want to use nearest neighbor interpolation:
```jldoctest intro
julia> itp = interpolate(NearestNeighbor(), points, samples);

julia> interpolated = evaluate(itp, [0.6; 0.6])
1×1 Matrix{Float64}:
 0.5000

```

### Gridding of data
A common use case for scattered interpolation is gridding of data, i.e. interpolation of
scattered data to a grid. Using the same data as above, we can interpolate it to a 5x5 grid

```@meta
DocTestSetup = quote
    using Printf
    Base.show(io::IO, f::Float64) = @printf(io, "%.4f", f)
    using ScatteredInterpolation
    samples = [0.0; 0.5; 0.5; 0.5; 1.0];
    points = [0.0 0.0; 0.0 1.0; 1.0 0.0; 0.5 0.5; 1.0 1.0]';
end
```

```jldoctest gridding; filter = r"-?0.0000"
n = 5
x = range(0, stop = 1, length = n)
y = range(0, stop = 1, length = n)
X = repeat(x, n)[:]
Y = repeat(y', n)[:]
gridPoints = [X Y]'

itp = interpolate(Multiquadratic(), points, samples)
interpolated = evaluate(itp, gridPoints)
gridded = reshape(interpolated, n, n)

# output

5×5 Matrix{Float64}:
 -0.0000  0.1081  0.2365  0.3703  0.5000
  0.1081  0.2263  0.3615  0.4975  0.6246
  0.2365  0.3615  0.5000  0.6356  0.7582
  0.3703  0.4975  0.6356  0.7688  0.8868
  0.5000  0.6246  0.7582  0.8868  1.0000
```
