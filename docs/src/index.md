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
```jldoctest intro
julia> using ScatteredInterpolation

julia> samples = [0.0; 0.5; 0.5; 0.5; 1.0];

julia> points = [0.0 0.0; 0.0 1.0; 1.0 0.0; 0.5 0.5; 1.0 1.0]';

julia> itp = interpolate(Multiquadratic(), points, samples);

julia> interpolated = evaluate(itp, [0.6; 0.6])
1-element Array{Float64,1}:
 0.6105036860019827
```

If we instead want to use nearest neighbor interpolation:
```jldoctest intro
julia> itp = interpolate(NearestNeighbor(), points, samples);

julia> interpolated = evaluate(itp, [0.6; 0.6])
1×1 Array{Float64,2}:
 0.5

```

### Gridding of data
A common use case for scattered interpolation is gridding of data, i.e. interpolation of
scattered data to a grid. Using the same data as above, we can interpolate it to a 5x5 grid

```@meta
DocTestSetup = quote
    using ScatteredInterpolation
    samples = [0.0; 0.5; 0.5; 0.5; 1.0];
    points = [0.0 0.0; 0.0 1.0; 1.0 0.0; 0.5 0.5; 1.0 1.0]';
end
```

```jldoctest gridding
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

5×5 Array{Float64,2}:
 -2.22045e-16  0.108133  0.236473  0.370317  0.5     
  0.108133     0.226333  0.361499  0.497542  0.62459 
  0.236473     0.361499  0.5       0.635589  0.758248
  0.370317     0.497542  0.635589  0.768751  0.886774
  0.5          0.62459   0.758248  0.886774  1.0
```
