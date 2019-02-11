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
using ScatteredInterpolation
```

Interpolate data defined on the vertices and in the center of a square using a
multiquadratic radial basis function:
```@example 1
samples = [0.0; 0.5; 0.5; 0.5; 1.0];
points = [0.0 0.0; 0.0 1.0; 1.0 0.0; 0.5 0.5; 1.0 1.0]';
itp = interpolate(Multiquadratic(), points, samples)
interpolated = evaluate(itp, [0.6; 0.6])
```

If we instead want to use nearest neighbor interpolation, we can run
```@example 1
itp = interpolate(NearestNeighbor(), points, samples)
interpolated = evaluate(itp, [0.6; 0.6])
```

### Gridding of data
A common use case for scattered interpolation is gridding of data, i.e. interpolation of
scattered data to a grid. Using the same data as above, we can interpolate it to a 5x5 grid

```@example 1
n = 5
x = range(0, stop = 1, length = n)
y = range(0, stop = 1, length = n)
X = repeat(x, n)[:]
Y = repeat(y', n)[:]
gridPoints = [X Y]'

itp = interpolate(Multiquadratic(), points, samples)
interpolated = evaluate(itp, gridPoints)
gridded = reshape(interpolated, n, n)
```
