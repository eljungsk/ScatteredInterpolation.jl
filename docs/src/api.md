# API

```@index
Order   = [:function, :type]
```

## Functions
```@docs
interpolate
evaluate
addpoints!
```

## Types

### Radial Basis Functions

```@docs
Multiquadratic
InverseMultiquadratic
Gaussian
InverseQuadratic
Polyharmonic
ThinPlate
GeneralizedMultiquadratic
GeneralizedPolyharmonic
Wendland
```

### Compactly supported RBFs (sparse path)

[`Wendland`](@ref) is the only compactly supported basis function, and the only one
eligible for the `sparse` interpolation path — see [Compactly supported RBFs and the
sparse path](@ref) for the full picture of the `sparse`/`neighbors` keywords, the
``r = 1/\varepsilon`` relation, and the accuracy/sparsity trade-off.

```@docs
CompactSupportRBFInterpolant
```

### Partition of Unity

```@docs
PartitionOfUnity
```

### Inverse Distance Weighting (Shepard)

```@docs
Shepard
```

### Nearest Neighbor

```@docs
NearestNeighbor
```
