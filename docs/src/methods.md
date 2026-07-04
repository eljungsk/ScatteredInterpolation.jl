# Supported methods
Currently, four different interpolation methods are available; Radial Basis Functions,
Radial Basis Function Partition of Unity, Inverse Distance Weighting and Nearest
Neighbor.

## Radial Basis Functions 

For radial basis function interpolation, the interpolated value at some point 
``\mathbf{x}`` is given by
```math
u(\mathbf{x}) = \displaystyle \sum_{i = 1}^{N}{ w_i \phi(||\mathbf{x} - \mathbf{x_i}||)}
```
where ``||\mathbf{x} - \mathbf{x_i}||`` is the distance between ``\mathbf{x}`` and 
``\mathbf{x}_i``, and ``\phi(r)`` is one of the basis functions defined below.

To use radial basis function interpolation, pass one of the available basis functions as 
`method` to `interpolate`.

If a `GeneralizedRadialBasisFunction` is used, an additional polynomial term is added in
order for the resulting matrix to be positive definite:
```math
u(\mathbf{x}) = \displaystyle \sum_{i = 1}^{N}{ w_i \phi(||\mathbf{x} - \mathbf{x_i}||)} + \mathbf{P}(\mathbf{x})\mathbf{λ}
```
where ``\mathbf{P}(\mathbf{x})`` is the matrix defining a complete homogeneous symmetric
polynomial of degree `degree`, and ``\mathbf{λ}`` is a vector containing the polynomial
coefficients.

### Available basis functions

  * [`Multiquadratic`](@ref)

    ```math
    ϕ(r) = \sqrt{1 + (εr)^2}
    ```

  * [`InverseMultiquadratic`](@ref)

    ```math
    ϕ(r) = \frac{1}{\sqrt{1 + (εr)^2}}
    ```

  * [`Gaussian`](@ref)

    ```math
    ϕ(r) = e^{-(εr)^2}
    ```

  * [`InverseQuadratic`](@ref)

    ```math
    ϕ(r) = \frac{1}{1 + (εr)^2}
    ```

  * [`Polyharmonic`](@ref) spline

    ```math
    ϕ(r) = 
    \begin{cases}
        \begin{align*}
            &r^k                    &   k = 1, 3, 5, ... \\
            &r^k \mathrm{ln}(r)     &   k = 2, 4, 6, ...
        \end{align*}
    \end{cases}
    ```

  * [`ThinPlate`](@ref) spline

    A thin plate spline is the special case ``k = 2`` of the polyharmonic splines.
    `ThinPlate()` is a shorthand for `Polyharmonic(2)`.

  * [`GeneralizedMultiquadratic`](@ref)

    ```math
    ϕ(r) = \left(1 + (εr)^2\right)^\beta
    ```
    The generalzized multiquadratic results in a positive definite system for polynomials of
    `degree` ``m \geq \lceil\beta\rceil``.

  * [`GeneralizedPolyharmonic`](@ref) spline

    ```math
    ϕ(r) = 
    \begin{cases}
        \begin{align*}
            &r^k                    &   k = 1, 3, 5, ... \\
            &r^k \mathrm{ln}(r)     &   k = 2, 4, 6, ...
        \end{align*}
    \end{cases}
    ```
    The generalized polyharmonic spline results in a positive definite system for
    polynomials of `degree`
    ```math
    \begin{cases}
        \begin{align*}
            m &\geq \left\lceil\frac{\beta}{2}\right\rceil          &   k = 1, 3, 5, ... \\
            m &= \beta + 1                                          &   k = 2, 4, 6, ...
        \end{align*}
    \end{cases}
    ```

  * [`Wendland`](@ref)

    ```math
    ϕ(r) = \max(1 - εr,\, 0)^{α(v, m)} \, f_{v,m}(εr)
    ```
    A compactly supported basis function: ``ϕ(r) = 0`` for ``r \geq 1/ε``. The kernel
    is positive definite for data of dimension up to `dim` and is ``2v`` times
    continuously differentiable, where ``v`` = `degree` ``\in \{0, 1, 2, 3\}`` and
    ``f_{v,m}`` is a polynomial of degree ``v`` (provided by
    `KernelFunctions.PiecewisePolynomialKernel`). For `dim` ``\in \{2, 3\}`` and
    `degree = 1` this is the classic C² function ``ϕ(r) = (1 - εr)_+^4 (4εr + 1)``.

## Radial Basis Function Partition of Unity

For large datasets, solving the single dense system of the plain radial basis function
method becomes infeasible. The partition of unity method ([`PartitionOfUnity`](@ref))
covers the bounding box of the data with a regular grid of overlapping spherical
patches, solves a small independent RBF interpolant on the points of each patch, and
blends the local interpolants:

```math
u(\mathbf{x}) = \displaystyle \sum_{j = 1}^{P} \bar{w}_j(\mathbf{x}) \, s_j(\mathbf{x}),
\qquad
\bar{w}_j(\mathbf{x}) =
    \frac{ψ\!\left(||\mathbf{x} - \mathbf{c}_j|| / ρ\right)}
         {\sum_{k} ψ\!\left(||\mathbf{x} - \mathbf{c}_k|| / ρ\right)}
```

where ``s_j`` is the local RBF interpolant of patch ``j`` with center ``\mathbf{c}_j``,
``ρ`` is the patch radius, and ``ψ`` is a compactly supported weight function (by
default the C² Wendland function of the data dimension, making the blend C²
continuous). Since each local interpolant reproduces the data of its patch and the
weights sum to one, the blended interpolant is still exact at the data points.

The number of patches is chosen so each patch holds `pointsperpatch` points on
average, and the patch radius is `overlap` times the grid cell half-diagonal
(`overlap > 1` guarantees full coverage). Patches near the boundary of the data that
would end up with only a few points are topped up with their nearest data points, so
every local system is well determined. Construction and evaluation are threaded;
start Julia with multiple threads to benefit.

Because each patch spans only a small part of the domain, basis functions with a
fixed shape parameter behave differently than in the global method: a kernel like
`Gaussian(ε)` that is well conditioned globally is nearly flat across a small patch,
which degrades accuracy at large point counts. Either scale ``ε`` to the patch size,
or — simpler and often best for large datasets — use a scale-free polyharmonic
spline with polynomial augmentation such as `GeneralizedPolyharmonic(3, 1)` as the
local method.

The `smooth` and `linsolve` keywords of `interpolate` are forwarded to every local
solve, so per-patch ridge-regression smoothing works exactly as in the global method.
Only the `Euclidean` metric is supported.

Evaluation points outside the bounding box of the data (or in interior regions that
contained no data) are extrapolated using the nearest patch's local interpolant.

New points can be added to an existing interpolant with [`addpoints!`](@ref), which
re-solves only the affected patches. The patch grid is fixed at construction, so the
new points must lie within the original data's bounding box.

To use partition of unity interpolation, pass a [`PartitionOfUnity`](@ref) object
wrapping any radial basis function to `interpolate`:

```julia
itp = interpolate(PartitionOfUnity(GeneralizedPolyharmonic(3, 1)), points, samples)
evaluate(itp, x)
addpoints!(itp, newpoints, newsamples)
```

## Inverse Distance Weighting
Also called Shepard interpolation, the basic version computes the interpolated value at
some point ``\mathbf{x}`` by
```math
u(\mathbf{x}) = 
\begin{cases} 
    \frac{\displaystyle \sum_{i = 1}^{N}{ w_i(\mathbf{x}) u_i } } 
        { \displaystyle \sum_{i = 1}^{N}{ w_i(\mathbf{x}) } }, 
         & \text{if } ||\mathbf{x} - \mathbf{x_i}|| \neq 0 \text{ for all } i \\ 
    u_i, & \text{if } ||\mathbf{x} - \mathbf{x_i}|| = 0 \text{ for some } i
\end{cases}
```
where ``||\mathbf{x} - \mathbf{x_i}||`` is the distance between ``\mathbf{x}`` and 
``\mathbf{x}_i``, and ``w_i(\mathbf{x}) = \frac{1}{||\mathbf{x} - \mathbf{x_i}||^P}``.

This model is selected by passing a [`Shepard`](@ref) object to `interpolate`.

## Nearest Neighbor
Nearest neighbor interpolation produces piecewise constant interpolations by returning the 
data value of the nearest sample point.

To use nearest neighbor interpolation, pass a [`NearestNeighbor`](@ref) object to 
`interpolate`.

