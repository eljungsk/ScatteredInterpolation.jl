# Supported methods
Currently, three different interpolation methods are available; Radial Basis Functions,
Inverse Distance Weighting and Nearest Neighbor.

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
    ϕ(r) = \sqrt{1 + (ɛr)^2}
    ```

  * [`InverseMultiquadratic`](@ref)

    ```math
    ϕ(r) = \frac{1}{\sqrt{1 + (ɛr)^2}}
    ```

  * [`Gaussian`](@ref)

    ```math
    ϕ(r) = e^{-(ɛr)^2}
    ```

  * [`InverseQuadratic`](@ref)

    ```math
    ϕ(r) = \frac{1}{1 + (ɛr)^2}
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
    ϕ(r) = \left(1 + (ɛr)^2\right)^\beta
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

