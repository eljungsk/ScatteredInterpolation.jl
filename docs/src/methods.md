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

Instead of picking one fixed `ε` for the whole domain, `PartitionOfUnity(method;
tune = :loocv)` chooses each patch's shape parameter automatically by exact
leave-one-out cross-validation (Rippa's method): for every patch, a handful of
candidate `ε` values scaled to that patch's point spacing are scored by how well
each one predicts a held-out point, and the best-scoring, numerically trustworthy
candidate is kept (falling back to the kernel's own `ε` when nothing scores better).
This helps when a single fixed `ε` is well-scaled in some parts of the domain but not
others — for example, data whose local length scale (feature width, oscillation
frequency) varies across the domain. It does not help, and mostly just costs time,
when a well-chosen fixed `ε` is already close to optimal everywhere, since no local
retuning can beat an optimum that is already global.

**This comes at a real, measured performance cost.** Rippa's formula needs the full
diagonal of the local system's inverse, which costs as much as factorizing the patch
from scratch, and several candidates are evaluated per patch — so `tune = :loocv`
builds are substantially slower than `tune = :none`, not a small constant-factor
overhead. Measured on a target-scale benchmark (10⁵ points, 3D, on data with a
genuinely mis-scaled fixed kernel): build time increased by roughly 150× while
off-node accuracy improved by roughly 19× (comfortably better than the mis-scaled
fixed kernel, and mis-scaled data is precisely the case tuning is for). On smooth
data without spatially varying structure, tuning was measured to add the same
order-of-magnitude build cost for **no** accuracy benefit — a fixed kernel already
near its own optimum cannot be reliably beaten by per-patch retuning, so `:loocv` is
a targeted tool for data known (or suspected) to need spatially varying `ε`, not a
default-on accuracy upgrade.

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

To let each patch pick its own shape parameter automatically instead (see the
performance note above before enabling this by default):

```julia
itp = interpolate(PartitionOfUnity(Gaussian(2); tune = :loocv), points, samples)
```

### Choosing a linear solver

Fitting the RBF weights requires solving a linear system ``\mathbf{A}\mathbf{w} =
\mathbf{s}`` (or, for `GeneralizedRadialBasisFunction`, the blocked system that also
determines the polynomial coefficients ``\mathbf{\lambda}``). This solve is performed
through [LinearSolve.jl](https://docs.sciml.ai/LinearSolve/stable/), and the algorithm
can be selected with the `linsolve` keyword to `interpolate`:

```julia
using LinearSolve

itp = interpolate(rbf, points, samples; linsolve = LUFactorization())
```

`linsolve = nothing` (the default) lets LinearSolve.jl pick an algorithm based on the
matrix type. Any concrete algorithm from LinearSolve.jl's
[solver list](https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/) can be passed,
e.g. `QRFactorization()` for a more numerically robust (but slower) solve of an
ill-conditioned system, or `KrylovJL_GMRES()` for large, sparse, or matrix-free problems.

The chosen algorithm is reused across every right-hand side needed for a given
`interpolate` call (multiple sample columns, and, for generalized RBFs, the polynomial
and Schur-complement solves), so the factorization is only computed once per matrix.

## Compactly supported RBFs and the sparse path

Every basis function above is evaluated at every pairwise distance, so the RBF
interpolation matrix is dense in general. [`Wendland`](@ref) is the exception: it is a
*compactly supported* kernel, identically zero beyond its support radius, and
`interpolate` can exploit that to build and factorize a sparse matrix instead of a
dense one.

| Kernel | Support | `sparse` eligible |
|--------|---------|-------------------|
| [`Wendland`](@ref) | Compact: zero for ``r > 1/\varepsilon`` | Yes |
| [`Gaussian`](@ref), [`Multiquadratic`](@ref), [`InverseQuadratic`](@ref), [`InverseMultiquadratic`](@ref), [`Polyharmonic`](@ref)/[`ThinPlate`](@ref), [`GeneralizedMultiquadratic`](@ref), [`GeneralizedPolyharmonic`](@ref) | Global: nonzero at all distances | No |

A globally supported kernel is nonzero at every distance — its interpolation matrix
has no zero entries, so `sparse` is mathematically meaningless for it, and
`sparse = true` throws an error. For local behavior with globally supported kernels,
use [`PartitionOfUnity`](@ref) instead.

### Support radius and `ε`

A Wendland kernel's shape parameter `ε` sets its support radius directly:
```math
ϕ(r) = 0 \quad \text{for} \quad r > 1/\varepsilon
```
so a larger `ε` means a smaller support radius, and vice versa.

`ε` can also be left unset on construction (`Wendland(dim, degree)`, with no `ε`).
In that case `interpolate` calibrates it from the training data: `1/ε` is set to the
median distance to the `neighbors`-th nearest neighbor, sampled over a subset of the
training points. Passing `ε` explicitly always takes precedence, in which case
`neighbors` is ignored.

This `neighbors` resolves something different from SciPy's `neighbors` keyword.
SciPy's `RBFInterpolator(neighbors=k)` solves a small local system per query point —
an approximation to the full interpolant. Here, `neighbors` only calibrates the
cutoff radius of the kernel used in one exact global system; it never changes what
is being solved. For the local-system approach analogous to SciPy's `neighbors`, use
[`PartitionOfUnity`](@ref).

### Sparsity, accuracy and conditioning

A small support radius (large `ε`) produces a sparser, faster-to-factorize system,
but sacrifices accuracy: data points outside each other's support do not interact at
all, so the interpolant can only capture behavior at the scale of `1/ε`. A large
support (small `ε`) gives accuracy approaching a dense global solve but loses the
sparsity benefit — and, as with any RBF, a support radius (or `ε`) chosen badly
relative to the point spacing can also leave the system ill-conditioned. For very
large datasets where even a sparse global system is too expensive,
[`PartitionOfUnity`](@ref) remains the primary method.

Query points that fall outside the support of every training point evaluate to
exactly zero. This is the mathematically correct value of the interpolant there, not
an approximation or a missing-data placeholder.

### Choosing `sparse` and the linear solver

`sparse` (default `:auto`) controls whether `interpolate` takes the sparse path:
`:auto` uses it when the problem is large and the resulting matrix sufficiently
sparse, `true` forces it (throwing if the kernel or metric makes sparsity
impossible, e.g. a globally supported kernel), and `false` always uses the dense
path.

The default solver for the sparse path is `CHOLMODFactorization()` from
[LinearSolve.jl](https://docs.sciml.ai/LinearSolve/stable/), which exploits the
symmetric positive definite structure of the Wendland interpolation matrix. As with
the dense path, this can be overridden with the `linsolve` keyword, e.g.
`linsolve = KrylovJL_CG()` for a matrix-free iterative solve on very large problems:

```julia
using LinearSolve

itp = interpolate(Wendland(3, 1), points, samples; linsolve = KrylovJL_CG())
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

