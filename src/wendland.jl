# Compactly supported Wendland radial basis functions, provided by wrapping
# KernelFunctions.PiecewisePolynomialKernel (which implements the Wendland family).

export Wendland

"""
    Wendland(dim, degree; ε = 1)

Define a compactly supported Wendland Radial Basis Function. The resulting kernel is
positive definite for data of dimension up to `dim`, is `2*degree` times continuously
differentiable, and vanishes identically for `r ≥ 1/ε`.

```math
ϕ(r) = \\max(1 - εr, 0)^{α(v, m)} \\, f_{v,m}(εr)
```

where ``v`` = `degree` ∈ {0, 1, 2, 3}, ``m`` = `dim`, and ``f_{v,m}`` is a polynomial of
degree ``v``. The kernel evaluation is provided by
`KernelFunctions.PiecewisePolynomialKernel`; see its documentation for the exact
coefficients. For `dim` ∈ {2, 3} and `degree = 1` this is the classic C² function
``ϕ(r) = (1 - εr)_+^4 (4εr + 1)``.
"""
struct Wendland{K <: KernelFunctions.PiecewisePolynomialKernel, T <: Real} <: RadialBasisFunction
    kernel::K
    ε::T
end

function Wendland(dim::Integer, degree::Integer; ε::Real = 1)
    0 <= degree <= 3 || throw(ArgumentError(
        "Wendland degree must be 0, 1, 2 or 3, got $degree"))
    dim >= 1 || throw(ArgumentError(
        "Wendland dim must be a positive integer, got $dim"))
    ε > 0 || throw(ArgumentError(
        "Wendland shape parameter ε must be positive, got $ε"))

    kernel = KernelFunctions.PiecewisePolynomialKernel(; degree = Int(degree), dim = Int(dim))
    Wendland(kernel, ε)
end

(w::Wendland)(r) = KernelFunctions.kappa(w.kernel, w.ε * r)

"""
    support_radius(rbf)

Radius beyond which `rbf(r)` is identically zero, or `Inf` for globally supported basis
functions. A finite support radius is the dispatch hook for a future sparse-assembly
interpolation path (see the RBF-PUM design spec, Future Work).
"""
support_radius(::AbstractRadialBasisFunction) = Inf
support_radius(w::Wendland) = 1 / w.ε
