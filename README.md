# tidu-rs

AD graph transforms for the tensor4all v2 stack.

Provides:
- `linearize` / `try_linearize` — JVP transform (resolved view → linear graph)
- `linear_transpose` / `try_linear_transpose` — reverse linear flow (linear graph → linear graph)
- `try_linear_transpose_with_builder` — helper for executing transposed linear graphs through a caller-provided builder
- `tidu::eager` — generic eager reverse-mode recording for `Primitive` frontends

Use the `try_*` APIs when downstream primitive sets can report unsupported
extension AD rules through `tidu::ADRuleError`.

Fully generic over `Op: Primitive`. References no specific primitives.

## Eager AD

`tidu::eager` provides a small generic recording layer for PyTorch-style eager
reverse-mode AD. Downstream frontends execute concrete primal operations, then
use a `Recorder` with input trace metadata and concrete output values. `tidu`
allocates stable input aliases and output keys, saves replay data, and shares
one opaque `Trace` across multi-output operations.

Downstream crates still own backend execution, tensor metadata, gradient slots,
and `BackwardExecutor` for concrete forward replay, transpose execution,
and cotangent addition.

## Complex number convention (JAX-compatible)

Forward and reverse modes follow the JAX convention:

- **`linearize` (JVP)** computes the full R-linear derivative:
  `df = (∂f/∂z)·dz + (∂f/∂z̄)·conj(dz)`
- **`linear_transpose` (VJP)** computes the adjoint w.r.t. the real inner product
  `⟨a, b⟩ = Re(conj(a)·b)`.

For a general function f: C → C, the VJP cotangent relates to Wirtinger
derivatives as:

```text
ct_z = ct_y · conj(∂f/∂z) + conj(ct_y) · (∂f/∂z̄)
```

Special cases:

| Case | Result |
|------|--------|
| Real loss (L: C→R), ct_y=1 | `ct_z = 2·(∂L/∂z̄)` |
| Holomorphic f, ∂f/∂z̄=0 | `ct_z = ct_y · conj(f'(z))` |
| conj(z), ∂f/∂z=0 | `ct_z = conj(ct_y)` |

For real-valued losses, this differs from PyTorch (which returns `∂L/∂z̄`
directly) by a factor of 2. The steepest-descent direction is the same.

## Higher-order AD

Higher-order derivatives are computed by repeated application of
`linearize` (and optionally `linear_transpose`). Each `linearize` call
consumes one tangent vector, so the output shape stays the same as the
original function output regardless of derivative order.

For f: R^n → R^m:

| Order | Computation | New tangent | Output shape |
|-------|-------------|-------------|-------------|
| 0 | f(x) | — | R^m |
| 1st (F) | J · dx₁ | dx₁ ∈ R^n | R^m |
| 2nd (FoF) | (∂J/∂x · dx₂) · dx₁ | dx₂ ∈ R^n | R^m |
| 3rd (FoFoF) | ∂³f/∂x³ · dx₁ · dx₂ · dx₃ | dx₃ ∈ R^n | R^m |

Each JVP contracts one tangent vector with the derivative tensor. The
k-th forward derivative is a rank-(k+1) tensor contracted with k tangent
vectors, producing a result in R^m.

To extract individual tensor components, use unit basis vectors as tangents:
`∂³f/∂xᵢ∂xⱼ∂xₖ = FoFoF(eᵢ, eⱼ, eₖ)`.

### Typical pipelines

```text
FoF:   build → resolve → linearize → resolve → linearize → materialize → compile → eval
FoFoF: build → (resolve → linearize) × 3 → materialize → compile → eval
FoR:   build → resolve → linearize → linear_transpose → resolve → linearize → materialize → compile → eval
```

Each `linearize` call requires a unique `DiffPassId` and a preceding
`resolve` to make earlier fragments traceable.

## Part of the tensor4all v2 stack

```text
computegraph-rs    graph engine
tidu-rs            Primitive trait, linearize, linear_transpose
tenferro-rs        concrete primitives
```
