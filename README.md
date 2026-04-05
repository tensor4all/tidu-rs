# tidu-rs

AD graph transforms for the tensor4all v2 stack.

Provides:
- `differentiate` — JVP transform (resolved view → linear fragment)
- `transpose` — reverse linear flow (linear fragment → linear fragment)

Fully generic over `Op: PrimitiveOp`. References no specific primitives.

## Complex number convention (JAX-compatible)

Forward and reverse modes follow the JAX convention:

- **`differentiate` (JVP)** computes the full R-linear derivative:
  `df = (∂f/∂z)·dz + (∂f/∂z̄)·conj(dz)`
- **`transpose` (VJP)** computes the adjoint w.r.t. the real inner product
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
`differentiate` (and optionally `transpose`). Each `differentiate` call
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
FoF:   build → resolve → differentiate → resolve → differentiate → materialize → compile → eval
FoFoF: build → (resolve → differentiate) × 3 → materialize → compile → eval
FoR:   build → resolve → differentiate → transpose → resolve → differentiate → materialize → compile → eval
```

Each `differentiate` call requires a unique `DiffPassId` and a preceding
`resolve` to make earlier fragments traceable.

## Part of the tensor4all v2 stack

```text
computegraph-rs    graph engine
chainrules-rs      PrimitiveOp trait
tidu-rs        <-- this crate (differentiate, transpose)
tenferro-rs        concrete primitives
```
