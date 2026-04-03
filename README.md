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

## Part of the tensor4all v2 stack

```text
computegraph-rs    graph engine
chainrules-rs      PrimitiveOp trait
tidu-rs        <-- this crate (differentiate, transpose)
tenferro-rs        concrete primitives
```
