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

For a real-valued loss L, the VJP cotangent satisfies:

```text
ct_z = 2·(∂L/∂z̄)      (= 2·conj(∂L/∂z))
```

This differs from PyTorch, which returns `∂L/∂z̄` directly (factor of 2).
The steepest-descent direction is the same in both conventions.

## Part of the tensor4all v2 stack

```text
computegraph-rs    graph engine
chainrules-rs      PrimitiveOp trait
tidu-rs        <-- this crate (differentiate, transpose)
tenferro-rs        concrete primitives
```
