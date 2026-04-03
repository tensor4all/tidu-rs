# tidu-rs

AD graph transforms for the tensor4all v2 stack.

Provides:
- `differentiate` — JVP transform (resolved view → linear fragment)
- `transpose` — reverse linear flow (linear fragment → linear fragment)

Fully generic over `Op: PrimitiveOp`. References no specific primitives.

## Part of the tensor4all v2 stack

```text
computegraph-rs    graph engine
chainrules-rs      PrimitiveOp trait
tidu-rs        <-- this crate (differentiate, transpose)
tenferro-rs        concrete primitives
```
