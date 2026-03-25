# tidu-rs

`tidu-rs` is a general-purpose, tape-based automatic-differentiation engine.

It originated in the tensor4all stack, but it is designed to work with any
downstream differentiable value type that implements the core AD traits from
[`chainrules-rs`](https://github.com/tensor4all/chainrules-rs).

The name **tidu** comes from the Chinese word **梯度**, written in pinyin as
**tī dù**, meaning "gradient".

## Getting Started

Add `tidu` and its companion crate `chainrules` (which provides scalar
differentiation rules such as `powf_rrule` and `sin_frule`) to your
`Cargo.toml`:

```toml
[dependencies]
tidu       = { git = "https://github.com/tensor4all/tidu-rs" }
chainrules = { git = "https://github.com/tensor4all/chainrules-rs" }
```

`tidu` re-exports the core traits (`Differentiable`, `ReverseRule`, `NodeId`,
etc.) from `chainrules-core`, so you only need to import `chainrules`
explicitly when you use its scalar rule helpers (e.g. `powf_rrule`,
`powf_frule`).

## Quick Example

Compute the gradient of f(x) = x³ at x = 2 using reverse-mode AD:

```rust
use chainrules::powf_rrule;
use tidu::{AdResult, NodeId, ReverseRule, Tape};

// 1. Define a reverse rule for f(x) = x^exponent.
struct PowfRule { input: NodeId, x: f64, exponent: f64 }

impl ReverseRule<f64> for PowfRule {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        Ok(vec![(self.input, powf_rrule(self.x, self.exponent, *cotangent))])
    }
    fn inputs(&self) -> Vec<NodeId> { vec![self.input] }
}

// 2. Build the computation graph.
let tape = Tape::<f64>::new();
let x = tape.leaf(2.0);
let y = tape.record_op(
    8.0,                                         // forward value: 2^3
    Box::new(PowfRule { input: x.node_id().unwrap(), x: 2.0, exponent: 3.0 }),
    None,                                        // no tangent (only for HVP)
);

// 3. Run reverse-mode pullback.
let grads = tape.pullback(&y).unwrap();
assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 12.0); // dy/dx = 3·2² = 12
```

See the [crate-level rustdoc](https://tensor4all.org/tidu-rs/tidu/) for
forward-mode, HVP, and custom-type examples.

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                       Tape<V>                       │
│  Shared, ref-counted autograd graph.                │
│  Records leaves and operations as graph nodes.      │
├─────────────────────────────────────────────────────┤
│                  TrackedValue<V>                     │
│  A value + its NodeId + a ref to the Tape.          │
│  Returned by tape.leaf() and tape.record_op().      │
├─────────────────────────────────────────────────────┤
│                   Gradients<V>                      │
│  Leaf-only gradient map returned by pullback.       │
│  Look up by NodeId: grads.get(node_id).             │
├─────────────────────────────────────────────────────┤
│                  DualValue<V>                       │
│  Primal + tangent pair for forward-mode AD.         │
│  Independent of the tape — no graph involved.       │
└─────────────────────────────────────────────────────┘

Traits (from chainrules-core, re-exported by tidu):
  Differentiable   — tangent algebra for a value type
  ReverseRule<V>   — pullback logic for one operation
```

## What Lives Here

- `tidu`: reverse-mode tape execution and dual-number forward mode
- `TrackedValue` and `DualValue`
- pullback planning, gradient extraction, and Hessian-vector-product support

## Layering

`tidu-rs` depends on the engine-independent traits in
[`chainrules-rs`](https://github.com/tensor4all/chainrules-rs), especially
`chainrules-core`.

That split is deliberate:

- `chainrules-rs` defines reusable traits and scalar rules
- `tidu-rs` provides one concrete engine runtime
- downstream tensor libraries can reuse the rules without depending on this
  specific engine

## Design Goals

- Keep the engine generic over downstream differentiable value types
- Preserve strict layering between rules and runtime execution
- Prefer root-cause fixes, DRY abstractions, and small focused modules

## Testing

```bash
cargo test --workspace --release
cargo doc --workspace --no-deps
```

## Solve-Bug Entrypoints

Use `bash ai/run-codex-solve-bug.sh` or `bash ai/run-claude-solve-bug.sh` when
you want a headless agent to pick one actionable bug or bug-like issue, fix it,
and drive the repository-local PR workflow.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT license ([LICENSE-MIT](./LICENSE-MIT))

at your option.
