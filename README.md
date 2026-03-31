# tidu-rs

`tidu-rs` is a general-purpose automatic-differentiation engine with a
value-centered, linearize-first public API.

It originated in the tensor4all stack, but it is designed to work with any
downstream differentiable value type that implements the core AD traits from
[`chainrules-rs`](https://github.com/tensor4all/chainrules-rs).

The name **tidu** comes from the Chinese word **梯度**, written in pinyin as
**tī dù**, meaning "gradient".

## Getting Started

Add `tidu` and its companion crate `chainrules` (which provides scalar
differentiation rules such as `powf_rrule` and `sin_rrule`) to your
`Cargo.toml`:

```toml
[dependencies]
tidu       = { git = "https://github.com/tensor4all/tidu-rs" }
chainrules = { git = "https://github.com/tensor4all/chainrules-rs" }
```

`tidu` re-exports the core AD traits needed by the normal public surface,
including `Differentiable`, `AdResult`, and `AutodiffError`. The intended
public extension points are `Value`, `LinearizableOp`, `LinearizedOp`,
`Schema`, `SlotSchema`, `CheckpointMode`, `AdExecutionPolicy`,
`CheckpointHint`, and `with_ad_policy(...)`.

## Quick Example

Compute the gradient of `f(x) = x^3` at `x = 2` and inspect a local
directional derivative from the same linearized object:

```rust
use tidu::{LinearizableOp, LinearizedOp, Schema, SlotSchema, Value};

#[derive(Clone, Copy)]
struct Cube;

struct CubeLinearized {
    x: f64,
}

impl LinearizedOp<f64> for CubeLinearized {
    fn jvp(&self, input_tangents: &[Option<f64>]) -> tidu::AdResult<Vec<Option<f64>>> {
        Ok(vec![input_tangents[0].map(|dx| 3.0 * self.x * self.x * dx)])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<f64>],
        input_grad_mask: &[bool],
    ) -> tidu::AdResult<Vec<Option<f64>>> {
        assert_eq!(input_grad_mask, &[true]);
        let g = output_cotangents[0].unwrap_or(0.0);
        Ok(vec![Some(3.0 * self.x * self.x * g)])
    }
}

impl LinearizableOp<f64> for Cube {
    type Linearized = CubeLinearized;

    fn primal(&self, inputs: &[&f64]) -> tidu::AdResult<Vec<f64>> {
        Ok(vec![*inputs[0] * *inputs[0] * *inputs[0]])
    }

    fn input_schema(&self, _inputs: &[&f64]) -> tidu::AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn linearize(
        &self,
        inputs: &[&f64],
        _outputs: &[f64],
    ) -> tidu::AdResult<Self::Linearized> {
        Ok(CubeLinearized { x: *inputs[0] })
    }
}

let x = Value::new(2.0).requires_grad_(true);
let y = Cube.apply_one(&[&x]).unwrap();
y.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap(), 12.0);

let lin = Cube.linearize(&[x.primal()], &[*y.primal()]).unwrap();
assert_eq!(lin.jvp(&[Some(1.0)]).unwrap(), vec![Some(12.0)]);
```

Checkpointing is controlled with a small public policy scope:

```rust
use tidu::{AdExecutionPolicy, CheckpointMode, with_ad_policy};

let policy = AdExecutionPolicy {
    checkpoint_mode: CheckpointMode::Conservative,
};

with_ad_policy(policy, || -> tidu::AdResult<()> {
    // Record and differentiate values inside this scope.
    Ok(())
})
.unwrap();
```

`CheckpointHint` is an advanced retain-vs-replay hint for custom ops. Most
downstream code only needs `CheckpointMode`, `AdExecutionPolicy`, and
`with_ad_policy(...)`.

See the [crate-level rustdoc](https://tensor4all.org/tidu-rs/tidu/) for
`Value`, `LinearizableOp`, `LinearizedOp`, and checkpoint policy examples.

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                      Value<V>                       │
│  Public value handle for eager reverse-mode AD.     │
│  Exposes requires_grad_, backward, and grad().      │
├─────────────────────────────────────────────────────┤
│                LinearizableOp<V>                    │
│  High-level custom op API: primal + linearize.      │
│  The normal extension path for downstream users.    │
├─────────────────────────────────────────────────────┤
│                 LinearizedOp<V>                     │
│  Shared first-order object exposing jvp + vjp.      │
│  Retained or replayed internally by the runtime.    │
├─────────────────────────────────────────────────────┤
│      CheckpointMode / AdExecutionPolicy scope       │
│  Small public policy surface over retain/replay.    │
└─────────────────────────────────────────────────────┘
```

## What Lives Here

- `tidu`: eager reverse mode with a linearize-first core
- `Value`, `LinearizableOp`, and `LinearizedOp`
- checkpoint policy scope via `CheckpointMode`, `AdExecutionPolicy`, and `with_ad_policy(...)`
- retained or replayed linearizations kept internal to the runtime

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
- Keep the normal public API torch-like and value-centered
- Make future forward-on-reverse straightforward without exposing higher-order execution now
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
