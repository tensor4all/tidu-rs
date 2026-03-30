# tidu-rs

`tidu-rs` is a general-purpose automatic-differentiation engine with a
value-centered public API.

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

`tidu` re-exports the core AD traits needed by the normal public surface,
including `Differentiable`, `AdResult`, and `AutodiffError`. Low-level graph
types and rule traits now live under `tidu::expert`, while scalar rule helpers
such as `powf_rrule` and `powf_frule` still come from `chainrules`.

## Quick Example

Compute the gradient of `f(x) = x^3` at `x = 2` using the high-level public
API:

```rust
use tidu::{Function, GradInputs, Value};

struct Cube;

impl Function<f64> for Cube {
    type Saved = f64;

    fn primal(inputs: &[&f64]) -> tidu::AdResult<f64> {
        Ok(*inputs[0] * *inputs[0] * *inputs[0])
    }

    fn save_for_backward(inputs: &[&f64], _output: &f64) -> tidu::AdResult<Self::Saved> {
        Ok(*inputs[0])
    }

    fn backward(saved: &Self::Saved, grad_out: &f64) -> tidu::AdResult<GradInputs<f64>> {
        Ok(GradInputs::from(vec![Some(3.0 * *saved * *saved * *grad_out)]))
    }
}

let x = Value::new(2.0).requires_grad_(true);
let y = Cube::apply(&[&x]).unwrap();
y.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap(), 12.0);
```

See the [crate-level rustdoc](https://tensor4all.org/tidu-rs/tidu/) for
forward-mode, HVP, custom `Function` definitions, and advanced low-level usage.

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                      Value<V>                       │
│  Public value handle for eager reverse-mode AD.     │
│  Exposes requires_grad_, backward, and grad().      │
├─────────────────────────────────────────────────────┤
│                    Function<V>                      │
│  High-level custom op API: primal/save/backward.    │
│  The normal extension path for downstream users.    │
├─────────────────────────────────────────────────────┤
│                    DualValue<V>                     │
│  Primal + tangent pair for forward-mode AD.         │
│  Independent of the reverse graph runtime.          │
├─────────────────────────────────────────────────────┤
│                    tidu::expert                     │
│  Low-level tape/rule APIs kept for advanced use.    │
└─────────────────────────────────────────────────────┘
```

## What Lives Here

- `tidu`: value-centered reverse mode and dual-number forward mode
- `Value`, `Function`, and `DualValue`
- `tidu::expert` for advanced tape/rule access
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
- Keep the normal public API torch-like and value-centered
- Preserve a separate expert path for low-level graph access
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
