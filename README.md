# tidu-rs

`tidu-rs` is a tape-based automatic-differentiation engine for the tensor4all
ecosystem.

The name **tidu** comes from the Chinese word **梯度**, written in pinyin as
**tī dù**, meaning "gradient".

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
