# Tidu Rustdoc Examples Redesign

## Summary

The current public rustdoc examples for `tidu` overfit to the `tenferro`
stack. That makes the crate look less generic than it really is and obscures
the actual layering: `tidu` is an AD engine runtime, not a `tenferro`-specific
library.

The redesign should make three things obvious from the docs homepage:

1. how to use `tidu` for scalar reverse-mode, forward-mode, and HVP,
2. how `chainrules` scalar AD helpers fit into that workflow, and
3. how downstream crates can plug in their own differentiable value types.

## Approved Structure

The crate-level docs in `crates/tidu/src/lib.rs` will become the main entry
point. They will include a short table of contents and split `# Examples` into
four sections:

- `Scalar Reverse Mode`
- `Scalar Forward Mode`
- `Scalar Hessian-Vector Product`
- `Custom Value Type`

The first three sections will use scalar examples so a new user can understand
the three AD modes without needing any tensor library. The reverse-mode and
forward-mode scalar examples will use `chainrules` helpers such as
`chainrules::powf_rrule` and `chainrules::powf_frule` so the docs explain how
`tidu` and `chainrules` fit together.

The HVP example will remain scalar but use a small custom `ReverseRule<f64>`
implementation with `pullback_with_tangents`, because `chainrules` scalar
helpers do not directly provide a `tidu`-ready HVP rule object.

The last section will show a downstream custom value type using a small fixed
size `Vec2` example and `pullback_with_seed`, demonstrating that `tidu` does
not depend on `tenferro` and can operate over independently defined
`Differentiable` types.

## Placement

The following files will be updated:

- `crates/tidu/src/lib.rs`
  Replace the existing `tenferro_*`-based crate examples with the new
  four-section layout and top-of-page table of contents.
- `crates/tidu/src/engine/tape.rs`
  Replace the type-level `Tape` example with a minimal scalar reverse-mode
  example.
- `crates/tidu/src/engine/tracked.rs`
  Replace the type-level `TrackedValue` example with a minimal scalar example.
- `crates/tidu/src/engine/results.rs`
  Replace the `HvpResult` example with a scalar HVP example.

`DualValue` item docs already use scalar examples and can stay mostly as-is.

## Dependency Plan

If runnable doctests need direct access to `chainrules` helpers, wire the
dependency explicitly in the workspace manifests instead of relying on hidden
or local-only setup:

- add `chainrules` under `[workspace.dependencies]` in the root `Cargo.toml`
- reference it from `crates/tidu/Cargo.toml` with `chainrules.workspace = true`

This keeps the docs examples honest and reproducible in CI.

## Verification Strategy

The redesign is successful when all of the following are true:

- public rustdoc for `tidu` no longer advertises `tenferro_*` usage as the main
  example path,
- the crate-level docs clearly explain reverse-mode, forward-mode, and HVP,
- the crate-level docs show how to use `chainrules` scalar helpers with `tidu`,
- the docs still show how a downstream crate can define a custom value type,
- doctests pass where the examples are meant to be runnable.

Verification should include:

- `cargo test --doc --release --workspace`
- `cargo nextest run --release --workspace --no-fail-fast`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `bash scripts/build_docs_site.sh`

## Risks

- Adding `chainrules` as a dependency for doctests could expand the visible
  dependency surface; the manifest change should be kept explicit and minimal.
- The HVP example needs a brief explanation that it is a `tidu`-specific
  `ReverseRule` example rather than a direct wrapper around a `chainrules`
  helper.
- Similar `tenferro_*` references may exist elsewhere in public docs, so the
  implementation should search for and inspect related files rather than only
  editing the top page.
