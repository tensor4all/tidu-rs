# Tidu Rustdoc Examples Refresh Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `tenferro`-centric public rustdoc examples with runnable, generic `tidu` examples that cover scalar reverse-mode, scalar forward-mode, scalar HVP, and one custom downstream value type example.

**Architecture:** Keep `tidu` responsible for execution and graph traversal, and use `chainrules` only where the examples need reusable scalar `frule`/`rrule` helpers. Make the crate-level docs the primary learning path, then align type-level docs with the same examples so the public rustdoc surface tells one consistent story.

**Tech Stack:** Rust 2021, rustdoc doctests, `tidu`, `chainrules`, Cargo workspace manifests, `cargo test --doc`, `cargo doc`, docs-site validation scripts.

---

### Task 1: Add a regression guard for public docs drift

**Files:**
- Create: `crates/tidu/tests/public_rustdoc_examples.rs`
- Test: `crates/tidu/tests/public_rustdoc_examples.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn public_rustdoc_examples_no_longer_reference_tenferro() {
    let tracked = include_str!("../src/engine/tracked.rs");
    let tape = include_str!("../src/engine/tape.rs");
    let results = include_str!("../src/engine/results.rs");
    let lib = include_str!("../src/lib.rs");

    for text in [tracked, tape, results, lib] {
        assert!(
            !text.contains("tenferro_"),
            "public rustdoc should not use tenferro-specific examples"
        );
    }

    assert!(lib.contains("Scalar Reverse Mode"));
    assert!(lib.contains("Scalar Forward Mode"));
    assert!(lib.contains("Scalar Hessian-Vector Product"));
    assert!(lib.contains("Custom Value Type"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test public_rustdoc_examples`

Expected: FAIL because the current public docs still contain `tenferro_*` and
the new section headings do not exist yet.

**Step 3: Write minimal implementation**

Create `crates/tidu/tests/public_rustdoc_examples.rs` with the exact test above.

**Step 4: Run test to verify it passes later**

Run: `cargo nextest run --release -p tidu --test public_rustdoc_examples`

Expected: still FAIL until the rustdoc updates land in Tasks 2-4.

**Step 5: Commit**

```bash
git add crates/tidu/tests/public_rustdoc_examples.rs
git commit -m "test: guard public rustdoc examples"
```

### Task 2: Wire `chainrules` for runnable rustdoc examples

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/tidu/Cargo.toml`

**Step 1: Write the failing doc import change**

Add a runnable crate-level snippet in `crates/tidu/src/lib.rs` that imports
`chainrules::powf_frule` or `chainrules::powf_rrule` before the dependency is
wired.

Example snippet to introduce:

```rust
use chainrules::powf_frule;

let (y, dy) = powf_frule(2.0_f64, 3.0, 1.0);
assert_eq!(y, 8.0);
assert_eq!(dy, 12.0);
```

**Step 2: Run doctests to verify they fail**

Run: `cargo test --doc --release -p tidu`

Expected: FAIL with an unresolved import for `chainrules` or equivalent missing
dependency error.

**Step 3: Write minimal implementation**

Update the manifests so doctests can use `chainrules` through normal Cargo
resolution:

```toml
# Cargo.toml
[workspace.dependencies]
chainrules-core = { git = "https://github.com/tensor4all/chainrules-rs", rev = "b1a17ae458bfd9f8976a12d7302c6abd3db31048" }
chainrules = { git = "https://github.com/tensor4all/chainrules-rs", rev = "b1a17ae458bfd9f8976a12d7302c6abd3db31048" }
```

```toml
# crates/tidu/Cargo.toml
[dependencies]
chainrules-core.workspace = true
chainrules.workspace = true
```

**Step 4: Run doctests to verify dependency wiring works**

Run: `cargo test --doc --release -p tidu`

Expected: the unresolved import failure disappears; some doc examples may still
fail until Tasks 3-4 complete.

**Step 5: Commit**

```bash
git add Cargo.toml crates/tidu/Cargo.toml
git commit -m "build: add chainrules for rustdoc examples"
```

### Task 3: Rewrite crate-level docs into the new four-section layout

**Files:**
- Modify: `crates/tidu/src/lib.rs`

**Step 1: Write the failing crate-level docs update**

Replace the old crate docs with a table of contents and these headings:

```rust
//! ## Table of Contents
//! - [Scalar Reverse Mode](#scalar-reverse-mode)
//! - [Scalar Forward Mode](#scalar-forward-mode)
//! - [Scalar Hessian-Vector Product](#scalar-hessian-vector-product)
//! - [Custom Value Type](#custom-value-type)
```

Use these example shapes:

```rust
//! ## Scalar Reverse Mode
//! ```rust
//! use chainrules::powf_rrule;
//! use tidu::{AdResult, NodeId, ReverseRule, Tape};
//!
//! struct PowfRule {
//!     input: NodeId,
//!     x: f64,
//!     exponent: f64,
//! }
//!
//! impl ReverseRule<f64> for PowfRule {
//!     fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
//!         Ok(vec![(self.input, powf_rrule(self.x, self.exponent, *cotangent))])
//!     }
//!
//!     fn inputs(&self) -> Vec<NodeId> {
//!         vec![self.input]
//!     }
//! }
//!
//! let tape = Tape::<f64>::new();
//! let x = tape.leaf(2.0);
//! let y = tape.record_op(
//!     8.0,
//!     Box::new(PowfRule {
//!         input: x.node_id().unwrap(),
//!         x: 2.0,
//!         exponent: 3.0,
//!     }),
//!     None,
//! );
//! let grads = tape.pullback(&y).unwrap();
//! assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 12.0);
//! ```
```

```rust
//! ## Scalar Forward Mode
//! ```rust
//! use chainrules::powf_frule;
//! use tidu::DualValue;
//!
//! let x = DualValue::with_tangent(2.0_f64, 1.0_f64).unwrap();
//! let (y, dy) = powf_frule(*x.primal(), 3.0, *x.tangent().unwrap());
//! assert_eq!(y, 8.0);
//! assert_eq!(dy, 12.0);
//! ```
```

```rust
//! ## Scalar Hessian-Vector Product
//! ```rust
//! use tidu::{AdResult, NodeId, ReverseRule, Tape};
//!
//! struct SquareRuleHvp {
//!     input: NodeId,
//!     x: f64,
//!     dx: f64,
//! }
//!
//! impl ReverseRule<f64> for SquareRuleHvp {
//!     fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
//!         Ok(vec![(self.input, 2.0 * self.x * *cotangent)])
//!     }
//!
//!     fn inputs(&self) -> Vec<NodeId> {
//!         vec![self.input]
//!     }
//!
//!     fn pullback_with_tangents(
//!         &self,
//!         cotangent: &f64,
//!         cotangent_tangent: &f64,
//!     ) -> AdResult<Vec<(NodeId, f64, f64)>> {
//!         Ok(vec![(
//!             self.input,
//!             2.0 * self.x * *cotangent,
//!             2.0 * self.dx * *cotangent + 2.0 * self.x * *cotangent_tangent,
//!         )])
//!     }
//! }
//!
//! let tape = Tape::<f64>::new();
//! let x = tape.leaf_with_tangent(3.0, 1.0).unwrap();
//! let y = tape.record_op(
//!     9.0,
//!     Box::new(SquareRuleHvp {
//!         input: x.node_id().unwrap(),
//!         x: 3.0,
//!         dx: 1.0,
//!     }),
//!     None,
//! );
//! let hvp = tape.hvp(&y).unwrap();
//! assert_eq!(*hvp.gradients.get(x.node_id().unwrap()).unwrap(), 6.0);
//! assert_eq!(*hvp.hvp.get(x.node_id().unwrap()).unwrap(), 2.0);
//! ```
```

```rust
//! ## Custom Value Type
//! ```rust
//! use tidu::{AdResult, Differentiable, NodeId, ReverseRule, Tape};
//!
//! #[derive(Clone, Copy, Debug, PartialEq)]
//! struct Vec2([f64; 2]);
//!
//! impl Differentiable for Vec2 {
//!     type Tangent = Self;
//!
//!     fn zero_tangent(&self) -> Self::Tangent {
//!         Self([0.0, 0.0])
//!     }
//!
//!     fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
//!         Self([a.0[0] + b.0[0], a.0[1] + b.0[1]])
//!     }
//!
//!     fn num_elements(&self) -> usize {
//!         2
//!     }
//!
//!     fn seed_cotangent(&self) -> Self::Tangent {
//!         Self([1.0, 1.0])
//!     }
//! }
//!
//! struct ScaleByTwoRule {
//!     input: NodeId,
//! }
//!
//! impl ReverseRule<Vec2> for ScaleByTwoRule {
//!     fn pullback(&self, cotangent: &Vec2) -> AdResult<Vec<(NodeId, Vec2)>> {
//!         Ok(vec![(
//!             self.input,
//!             Vec2([2.0 * cotangent.0[0], 2.0 * cotangent.0[1]]),
//!         )])
//!     }
//!
//!     fn inputs(&self) -> Vec<NodeId> {
//!         vec![self.input]
//!     }
//! }
//!
//! let tape = Tape::<Vec2>::new();
//! let x = tape.leaf(Vec2([3.0, -1.0]));
//! let y = tape.record_op(6.0.into(), Box::new(ScaleByTwoRule { input: x.node_id().unwrap() }), None);
//! ```
```

Do not keep the malformed `6.0.into()` line above in the final code. Replace it
with a valid `Vec2([6.0, -2.0])` output and complete the example with
`pullback_with_seed`.

**Step 2: Run doctests to verify they fail for the right reasons**

Run: `cargo test --doc --release -p tidu`

Expected: FAIL on incomplete or not-yet-updated example blocks in `lib.rs`.

**Step 3: Write minimal implementation**

Finish the crate docs so the final custom example is:

```rust
//! let y = tape.record_op(
//!     Vec2([6.0, -2.0]),
//!     Box::new(ScaleByTwoRule {
//!         input: x.node_id().unwrap(),
//!     }),
//!     None,
//! );
//! let grads = tape.pullback_with_seed(&y, Vec2([1.0, -1.0])).unwrap();
//! assert_eq!(
//!     *grads.get(x.node_id().unwrap()).unwrap(),
//!     Vec2([2.0, -2.0]),
//! );
//! ```
```

Also add a short prose explanation above the HVP section that this example
implements a `tidu`-specific HVP-aware reverse rule because `chainrules`
provides scalar `frule`/`rrule` helpers rather than a ready-made `ReverseRule`
object with `pullback_with_tangents`.

**Step 4: Run doctests to verify the crate-level page passes**

Run: `cargo test --doc --release -p tidu`

Expected: the crate-level docs compile and pass; item-level examples may still
need updates in Task 4.

**Step 5: Commit**

```bash
git add crates/tidu/src/lib.rs
git commit -m "docs: rewrite crate-level tidu examples"
```

### Task 4: Align type-level docs with the crate-level story

**Files:**
- Modify: `crates/tidu/src/engine/tape.rs`
- Modify: `crates/tidu/src/engine/tracked.rs`
- Modify: `crates/tidu/src/engine/results.rs`

**Step 1: Write the failing docs update**

Replace the old type-level examples with these smaller examples:

For `Tape`:

```rust
/// ```rust
/// use chainrules::powf_rrule;
/// use tidu::{AdResult, NodeId, ReverseRule, Tape};
///
/// struct PowfRule {
///     input: NodeId,
///     x: f64,
/// }
///
/// impl ReverseRule<f64> for PowfRule {
///     fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
///         Ok(vec![(self.input, powf_rrule(self.x, 3.0, *cotangent))])
///     }
///
///     fn inputs(&self) -> Vec<NodeId> {
///         vec![self.input]
///     }
/// }
///
/// let tape = Tape::<f64>::new();
/// let x = tape.leaf(2.0);
/// let y = tape.record_op(
///     8.0,
///     Box::new(PowfRule {
///         input: x.node_id().unwrap(),
///         x: 2.0,
///     }),
///     None,
/// );
/// let grads = tape.pullback(&y).unwrap();
/// assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 12.0);
/// ```
```

For `TrackedValue`:

```rust
/// ```rust
/// use tidu::Tape;
///
/// let tape = Tape::<f64>::new();
/// let x = tape.leaf(3.0);
/// assert!(x.requires_grad());
/// assert_eq!(*x.value(), 3.0);
/// ```
```

For `HvpResult`:

```rust
/// ```rust
/// use tidu::{AdResult, HvpResult, NodeId, ReverseRule, Tape};
///
/// struct SquareRuleHvp {
///     input: NodeId,
///     x: f64,
///     dx: f64,
/// }
///
/// impl ReverseRule<f64> for SquareRuleHvp {
///     fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
///         Ok(vec![(self.input, 2.0 * self.x * *cotangent)])
///     }
///
///     fn inputs(&self) -> Vec<NodeId> {
///         vec![self.input]
///     }
///
///     fn pullback_with_tangents(
///         &self,
///         cotangent: &f64,
///         cotangent_tangent: &f64,
///     ) -> AdResult<Vec<(NodeId, f64, f64)>> {
///         Ok(vec![(
///             self.input,
///             2.0 * self.x * *cotangent,
///             2.0 * self.dx * *cotangent + 2.0 * self.x * *cotangent_tangent,
///         )])
///     }
/// }
///
/// let tape = Tape::<f64>::new();
/// let x = tape.leaf_with_tangent(3.0, 1.0).unwrap();
/// let y = tape.record_op(
///     9.0,
///     Box::new(SquareRuleHvp {
///         input: x.node_id().unwrap(),
///         x: 3.0,
///         dx: 1.0,
///     }),
///     None,
/// );
/// let result: HvpResult<f64> = tape.hvp(&y).unwrap();
/// assert_eq!(*result.gradients.get(x.node_id().unwrap()).unwrap(), 6.0);
/// assert_eq!(*result.hvp.get(x.node_id().unwrap()).unwrap(), 2.0);
/// ```
```

**Step 2: Run doctests to verify they fail until all replacements are complete**

Run: `cargo test --doc --release -p tidu`

Expected: FAIL while some type-level examples still reference `tenferro_*` or
contain outdated `ignore` examples.

**Step 3: Write minimal implementation**

Update the three files so all public type-level examples align with the crate
story and use no `tenferro_*` imports.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo nextest run --release -p tidu --test public_rustdoc_examples
cargo test --doc --release -p tidu
```

Expected: both commands PASS.

**Step 5: Commit**

```bash
git add crates/tidu/src/engine/tape.rs crates/tidu/src/engine/tracked.rs crates/tidu/src/engine/results.rs
git commit -m "docs: align type-level rustdoc examples"
```

### Task 5: Run full verification and prepare the branch for review

**Files:**
- Modify: none
- Verify: workspace manifests, rustdoc output, docs-site artifacts

**Step 1: Run formatting**

Run: `cargo fmt --all`

Expected: command succeeds with no diff afterward.

**Step 2: Run lint and tests**

Run:

```bash
cargo clippy --workspace
cargo nextest run --release --workspace --no-fail-fast
cargo test --doc --release --workspace
```

Expected: all commands PASS.

**Step 3: Run documentation verification**

Run:

```bash
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
bash scripts/build_docs_site.sh
```

Expected: rustdoc builds, docs-site validation passes, and the generated API
landing page contains the workspace crate link(s).

**Step 4: Inspect for related doc drift**

Run: `rg -n "tenferro_" crates/tidu/src README.md docs -S`

Expected: no unexpected public-doc references remain. If new hits appear in
nearby rustdoc, fix them before opening a PR.

**Step 5: Commit**

```bash
git add -A
git commit -m "docs: refresh public tidu rustdoc examples"
```
