# Edge-Based Public Autograd Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `tidu-rs`'s public tape-centered AD surface with a `torch`-like value-centered API while keeping the engine generic over arbitrary `V: Differentiable`.

**Architecture:** Keep an internal graph store, but hide `Tape` and `NodeId` from the normal public API. Introduce a `Value<V>` surface with `requires_grad_`, `backward`, `.grad()`, and value-oriented HVP, and make a high-level `Function` API the default custom-op mechanism. Move raw graph/rule APIs into `tidu::expert`.

**Tech Stack:** Rust 2021, `tidu`, `chainrules-core`, cargo workspace tests, `cargo nextest --release`, rustdoc examples, README updates.

---

### Task 1: Lock down the new public surface with failing smoke tests

**Files:**
- Create: `crates/tidu/tests/value_api_tests.rs`
- Test: `crates/tidu/tests/value_api_tests.rs`

**Step 1: Write the failing tests**

Create `crates/tidu/tests/value_api_tests.rs` with focused public-API coverage:

```rust
#[test]
fn backward_accumulates_leaf_gradients_without_explicit_tape() {
    let x = Value::new(2.0).requires_grad_(true);
    let y = square(&x)?;
    y.backward()?;
    assert_eq!(x.grad()?.unwrap(), 4.0);
}

#[test]
fn zero_grad_clears_leaf_gradient() {
    let x = Value::new(3.0).requires_grad_(true);
    let y = square(&x)?;
    y.backward()?;
    x.zero_grad()?;
    assert!(x.grad()?.is_none());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test value_api_tests`

Expected: FAIL because `Value`, `requires_grad_`, and `backward` do not exist.

**Step 3: Write minimal implementation**

Add the exact failing tests and helper functions only. Do not implement the
API yet.

**Step 4: Run test to verify the failure is the missing feature**

Run: `cargo nextest run --release -p tidu --test value_api_tests`

Expected: FAIL with missing-type or missing-method errors tied to the new
public API.

**Step 5: Commit**

```bash
git add crates/tidu/tests/value_api_tests.rs
git commit -m "test: add value API smoke coverage"
```

### Task 2: Refactor the runtime around hidden graph edges instead of public tapes

**Files:**
- Create: `crates/tidu/src/engine/graph.rs`
- Create: `crates/tidu/src/engine/node.rs`
- Modify: `crates/tidu/src/engine/context.rs`
- Modify: `crates/tidu/src/engine/mod.rs`
- Modify: `crates/tidu/src/engine/results.rs`
- Modify: `crates/tidu/src/engine/tracked.rs`
- Modify: `crates/tidu/src/engine/tape.rs`
- Test: `crates/tidu/tests/chainrules_tests.rs`

**Step 1: Write the failing structural test**

Add a focused regression test in `crates/tidu/tests/chainrules_tests.rs` that
asserts the public engine no longer needs explicit tape ownership for basic
composition:

```rust
#[test]
fn hidden_graph_store_can_connect_values_without_public_tape_identity() {
    // Create two grad-tracking values through the new surface and ensure a
    // binary op can connect them without user-visible tape plumbing.
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test chainrules_tests hidden_graph_store_can_connect_values_without_public_tape_identity`

Expected: FAIL because the current runtime is still public-tape-centered.

**Step 3: Write minimal implementation**

Split the runtime into explicit graph/node modules. Keep an internal graph
store, but refactor tracked state so hidden edge handles, not public `Tape`,
define composition:

```rust
pub(crate) struct GraphStore<V: Differentiable> { /* ... */ }
pub(crate) struct EdgeHandle { /* ... */ }
pub(crate) struct Node<V: Differentiable> { /* ... */ }
```

Do not finish the public `Value` API yet; just make the runtime ready for it.

**Step 4: Run targeted tests**

Run: `cargo nextest run --release -p tidu --test chainrules_tests`

Expected: existing low-level tests still compile or fail only where the public
surface is intentionally changing.

**Step 5: Commit**

```bash
git add crates/tidu/src/engine/graph.rs crates/tidu/src/engine/node.rs crates/tidu/src/engine/context.rs crates/tidu/src/engine/mod.rs crates/tidu/src/engine/results.rs crates/tidu/src/engine/tracked.rs crates/tidu/src/engine/tape.rs crates/tidu/tests/chainrules_tests.rs
git commit -m "refactor: hide graph edges behind internal runtime"
```

### Task 3: Introduce the public `Value<V>` API

**Files:**
- Create: `crates/tidu/src/value.rs`
- Modify: `crates/tidu/src/lib.rs`
- Modify: `crates/tidu/src/engine/results.rs`
- Test: `crates/tidu/tests/value_api_tests.rs`

**Step 1: Extend the failing tests**

Add coverage for the final public methods:

```rust
#[test]
fn grad_returns_none_for_non_leaf_or_non_tracking_values() { /* ... */ }

#[test]
fn backward_with_seed_supports_non_scalar_outputs() { /* ... */ }
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test value_api_tests`

Expected: FAIL because `Value::grad`, `Value::backward`, and seeded backward
are not implemented.

**Step 3: Write minimal implementation**

Create `crates/tidu/src/value.rs` and expose:

```rust
pub struct Value<V: Differentiable> { /* hidden graph edge + primal */ }

impl<V: Differentiable> Value<V> {
    pub fn new(primal: V) -> Self;
    pub fn primal(&self) -> &V;
    pub fn requires_grad(&self) -> bool;
    pub fn requires_grad_(self, enabled: bool) -> Self;
    pub fn grad(&self) -> Result<Option<V::Tangent>>;
    pub fn zero_grad(&self) -> Result<()>;
    pub fn backward(&self) -> Result<()>;
    pub fn backward_with_seed(&self, seed: V::Tangent) -> Result<()>;
}
```

Wire these methods into the hidden graph runtime.

**Step 4: Run targeted tests**

Run: `cargo nextest run --release -p tidu --test value_api_tests`

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tidu/src/value.rs crates/tidu/src/lib.rs crates/tidu/src/engine/results.rs crates/tidu/tests/value_api_tests.rs
git commit -m "feat: add public value-centered reverse API"
```

### Task 4: Make high-level `Function` the default custom-op path

**Files:**
- Create: `crates/tidu/src/function.rs`
- Modify: `crates/tidu/src/lib.rs`
- Test: `crates/tidu/tests/function_api_tests.rs`

**Step 1: Write the failing tests**

Create `crates/tidu/tests/function_api_tests.rs` with generic custom-op
coverage:

```rust
#[test]
fn custom_function_supports_scalar_value_types() {
    // Define a square function via Function<f64> and verify backward.
}

#[test]
fn custom_function_supports_user_defined_differentiable_types() {
    // Define a tiny custom value type and a Function implementation for it.
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test function_api_tests`

Expected: FAIL because `Function`, `Context`, and `GradInputs` do not exist.

**Step 3: Write minimal implementation**

Add `crates/tidu/src/function.rs` with the high-level extension API and an
`apply` path that records hidden graph edges without exposing low-level rule
plumbing.

**Step 4: Run targeted tests**

Run: `cargo nextest run --release -p tidu --test function_api_tests`

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tidu/src/function.rs crates/tidu/src/lib.rs crates/tidu/tests/function_api_tests.rs
git commit -m "feat: add high-level function API"
```

### Task 5: Add value-oriented HVP

**Files:**
- Modify: `crates/tidu/src/value.rs`
- Modify: `crates/tidu/src/engine/forward.rs`
- Modify: `crates/tidu/src/engine/context.rs`
- Test: `crates/tidu/tests/hvp_api_tests.rs`

**Step 1: Write the failing tests**

Create `crates/tidu/tests/hvp_api_tests.rs`:

```rust
#[test]
fn hvp_uses_value_and_direction_pairs() {
    // Build a scalar loss and compute H·v through loss.hvp(...).
}

#[test]
fn hvp_does_not_require_public_tape_or_node_ids() {
    // Assert the public API uses Value references only.
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test hvp_api_tests`

Expected: FAIL because the public HVP API is still tape/node based.

**Step 3: Write minimal implementation**

Expose `Value::hvp(...)` and adapt the runtime so hidden graph edges provide
the necessary leaf tangent mapping internally.

**Step 4: Run targeted tests**

Run: `cargo nextest run --release -p tidu --test hvp_api_tests`

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tidu/src/value.rs crates/tidu/src/engine/forward.rs crates/tidu/src/engine/context.rs crates/tidu/tests/hvp_api_tests.rs
git commit -m "feat: add value-oriented hvp API"
```

### Task 6: Move raw graph APIs behind `tidu::expert`

**Files:**
- Create: `crates/tidu/src/expert.rs`
- Modify: `crates/tidu/src/lib.rs`
- Modify: `crates/tidu/src/engine/tape.rs`
- Modify: `crates/tidu/src/engine/tracked.rs`
- Test: `crates/tidu/tests/organization.rs`

**Step 1: Write the failing organization tests**

Add public-surface checks to `crates/tidu/tests/organization.rs`:

```rust
#[test]
fn tape_and_tracked_value_are_not_reexported_from_root() {
    // Compile-fail or doc-based surface check.
}

#[test]
fn expert_module_reexports_low_level_graph_controls() {
    // Ensure low-level escape hatches still exist under tidu::expert.
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test organization`

Expected: FAIL because low-level APIs are still on the root surface.

**Step 3: Write minimal implementation**

Add `crates/tidu/src/expert.rs`, move or re-export low-level runtime types
there, and remove them from the root `tidu` namespace.

**Step 4: Run targeted tests**

Run: `cargo nextest run --release -p tidu --test organization`

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tidu/src/expert.rs crates/tidu/src/lib.rs crates/tidu/src/engine/tape.rs crates/tidu/src/engine/tracked.rs crates/tidu/tests/organization.rs
git commit -m "refactor: isolate low-level graph APIs under expert"
```

### Task 7: Rewrite docs and examples around the new high-level path

**Files:**
- Modify: `README.md`
- Modify: `crates/tidu/src/lib.rs`
- Modify: `crates/tidu/tests/public_rustdoc_examples.rs`

**Step 1: Write the failing doc expectations**

Update the public rustdoc examples test so the expected snippets use:

- `Value::new(...)`
- `.requires_grad_(true)`
- `.backward()`
- `.grad()`
- `Function` for custom ops

**Step 2: Run doc/example verification to see the mismatch**

Run: `cargo nextest run --release -p tidu --test public_rustdoc_examples`

Expected: FAIL or stale examples still describing `Tape` / `NodeId`.

**Step 3: Write minimal implementation**

Rewrite `README.md` and crate-level rustdoc to make the high-level path the
only default documentation path. Mention `tidu::expert` only in an advanced
section.

**Step 4: Run targeted verification**

Run: `cargo nextest run --release -p tidu --test public_rustdoc_examples`

Expected: PASS.

**Step 5: Commit**

```bash
git add README.md crates/tidu/src/lib.rs crates/tidu/tests/public_rustdoc_examples.rs
git commit -m "docs: lead with value-centered autograd API"
```

### Task 8: Verify locally, then hand off immediately to `tenferro-rs`

**Files:**
- Modify: `Cargo.toml` (only if needed for local path wiring)
- Test: `crates/tidu/tests/value_api_tests.rs`
- Test: `crates/tidu/tests/function_api_tests.rs`
- Test: `crates/tidu/tests/hvp_api_tests.rs`

**Step 1: Run the focused `tidu-rs` verification set**

Run:

```bash
cargo fmt --all
cargo nextest run --release -p tidu --test value_api_tests --test function_api_tests --test hvp_api_tests
```

Expected: PASS.

**Step 2: Run the broader `tidu-rs` verification**

Run:

```bash
cargo nextest run --release -p tidu
```

Expected: PASS.

**Step 3: Wire the local `tidu-rs` checkout into `tenferro-rs`**

Update `tenferro-rs` locally to depend on the redesigned `tidu-rs` checkout,
then begin the `tenferro-rs` redesign/implementation immediately. Do **not**
open a PR yet.

**Step 4: Record the handoff status**

Document in the working notes or commit message that:

- `tidu-rs` is locally green,
- no PR has been opened,
- the next action is `tenferro-rs` adaptation on top of the local `tidu-rs`
  redesign.

**Step 5: Commit**

```bash
git add README.md crates/tidu/src lib.rs crates/tidu/tests
git commit -m "feat: redesign tidu public autograd API"
```
