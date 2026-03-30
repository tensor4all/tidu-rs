# Function Primal Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign `tidu::Function<V>` so primal execution, save-for-backward policy, and reverse pullback are separate, eliminating no-grad overhead and avoiding duplicated primal implementations.

**Architecture:** Replace `Function::forward(ctx, ...)` with `Function::primal(...)` plus `Function::save_for_backward(...)`, update `Function::apply(...)` to skip save work when no input requires gradients, and remove `Context` from the normal public surface. Keep `tidu::expert` unchanged.

**Tech Stack:** Rust 2021, `tidu`, `chainrules-core`, workspace rustdoc/README examples, `cargo nextest --release`, `cargo fmt`.

---

### Task 1: Lock down the redesign with failing high-level API tests

**Files:**
- Modify: `crates/tidu/tests/function_api_tests.rs`
- Modify: `crates/tidu/tests/value_api_tests.rs`

**Step 1: Write the failing tests**

Update `crates/tidu/tests/function_api_tests.rs` so every custom function uses
the new API:

```rust
impl Function<f64> for Square {
    type Saved = f64;

    fn primal(inputs: &[&f64]) -> tidu::AdResult<f64> {
        Ok(inputs[0] * inputs[0])
    }

    fn save_for_backward(inputs: &[&f64], _output: &f64) -> tidu::AdResult<f64> {
        Ok(*inputs[0])
    }

    fn backward(saved: &f64, grad_out: &f64) -> tidu::AdResult<GradInputs<f64>> {
        Ok(GradInputs::from(vec![Some(2.0 * *saved * *grad_out)]))
    }
}
```

Add a new regression test that proves `save_for_backward()` is skipped on the
no-grad path by incrementing an atomic counter only inside
`save_for_backward()`.

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test function_api_tests`

Expected: FAIL because `Function` still requires `forward(ctx, ...)`.

**Step 3: Write minimal implementation**

Do not touch production code yet. Only update tests to the new API and add the
new no-grad regression.

**Step 4: Run the targeted test again**

Run: `cargo nextest run --release -p tidu --test function_api_tests`

Expected: FAIL for the intended missing API symbols only.

### Task 2: Redesign the `Function` trait and remove `Context`

**Files:**
- Modify: `crates/tidu/src/function.rs`
- Modify: `crates/tidu/src/lib.rs`

**Step 1: Replace the trait surface**

Rewrite `crates/tidu/src/function.rs` so `Function<V>` becomes:

```rust
pub trait Function<V: Differentiable + Send + Sync + 'static>: Send + Sync + 'static {
    type Saved: Send + Sync + 'static;

    fn primal(inputs: &[&V]) -> AdResult<V>;
    fn save_for_backward(inputs: &[&V], output: &V) -> AdResult<Self::Saved>;
    fn backward(saved: &Self::Saved, grad_out: &V::Tangent) -> AdResult<GradInputs<V>>;
}
```

Delete the public `Context` type from the high-level API and remove its
re-export from `crates/tidu/src/lib.rs`.

**Step 2: Update `apply()`**

Make `Function::apply()`:

1. gather `&V` primals,
2. compute `output = Self::primal(&primals)?`,
3. return `Value::new(output)` immediately when no input requires grad,
4. otherwise compute `saved = Self::save_for_backward(&primals, &output)?`,
5. register the reverse rule and return the reverse-tracked `Value<V>`.

**Step 3: Run the focused tests**

Run: `cargo nextest run --release -p tidu --test function_api_tests`

Expected: PASS.

### Task 3: Update public docs and examples to the new high-level API

**Files:**
- Modify: `README.md`
- Modify: `crates/tidu/src/lib.rs`
- Modify: `crates/tidu/tests/public_rustdoc_examples.rs`

**Step 1: Rewrite examples**

Replace every `forward(ctx, ...)` example with the new `primal +
save_for_backward + backward` shape in:

- README quick example
- crate-level rustdoc
- rustdoc smoke tests

**Step 2: Run doc-facing verification**

Run: `cargo test -p tidu --doc`

Expected: PASS.

### Task 4: Run full targeted verification and leave the branch ready for downstream work

**Files:**
- Modify: `crates/tidu/tests/function_api_tests.rs`
- Modify: `crates/tidu/tests/value_api_tests.rs`
- Modify: `README.md`
- Modify: `crates/tidu/src/function.rs`
- Modify: `crates/tidu/src/lib.rs`

**Step 1: Format**

Run: `cargo fmt --all`

Expected: formatting is clean.

**Step 2: Run targeted test suite**

Run:

```bash
cargo nextest run --release -p tidu --test function_api_tests
cargo nextest run --release -p tidu --test value_api_tests
cargo nextest run --release -p tidu
cargo test -p tidu --doc
```

Expected: PASS.

**Step 3: Stop before PR**

Do not create a PR. Leave the updated `tidu-rs` worktree ready, then move on
to `tenferro-rs` for downstream integration.
