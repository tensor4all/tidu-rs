# Shared Primal Ownership Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the new `V: Clone` requirements introduced by checkpoint replay by sharing retained primals between the graph and attached `TrackedValue` handles.

**Architecture:** Keep the checkpoint replay design, but replace duplicated retained primals with shared `Arc<V>` storage. Graph nodes retain `Arc<V>` when needed, and attached `TrackedValue` instances reuse that shared primal instead of cloning `V`.

**Tech Stack:** Rust 2021, `std::sync::Arc`, `tidu`, `cargo nextest`, doctests, coverage checks, rustdoc.

---

### Task 1: Lock down the ownership regression with tests

**Files:**
- Modify: `crates/tidu/tests/chainrules_tests.rs`
- Test: `crates/tidu/tests/chainrules_tests.rs`

**Step 1: Write the failing test**

Add focused tests that instantiate a non-`Clone` differentiable scalar and
assert:

- `Tape::leaf` compiles and returns the original value via `TrackedValue::value()`
- `Tape::record_op` compiles for a non-`Clone` primal
- checkpoint pullback behavior still matches existing semantics

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test chainrules_tests`

Expected: FAIL because the tape constructors still require `V: Clone`.

**Step 3: Write minimal implementation**

Add the non-`Clone` coverage and keep the test data tiny and deterministic.

**Step 4: Run the targeted test**

Run: `cargo nextest run --release -p tidu --test chainrules_tests`

Expected: compile failure tied directly to the `Clone` bounds.

**Step 5: Commit**

```bash
git add crates/tidu/tests/chainrules_tests.rs
git commit -m "test: cover retained primal ownership without Clone"
```

### Task 2: Refactor retained primal storage to shared ownership

**Files:**
- Modify: `crates/tidu/src/engine/node.rs`
- Modify: `crates/tidu/src/engine/context.rs`
- Modify: `crates/tidu/src/engine/execution.rs`
- Modify: `crates/tidu/src/engine/tracked.rs`
- Modify: `crates/tidu/src/engine/tape.rs`

**Step 1: Write the minimal shared-ownership types**

Introduce explicit shared-primal storage:

```rust
enum TrackedPrimal<V> {
    Owned(V),
    Shared(Arc<V>),
}
```

and change graph-side retained primals to `Arc<V>`.

**Step 2: Update graph and replay accessors**

Make graph nodes expose retained primals as `&V`, backed by `Arc<V>`, and make
replay caches continue to work without cloning retained inputs eagerly.

**Step 3: Update tape constructors**

Change `leaf`, `leaf_with_tangent`, `placeholder`, and `record_op` to create one
shared `Arc<V>` instead of cloning `V`.

**Step 4: Run targeted tests**

Run: `cargo nextest run --release -p tidu --test chainrules_tests`

Expected: the non-`Clone` tests now compile and pass.

**Step 5: Commit**

```bash
git add crates/tidu/src/engine/node.rs crates/tidu/src/engine/context.rs crates/tidu/src/engine/execution.rs crates/tidu/src/engine/tracked.rs crates/tidu/src/engine/tape.rs crates/tidu/tests/chainrules_tests.rs
git commit -m "refactor: share retained primals across graph and handles"
```

### Task 3: Update consuming APIs and public docs

**Files:**
- Modify: `crates/tidu/src/engine/tracked.rs`
- Modify: `crates/tidu/src/lib.rs`
- Modify: `README.md`
- Modify: `crates/tidu/tests/public_rustdoc_examples.rs`

**Step 1: Define the consuming API behavior**

Keep `TrackedValue::value()` returning `&V`. For consuming methods that cannot
move out of shared state without cloning, add the minimal required bounds or
adjust semantics cleanly.

**Step 2: Update docs**

Document that retained primals are shared rather than cloned, and explain that
checkpointed nodes still trade stored rule state for replay.

**Step 3: Run docs-focused tests**

Run:

```bash
cargo test --doc --release -p tidu
cargo nextest run --release -p tidu --test public_rustdoc_examples
```

Expected: PASS.

**Step 4: Commit**

```bash
git add crates/tidu/src/engine/tracked.rs crates/tidu/src/lib.rs README.md crates/tidu/tests/public_rustdoc_examples.rs
git commit -m "docs: describe shared retained primals"
```

### Task 4: Run full verification

**Files:**
- Modify: none

**Step 1: Format**

Run: `cargo fmt --all`

**Step 2: Run full verification**

Run:

```bash
cargo fmt --all --check
cargo clippy --workspace
cargo nextest run --release --workspace --no-fail-fast
cargo test --doc --release --workspace
cargo llvm-cov nextest --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
bash scripts/build_docs_site.sh
```

Expected: PASS for all commands.

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: remove retained primal Clone requirements"
```
