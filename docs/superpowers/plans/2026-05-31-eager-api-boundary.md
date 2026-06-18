# Eager API Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `tidu`'s eager reverse-mode AD helpers into a narrow `tidu::eager` public module and move linear-fragment emitter helpers into `tidu::emit`.

**Architecture:** Keep the existing generic trace algorithm, but hide trace node and edge internals behind an opaque `Trace<Op>`. Route recording through `Recorder`, route backward through `backward`, and expose only the downstream runtime hooks needed for concrete replay and transpose execution.

**Tech Stack:** Rust 2021, `computegraph`, `tidu::PrimitiveOp`, `cargo nextest --release`, doctests.

---

### Task 1: Lock Down The New Public API

**Files:**
- Modify: `tests/eager_record_tests.rs`
- Modify: `tests/eager_backward_tests.rs`
- Modify: `tests/fallible_ad_tests.rs`

- [ ] **Step 1: Write failing tests and imports**

Update eager tests so they import `tidu::eager::{...}` and `tidu::emit::linear_transpose_with_builder` instead of root eager symbols.

- [ ] **Step 2: Run targeted tests**

Run: `cargo nextest run --release --test eager_record_tests --test eager_backward_tests --test fallible_ad_tests`

Expected: FAIL with unresolved `tidu::eager` and `tidu::emit` items.

### Task 2: Introduce `tidu::emit`

**Files:**
- Create: `src/emit.rs`
- Modify: `src/lib.rs`
- Delete: `src/eager_transpose.rs`

- [ ] **Step 1: Move the fallible emitter helper**

Move the fallible eager transpose helper to `emit` and drop the infallible wrapper.

- [ ] **Step 2: Run emitter-focused tests**

Run: `cargo nextest run --release --test eager_backward_tests emit_linear_transpose_with_builder_fan_out_accumulation`

Expected: PASS after tests call the new helper.

### Task 3: Introduce `tidu::eager`

**Files:**
- Create: `src/eager/mod.rs`
- Create: `src/eager/record.rs`
- Create: `src/eager/backward.rs`
- Create: `src/eager/trace.rs`
- Modify: `src/lib.rs`
- Delete: `src/eager_record.rs`
- Delete: `src/backward.rs`
- Delete: `src/grad_node.rs`

- [ ] **Step 1: Add opaque trace and public input/output types**

Add `Trace`, `Input`, `Output`, `KeySource`, and `Recorder`.

- [ ] **Step 2: Add `BackwardExecutor` and `backward`**

Move the current reverse traversal behind `backward`, with trace sorting internal to the module.

- [ ] **Step 3: Run eager tests**

Run: `cargo nextest run --release --test eager_record_tests --test eager_backward_tests`

Expected: PASS.

### Task 4: Refresh Docs And Verify

**Files:**
- Modify: `README.md`
- Modify: `src/lib.rs`
- Test: all tests

- [ ] **Step 1: Update public docs**

Document `tidu::eager` and `tidu::emit` without exposing trace internals.

- [ ] **Step 2: Format and verify**

Run:

```bash
cargo fmt --all
cargo clippy --workspace
cargo nextest run --release --workspace --no-fail-fast
cargo test --doc --release --workspace
```

Expected: PASS.
