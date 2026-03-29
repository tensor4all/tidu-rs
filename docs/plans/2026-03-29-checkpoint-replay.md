# Checkpoint Replay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add first-class checkpointed operation replay to `tidu`, including HVP support, so callers can trade memory for replayed backward/HVP state without changing the tape's append-only execution model.

**Architecture:** Keep the tape as an append-only `Vec<Node>` ordered by `NodeId`, refactor node storage to distinguish leaf/materialized/replayable execution kinds, and introduce execution-local replay contexts that lazily rebuild checkpointed nodes. For HVP, replace the current eager whole-graph tangent array with phase-local, demand-driven tangent lookup so checkpoint replay remains memory-oriented.

**Tech Stack:** Rust 2021, `tidu`, `chainrules-core::ReverseRule`, Cargo workspace tests, `cargo nextest`, doctests, rustdoc.

---

### Task 1: Lock down checkpoint pullback semantics with failing tests

**Files:**
- Create: `crates/tidu/tests/checkpoint_pullback_tests.rs`
- Test: `crates/tidu/tests/checkpoint_pullback_tests.rs`

**Step 1: Write the failing test**

Create integration tests that define a tiny replay-counting recipe and assert:

```rust
#[test]
fn checkpointed_pullback_matches_materialized_pullback() {
    // Build the same scalar graph once with record_op and once with
    // record_checkpointed_op. Assert equal gradients.
}

#[test]
fn checkpointed_pullback_replays_lazily() {
    // Replay counter stays at 0 after forward and increases only when pullback
    // reaches the checkpointed node.
}

#[test]
fn checkpointed_pullback_does_not_persist_replayed_state_on_tape() {
    // Two pullback calls should trigger replay twice for a replayable node.
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test checkpoint_pullback_tests`

Expected: FAIL because `record_checkpointed_op` and replay recipes do not exist.

**Step 3: Write minimal implementation**

Add `crates/tidu/tests/checkpoint_pullback_tests.rs` with the exact replay
counter helpers and failing assertions.

**Step 4: Run test to verify the failure is the missing feature**

Run: `cargo nextest run --release -p tidu --test checkpoint_pullback_tests`

Expected: FAIL with missing API or type errors tied to checkpoint support.

**Step 5: Commit**

```bash
git add crates/tidu/tests/checkpoint_pullback_tests.rs
git commit -m "test: add checkpoint pullback coverage"
```

### Task 2: Refactor graph storage into explicit node execution kinds

**Files:**
- Create: `crates/tidu/src/engine/node.rs`
- Modify: `crates/tidu/src/engine/mod.rs`
- Modify: `crates/tidu/src/engine/context.rs`
- Modify: `crates/tidu/src/engine/tape.rs`
- Test: `crates/tidu/tests/chainrules_tests.rs`

**Step 1: Write the failing structural test**

Add a focused regression test in `crates/tidu/tests/chainrules_tests.rs` that
asserts placeholder `None` no longer needs to represent checkpointed ops:

```rust
#[test]
fn placeholder_and_leaf_states_remain_distinct() {
    // Create a leaf and a placeholder; ensure the internal node model keeps
    // their states separate once the explicit node-kind refactor lands.
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test chainrules_tests placeholder_and_leaf_states_remain_distinct`

Expected: FAIL because the current node model cannot express the distinction.

**Step 3: Write minimal implementation**

Introduce a dedicated node model in `crates/tidu/src/engine/node.rs`:

```rust
pub(crate) enum NodeExec<V: Differentiable> {
    Leaf,
    Materialized(Box<dyn ReverseRule<V>>),
    Replayable(Box<dyn CheckpointRecipe<V>>),
    Placeholder,
}
```

Update `AutogradGraph` to store `Node<V>` values with explicit `inputs`,
`NodeExec`, and primal retention policy instead of a bare `Option<Box<...>>`.

**Step 4: Run targeted tests**

Run: `cargo nextest run --release -p tidu --test chainrules_tests`

Expected: existing tape tests still compile and the new structural test passes.

**Step 5: Commit**

```bash
git add crates/tidu/src/engine/node.rs crates/tidu/src/engine/mod.rs crates/tidu/src/engine/context.rs crates/tidu/src/engine/tape.rs crates/tidu/tests/chainrules_tests.rs
git commit -m "refactor: model node execution kinds explicitly"
```

### Task 3: Introduce replay recipes and a checkpoint recording API

**Files:**
- Create: `crates/tidu/src/engine/replay.rs`
- Modify: `crates/tidu/src/engine/mod.rs`
- Modify: `crates/tidu/src/engine/tape.rs`
- Modify: `crates/tidu/src/lib.rs`
- Test: `crates/tidu/tests/checkpoint_pullback_tests.rs`

**Step 1: Write the failing API usage**

Extend the new checkpoint tests to use an explicit replay recipe:

```rust
let y = tape.record_checkpointed_op(
    8.0,
    Box::new(TestCheckpointRecipe::new(vec![x.node_id().unwrap()], 3.0)),
    None,
);
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test checkpoint_pullback_tests`

Expected: FAIL because `CheckpointRecipe` and `record_checkpointed_op` do not
exist.

**Step 3: Write minimal implementation**

Add replay support types in `crates/tidu/src/engine/replay.rs`:

```rust
pub struct ReplayResult<V: Differentiable> {
    pub output_primal: V,
    pub rule: Box<dyn ReverseRule<V>>,
}

pub trait CheckpointRecipe<V: Differentiable>: Send + Sync {
    fn inputs(&self) -> Vec<NodeId>;
    fn replay(&self, ctx: &mut ReplayContext<V>) -> AdResult<ReplayResult<V>>;
}
```

Expose the trait publicly from `tidu`, and add `Tape::record_checkpointed_op`
as the checkpointed sibling of `record_op`.

**Step 4: Run targeted tests**

Run: `cargo nextest run --release -p tidu --test checkpoint_pullback_tests`

Expected: tests compile farther, but still fail until replay execution exists.

**Step 5: Commit**

```bash
git add crates/tidu/src/engine/replay.rs crates/tidu/src/engine/mod.rs crates/tidu/src/engine/tape.rs crates/tidu/src/lib.rs crates/tidu/tests/checkpoint_pullback_tests.rs
git commit -m "feat: add checkpoint replay recipe API"
```

### Task 4: Execute checkpointed pullback through an execution-local replay context

**Files:**
- Create: `crates/tidu/src/engine/execution.rs`
- Modify: `crates/tidu/src/engine/context.rs`
- Modify: `crates/tidu/src/engine/mod.rs`
- Modify: `crates/tidu/src/engine/results.rs`
- Test: `crates/tidu/tests/checkpoint_pullback_tests.rs`

**Step 1: Write the failing replay behavior assertions**

Add assertions that two pullback calls on the same checkpointed tape replay the
node twice and return identical gradients.

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test checkpoint_pullback_tests`

Expected: FAIL because pullback still assumes permanent materialized rules.

**Step 3: Write minimal implementation**

Create an execution-local replay context:

```rust
pub(crate) struct ReplayContext<'g, V: Differentiable> {
    graph: &'g AutogradGraph<V>,
    primals: HashMap<NodeId, V>,
    replayed_rules: HashMap<NodeId, Box<dyn ReverseRule<V>>>,
}
```

Update pullback execution so replayable nodes call `recipe.replay(...)` only
when their node is reached during reverse traversal. Do not write replayed
rules or primals back into `AutogradGraph`.

**Step 4: Run targeted tests**

Run: `cargo nextest run --release -p tidu --test checkpoint_pullback_tests`

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tidu/src/engine/execution.rs crates/tidu/src/engine/context.rs crates/tidu/src/engine/mod.rs crates/tidu/src/engine/results.rs crates/tidu/tests/checkpoint_pullback_tests.rs
git commit -m "feat: execute checkpointed pullback via replay context"
```

### Task 5: Lock down checkpointed HVP behavior with failing tests

**Files:**
- Create: `crates/tidu/tests/checkpoint_hvp_tests.rs`
- Test: `crates/tidu/tests/checkpoint_hvp_tests.rs`

**Step 1: Write the failing test**

Create HVP coverage for replayable nodes:

```rust
#[test]
fn checkpointed_hvp_matches_materialized_hvp() {
    // Compare gradients and hvp results between materialized and replayable
    // versions of the same scalar graph.
}

#[test]
fn checkpointed_hvp_replays_each_phase_independently() {
    // Use counters to assert one replay during tangent computation and one
    // replay during reverse HVP.
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test checkpoint_hvp_tests`

Expected: FAIL because HVP still depends on the eager tangent array and
permanent materialized rules.

**Step 3: Write minimal implementation**

Add `crates/tidu/tests/checkpoint_hvp_tests.rs` with scalar-only deterministic
recipes and replay counters.

**Step 4: Run test to verify the failure is specific**

Run: `cargo nextest run --release -p tidu --test checkpoint_hvp_tests`

Expected: FAIL with HVP replay support errors rather than unrelated failures.

**Step 5: Commit**

```bash
git add crates/tidu/tests/checkpoint_hvp_tests.rs
git commit -m "test: add checkpoint hvp coverage"
```

### Task 6: Replace eager tangent arrays with demand-driven HVP execution

**Files:**
- Modify: `crates/tidu/src/engine/execution.rs`
- Modify: `crates/tidu/src/engine/context.rs`
- Modify: `crates/tidu/src/engine/tape.rs`
- Test: `crates/tidu/tests/checkpoint_hvp_tests.rs`
- Test: `crates/tidu/tests/chainrules_tests.rs`

**Step 1: Write the failing demand-driven assertion**

Extend the HVP tests so replay counters show no unconditional whole-graph
replay during tangent setup.

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test checkpoint_hvp_tests`

Expected: FAIL because the current HVP implementation eagerly computes all
tangents.

**Step 3: Write minimal implementation**

Move HVP to phase-local demand-driven lookup inside `execution.rs`:

```rust
fn tangent(&mut self, node: NodeId) -> AdResult<Option<V::Tangent>> {
    // leaf lookup -> cached tangent -> materialize recipe if needed ->
    // compute forward_tangents lazily for this node only
}
```

Use one replay/tangent cache for the forward-tangent phase and a separate cache
for reverse HVP. Do not share heavy replay state across phases.

**Step 4: Run targeted tests**

Run: `cargo nextest run --release -p tidu --test checkpoint_hvp_tests`

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tidu/src/engine/execution.rs crates/tidu/src/engine/context.rs crates/tidu/src/engine/tape.rs crates/tidu/tests/checkpoint_hvp_tests.rs crates/tidu/tests/chainrules_tests.rs
git commit -m "feat: make checkpointed hvp demand-driven"
```

### Task 7: Update public docs for retained vs checkpointed operations

**Files:**
- Modify: `README.md`
- Modify: `crates/tidu/src/lib.rs`
- Modify: `crates/tidu/src/engine/tape.rs`
- Modify: `crates/tidu/src/engine/tracked.rs`

**Step 1: Write the failing doc expectation**

Add a doc-focused assertion to an existing public-doc test that checks for the
new checkpoint API name and a short explanation of the replay tradeoff.

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tidu --test public_rustdoc_examples`

Expected: FAIL because checkpoint docs do not exist yet.

**Step 3: Write minimal implementation**

Document:

- when to use `record_op`,
- when to use `record_checkpointed_op`,
- that checkpointing trades memory for replayed backward/HVP work,
- that HVP replay is phase-local by default.

Include a minimal scalar example in rustdoc.

**Step 4: Run targeted tests**

Run: `cargo test --doc --release -p tidu`

Expected: PASS.

**Step 5: Commit**

```bash
git add README.md crates/tidu/src/lib.rs crates/tidu/src/engine/tape.rs crates/tidu/src/engine/tracked.rs
git commit -m "docs: explain checkpoint replay API"
```

### Task 8: Run the full verification suite and prepare the PR

**Files:**
- Modify: any files touched above as needed

**Step 1: Run formatting**

Run: `cargo fmt --all`

Expected: PASS with no diff remaining after formatting.

**Step 2: Run unit and integration tests**

Run: `cargo nextest run --release --workspace --no-fail-fast`

Expected: PASS.

**Step 3: Run doctests**

Run: `cargo test --doc --release --workspace`

Expected: PASS.

**Step 4: Run coverage and docs checks**

Run: `cargo llvm-cov nextest --workspace --release --json --output-path coverage.json`

Expected: PASS and produce `coverage.json`.

Run: `python3 scripts/check-coverage.py coverage.json`

Expected: PASS.

Run: `cargo doc --workspace --no-deps`

Expected: PASS.

Run: `python3 scripts/check-docs-site.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add checkpoint replay support"
```
