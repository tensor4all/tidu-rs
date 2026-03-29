# Checkpoint Replay Design for Tidu

## Summary

`tidu` currently stores a fully prepared `ReverseRule` on each operation node.
That is simple for pullback and the current deferred-HVP implementation, but it
prevents activation-checkpoint-style memory tradeoffs because the heavy
backward/HVP state is retained on the tape for the lifetime of the graph.

This design introduces checkpointing as a first-class node execution mode for
`tidu` itself, not as a PyTorch-style whole-region autograd hook. The tape will
continue to be an append-only `Vec<Node>` in topological order, but operation
nodes will distinguish between permanently materialized execution state and
replayable recipes that can rebuild execution state on demand.

The design includes HVP from the start. The key constraint is that HVP replay
artifacts should remain phase-local by default: forward-tangent replay and
reverse HVP replay must not share heavy caches across phases unless a later
optimization proves worthwhile.

## Why Not Copy PyTorch

PyTorch's `torch.utils.checkpoint` is built around eager autograd,
`saved_tensors_hooks`, and whole-region replay during backward. `tidu` has a
different execution model:

- forward values are computed explicitly by the caller,
- the tape stores per-node rule objects rather than tracing arbitrary Python
  code,
- pullback and HVP are explicit graph traversals over `NodeId` order.

Because of that, the natural design in `tidu` is per-node replay rather than
whole-region replay. A checkpointed node should store a lightweight replay
recipe plus structural graph information, and execution should materialize the
heavy per-node state only when a traversal reaches that node.

## Approved Direction

### 1. Keep the tape storage linear

Do not move the runtime graph to `petgraph`. The tape is append-only and
already topologically ordered by `NodeId`, so reverse traversal stays a simple
reverse index walk over `Vec<Node>`. Generic graph crates are only attractive
for analysis passes; they add little value to the runtime execution path.

### 2. Split node execution kinds explicitly

The graph should represent these cases directly instead of overloading
`Option<Box<dyn ReverseRule<V>>>`:

- `Leaf`
- `Materialized`
- `Replayable`

This avoids conflating leaf nodes, placeholders, and checkpointed operations.

### 3. Separate replay specs from replay results

Each replayable node stores a lightweight recipe that can rebuild execution
state later. The replay result is a temporary execution object used only inside
one pullback/HVP call.

For this PR, the cleanest compatibility path is:

- keep `chainrules_core::ReverseRule` as the materialized execution trait,
- add a `CheckpointRecipe` trait in `tidu`,
- let the recipe replay return both `output_primal` and a freshly prepared
  `Box<dyn ReverseRule<V>>`.

That keeps the public dependency layering intact while moving the runtime
toward the cleaner "recipe vs prepared state" architecture.

### 4. Move replay state into execution-local contexts

Recomputed primals and freshly prepared rules must not be written back to the
graph permanently. They belong to a single pullback or HVP invocation.

Execution should therefore build an `ExecutionContext` that owns temporary
caches such as:

- replayed output primals,
- replayed `ReverseRule` objects,
- per-phase tangent caches.

This keeps repeated backward calls isolated and avoids hidden tape mutation.

### 5. Make HVP phase-local by default

Checkpointing increases HVP compute cost. That is expected and acceptable. The
correct baseline is:

- HVP forward-tangent replay uses one phase-local cache,
- HVP reverse replay uses a separate phase-local cache,
- heavy replay artifacts are not shared across phases by default.

This preserves checkpoint's memory-saving intent and avoids turning replay
results into long-lived HVP state.

### 6. Move tangents toward demand-driven execution

The current HVP implementation eagerly allocates `Vec<Option<V::Tangent>>` for
all nodes. That conflicts with checkpoint's "recompute only what is needed"
model.

The target end state is a demand-driven tangent interface in the execution
context:

- request an output tangent by `NodeId`,
- recursively materialize input tangents only when needed,
- cache tangents only within the current HVP phase.

This PR should aim for that end state rather than bolting checkpoint replay on
top of the eager tangent array.

## Data Model Sketch

The runtime shape should move toward:

```rust
struct Node<V: Differentiable> {
    inputs: Vec<NodeId>,
    exec: NodeExec<V>,
    primal: PrimalStorage<V>,
}

enum NodeExec<V: Differentiable> {
    Leaf,
    Materialized(Box<dyn ReverseRule<V>>),
    Replayable(Box<dyn CheckpointRecipe<V>>),
}

enum PrimalStorage<V> {
    Retained(V),
    Evicted,
}
```

`TrackedValue<V>` may still carry the immediate forward value returned to the
caller, but graph-level primal storage should become an explicit policy rather
than an implicit duplicate of `TrackedValue::value`.

## Public API Direction

This PR should keep existing users working while introducing the new path:

- keep `Tape::record_op(...)` for materialized operations,
- add `Tape::record_checkpointed_op(...)`,
- add a replay recipe trait and any supporting replay result types in `tidu`,
- update rustdoc to explain that `record_op` retains backward state while
  `record_checkpointed_op` trades memory for replay.

If the implementation proves cleaner, `record_op` can later become a thin
wrapper over a more explicit internal node builder.

## Testing Strategy

The design is only acceptable if these behaviors are covered by tests:

- checkpointed pullback matches materialized pullback,
- checkpointed HVP matches materialized HVP,
- replay happens lazily and only when the traversal reaches the node,
- replayed state is execution-local rather than permanently reinstalled on the
  tape,
- nested replayable dependencies work,
- demand-driven HVP does not require a full eager tangent vector.

Tests should stay small and deterministic, use scalar examples first, and add
replay counters in test-only recipes to validate replay behavior directly.

## Main Risks

- The current `TrackedValue` and graph node storage model duplicate primal
  ownership; the refactor must leave one clear source of truth.
- HVP demand-driven replay is the highest-risk part of this PR because it
  changes the current eager tangent algorithm.
- Public API docs must be updated together with the implementation so users can
  understand when to choose retained vs checkpointed ops.
