# Shared Primal Ownership Design for Tidu

## Summary

The checkpoint-replay refactor introduced retained graph primals so replayable
nodes can recover direct input values during pullback and HVP. The first cut
stored the same primal twice:

- once in `TrackedValue<V>`,
- once in the graph node when the node retained its primal.

That duplication forced `Tape::leaf`, `Tape::placeholder`, and
`Tape::record_op` to require `V: Clone`.

This addendum removes that constraint by switching retained primals to shared
ownership. Instead of cloning `V`, the graph and attached `TrackedValue`
instances will share the same `Arc<V>` when a primal must stay available after
recording.

## Why Not Pure Graph-Owned Handles

A pure handle design looked attractive at first, but it collides with two facts
in the current API:

- `TrackedValue::value()` returns `&V`, while attached values would need to
  borrow through `Tape`'s `Mutex`.
- `record_checkpointed_op(...)` must return a forward value immediately even
  though the graph intentionally evicts the checkpointed node's retained primal.

That means a fully graph-owned handle design would force a broader public API
break in the same PR. Shared ownership fixes the immediate root cause without
derailing the checkpoint work.

## Approved Direction

### 1. Share retained primals with `Arc<V>`

When a node retains its primal, store it as `Arc<V>` in the graph. Any attached
`TrackedValue` pointing at that node should hold the same `Arc<V>` rather than a
second copy.

### 2. Keep checkpointed outputs forward-usable

Checkpointed nodes may still evict the graph-side retained primal. Their
returned `TrackedValue` should therefore keep an owned/shared forward primal so
callers can continue eager forward computation.

### 3. Represent tracked primals explicitly

`TrackedValue` should distinguish between:

- detached owned values,
- attached shared primals,
- checkpoint-only forward values not retained in the graph.

The runtime should not infer ownership policy from optional `Tape` or `NodeId`
fields alone.

### 4. Preserve user-facing read ergonomics

`TrackedValue::value()` should keep returning `&V` so existing examples and
downstream code do not need a closure-based accessor just to read a primal.

Methods that consume the primal (`into_value`, `detach`) may need a `V: Clone`
bound when the underlying storage is shared, because moving out of shared state
is impossible.

### 5. Update helper APIs to the new ownership model

`tracked_existing(...)` and any internal graph helpers should stop assuming the
caller must always provide a full owned primal copy for attached nodes. If a
retained primal already exists in the graph, reconstruct the handle from that
shared state.

## Data Model Sketch

```rust
enum TrackedPrimal<V> {
    Owned(V),
    Shared(Arc<V>),
}

enum PrimalStorage<V> {
    Retained(Arc<V>),
    Evicted,
}
```

The graph stores `PrimalStorage<V>`. `TrackedValue` stores `TrackedPrimal<V>`.
Attached materialized/leaf/placeholder values normally use `Shared`. Detached
values use `Owned`. Checkpointed values may use `Shared` or `Owned` depending on
whether the graph retains the primal.

## Testing Strategy

The refactor is acceptable only if it proves all of these:

- `Tape::leaf`, `Tape::placeholder`, and `Tape::record_op` no longer require
  `V: Clone`,
- existing pullback and HVP behavior remains unchanged,
- checkpointed pullback and HVP still work,
- `TrackedValue::value()` keeps exposing the same forward result,
- consuming APIs (`into_value`, `detach`) behave sensibly for shared primals.

## Main Risk

The main risk is partially converting the ownership model and leaving graph
storage, `TrackedValue`, and docs out of sync. The implementation should move
all retained-primal paths together and verify the public examples immediately.
