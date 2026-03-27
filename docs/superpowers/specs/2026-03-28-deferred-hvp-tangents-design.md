# Deferred Leaf Tangent Injection for HVP

**Issue:** #6
**Date:** 2026-03-28
**Status:** Approved

## Summary

Replace the current "tangents baked into rules at construction time" HVP
workflow with a fully deferred tangent injection model. Leaf tangent
directions **v** are passed at `hvp()` call time, not at graph
construction time.

Changes span two crates:

1. `chainrules-core` — update `ReverseRule` trait (add
   `forward_tangents`, update `pullback_with_tangents` signature)
2. `tidu` — update `Tape::hvp` signature, add forward tangent
   propagation pass in `AutogradGraph`

## Trait Changes (`chainrules-core`)

### New method: `forward_tangents`

Computes the output tangent from input tangents (frule). Used by the
forward tangent propagation pass before HVP.

```rust
fn forward_tangents(
    &self,
    input_tangents: &dyn Fn(NodeId) -> Option<&V::Tangent>,
) -> AdResult<Option<V::Tangent>> {
    let _ = input_tangents;
    Err(AutodiffError::HvpNotSupported)
}
```

Returns `Ok(None)` when all inputs have zero tangent (the rule can skip
computation). Returns `Ok(Some(tangent))` otherwise.

### Updated method: `pullback_with_tangents`

Now receives input tangents via a provider instead of storing them in the
rule.

```rust
fn pullback_with_tangents(
    &self,
    cotangent: &V::Tangent,
    cotangent_tangent: &V::Tangent,
    input_tangents: &dyn Fn(NodeId) -> Option<&V::Tangent>,
) -> AdResult<Vec<PullbackWithTangentsEntry<V>>> {
    let _ = (cotangent, cotangent_tangent, input_tangents);
    Err(AutodiffError::HvpNotSupported)
}
```

Both methods have default implementations returning `HvpNotSupported`, so
existing rules that do not need HVP are unaffected.

## Tape API Changes (`tidu`)

### `Tape::hvp` — new signature

```rust
pub fn hvp(
    &self,
    loss: &TrackedValue<V>,
    leaf_tangents: &HashMap<NodeId, V::Tangent>,
) -> AdResult<HvpResult<V>>
```

The caller passes the tangent direction **v** as a map from leaf `NodeId`
to tangent value. Leaves not present in the map are treated as having zero
tangent (`None`).

### Unchanged APIs

- `Tape::leaf` — unchanged.
- `Tape::leaf_with_tangent` — kept for forward-mode (JVP) use cases.
- `Tape::record_op` — the `output_tangent` parameter stays (used by
  downstream code that computes tangents eagerly during graph construction
  for non-HVP purposes).
- `Tape::pullback` / `pullback_with_seed` — unchanged.

## HVP Traversal Algorithm

The new `hvp_from` in `AutogradGraph` runs in two phases.

### Phase 1 — Forward tangent propagation

Walk nodes `0..=output_node` in topological (forward) order. For each
node:

- **Leaf node:** look up tangent from `leaf_tangents` map; if absent,
  store `None` (meaning zero tangent).
- **Op node:** call `rule.forward_tangents(|id| tangents[id])` to compute
  output tangent.

Result: `Vec<Option<V::Tangent>>` indexed by node index.

### Phase 2 — Reverse pass with tangent provider

Walk nodes `output_node..=0` in reverse order. For each op node with a
cotangent:

- Call `rule.pullback_with_tangents(&cot, &cot_tan, |id| tangents[id])`
  where `tangents` is the vec from Phase 1.
- Accumulate both cotangents and cotangent-tangents to input nodes.

Result: `(Vec<Option<V::Tangent>>, Vec<Option<V::Tangent>>)` —
cotangents and cotangent-tangents.

### None semantics

`None` in the tangents vec means "zero tangent". The closure
`&dyn Fn(NodeId) -> Option<&V::Tangent>` naturally represents this. Rules
can optimize zero cases (e.g., skip einsum terms when an input tangent is
`None`).

## Error Handling

No new error variants. Existing `AutodiffError` covers all cases:

- `HvpNotSupported` — if any rule's `forward_tangents` or
  `pullback_with_tangents` hits the default impl.
- `MissingNode` — if the loss node is not on the tape.
- `NonScalarLoss` — if `loss.num_elements() != 1`.
- `GraphFreed` — if the graph has been freed.

`leaf_tangents` keys that are not leaves or not on the tape are silently
ignored (never looked up).

## Migration & Breaking Changes

### `chainrules-core` (breaking)

- `pullback_with_tangents` gains a third parameter `input_tangents`. All
  downstream implementors must update their signature.
- New `forward_tangents` method has a default impl — non-breaking for
  rules that do not need HVP.

### `tidu` (breaking)

- `Tape::hvp` signature changes: adds
  `leaf_tangents: &HashMap<NodeId, V::Tangent>` parameter.
- All existing `hvp()` call sites must pass the map.

### Migration for existing rule implementors

1. Remove stored tangent fields (e.g., `saved_dx`, `input_tangents`).
2. Implement `forward_tangents`: compute output tangent from input
   tangents via the provider.
3. Update `pullback_with_tangents`: read input tangents from the closure
   instead of `self` fields.

### Migration for existing `hvp()` callers

1. Build a `HashMap<NodeId, V::Tangent>` with leaf tangent directions.
2. Pass it to `tape.hvp(&loss, &leaf_tangents)`.
3. No longer need `leaf_with_tangent()` for HVP (still available for
   JVP).

### Test updates in tidu

- `SquareRuleHvp` and `AddRuleHvp`: remove `saved_dx`, implement
  `forward_tangents`, update `pullback_with_tangents` to use the closure.
- `hvp_square_function` and `hvp_dag_merge_point` tests: pass
  `leaf_tangents` map instead of using `leaf_with_tangent`.
