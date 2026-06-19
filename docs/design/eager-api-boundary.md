# Eager API Boundary

> **Design note (internal record).** This page captures a design decision and may
> describe historical or in-progress reasoning. It is not part of the user guide —
> see Getting Started and the Guides for current usage.

## Summary

`tidu` keeps a generic eager reverse-mode recording layer for downstream
frontends built on `computegraph` and `Primitive`, but the public surface is
limited to frontend-facing handles and executor traits.

The eager layer is not a tensor runtime. Downstream crates still own concrete
execution, tensor metadata, gradient slots, backend placement, extension
runtimes, and user-facing eager tensor types.

## Public Responsibilities

`tidu::eager` owns:

- recording an eager graph invocation into an opaque trace handle,
- allocating stable graph input keys and eager output keys,
- sharing one trace node across multi-output operations,
- sorting and walking the reverse trace during backward,
- linearizing recorded graphs during backward, and
- calling downstream executors to replay primal and `linear_transpose` work.

The root crate owns small helpers that execute AD-generated linearized graphs
through a caller-provided `PrimitiveBuilder`.

## Private Responsibilities

The following remain implementation details:

- trace nodes and trace edges,
- recorded graph key alignment,
- saved-forward map construction,
- reverse trace topological sorting.

Downstream frontends should not construct trace nodes, trace edges, or saved
forward maps by hand. They should build a `RecordedGraph`, pass eager inputs and
concrete outputs to a `Recorder`, store the returned output keys and trace
handles, then call `try_backward` from their public tensor API.

## Downstream Responsibilities

Downstream crates own:

- concrete eager operation execution,
- metadata and shape-guard registration,
- gradient slot storage,
- backend-specific cotangent accumulation,
- extension-rule/runtime dispatch, and
- the public eager tensor/value type.

The executor trait is the boundary where downstream runtime behavior enters
`tidu`'s generic backward traversal.

## Public Shape

The root crate exports graph-transform APIs and rule contracts. Eager-specific
items live under `tidu::eager`:

```rust
tidu::eager::{
    BackwardExecutor, EagerInput, EagerOutput, KeySource, RecordedGraph, Recorder, Trace,
    try_backward,
}
```

Builder-backed transpose helpers live at the crate root:

```rust
tidu::try_linear_transpose_with_builder(...)
```

This prevents two different "eager modes" from appearing at the root API:
there is one eager trace runtime, and one lower-level builder-backed helper.
