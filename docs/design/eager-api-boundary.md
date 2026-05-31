# Eager API Boundary

## Summary

`tidu` keeps a generic eager reverse-mode recording layer for downstream
frontends built on `computegraph` and `PrimitiveOp`, but the public surface is
limited to frontend-facing handles and executor traits.

The eager layer is not a tensor runtime. Downstream crates still own concrete
execution, tensor metadata, gradient slots, backend placement, extension
runtimes, and user-facing eager tensor types.

## Public Responsibilities

`tidu::eager` owns:

- recording an eager operation into an opaque trace handle,
- allocating stable input aliases and output keys,
- sharing one trace node across multi-output operations,
- sorting and walking the reverse trace during backward,
- building one-op linear fragments during backward, and
- calling downstream executors to replay primal and transpose work.

`tidu::emit` owns small helpers that execute AD-generated linear fragments
through a caller-provided `OpEmitter`.

## Private Responsibilities

The following remain implementation details:

- trace nodes and trace edges,
- saved-forward key derivation,
- saved-forward map construction,
- reverse trace topological sorting, and
- single-op linear-fragment construction.

Downstream frontends should not construct trace nodes, trace edges, or saved
forward maps by hand. They should pass eager inputs and concrete outputs to a
`Recorder`, store the returned output keys and trace handles, then call
`try_backward` from their public tensor API.

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
    BackwardExecutor, Input, KeySource, Output, Recorder, Trace, try_backward,
}
```

Linear-fragment emitter helpers live under `tidu::emit`:

```rust
tidu::emit::try_transpose_fragment(...)
```

This prevents two different "eager modes" from appearing at the root API:
there is one eager trace runtime, and one lower-level emitter helper module.
