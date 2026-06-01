# Eager Integration

`tidu::eager` is for downstream frontends that execute primitive operations
immediately and want a reverse-mode `backward()` workflow.

## Recording

Use `Recorder` to record each primitive execution. Each input is described with
`EagerInput`:

- `key` is the user-visible value key used for cotangent accumulation.
- `trace` points to the operation that produced the value, if any.
- `requires_grad` controls whether cotangents should flow through the value.
- `data` stores concrete primal data for later replay.

`Recorder::record` returns one `EagerOutput` per primitive output.

## Backward Execution

The downstream runtime implements `BackwardExecutor`.

`tidu` calls it to:

- replay concrete primal values for a primitive graph,
- run a transposed linear graph with cotangent seeds,
- add concrete cotangents when multiple paths meet.

The downstream runtime still owns tensor allocation, gradient storage, device
selection, shape metadata, and user-facing error reporting.
