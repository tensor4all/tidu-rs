# Eager Integration

`tidu::eager` is for downstream frontends that execute operations immediately
and want a reverse-mode `backward()` workflow.

## Recording

Use `Recorder` to record each eager graph invocation. A single primitive eager
operation is represented as a one-operation `RecordedGraph`; composite eager
operations can record a larger primitive graph as one tape node. Each input is
described with `EagerInput`:

- `key` is the user-visible value key used for cotangent accumulation.
- `trace` points to the graph invocation that produced the value, if any.
- `requires_grad` controls whether cotangents should flow through the value.
- `data` stores concrete primal data for later replay.

`Recorder::record_graph` returns one `EagerOutput` per recorded graph output.

## Backward Execution

The downstream runtime implements `BackwardExecutor`.

`tidu` calls it to:

- replay concrete primal values for a primitive graph,
- run a transposed linear graph with cotangent seeds,
- add concrete cotangents when multiple paths meet.

The downstream runtime still owns tensor allocation, gradient storage, device
selection, shape metadata, and user-facing error reporting.
