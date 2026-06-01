# Computegraph Integration

This page names lower-level graph storage terms for implementers who need them.
They are not required terminology for the getting-started path.

`tidu` stores graph transform results using `computegraph` data structures. The
important public wrappers are `LinearizedGraph` and `PrimitiveGraph`, but
advanced runtimes may need raw access.

## Storage Terms

- `Fragment` is the lower-level graph container.
- `LocalValId` identifies a value within one graph container.
- `GlobalValKey` identifies an input or derived value across graph boundaries.
- `ValRef` represents an operation input, either local or external.
- `OpMode` marks primal operations and linear operations.

## Why These Appear

Graph transforms often need to reference primal values from a linear rule. Those
references are represented as external graph values at the storage layer.

Downstream runtimes can use `as_graph()` or `into_graph()` when compiling,
materializing, or merging transform results with their existing graph execution
pipeline.
