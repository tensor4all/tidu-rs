# Public Boundaries

`tidu` exposes transform-oriented wrappers and traits.

## Main Boundary

- `Primitive` defines local AD behavior for a downstream operation set.
- `PrimitiveBuilder` is the rule-emission interface used by JVP and transpose
  rules.
- `LinearizedGraph` is the public transform result.
- `PrimitiveGraph` is a borrowed graph view passed to eager replay hooks.
- `tidu::eager` records and walks eager graph-invocation traces.

## Downstream Responsibilities

Downstream crates own:

- concrete primitive execution,
- tensor or scalar storage,
- metadata such as shape and dtype,
- device placement,
- gradient slots,
- user-facing frontend APIs.

## Advanced Access

`LinearizedGraph::as_graph()` and `LinearizedGraph::into_graph()` expose the
lower-level graph representation for runtimes that need to compile or merge it.
These methods are advanced integration points, not the first-read model for
using `tidu`.
