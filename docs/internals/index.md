# Internals

Most downstream implementers should not need internals.

Use this section when debugging `tidu` itself or when integrating a runtime that
must inspect lower-level graph storage.

Key implementation areas:

- `linearize` walks a resolved graph and asks primitive JVP rules to emit
  tangent graph structure.
- `linear_transpose` walks linear graph structure in reverse and asks primitive
  transpose rules to emit cotangent graph structure.
- `tidu::eager` stores recorded graph invocation traces, linearizes those
  graphs during backward, and delegates concrete execution to
  `BackwardExecutor`.

The public tutorials intentionally avoid these implementation details until they
are needed.
