# Architecture

> See [How tidu Works](overview.md) for diagrams of the layers, the transform
> pipeline, and the primitive contract.

`tidu` sits between downstream primitive sets and graph runtimes.

```text
downstream primitive set
  -> Primitive, JVP rules, transpose rules
tidu
  -> linearize, linear_transpose, eager trace walking
downstream runtime
  -> concrete execution, metadata, storage, gradient slots
```

The public boundary is intentionally expressed in AD terms: primitive operation,
primitive computation graph, linearization, linear transpose, JVP rule, and
transpose rule.

Lower-level graph storage details are documented only for implementers who need
advanced integration points.
