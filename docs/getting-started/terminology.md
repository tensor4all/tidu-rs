# Terminology

`tidu` uses names that line up with JAX where the ideas match, but the concepts
below do not require JAX knowledge.

## Primitive Operation

A primitive operation is an atomic operation supplied by a downstream crate. It
could be scalar add, tensor multiply, matrix multiply, exponential, convolution,
or a domain-specific operation.

`tidu` does not know how to execute a primitive by itself. The downstream crate
defines the primitive, its input and output arity, how to run it in a concrete
runtime, and how AD should transform it.

## Primitive Computation Graph

A primitive computation graph is a directed acyclic program made from:

- input values,
- primitive operation applications,
- data dependencies between them,
- selected output values.

In this documentation, `graph` means this primitive computation graph unless a
page explicitly says it is discussing lower-level storage internals.

For example, `f(x) = x * x` can be represented as a graph with one input `x`,
one primitive multiply operation, and one output `y`.

## Linearization

Linearization builds a new graph that computes how selected outputs change for
selected input tangents. This is a graph-level Jacobian-vector product (JVP).

To linearize a graph, `tidu` walks the primitive applications and asks each
primitive for its local JVP rule. The result is a graph that reuses primal
values and accepts tangent inputs such as `dx`.

```text
f(x) = x * x

linearize f at x with tangent dx:
  y  = x * x
  dy = x * dx + dx * x

linear_transpose of dy = J dx with seed ct_y:
  ct_x = x * ct_y + x * ct_y
```

The original graph still computes `y`. The linearized graph computes `dy`, the
change in `y` caused by the chosen tangent `dx`.

## Linear Transpose

A linear transpose takes a linearized graph representing `dy = J dx` and builds
a new graph for cotangent propagation. With a cotangent seed `ct_y`, the
transposed graph computes `ct_x = J^T ct_y`.

This is the graph transform used to build reverse-mode flows after
linearization. The downstream runtime still decides how concrete values are
stored and evaluated.

## JVP Rule

A JVP rule is a local rule for one primitive operation. It receives primal
inputs, primal outputs, and optional tangent inputs. It emits tangent outputs.

The rule must be linear in its tangent inputs. For multiply, the rule is:

```text
d(lhs * rhs) = d_lhs * rhs + lhs * d_rhs
```

## Transpose Rule

A transpose rule is a local rule for one linear primitive operation. It receives
cotangents of the primitive outputs and emits cotangents of the primitive
inputs.

Transpose rules are the local pieces used by `linear_transpose` to reverse a
linearized graph.

## Eager Integration

Eager integration is for downstream frontends that execute primitives
immediately and want to expose a reverse-mode `backward()` workflow.

The downstream frontend records graph invocations with `tidu::eager::Recorder`
while it runs concrete operations. A single primitive can be recorded as a
one-operation graph, while a composite eager operation can record a larger graph
as one tape node. Later, `tidu::eager::try_backward` walks that trace, asks the
downstream runtime to replay needed primal values, transposes graph
linearizations, and accumulates cotangents.

`tidu` does not own tensor data, gradient slots, device placement, or user-facing
tensor objects.
