# Implementing Primitives

Downstream crates implement `tidu::Primitive` for their operation enum or
operation descriptor type.

## Operation Contract

A primitive operation must first implement `computegraph::GraphOp`:

- `Operand` is the concrete value type used by the runtime.
- `Context` is runtime evaluation context.
- `InputKey` identifies graph inputs and must implement `tidu::ADKey`.
- `n_inputs` and `n_outputs` describe operation arity.

`tidu::Primitive` adds AD-specific requirements:

- `add()` returns the primitive used to accumulate cotangents.
- `jvp_rule()` emits tangent outputs for linearization.
- `transpose_rule()` emits input cotangents for transposed linear maps.
- `try_jvp_rule()` and `try_linear_transpose_rule()` can report missing rules
  with `ADRuleError`.

## Rule Closure

Rules emit primitive applications through `PrimitiveBuilder`. Every operation
that a rule emits must also be part of the same primitive set and must also have
the AD rules needed by later transforms.

For example, a multiply JVP usually emits multiply and add. That means multiply
and add must both be valid primitives in the downstream set.

## Active And Inactive Inputs

In rule signatures, `LocalValId` is the graph-local identifier for a value
created while building the transformed primitive computation graph.

JVP rules receive `Option<LocalValId>` tangent inputs. `None` means the
corresponding primal input is not active for the current transform. Rules should
avoid emitting work for inactive inputs.

Transpose rules receive optional output cotangents. If an output cotangent is
`None`, no cotangent flows through that output slot.
