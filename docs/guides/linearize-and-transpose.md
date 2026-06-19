# Linearize And Transpose

`linearize` and `linear_transpose` are separate graph transforms.

## Linearize

`linearize` takes a resolved primitive computation graph, selected output keys,
selected input keys, a `DiffPassId`, mutable AD context, and input aliases. It
returns a `LinearizedGraph`.

The returned graph has tangent inputs for the selected primal inputs and tangent
outputs for the selected primal outputs. Use `try_linearize` when primitive
rules can fail.

`DiffPassId` is a `u64` that tags one linearization pass; it is woven into the
tangent input keys (via `ADKey::tangent_of`) so repeated transforms do not
collide. Pass a fresh value per `linearize` call (see Repeated Transforms).

The input-aliases map (`HashMap<InputKey, ValueKey>`) is empty in the common
case. A non-empty entry says "treat this graph input as identical to that
existing value," so the input inherits the aliased value's tangent instead of
being an independent leaf — used for checkpointing and shared sub-graphs.

## Linear Transpose

`linear_transpose` takes a `LinearizedGraph` and returns another
`LinearizedGraph` whose inputs are cotangent seeds and whose outputs are
cotangents for the original active inputs.

Use `try_linear_transpose` when transpose rules can fail. Use
`try_linear_transpose_with_builder` when a downstream eager runtime wants to
execute the transposed linear map directly through a concrete builder.

## Cotangent Accumulation

When a single value feeds more than one consumer in the primal graph, the
transposed graph receives a separate cotangent contribution from each consumer.
`linear_transpose` sums these by emitting `add` nodes — obtained from
`Primitive::add()` — so the transposed map yields one accumulated cotangent per
active input.

For example, if `x` feeds both a `Mul` and an `Add`, transposition produces a
cotangent for `x` from each, and an `add` node combines them before the value is
reported as an input cotangent. Eager runtimes perform the same accumulation at
execution time through `BackwardExecutor::add_operands`.

## Repeated Transforms

Each `linearize` call needs a unique `DiffPassId`. If a transform result should
be transformed again, resolve the graph collection that includes the previous
result before calling `linearize` again.
