# Linearize And Transpose

`linearize` and `linear_transpose` are separate graph transforms.

## Linearize

`linearize` takes a resolved primitive computation graph, selected output keys,
selected input keys, a `DiffPassId`, mutable AD context, and input aliases. It
returns a `LinearizedGraph`.

The returned graph has tangent inputs for the selected primal inputs and tangent
outputs for the selected primal outputs. Use `try_linearize` when primitive
rules can fail.

## Linear Transpose

`linear_transpose` takes a `LinearizedGraph` and returns another
`LinearizedGraph` whose inputs are cotangent seeds and whose outputs are
cotangents for the original active inputs.

Use `try_linear_transpose` when transpose rules can fail. Use
`try_linear_transpose_with_builder` when a downstream eager runtime wants to
execute the transposed linear map directly through a concrete builder.

## Repeated Transforms

Each `linearize` call needs a unique `DiffPassId`. If a transform result should
be transformed again, resolve the graph collection that includes the previous
result before calling `linearize` again.
