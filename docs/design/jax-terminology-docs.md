# JAX-Aligned Public Vocabulary And Documentation

Date: 2026-06-01

## Summary

`tidu` should present itself as a generic Rust crate for automatic
differentiation transforms over primitive computation graphs. It should not be
described as tensor4all-specific infrastructure, and it should not require
readers to know internal `computegraph` terminology before they can understand
the public API.

The primary audience is downstream implementers: crates that define primitive
operations, AD rules, graph runtimes, or eager tensor frontends.

The public vocabulary should align with JAX where the concepts match:
primitive, JVP rule, linearization, transpose rule, and transposed linear maps.
Implementation words such as fragment and emitter should not appear in the
main public API or in first-read documentation.

## Audience

The documentation targets downstream implementers who need to:

- define a primitive operation set,
- implement local JVP and transpose rules for those primitives,
- run graph-level AD transforms,
- integrate immediate execution with reverse-mode `backward()`, or
- understand where a runtime must provide concrete execution and metadata.

End users who only want tensor operations should normally read the downstream
tensor/runtime crate documentation instead of starting with `tidu`.

## Core Concepts

The documentation should define these terms without assuming JAX knowledge:

- **Primitive operation**: an atomic operation supplied by a downstream crate,
  such as add, multiply, sine, matrix multiplication, or a domain-specific
  extension operation.
- **Primitive computation graph**: a directed acyclic program made from inputs,
  primitive applications, and outputs. This plays a similar role to JAX's
  `jaxpr`, but the primary term in `tidu` docs should be self-contained.
- **Linearization**: a transform that builds a new graph computing a
  Jacobian-vector product for selected inputs. For `f(x) = x * x`, linearizing
  with tangent `dx` produces `dy = x * dx + dx * x`.
- **Linear transpose**: a transform that takes a linearized graph representing
  `dy = J dx` and builds a graph for cotangent propagation, `ct_x = J^T ct_y`.
- **JVP rule**: a local primitive rule that emits tangent outputs from primal
  inputs, primal outputs, and tangent inputs.
- **Transpose rule**: a local primitive rule that emits input cotangents from
  output cotangents.

The first-read docs should avoid `Fragment`, `OpEmitter`, `LocalValId`,
`ValRef`, and similar storage-level words. If those names remain necessary for
interoperability with `computegraph`, isolate them in architecture or internals
documentation and provide explicit escape hatches rather than making them the
main abstraction.

## Public API Direction

This is a breaking-change direction. Keeping multiple public vocabularies in
the crate would make the docs harder to learn.

The root graph-transform API should be documented with the current
JAX-aligned names:

- `linearize` and `try_linearize` build a linearized graph for selected inputs.
- `linear_transpose` and `try_linear_transpose` transpose a linearized graph
  for cotangent propagation.
- `LinearizedGraph` is the public wrapper for graph transform results.
- `Primitive` is the public trait implemented by downstream operation sets.

`LinearizedGraph` should not expose a public `fragment` field. It should own
the lower-level computegraph representation privately and expose use-case
oriented methods. If downstream crates need raw access, provide explicit
advanced methods such as `as_graph()` or `into_graph()` and keep those methods
out of the tutorial path.

The rule-building boundary should avoid exposing `computegraph::OpEmitter` as a
primary concept. Add a `tidu`-owned `PrimitiveBuilder` trait that local JVP and
transpose rules use to append primitive applications. Internally, `tidu` can
adapt this trait to `computegraph`.

## Eager Integration

`tidu::eager` should remain, but it should be documented as eager integration,
not as a second top-level AD mode.

The eager module is for downstream runtimes that execute primitive operations
immediately and want to expose a PyTorch-style `backward()` workflow. `tidu`
does not execute tensors, own gradient slots, infer tensor metadata, manage
devices, or define a user-facing tensor type.

Recommended eager public surface:

```rust
tidu::eager::{
    BackwardExecutor,
    EagerInput,
    EagerOutput,
    KeySource,
    Recorder,
    Trace,
    try_backward,
}
```

`Recorder`, `Trace`, `EagerInput`, and `EagerOutput` are the public eager
recording terms. `BackwardExecutor` remains acceptable because it describes
the downstream hook that performs concrete replay, transpose execution, and
cotangent addition.

The eager executor boundary should also stop exposing raw graph storage values.
Pass wrapper types instead: `PrimitiveGraph` for primal replay and
`LinearizedGraph` for transpose execution. These wrappers hide storage layout
while still giving downstream executors enough information to run primitive
applications.

## Documentation Structure

Use README as a short front door and put the learning path under `docs/`.

Recommended structure:

```text
README.md
docs/
  _quarto.yml
  index.md
  getting-started/
    index.md
    terminology.md
  tutorials/
    index.md
    primitive-linearization.md
    eager-reverse-mode.md
  guides/
    implementing-primitives.md
    linearize-and-transpose.md
    eager-integration.md
    complex-ad.md
    higher-order-ad.md
  architecture/
    index.md
    public-boundaries.md
    computegraph-integration.md
  api/
    index.md
  internals/
    index.md
```

### README

README should answer only:

- what `tidu` is,
- who should read it,
- what transforms it provides,
- where to go next.

It should avoid long derivations, internal diagrams, and tensor4all-specific
positioning.

Suggested opening:

```text
tidu builds automatic-differentiation transforms for primitive computation
graphs in Rust.

Downstream crates provide primitive operations, local AD rules, and concrete
runtimes. tidu builds new graphs for linearization, transposed linear maps, and
optional eager reverse-mode integration.
```

### Getting Started

The first getting-started page should explain primitive computation graphs and
linearization from scratch. It should include a small symbolic example before
any Rust API appears.

### Tutorials

Provide two runnable tutorial paths:

1. **Primitive linearization tutorial**: define a tiny primitive set, implement
   `Primitive`, run `linearize`, and run `linear_transpose`.
2. **Eager reverse-mode tutorial**: use `Recorder`, `Trace`,
   `BackwardExecutor`, and `try_backward` to connect immediate downstream
   execution to reverse-mode AD.

Tutorial code should live in `examples/` or test modules and be executed by CI,
so docs do not drift from the public API.

### Guides

Guides should be broader than tutorials and explain contracts:

- how primitive identity and graph keys work,
- how JVP and transpose rules compose,
- how `linearize` and `linear_transpose` build graph-level transforms,
- how eager recording relates to the graph transforms,
- complex-number conventions, and
- higher-order AD through repeated transforms.

### Architecture And Internals

Architecture docs may explain the dependency on `computegraph` and the exact
storage model. They should explicitly label `Fragment`, `OpEmitter`, and
related terms as lower-level integration details.

Internals docs should be optional. A downstream implementer should be able to
write primitives and eager integration without starting there.

## Migration Plan Shape

The implementation should be staged:

1. Add tests that describe the new names and the absence of old root-level
   names from the intended public path.
2. Rename graph transform APIs and public types.
3. Introduce `PrimitiveBuilder`, replacing `OpEmitter` in public rule
   signatures.
4. Wrap or hide computegraph fragments behind `LinearizedGraph` and any
   required primitive graph wrapper.
5. Align eager recording around `EagerInput`/`EagerOutput` and adjust executor
   signatures to avoid raw graph-storage exposure.
6. Rebuild README, rustdoc, and the Quarto docs site around the new vocabulary.
7. Migrate downstream crates such as tenferro after the `tidu` changes merge.

Because this is a pre-1.0 crate and the current vocabulary leaks implementation
details, this plan intentionally prefers one coherent breaking migration over
additive compatibility aliases.

## Concrete API Decisions To Validate

- Use `PrimitiveBuilder` as the `tidu`-owned rule-building trait name.
- Use `as_graph()` and `into_graph()` for explicit advanced access to the
  lower-level graph representation, if downstream crates still need raw
  computegraph interoperability.
- Use wrapper graph types in eager executor signatures instead of narrower
  single-operation callbacks. This keeps eager backward aligned with the graph
  transform API and avoids leaking storage-level `Fragment` names.
