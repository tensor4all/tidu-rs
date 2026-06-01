# tidu Documentation

`tidu` builds automatic-differentiation transforms for primitive computation
graphs in Rust.

Downstream crates bring the operation set, local AD rules, and concrete runtime.
`tidu` builds new graphs for linearization, transposed linear maps, and optional
eager reverse-mode integration.

## Where To Start

- Read [Terminology](getting-started/terminology.md) if the graph transform
  vocabulary is new.
- Work through [Primitive Linearization](tutorials/primitive-linearization.md)
  to define a tiny primitive set and run graph transforms.
- Work through [Eager Reverse Mode](tutorials/eager-reverse-mode.md) to connect
  immediate execution to `backward()`.
- Use the [Guides](guides/implementing-primitives.md) when implementing a real
  downstream primitive set.

## What tidu Owns

`tidu` owns the AD transform contracts. It asks downstream crates to provide
primitive operations, JVP rules, transpose rules, concrete execution, metadata,
and user-facing tensor APIs.
