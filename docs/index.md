# tidu Documentation

`tidu` builds automatic-differentiation transforms for primitive computation
graphs in Rust.

Downstream crates bring the operation set, local AD rules, and concrete runtime.
`tidu` builds new graphs for linearization, transposed linear maps, and optional
eager reverse-mode integration.

## Name

The name `tidu` comes from the Chinese word 梯度, whose pinyin romanization is
`tidu` and whose meaning is "gradient".

## Where To Start

- New to the graph-transform vocabulary? Read
  [Terminology](getting-started/terminology.md) first.
- Then read [How tidu Works](architecture/overview.qmd) for the whole-system
  picture: the layers, the transform pipeline, and the primitive contract.
- Work through [Primitive Linearization](tutorials/primitive-linearization.md)
  to define a tiny primitive set and run graph transforms.
- Work through [Eager Reverse Mode](tutorials/eager-reverse-mode.md) to connect
  immediate execution to `backward()`.
- Use the [Guides](guides/implementing-primitives.qmd) when implementing a real
  downstream primitive set; the [Eager Integration](guides/eager-integration.qmd)
  guide covers the `backward()` workflow. The public API lives in the
  `tidu::rules` and `tidu::eager` modules.

## What tidu Owns

`tidu` owns the AD transform contracts. It asks downstream crates to provide
primitive operations, JVP rules, transpose rules, concrete execution, metadata,
and user-facing tensor APIs.
