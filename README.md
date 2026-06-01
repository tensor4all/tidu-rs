# tidu-rs

tidu builds automatic-differentiation transforms for primitive computation
graphs in Rust.

Downstream crates provide primitive operations, local AD rules, and concrete
runtimes. tidu builds new graphs for linearization, transposed linear maps, and
optional eager reverse-mode integration.

## Who This Is For

Read tidu docs if you are implementing a primitive operation set, AD rules, a
graph runtime, or an eager tensor frontend. If you only want tensor operations,
start with the downstream tensor/runtime crate that uses tidu.

## Documentation

- [Getting started](docs/getting-started/index.md)
- [Terminology](docs/getting-started/terminology.md)
- [Tutorials](docs/tutorials/index.md)
- [Guides](docs/guides/implementing-primitives.md)
- [API map](docs/api/index.md)
