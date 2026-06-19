# tidu-rs

tidu builds automatic-differentiation transforms for primitive computation
graphs in Rust.

Downstream crates provide primitive operations, local AD rules, and concrete
runtimes. tidu builds new graphs for linearization, transposed linear maps, and
optional eager reverse-mode integration.

## Name

The name `tidu` comes from the Chinese word 梯度, whose pinyin romanization is
`tidu` and whose meaning is "gradient".

## Who This Is For

Read tidu docs if you are implementing a primitive operation set, AD rules, a
graph runtime, or an eager tensor frontend. If you only want tensor operations,
start with the downstream tensor/runtime crate that uses tidu.

## Documentation

Hosted documentation: <https://tensor4all.org/tidu-rs/>

Start with [How tidu Works](https://tensor4all.org/tidu-rs/docs/architecture/overview.html)
for the architecture at a glance.
