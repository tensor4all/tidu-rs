> **Archived — superseded.** tenferro's automatic differentiation lives in
> `tenferro-ad` inside [tenferro-rs](https://github.com/tensor4all/tenferro-rs), which is the
> implementation this repository was meant to be extracted *from*. That extraction was
> considered and explicitly deferred (tenferro-rs#1695 → #974, "not now"), and the AD
> vocabulary is now owned by tenferro rather than by a shared traits crate.
> The published `tidu` crate stays installable; nothing here will receive changes.
> See tidu-rs#48 for the decision and the crates.io check.

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

## Development

Run the test suite with nextest and keep doctests on `cargo test`:

```sh
cargo nextest run --workspace --release --no-fail-fast   # unit and integration tests
cargo test --doc --workspace --release                   # doctests
cargo llvm-cov nextest --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json            # per-file thresholds
python3 -m unittest discover -s scripts/tests              # script tests
```
