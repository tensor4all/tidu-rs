# Tutorials

The tutorials are backed by runnable examples.

- [Primitive Linearization](primitive-linearization.md) defines a tiny scalar
  primitive set, then runs `linearize` and `linear_transpose`.
- [Eager Reverse Mode](eager-reverse-mode.md) records immediate scalar
  operations and runs `tidu::eager::backward`.

Run both with:

```bash
cargo run --example primitive_linearization
cargo run --example eager_reverse_mode
```
