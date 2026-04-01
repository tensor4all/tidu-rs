# Function Primal Split Redesign for Tidu

## Summary

The current high-level `tidu::Function<V>` API combines primal execution and
backward-state capture in one method:

- `fn forward(ctx: &mut Context<Saved>, inputs: &[&Value<V>]) -> AdResult<V>`
- `fn backward(saved: &Saved, grad_out: &V::Tangent) -> AdResult<GradInputs<V>>`

That shape is convenient for toy examples, but it has two design problems:

1. Non-AD execution still pays for backward-state preparation because `apply()`
   always calls `forward()` before it checks whether any input requires grad.
2. Real downstream libraries tend to implement the primal operation once for
   ordinary eager execution and again inside `Function::forward`, which creates
   avoidable duplication.

The approved redesign keeps `tidu` homogeneous over a single `V`, but splits
the high-level custom-op API into:

- `primal(inputs: &[&V]) -> AdResult<V>`
- `save_for_backward(inputs: &[&V], output: &V) -> AdResult<Saved>`
- `backward(saved: &Saved, grad_out: &V::Tangent) -> AdResult<GradInputs<V>>`

This lets `Function::apply()` compute the primal result once, skip
`save_for_backward()` entirely when gradients are not needed, and share the
same primal implementation between AD and non-AD call paths.

## Goals

- Remove unnecessary backward-state work from the no-grad path.
- Make the primal implementation the single source of truth for a custom op.
- Keep the public high-level API torch-like and homogeneous over one `V`.
- Preserve `tidu::expert` unchanged as the advanced low-level path.
- Keep the implementation generic over arbitrary `V: Differentiable`.

## Non-Goals

- Supporting heterogeneous custom functions such as `Vec2 -> ScalarBox`.
- Changing `Value<V>`, `backward()`, or `.grad()` semantics.
- Changing the `tidu::expert` low-level tape/rule interfaces.
- Adding new in-place, alias, or view semantics.

## Problem with the Current API

The current trait requires users to put primal work and save-for-backward
policy into one method:

```rust
fn forward(ctx: &mut Context<Saved>, inputs: &[&Value<V>]) -> AdResult<V>;
```

This creates three issues.

### 1. No-grad execution does unnecessary work

`Function::apply()` currently calls `forward()` first and only afterwards checks
whether any input requires gradients. If `forward()` clones large values or
computes auxiliary state solely for reverse mode, the no-grad path still pays
for that work.

### 2. Downstream ops duplicate primal implementations

A real library usually wants both:

- a plain eager operation on `&V`
- an AD-tracked operation through `Function::apply(&[&Value<V>])`

When the high-level AD API only exposes `forward(ctx, ...)`, downstream code
often copies the primal logic into `forward()` rather than reusing the plain
implementation.

### 3. `Context` leaks backward-save policy into primal code

`Context` makes the user think about reverse-mode save policy while writing the
primal operation. That is the wrong abstraction boundary. The clean separation
is:

- `primal()` computes the result
- `save_for_backward()` decides what reverse mode should keep
- `backward()` consumes the saved state

## Approved Direction

### New `Function<V>` shape

The public high-level custom-op trait becomes:

```rust
pub trait Function<V: Differentiable + Send + Sync + 'static>: Send + Sync + 'static {
    type Saved: Send + Sync + 'static;

    fn primal(inputs: &[&V]) -> AdResult<V>;

    fn save_for_backward(inputs: &[&V], output: &V) -> AdResult<Self::Saved>;

    fn backward(saved: &Self::Saved, grad_out: &V::Tangent) -> AdResult<GradInputs<V>>;

    fn apply(inputs: &[&Value<V>]) -> AdResult<Value<V>>
    where
        Self: Sized;
}
```

### `apply()` execution flow

`Function::apply()` should work like this:

1. Borrow input primals from `Value<V>`.
2. Compute the output with `Self::primal(&primals)`.
3. If no input requires gradients, return `Value::new(output)` immediately.
4. Otherwise call `Self::save_for_backward(&primals, &output)`.
5. Attach leaves to the common reverse graph and register the reverse rule.
6. Return a reverse-tracked `Value<V>`.

This preserves the current graph wiring behavior while fixing the no-grad path.

### Remove `Context` from the high-level public API

`Context` is no longer needed on the normal `Function` path and should be
removed from the public surface. The saved state becomes an explicit return
value of `save_for_backward()`.

This makes the high-level API easier to read and keeps save policy separate
from primal semantics.

### Keep errors as `AdResult`

`primal()` should continue returning `AdResult<V>` rather than `V` directly.
The primal computation itself may fail for domain, shape, or runtime reasons,
and the high-level API should not force users into panic-only or custom error
channels for normal operation failures.

### Keep homogeneous typing

The redesign intentionally remains `Function<V>` rather than `Function<Input,
Output>`. The goal is to improve the existing homogeneous API, not broaden it.

## Example After Redesign

```rust
impl Function<ScalarBox> for Multiply {
    type Saved = (ScalarBox, ScalarBox);

    fn primal(inputs: &[&ScalarBox]) -> AdResult<ScalarBox> {
        Ok(ScalarBox(inputs[0].0 * inputs[1].0))
    }

    fn save_for_backward(inputs: &[&ScalarBox], _output: &ScalarBox) -> AdResult<Self::Saved> {
        Ok((*inputs[0], *inputs[1]))
    }

    fn backward(saved: &Self::Saved, grad_out: &ScalarBox) -> AdResult<GradInputs<ScalarBox>> {
        let (x, y) = *saved;
        Ok(GradInputs::from(vec![
            Some(ScalarBox(grad_out.0 * y.0)),
            Some(ScalarBox(grad_out.0 * x.0)),
        ]))
    }
}
```

This gives one primal implementation that can be reused both inside and
outside autograd.

## Testing Strategy

The redesign needs focused regression coverage for the approved goal.

- Update the existing scalar and custom-type `Function` tests to the new API.
- Add a no-grad-path regression test proving `save_for_backward()` is not
  called when all inputs are detached.
- Keep the existing wrong-gradient-count validation coverage.
- Update rustdoc and README examples to the new trait shape.

## Consequences

The resulting API is slightly more explicit, but the tradeoff is worthwhile:

- no extra save work when gradients are off,
- no need to duplicate primal implementations,
- clearer separation between forward semantics and reverse-mode bookkeeping.

That is the right long-term shape for a library-facing high-level AD API.
