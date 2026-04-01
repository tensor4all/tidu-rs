# Edge-Based Public Autograd Redesign for Tidu

## Summary

`tidu-rs` currently exposes a tape-centered public API:

- users create a `Tape<V>`,
- leaves and operations are recorded onto that tape,
- gradients are read back through `NodeId`.

That model is generic and small, but it leaks graph-container policy into
downstream libraries. The most visible failure mode is mixed-tape behavior in
multi-input eager tensor operations: downstream code starts caring about when
reverse leaves are attached and which tape "owns" a composed graph.

The approved direction is to move `tidu-rs` to a `torch`-like public model:

- public API becomes edge-based and value-centered,
- `Tape` and `NodeId` disappear from the normal user surface,
- `backward()` and `.grad()` become the primary reverse-mode interface,
- custom ops use a high-level `Function`-style API,
- low-level graph/rule access is retained only under `tidu::expert`.

Internally, `tidu-rs` may still use an arena or tape-like graph store. The key
change is that graph ownership becomes an implementation detail rather than a
public semantic concept.

## Goals

- Make the public reverse-mode model `torch`-like.
- Keep `tidu-rs` generic over arbitrary `V: Differentiable`.
- Remove graph-container policy from downstream crates such as `tenferro-rs`
  and `tensor4all-rs`.
- Preserve an expert-level escape hatch for engine experimentation without
  making it the default path.
- Make future `view` / alias / version-counter / in-place semantics possible
  without another public API reset.

## Non-Goals for This Redesign

- Full `torch` parity for `view`, aliasing, version counters, and in-place
  correctness.
- Public graph-inspection APIs in the first iteration.
- Backward compatibility with the current `Tape`-first public API.
- Solving all higher-order execution/lifetime questions in the default API.

The redesign must leave room for those features, but it does not need to
implement them immediately.

## Why the Current Tape-Centered Surface Is the Wrong Boundary

`tidu` itself is a low-level engine, so an internal graph store is expected and
useful. The problem is not the existence of a tape-like runtime container. The
problem is that the public model currently says:

- a tracked value is identified by `(value, node_id, tape)`,
- operations are explicitly recorded on one `Tape`,
- downstream libraries must align graph ownership before they can compose
  eager multi-input operations.

That forces policy decisions about graph construction into upper layers.
`tenferro-rs` then needs helper APIs to attach pending reverse leaves to a
common tape before einsum / QR / SVD composition, and `tensor4all-rs` starts to
inherit those concerns. That is the wrong layering boundary.

The clean boundary is:

- `tidu-rs` owns graph construction semantics,
- `tenferro-rs` owns tensor AD surface semantics,
- `tensor4all-rs` should remain oblivious to AD graph ownership.

## Approved Direction

### 1. Public API becomes value-centered

The normal user entrypoint becomes a `Value<V>`-like handle:

```rust
let x = Value::new(2.0).requires_grad_(true);
let y = x.powf(3.0)?;
y.backward()?;
assert_eq!(x.grad()?.unwrap(), 12.0);
```

The public mental model is:

- values carry hidden reverse edges,
- leaves accumulate gradients,
- operations derive new values from input values,
- backward starts from an output value rather than from a separate tape object.

### 2. `Tape` and `NodeId` leave the main public surface

`Tape`, `TrackedValue`, explicit `record_op`, and raw `NodeId` are no longer
part of the normal `tidu` API. If retained, they move under an expert-only
module such as:

```rust
tidu::expert
```

That module is intentionally non-default:

- excluded from README quick starts,
- excluded from top-level rustdoc examples,
- documented as advanced / engine-level usage.

### 3. Keep `tidu-rs` generic over arbitrary `Differentiable` values

This redesign does **not** make `tidu` tensor-specific. The public API remains
generic over `V: Differentiable`.

Users must still be able to:

- define custom differentiable value types,
- define custom operations on those types,
- use reverse mode, forward mode, and HVP generically.

The redesign changes the graph construction model, not the type-generic nature
of the engine.

### 4. High-level custom op API becomes the primary extension mechanism

The normal extension point becomes a `Function`-style API similar in spirit to
PyTorch custom autograd functions:

```rust
struct MyOp;

impl Function<MyValue> for MyOp {
    type Saved = MySaved;

    fn forward(ctx: &mut Context<Self::Saved>, inputs: &[&Value<MyValue>])
        -> Result<Value<MyValue>>;

    fn backward(saved: &Self::Saved, grad_out: &MyValue::Tangent)
        -> Result<GradInputs<MyValue>>;

    fn jvp(saved: &Self::Saved, tangents: TangentInputs<MyValue>)
        -> Result<MyValue::Tangent>;
}
```

This keeps the common path high-level while still supporting generic custom
types.

### 5. Low-level rule APIs remain available only for experts

The existing `ReverseRule` / `ForwardRule` style APIs should remain available
for expert users because they still matter for:

- engine experimentation,
- two-phase recording,
- nonstandard graph construction,
- low-level performance work.

However, they move behind `tidu::expert`, and the main docs should treat them
as advanced APIs rather than the default programming model.

### 6. Reverse-mode primary API is `backward()` plus `.grad()`

The main reverse-mode path should look `torch`-like:

- mark leaves with `requires_grad_(true)`,
- compute outputs eagerly,
- call `loss.backward()`,
- inspect gradients with `.grad()`.

For non-scalar outputs, explicit cotangent-seeded methods may still exist, but
the main docs should lead with scalar-loss `backward()`.

### 7. HVP remains public but advanced

HVP should remain a first-class public capability, but not the first thing a
new user sees. The approved shape is:

- value-oriented HVP such as `loss.hvp(&[(&x, vx), (&y, vy)])`,
- optional functional helpers under `tidu::autograd::functional`,
- no need to expose tape identity to compute higher-order derivatives.

### 8. Graph lifetime policy should stay out of the default API

If `retain_graph`-like control is eventually needed, it should live in an
expert-facing or explicit-options path. The default high-level API should not
force ordinary users to think about graph container lifetime.

This keeps the primary public model closer to mathematics than to runtime
bookkeeping.

### 9. Future `view` / alias / in-place semantics must remain possible

This redesign does not implement those semantics yet, but it must avoid
blocking them. In particular:

- `Value<V>` should be able to grow hidden metadata beyond just primal + grad,
- internal graph nodes should be capable of storing alias/view provenance,
- mutation/version tracking should be addable without another public reset.

## Public API Sketch

The shape below is intentionally illustrative rather than final:

```rust
pub struct Value<V: Differentiable> {
    // public API does not expose graph internals
}

impl<V: Differentiable> Value<V> {
    pub fn new(primal: V) -> Self;
    pub fn primal(&self) -> &V;
    pub fn requires_grad(&self) -> bool;
    pub fn requires_grad_(self, enabled: bool) -> Self;
    pub fn grad(&self) -> Result<Option<V::Tangent>>;
    pub fn zero_grad(&self) -> Result<()>;
    pub fn backward(&self) -> Result<()>;
    pub fn backward_with_seed(&self, seed: V::Tangent) -> Result<()>;
    pub fn hvp(
        &self,
        tangents: &[(&Self, V::Tangent)],
    ) -> Result<Vec<Option<V::Tangent>>>;
}
```

The important semantic shift is that none of these methods mention `Tape` or
`NodeId`.

## Internal Runtime Shape

The internal implementation can still use a graph store very similar to the
current tape model. A plausible direction is:

```rust
struct GraphStore<V: Differentiable> {
    nodes: Vec<Node<V>>,
}

struct Node<V: Differentiable> {
    kind: NodeKind<V>,
    inputs: Vec<EdgeHandle>,
}

enum NodeKind<V: Differentiable> {
    Leaf(LeafState<V>),
    Function(FunctionState<V>),
    View(ViewState<V>),      // future
    InPlace(InPlaceState<V>),// future
}
```

This keeps runtime implementation freedom while ensuring public semantics stay
edge-based.

## Expert API Direction

The `expert` module may continue exposing:

- raw rule traits,
- node handles,
- explicit graph recording,
- graph debugging / inspection helpers,
- advanced saved-state policies.

That module is intentionally not the migration target for normal downstream
libraries. `tenferro-rs` should migrate to the new high-level `Value` /
`Function` model for ordinary eager tensor ops.

## Downstream Implications

### `tenferro-rs`

After this redesign, `tenferro-rs` should:

- stop owning pending-leaf common-tape policy,
- express eager tensor ops in terms of `Value<Tensor>`-style composition,
- build tensor-specific ergonomic wrappers on top of `tidu`'s new public
  model,
- keep any truly engine-level work inside `tidu::expert` only when unavoidable.

### `tensor4all-rs`

`tensor4all-rs` should only need:

- dependency updates,
- integration test adjustments,
- no AD-graph policy logic of its own.

That is the desired end state for the stack.

## Migration Strategy

The approved migration order is:

1. redesign and implement `tidu-rs` locally,
2. without opening a PR yet, move directly to `tenferro-rs`,
3. redesign/adapt `tenferro-rs` to the new `tidu` surface,
4. then update `tensor4all-rs`,
5. only after the full local stack is coherent should PRs be split and opened.

This keeps cross-repo churn local until the new boundary is proven to work.

## Testing Strategy

The redesign is only acceptable if tests cover:

- `Value::backward()` on scalar losses,
- `.grad()` accumulation and zeroing,
- custom `Function` definitions over non-tensor value types,
- HVP through the new value-oriented API,
- movement of old tape/rule APIs behind `tidu::expert`,
- rustdoc / README examples using the high-level path only.

The first implementation should rely on small scalar examples to validate the
API model before adapting tensor-heavy downstream users.
