# Primitive Linearization

This tutorial builds a tiny scalar example for `f(x) = x * x`, then asks
`tidu` to produce two derived primitive computation graphs:

- a linearized graph for the JVP `dy = 2 * x * dx`,
- a transposed linear graph for the cotangent `ct_x = 2 * x * ct_y`.

The complete runnable source is `examples/primitive_linearization.rs`. The
example also includes an `example_runs` test, so `cargo test --examples`
exercises the same assertions as the binary.

Run it locally with:

```bash
cargo run --example primitive_linearization
```

<!-- snippet-source: examples/primitive_linearization.rs -->

## Primitive Set

A downstream crate supplies the primitive operations. This tutorial uses scalar
addition, multiplication, negation, and exponential operations:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ScalarOp {
    Add,
    Mul,
    Neg,
    Exp,
}

impl GraphOp for ScalarOp {
    type Operand = f64;
    type Context = ();
    type InputKey = ScalarKey;

    fn n_inputs(&self) -> usize {
        match self {
            Self::Add | Self::Mul => 2,
            Self::Neg | Self::Exp => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }
}
```

The key type implements `ADKey` so `tidu` can allocate tangent keys for active
inputs:

```rust
impl ADKey for ScalarKey {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        Self::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}
```

## AD Rules

`Primitive` is where the downstream crate teaches `tidu` the local derivative
rules. The rule methods receive primal values, tangent or cotangent slots, and
a `PrimitiveBuilder` for appending primitive applications to the transformed
graph. `LocalValId` is the graph-local identifier returned by that builder for
a newly produced value.

For addition, the JVP is just the sum of active tangent inputs. For
multiplication, the rule emits `dx * y` when the left input is active and
`x * dy` when the right input is active, then sums the emitted terms:

```rust
let mut terms = Vec::new();
if let Some(dx) = tangent_inputs[0] {
    let term = builder.add_primitive(
        ScalarOp::Mul,
        vec![
            PrimitiveValue::Local(dx),
            PrimitiveValue::External(primal_inputs[1].clone()),
        ],
        OpMode::Linear {
            active_mask: vec![true, false],
        },
    );
    terms.push(term[0]);
}
if let Some(dy) = tangent_inputs[1] {
    let term = builder.add_primitive(
        ScalarOp::Mul,
        vec![
            PrimitiveValue::External(primal_inputs[0].clone()),
            PrimitiveValue::Local(dy),
        ],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    );
    terms.push(term[0]);
}
sum_tangent_terms(builder, terms)
```

The multiply transpose rule sends an output cotangent back through each active
input:

```rust
fn transpose_mul(
    builder: &mut impl PrimitiveBuilder<ScalarOp>,
    inputs: &[PrimitiveValue<ScalarOp>],
    ct: LocalValId,
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    let active_mask = match mode {
        OpMode::Linear { active_mask } => active_mask,
        OpMode::Primal => return vec![None, None],
    };

    let mut result = vec![None, None];
    if active_mask[0] {
        let out = builder.add_primitive(
            ScalarOp::Mul,
            vec![inputs[1].clone(), PrimitiveValue::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[0] = Some(out[0]);
    }
    if active_mask[1] {
        let out = builder.add_primitive(
            ScalarOp::Mul,
            vec![inputs[0].clone(), PrimitiveValue::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(out[0]);
    }
    result
}
```

## Evaluator

The example includes a small evaluator so the tutorial can assert numerical
results. It resolves the source graph plus the transformed graph, materializes
the requested outputs, compiles the result, and feeds concrete scalar inputs.
The key steps are:

```rust
let view = resolve(roots);
let graph = materialize_merge(&view, outputs);
let ordered_inputs: Vec<_> = graph
    .inputs
    .iter()
    .map(|key| {
        binding_map
            .get(key)
            .copied()
            .unwrap_or_else(|| panic!("missing value for input key {key:?}"))
    })
    .collect();
let ordered_refs: Vec<_> = ordered_inputs.iter().collect();
let program = compile(&graph);
program.eval(&mut (), &ordered_refs)
```

## Driver

The driver builds `x * x`, linearizes it with respect to `x`, and then
transposes the linearized graph:

```rust
let (primal, y_key) = build_x_squared();
let linear = linearize(
    &resolve(vec![primal.clone()]),
    std::slice::from_ref(&y_key),
    &[sk("x")],
    1,
    &mut (),
    &HashMap::new(),
);
let transposed = linear_transpose(&linear, &mut ());
```

At `x = 3` and `dx = 1.5`, the primal output is `9` and the JVP is also `9`.
With output cotangent `ct_y = 2`, the transposed graph returns `ct_x = 12`:

```rust
assert_close(primal_and_tangent[0], 9.0);
assert_close(primal_and_tangent[1], 9.0);
assert_close(cotangent[0], 12.0);
```

<!-- end-snippet-source -->
