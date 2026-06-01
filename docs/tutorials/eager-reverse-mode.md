# Eager Reverse Mode

This tutorial shows how a downstream eager frontend can record immediate
primitive execution and call `tidu::eager::try_backward`.

The complete runnable source is `examples/eager_reverse_mode.rs`. The example
also includes an `example_runs` test, so `cargo test --examples` exercises the
same assertions as the binary.

Run it locally with:

```bash
cargo run --example eager_reverse_mode
```

<!-- snippet-source: examples/eager_reverse_mode.rs -->

## Primitive Set

The eager example uses the same scalar primitive set as the linearization
tutorial. Downstream crates provide operation identity, arity, concrete
execution, and AD rules:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ScalarOp {
    Add,
    Mul,
    Neg,
    Exp,
}

impl GraphOperation for ScalarOp {
    type Operand = f64;
    type Context = ();
    type InputKey = ScalarKey;

    fn input_count(&self) -> usize {
        match self {
            Self::Add | Self::Mul => 2,
            Self::Neg | Self::Exp => 1,
        }
    }

    fn output_count(&self) -> usize {
        1
    }
}

impl EvaluableGraphOperation for ScalarOp {
    fn eval(&self, _ctx: &mut (), inputs: &[&f64]) -> Vec<f64> {
        match self {
            Self::Add => vec![inputs[0] + inputs[1]],
            Self::Mul => vec![inputs[0] * inputs[1]],
            Self::Neg => vec![-inputs[0]],
            Self::Exp => vec![inputs[0].exp()],
        }
    }
}
```

## AD Rules

The eager path still relies on the same `Primitive` JVP and transpose rules as
graph linearization. During backward, `tidu` builds small linearized graphs and
asks the downstream executor to run their transposes. The multiply JVP emits
`dx * y + x * dy`, and the multiply transpose rule maps an output cotangent
back to each active input. `LocalValueId` is the graph-local identifier returned
by the builder for values created while constructing transformed primitive
computation graphs.

```rust
let mut terms = Vec::new();
if let Some(dx) = tangent_inputs[0] {
    let term = builder.add_primitive(
        ScalarOp::Mul,
        vec![
            PrimitiveValue::Local(dx),
            PrimitiveValue::External(primal_inputs[1].clone()),
        ],
        OperationRole::Linearized {
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
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    );
    terms.push(term[0]);
}
sum_tangent_terms(builder, terms)
```

## Runtime Hooks

Eager integration needs a `KeySource` for fresh internal input names, a
`PrimitiveBuilder` implementation that can execute transposed linear work, and
a `BackwardExecutor` implementation that connects `tidu` to the downstream
runtime.

The example builder stores local scalar outputs and resolves external values
from the data map supplied by `tidu`:

```rust
struct ScalarBuilder {
    locals: Vec<Arc<f64>>,
    external_data: HashMap<ValueKey<ScalarOp>, Arc<f64>>,
}

impl PrimitiveBuilder<ScalarOp> for ScalarBuilder {
    fn add_primitive(
        &mut self,
        op: ScalarOp,
        inputs: Vec<PrimitiveValue<ScalarOp>>,
        _mode: OperationRole,
    ) -> Vec<LocalValueId> {
        let values: Vec<_> = inputs
            .iter()
            .map(|input| self.resolve_input(input))
            .collect();
        let refs: Vec<_> = values.iter().map(|value| value.as_ref()).collect();
        let outputs = op.eval(&mut (), &refs);
        let start = self.locals.len();
        self.locals.extend(outputs.into_iter().map(Arc::new));
        (start..self.locals.len()).collect()
    }
}
```

`BackwardExecutor::execute_forward` replays a `PrimitiveGraph`, which is the
public wrapper `tidu` passes for forward replay. User inputs must already have
concrete values. The example only defaults synthetic tangent inputs to zero:

```rust
for &input_id in graph.inputs() {
    let key = graph.values()[input_id].key.clone();
    if values.contains_key(&key) {
        continue;
    }
    match &key {
        ValueKey::Input(ScalarKey::Tangent { .. }) => {
            values.insert(key, Arc::new(0.0));
        }
        _ => panic!("missing concrete value for graph input {key:?}"),
    }
}
```

`run_transposed_linear` receives a `LinearizedGraph` and uses the root
`try_linear_transpose_with_builder` helper to execute the transpose with the
example builder:

```rust
let mut builder = ScalarBuilder::new(external_data.clone());
let seed_ids: Vec<_> = cotangent_outputs
    .iter()
    .map(|seed| seed.as_ref().map(|value| builder.push_value(value.clone())))
    .collect();

try_linear_transpose_with_builder(linear, &mut builder, &seed_ids, ctx)
```

## Driver

The driver records the eager multiply `x * x`, then asks `tidu` to propagate an
output cotangent of `1` back to `x`:

```rust
let mut recorder = Recorder::new(ExampleKeySource::default());
let x = eager_input("x", 3.0, true);
let inputs = vec![
    EagerInput {
        key: x.key.clone(),
        trace: x.trace.clone(),
        requires_grad: x.requires_grad,
        data: x.data.clone(),
    },
    x,
];
let outputs = recorder.record(ScalarOp::Mul, &inputs, &[arc(9.0)]);

let mut executor = ScalarBackwardExecutor;
let cotangents = eager::try_backward(
    &outputs[0].key,
    outputs[0].trace.as_ref(),
    arc(1.0),
    &mut executor,
    &mut (),
)?;
```

For `x = 3`, the gradient of `x * x` is `6`:

```rust
let gradient = cotangents
    .get(&input_key("x"))
    .expect("gradient for x")
    .as_ref();
assert_close(*gradient, 6.0);
```

<!-- end-snippet-source -->
