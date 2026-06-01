use std::collections::HashMap;
use std::sync::Arc;

use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use computegraph::{EvaluableGraphOperation, GraphOperation};
use tidu::eager::{self, BackwardExecutor, EagerInput, KeySource, Recorder};
use tidu::{
    try_linear_transpose_with_builder, ADKey, ADRuleResult, DiffPassId, LinearizedGraph, Primitive,
    PrimitiveBuilder, PrimitiveGraph, PrimitiveValue,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ScalarKey {
    User(String),
    Tangent {
        of: Box<ScalarKey>,
        pass: DiffPassId,
    },
}

impl ADKey for ScalarKey {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        Self::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}

#[allow(dead_code)]
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

impl Primitive for ScalarOp {
    type ADContext = ();

    fn add() -> Self {
        Self::Add
    }

    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        primal_inputs: &[ValueKey<Self>],
        primal_outputs: &[ValueKey<Self>],
        tangent_inputs: &[Option<LocalValueId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValueId>> {
        match self {
            Self::Add => sum_tangent_terms(builder, tangent_inputs.iter().filter_map(|id| *id)),
            Self::Mul => {
                let mut terms = Vec::new();
                if let Some(dx) = tangent_inputs[0] {
                    let term = builder.add_primitive(
                        Self::Mul,
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
                        Self::Mul,
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
            }
            Self::Neg => tangent_inputs[0].map_or_else(
                || vec![None],
                |dx| {
                    let out = builder.add_primitive(
                        Self::Neg,
                        vec![PrimitiveValue::Local(dx)],
                        OperationRole::Linearized {
                            active_mask: vec![true],
                        },
                    );
                    vec![Some(out[0])]
                },
            ),
            Self::Exp => {
                if let Some(dx) = tangent_inputs[0] {
                    let out = builder.add_primitive(
                        Self::Mul,
                        vec![
                            PrimitiveValue::External(primal_outputs[0].clone()),
                            PrimitiveValue::Local(dx),
                        ],
                        OperationRole::Linearized {
                            active_mask: vec![false, true],
                        },
                    );
                    vec![Some(out[0])]
                } else {
                    vec![None]
                }
            }
        }
    }

    fn transpose_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_outputs: &[Option<LocalValueId>],
        inputs: &[PrimitiveValue<Self>],
        role: &OperationRole,
        _ctx: &mut (),
    ) -> Vec<Option<LocalValueId>> {
        let Some(ct) = cotangent_outputs[0] else {
            return vec![None; self.input_count()];
        };

        match self {
            Self::Add => vec![Some(ct), Some(ct)],
            Self::Mul => transpose_mul(builder, inputs, ct, role),
            Self::Neg => {
                let out = builder.add_primitive(
                    Self::Neg,
                    vec![PrimitiveValue::Local(ct)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                vec![Some(out[0])]
            }
            Self::Exp => panic!("Exp should be linearized before linear_transpose"),
        }
    }
}

#[derive(Default)]
struct ExampleKeySource {
    next: usize,
}

impl KeySource<ScalarOp> for ExampleKeySource {
    fn fresh_input_key(&mut self) -> ScalarKey {
        let key = ScalarKey::User(format!("e{}", self.next));
        self.next += 1;
        key
    }
}

struct ScalarBuilder {
    locals: Vec<Arc<f64>>,
    external_data: HashMap<ValueKey<ScalarOp>, Arc<f64>>,
}

impl ScalarBuilder {
    fn new(external_data: HashMap<ValueKey<ScalarOp>, Arc<f64>>) -> Self {
        Self {
            locals: Vec::new(),
            external_data,
        }
    }

    fn push_value(&mut self, value: Arc<f64>) -> LocalValueId {
        let id = self.locals.len();
        self.locals.push(value);
        id
    }

    fn value(&self, id: LocalValueId) -> Arc<f64> {
        self.locals[id].clone()
    }

    fn resolve_input(&self, input: &PrimitiveValue<ScalarOp>) -> Arc<f64> {
        match input {
            PrimitiveValue::Local(local_id) => self.value(*local_id),
            PrimitiveValue::External(key) => self
                .external_data
                .get(key)
                .cloned()
                .unwrap_or_else(|| panic!("missing eager value for {key:?}")),
        }
    }
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

struct ScalarBackwardExecutor;

impl BackwardExecutor<ScalarOp> for ScalarBackwardExecutor {
    fn execute_forward(
        &mut self,
        graph: PrimitiveGraph<'_, ScalarOp>,
        initial_data: &HashMap<ValueKey<ScalarOp>, Arc<f64>>,
    ) -> HashMap<ValueKey<ScalarOp>, Arc<f64>> {
        let mut values = initial_data.clone();
        let graph = graph.as_graph();

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

        for op_node in graph.operations() {
            let inputs: Vec<Arc<f64>> = op_node
                .inputs
                .iter()
                .map(|input| match input {
                    ValueRef::Local(local_id) => values
                        .get(&graph.values()[*local_id].key)
                        .cloned()
                        .unwrap_or_else(|| {
                            panic!(
                                "missing concrete value for {:?}",
                                graph.values()[*local_id].key
                            )
                        }),
                    ValueRef::External(key) => values
                        .get(key)
                        .cloned()
                        .unwrap_or_else(|| panic!("missing concrete value for {key:?}")),
                })
                .collect();
            let refs: Vec<_> = inputs.iter().map(|value| value.as_ref()).collect();
            let outputs = op_node.operation.eval(&mut (), &refs);

            for (output_id, output) in op_node.outputs.iter().zip(outputs) {
                values.insert(graph.values()[*output_id].key.clone(), Arc::new(output));
            }
        }

        values
    }

    fn run_transposed_linear(
        &mut self,
        linear: &LinearizedGraph<ScalarOp>,
        cotangent_outputs: &[Option<Arc<f64>>],
        external_data: &HashMap<ValueKey<ScalarOp>, Arc<f64>>,
        ctx: &mut (),
    ) -> ADRuleResult<Vec<Option<Arc<f64>>>> {
        let mut builder = ScalarBuilder::new(external_data.clone());
        let seed_ids: Vec<_> = cotangent_outputs
            .iter()
            .map(|seed| seed.as_ref().map(|value| builder.push_value(value.clone())))
            .collect();

        try_linear_transpose_with_builder(linear, &mut builder, &seed_ids, ctx).map(|ids| {
            ids.into_iter()
                .map(|id| id.map(|local_id| builder.value(local_id)))
                .collect()
        })
    }

    fn add_operands(&mut self, a: &Arc<f64>, b: &Arc<f64>) -> Arc<f64> {
        Arc::new(**a + **b)
    }
}

fn sk(name: &str) -> ScalarKey {
    ScalarKey::User(name.to_string())
}

fn input_key(name: &str) -> ValueKey<ScalarOp> {
    ValueKey::Input(sk(name))
}

fn arc(value: f64) -> Arc<f64> {
    Arc::new(value)
}

fn eager_input(name: &str, value: f64, requires_grad: bool) -> EagerInput<ScalarOp> {
    EagerInput {
        key: input_key(name),
        trace: None,
        requires_grad,
        data: arc(value),
    }
}

fn sum_tangent_terms(
    builder: &mut impl PrimitiveBuilder<ScalarOp>,
    terms: impl IntoIterator<Item = LocalValueId>,
) -> Vec<Option<LocalValueId>> {
    let terms: Vec<_> = terms.into_iter().collect();
    match terms.as_slice() {
        [] => vec![None],
        [only] => vec![Some(*only)],
        [first, rest @ ..] => {
            let mut acc = *first;
            for term in rest {
                let out = builder.add_primitive(
                    ScalarOp::Add,
                    vec![PrimitiveValue::Local(acc), PrimitiveValue::Local(*term)],
                    OperationRole::Linearized {
                        active_mask: vec![true, true],
                    },
                );
                acc = out[0];
            }
            vec![Some(acc)]
        }
    }
}

fn transpose_mul(
    builder: &mut impl PrimitiveBuilder<ScalarOp>,
    inputs: &[PrimitiveValue<ScalarOp>],
    ct: LocalValueId,
    role: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    let active_mask = match role {
        OperationRole::Linearized { active_mask } => active_mask,
        OperationRole::Primary => return vec![None, None],
    };
    let mut result = vec![None, None];
    if active_mask[0] {
        let out = builder.add_primitive(
            ScalarOp::Mul,
            vec![inputs[1].clone(), PrimitiveValue::Local(ct)],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        );
        result[0] = Some(out[0]);
    }
    if active_mask[1] {
        let out = builder.add_primitive(
            ScalarOp::Mul,
            vec![inputs[0].clone(), PrimitiveValue::Local(ct)],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(out[0]);
    }
    result
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, got {actual}"
    );
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
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

    let gradient = cotangents
        .get(&input_key("x"))
        .expect("gradient for x")
        .as_ref();
    assert_close(*gradient, 6.0);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}

#[test]
fn example_runs() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
