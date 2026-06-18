use std::collections::HashMap;
use std::sync::Arc;

use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use computegraph::{EvaluableGraphOperation, GraphOperation};
use tidu::eager::{
    self, BackwardExecutor, EagerInput, EagerRecordError, KeySource, RecordedGraph, Recorder,
};
use tidu::{
    linear_transpose_with_builder, ADKey, DiffPassId, LinearizedGraph, Primitive, PrimitiveBuilder,
    PrimitiveGraph, PrimitiveValue,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Key {
    User(String),
    Tangent { of: Box<Key>, pass: DiffPassId },
}

impl ADKey for Key {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        Key::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum RecorderOp {
    Add,
    Mul,
    Neg,
    Split,
    Sum3,
}

impl GraphOperation for RecorderOp {
    type Operand = f64;
    type Context = ();
    type InputKey = Key;

    fn input_count(&self) -> usize {
        match self {
            Self::Add | Self::Mul => 2,
            Self::Neg | Self::Split => 1,
            Self::Sum3 => 3,
        }
    }

    fn output_count(&self) -> usize {
        match self {
            Self::Split => 2,
            Self::Add | Self::Mul | Self::Neg | Self::Sum3 => 1,
        }
    }
}

impl EvaluableGraphOperation for RecorderOp {
    fn eval(&self, _ctx: &mut (), inputs: &[&f64]) -> Vec<f64> {
        match self {
            Self::Add => vec![inputs[0] + inputs[1]],
            Self::Mul => vec![inputs[0] * inputs[1]],
            Self::Neg => vec![-inputs[0]],
            Self::Split => vec![*inputs[0], -*inputs[0]],
            Self::Sum3 => vec![inputs[0] + inputs[1] + inputs[2]],
        }
    }
}

impl Primitive for RecorderOp {
    type ADContext = ();

    fn add() -> Self {
        Self::Add
    }

    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        primal_in: &[ValueKey<Self>],
        _primal_out: &[ValueKey<Self>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut (),
    ) -> tidu::ADRuleResult<Vec<Option<LocalValueId>>> {
        match self {
            Self::Add => Ok(add_tangents(builder, tangent_in)),
            Self::Mul => {
                let mut terms = Vec::new();
                if let Some(dx) = tangent_in[0] {
                    let term = builder.add_primitive(
                        Self::Mul,
                        vec![
                            PrimitiveValue::Local(dx),
                            PrimitiveValue::External(primal_in[1].clone()),
                        ],
                        OperationRole::Linearized {
                            active_mask: vec![true, false],
                        },
                    );
                    terms.push(term[0]);
                }
                if let Some(dy) = tangent_in[1] {
                    let term = builder.add_primitive(
                        Self::Mul,
                        vec![
                            PrimitiveValue::External(primal_in[0].clone()),
                            PrimitiveValue::Local(dy),
                        ],
                        OperationRole::Linearized {
                            active_mask: vec![false, true],
                        },
                    );
                    terms.push(term[0]);
                }
                Ok(sum_terms(builder, terms))
            }
            Self::Neg => Ok(tangent_in[0].map_or_else(
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
            )),
            Self::Split => {
                let Some(dx) = tangent_in[0] else {
                    return Ok(vec![None, None]);
                };
                let neg = builder.add_primitive(
                    Self::Neg,
                    vec![PrimitiveValue::Local(dx)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                Ok(vec![Some(dx), Some(neg[0])])
            }
            Self::Sum3 => Ok(add_tangents(builder, tangent_in)),
        }
    }

    fn transpose_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[PrimitiveValue<Self>],
        role: &OperationRole,
        _ctx: &mut (),
    ) -> tidu::ADRuleResult<Vec<Option<LocalValueId>>> {
        let ct = match cotangent_out[0] {
            Some(ct) => ct,
            None => return Ok(vec![None; self.input_count()]),
        };

        match self {
            Self::Add => Ok(vec![Some(ct), Some(ct)]),
            Self::Mul => Ok(transpose_mul(builder, inputs, ct, role)),
            Self::Neg => {
                let out = builder.add_primitive(
                    Self::Neg,
                    vec![PrimitiveValue::Local(ct)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                Ok(vec![Some(out[0])])
            }
            Self::Split => panic!("Split is linearized before linear_transpose"),
            Self::Sum3 => Ok(vec![Some(ct), Some(ct), Some(ct)]),
        }
    }
}

fn add_tangents(
    builder: &mut impl PrimitiveBuilder<RecorderOp>,
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    let terms: Vec<_> = tangent_in.iter().filter_map(|id| *id).collect();
    sum_terms(builder, terms)
}

fn sum_terms(
    builder: &mut impl PrimitiveBuilder<RecorderOp>,
    terms: Vec<LocalValueId>,
) -> Vec<Option<LocalValueId>> {
    match terms.as_slice() {
        [] => vec![None],
        [only] => vec![Some(*only)],
        [first, rest @ ..] => {
            let mut acc = *first;
            for term in rest {
                let out = builder.add_primitive(
                    RecorderOp::Add,
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
    builder: &mut impl PrimitiveBuilder<RecorderOp>,
    inputs: &[PrimitiveValue<RecorderOp>],
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
            RecorderOp::Mul,
            vec![inputs[1].clone(), PrimitiveValue::Local(ct)],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        );
        result[0] = Some(out[0]);
    }
    if active_mask[1] {
        let out = builder.add_primitive(
            RecorderOp::Mul,
            vec![inputs[0].clone(), PrimitiveValue::Local(ct)],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(out[0]);
    }
    result
}

#[derive(Default)]
struct TestKeySource {
    next: usize,
}

impl KeySource<RecorderOp> for TestKeySource {
    fn fresh_input_key(&mut self) -> Key {
        let key = Key::User(format!("e{}", self.next));
        self.next += 1;
        key
    }
}

fn key(name: &str) -> ValueKey<RecorderOp> {
    ValueKey::Input(Key::User(name.to_string()))
}

fn eager_value(name: &str, value: f64, requires_grad: bool) -> EagerInput<RecorderOp> {
    EagerInput {
        key: key(name),
        trace: None,
        requires_grad,
        data: Arc::new(value),
    }
}

fn record_op(
    recorder: &mut Recorder<TestKeySource>,
    op: RecorderOp,
    inputs: &[EagerInput<RecorderOp>],
    outputs: &[Arc<f64>],
) -> Vec<tidu::eager::EagerOutput<RecorderOp>> {
    let graph_input_keys = recorder.fresh_input_keys::<RecorderOp>(inputs.len());
    let graph = RecordedGraph::from_primitive(op, graph_input_keys)
        .expect("test primitive graph metadata should be valid");
    let retained = graph
        .output_keys()
        .iter()
        .cloned()
        .zip(outputs.iter().cloned())
        .collect();
    recorder
        .record_graph(graph, inputs, outputs, retained)
        .expect("test eager recording metadata should be valid")
}

struct EagerBuilder {
    locals: Vec<Arc<f64>>,
    external_data: HashMap<ValueKey<RecorderOp>, Arc<f64>>,
}

impl EagerBuilder {
    fn new(external_data: HashMap<ValueKey<RecorderOp>, Arc<f64>>) -> Self {
        Self {
            locals: Vec::new(),
            external_data,
        }
    }

    fn push(&mut self, value: Arc<f64>) -> LocalValueId {
        let id = self.locals.len();
        self.locals.push(value);
        id
    }

    fn value(&self, id: LocalValueId) -> Arc<f64> {
        self.locals[id].clone()
    }

    fn resolve_primitive_input(&self, input: &PrimitiveValue<RecorderOp>) -> Arc<f64> {
        match input {
            PrimitiveValue::Local(local_id) => self.value(*local_id),
            PrimitiveValue::External(key) => self
                .external_data
                .get(key)
                .cloned()
                .unwrap_or_else(|| panic!("missing eager value for {:?}", key)),
        }
    }
}

impl PrimitiveBuilder<RecorderOp> for EagerBuilder {
    fn add_primitive(
        &mut self,
        op: RecorderOp,
        inputs: Vec<PrimitiveValue<RecorderOp>>,
        _mode: OperationRole,
    ) -> Vec<LocalValueId> {
        let resolved: Vec<_> = inputs
            .iter()
            .map(|input| self.resolve_primitive_input(input))
            .collect();
        let refs: Vec<_> = resolved.iter().map(|value| value.as_ref()).collect();
        let outputs = op.eval(&mut (), &refs);
        let start = self.locals.len();
        self.locals.extend(outputs.into_iter().map(Arc::new));
        (start..self.locals.len()).collect()
    }
}

#[derive(Default)]
struct Callbacks {
    last_initial_data: HashMap<ValueKey<RecorderOp>, Arc<f64>>,
}

impl BackwardExecutor<RecorderOp> for Callbacks {
    fn execute_forward(
        &mut self,
        graph: PrimitiveGraph<'_, RecorderOp>,
        initial_data: &HashMap<ValueKey<RecorderOp>, Arc<f64>>,
    ) -> HashMap<ValueKey<RecorderOp>, Arc<f64>> {
        self.last_initial_data = initial_data.clone();
        let mut all_values = initial_data.clone();
        let graph = graph.as_graph();

        for &input_id in graph.inputs() {
            let key = graph.values()[input_id].key.clone();
            all_values.entry(key).or_insert_with(|| Arc::new(0.0));
        }

        for op_node in graph.operations() {
            let resolved: Vec<_> = op_node
                .inputs
                .iter()
                .map(|input| match input {
                    ValueRef::Local(local_id) => all_values
                        .get(&graph.values()[*local_id].key)
                        .cloned()
                        .unwrap_or_else(|| {
                            panic!(
                                "missing concrete value for local key {:?}",
                                graph.values()[*local_id].key
                            )
                        }),
                    ValueRef::External(key) => all_values
                        .get(key)
                        .cloned()
                        .unwrap_or_else(|| panic!("missing concrete value for {:?}", key)),
                })
                .collect();
            let refs: Vec<_> = resolved.iter().map(|value| value.as_ref()).collect();
            let outputs = op_node.operation.eval(&mut (), &refs);
            for (output_id, output) in op_node.outputs.iter().zip(outputs) {
                all_values.insert(graph.values()[*output_id].key.clone(), Arc::new(output));
            }
        }

        all_values
    }

    fn run_transposed_linear(
        &mut self,
        linear: &LinearizedGraph<RecorderOp>,
        cotangent_out: &[Option<Arc<f64>>],
        external_data: &HashMap<ValueKey<RecorderOp>, Arc<f64>>,
        ctx: &mut (),
    ) -> tidu::ADRuleResult<Vec<Option<Arc<f64>>>> {
        let mut builder = EagerBuilder::new(external_data.clone());
        let seed_ids: Vec<_> = cotangent_out
            .iter()
            .map(|seed| seed.as_ref().map(|value| builder.push(value.clone())))
            .collect();
        linear_transpose_with_builder(linear, &mut builder, &seed_ids, ctx).map(|ids| {
            ids.into_iter()
                .map(|id| id.map(|id| builder.value(id)))
                .collect()
        })
    }

    fn add_operands(&mut self, a: &Arc<f64>, b: &Arc<f64>) -> Arc<f64> {
        Arc::new(**a + **b)
    }
}

#[test]
fn record_eager_binary_op_and_skips_inactive_input_cotangents() {
    let inputs = vec![eager_value("x", 3.0, true), eager_value("y", 4.0, false)];
    let mut recorder = Recorder::new(TestKeySource::default());
    let outputs = record_op(&mut recorder, RecorderOp::Mul, &inputs, &[Arc::new(12.0)]);
    let mut callbacks = Callbacks::default();
    let cotangents = eager::backward(
        &outputs[0].key,
        outputs[0].trace.as_ref(),
        Arc::new(1.0),
        &mut callbacks,
        &mut (),
    )
    .unwrap();

    assert_eq!(**cotangents.get(&key("x")).expect("cotangent for x"), 4.0);
    assert!(
        !cotangents.contains_key(&key("y")),
        "inactive input must not receive a cotangent"
    );
}

#[test]
fn recorded_graph_new_reports_mismatched_input_key_count() {
    let mut builder = computegraph::graph::GraphBuilder::<RecorderOp>::new();
    let x = builder.add_input(Key::User("x".into()));
    let y = builder.add_input(Key::User("y".into()));
    let out = builder.add_operation(
        RecorderOp::Add,
        vec![ValueRef::Local(x), ValueRef::Local(y)],
        OperationRole::Primary,
    );
    builder.set_outputs(out.clone());
    let graph = Arc::new(builder.build());
    let output_keys = out
        .iter()
        .map(|output_id| graph.values()[*output_id].key.clone())
        .collect();

    let err = match RecordedGraph::new(graph, vec![Key::User("x".into())], output_keys) {
        Ok(_) => panic!("RecordedGraph::new unexpectedly accepted mismatched input keys"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        EagerRecordError::CountMismatch {
            field: "RecordedGraph input keys",
            expected: 2,
            actual: 1
        }
    ));
}

#[test]
fn record_graph_reports_mismatched_input_count() {
    let mut recorder = Recorder::new(TestKeySource::default());
    let graph_inputs = recorder.fresh_input_keys::<RecorderOp>(1);
    let graph = RecordedGraph::from_primitive(RecorderOp::Neg, graph_inputs)
        .expect("test primitive graph metadata should be valid");

    let err = match recorder.record_graph(graph, &[], &[Arc::new(-1.0)], HashMap::new()) {
        Ok(_) => panic!("Recorder::record_graph unexpectedly accepted mismatched inputs"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        EagerRecordError::CountMismatch {
            field: "Recorder::record_graph inputs",
            expected: 1,
            actual: 0
        }
    ));
}

#[test]
fn record_eager_nary_op_builds_all_input_edges() {
    let inputs = vec![
        eager_value("a", 1.0, true),
        eager_value("b", 2.0, false),
        eager_value("c", 3.0, true),
    ];
    let mut recorder = Recorder::new(TestKeySource::default());
    let outputs = record_op(&mut recorder, RecorderOp::Sum3, &inputs, &[Arc::new(6.0)]);
    let mut callbacks = Callbacks::default();
    let cotangents = eager::backward(
        &outputs[0].key,
        outputs[0].trace.as_ref(),
        Arc::new(1.0),
        &mut callbacks,
        &mut (),
    )
    .unwrap();

    assert_eq!(**cotangents.get(&key("a")).expect("cotangent for a"), 1.0);
    assert!(
        !cotangents.contains_key(&key("b")),
        "inactive input must not receive a cotangent"
    );
    assert_eq!(**cotangents.get(&key("c")).expect("cotangent for c"), 1.0);
}

#[test]
fn record_eager_multi_output_op_uses_one_node_and_seeds_nonzero_slot() {
    let inputs = vec![eager_value("x", 3.0, true)];
    let mut recorder = Recorder::new(TestKeySource::default());
    let outputs = record_op(
        &mut recorder,
        RecorderOp::Split,
        &inputs,
        &[Arc::new(3.0), Arc::new(-3.0)],
    );

    assert_eq!(outputs.len(), 2);
    assert!(outputs[0].trace.is_some());
    assert!(outputs[1].trace.is_some());

    let mut callbacks = Callbacks::default();
    let cotangents = eager::backward(
        &outputs[1].key,
        outputs[1].trace.as_ref(),
        Arc::new(1.0),
        &mut callbacks,
        &mut (),
    )
    .unwrap();

    assert_eq!(**cotangents.get(&key("x")).expect("cotangent for x"), -1.0);
    assert!(
        callbacks
            .last_initial_data
            .iter()
            .any(|(saved_key, value)| matches!(
                saved_key,
                ValueKey::Derived { output_slot: 1, .. } if **value == -3.0
            )),
        "saved forward data should include the second derived output"
    );
    assert!(
        callbacks
            .last_initial_data
            .iter()
            .any(|(saved_key, value)| matches!(
                saved_key,
                ValueKey::Input(_) if **value == 3.0
            )),
        "saved forward data should include the input alias value"
    );
}
