use std::collections::HashMap;
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::resolve::resolve;
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use computegraph::{EvaluableGraphOperation, GraphOperation};
use tidu::eager::{
    self, BackwardExecutor, EagerInput, EagerOutput, KeySource, RecordedGraph, Recorder,
};
use tidu::{
    linearize, try_linear_transpose_with_builder, ADKey, DiffPassId, LinearizedGraph, Primitive,
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

fn sk(name: &str) -> ScalarKey {
    ScalarKey::User(name.to_string())
}

fn arc(value: f64) -> Arc<f64> {
    Arc::new(value)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ScalarOp {
    Add,
    Mul,
    Neg,
}

impl computegraph::GraphOperation for ScalarOp {
    type Operand = f64;
    type Context = ();
    type InputKey = ScalarKey;

    fn input_count(&self) -> usize {
        match self {
            Self::Add | Self::Mul => 2,
            Self::Neg => 1,
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
        primal_in: &[ValueKey<Self>],
        _primal_out: &[ValueKey<Self>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValueId>> {
        match self {
            Self::Add => scalar_add_tangents(builder, tangent_in),
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
                scalar_sum_terms(builder, terms)
            }
            Self::Neg => tangent_in[0].map_or_else(
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
        }
    }

    fn transpose_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[PrimitiveValue<Self>],
        role: &OperationRole,
        _ctx: &mut (),
    ) -> Vec<Option<LocalValueId>> {
        let ct = match cotangent_out[0] {
            Some(ct) => ct,
            None => return vec![None; self.input_count()],
        };

        match self {
            Self::Add => vec![Some(ct), Some(ct)],
            Self::Mul => scalar_transpose_mul(builder, inputs, ct, role),
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
        }
    }
}

fn scalar_add_tangents(
    builder: &mut impl PrimitiveBuilder<ScalarOp>,
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    let terms: Vec<_> = tangent_in.iter().filter_map(|id| *id).collect();
    scalar_sum_terms(builder, terms)
}

fn scalar_sum_terms(
    builder: &mut impl PrimitiveBuilder<ScalarOp>,
    terms: Vec<LocalValueId>,
) -> Vec<Option<LocalValueId>> {
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

fn scalar_transpose_mul(
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum TwoOutputOp {
    Add,
    Split,
}

impl computegraph::GraphOperation for TwoOutputOp {
    type Operand = f64;
    type Context = ();
    type InputKey = ScalarKey;

    fn input_count(&self) -> usize {
        match self {
            Self::Add => 2,
            Self::Split => 1,
        }
    }

    fn output_count(&self) -> usize {
        match self {
            Self::Add => 1,
            Self::Split => 2,
        }
    }
}

impl EvaluableGraphOperation for TwoOutputOp {
    fn eval(&self, _ctx: &mut (), inputs: &[&f64]) -> Vec<f64> {
        match self {
            Self::Add => vec![inputs[0] + inputs[1]],
            Self::Split => vec![*inputs[0], 2.0 * *inputs[0]],
        }
    }
}

impl Primitive for TwoOutputOp {
    type ADContext = ();

    fn add() -> Self {
        Self::Add
    }

    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        _primal_in: &[ValueKey<Self>],
        _primal_out: &[ValueKey<Self>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValueId>> {
        match self {
            Self::Add => match (tangent_in[0], tangent_in[1]) {
                (Some(lhs), Some(rhs)) => {
                    let sum = builder.add_primitive(
                        Self::Add,
                        vec![PrimitiveValue::Local(lhs), PrimitiveValue::Local(rhs)],
                        OperationRole::Linearized {
                            active_mask: vec![true, true],
                        },
                    )[0];
                    vec![Some(sum)]
                }
                (Some(lhs), None) => vec![Some(lhs)],
                (None, Some(rhs)) => vec![Some(rhs)],
                (None, None) => vec![None],
            },
            Self::Split => {
                let Some(dx) = tangent_in[0] else {
                    return vec![None, None];
                };
                let doubled = builder.add_primitive(
                    Self::Add,
                    vec![PrimitiveValue::Local(dx), PrimitiveValue::Local(dx)],
                    OperationRole::Linearized {
                        active_mask: vec![true, true],
                    },
                )[0];
                vec![Some(dx), Some(doubled)]
            }
        }
    }

    fn transpose_rule(
        &self,
        _builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_out: &[Option<LocalValueId>],
        _inputs: &[PrimitiveValue<Self>],
        _mode: &OperationRole,
        _ctx: &mut (),
    ) -> Vec<Option<LocalValueId>> {
        match self {
            Self::Add => vec![cotangent_out[0], cotangent_out[0]],
            Self::Split => panic!("Split should be linearized before linear_transpose"),
        }
    }
}

struct TwoOutputBuilder {
    locals: Vec<Arc<f64>>,
}

impl TwoOutputBuilder {
    fn push_value(&mut self, value: Arc<f64>) -> LocalValueId {
        let id = self.locals.len();
        self.locals.push(value);
        id
    }

    fn value(&self, id: LocalValueId) -> Arc<f64> {
        self.locals[id].clone()
    }
}

impl PrimitiveBuilder<TwoOutputOp> for TwoOutputBuilder {
    fn add_primitive(
        &mut self,
        op: TwoOutputOp,
        inputs: Vec<PrimitiveValue<TwoOutputOp>>,
        _mode: OperationRole,
    ) -> Vec<LocalValueId> {
        let values: Vec<Arc<f64>> = inputs
            .iter()
            .map(|input| match input {
                PrimitiveValue::Local(local_id) => self.value(*local_id),
                PrimitiveValue::External(key) => {
                    panic!("unexpected external input in two-output test: {key:?}")
                }
            })
            .collect();
        let refs: Vec<&f64> = values.iter().map(|value| value.as_ref()).collect();
        let outputs = op.eval(&mut (), &refs);
        let start = self.locals.len();
        self.locals.extend(outputs.into_iter().map(Arc::new));
        (start..self.locals.len()).collect()
    }
}

#[derive(Default)]
struct TestKeySource {
    next: usize,
}

impl TestKeySource {
    fn next_key(&mut self) -> ScalarKey {
        let key = ScalarKey::User(format!("e{}", self.next));
        self.next += 1;
        key
    }
}

impl KeySource<ScalarOp> for TestKeySource {
    fn fresh_input_key(&mut self) -> ScalarKey {
        self.next_key()
    }
}

impl KeySource<TwoOutputOp> for TestKeySource {
    fn fresh_input_key(&mut self) -> ScalarKey {
        self.next_key()
    }
}

fn scalar_input(name: &str, value: f64, requires_grad: bool) -> EagerInput<ScalarOp> {
    EagerInput {
        key: ValueKey::Input(sk(name)),
        trace: None,
        requires_grad,
        data: arc(value),
    }
}

fn scalar_input_from_output(
    output: &EagerOutput<ScalarOp>,
    value: f64,
    requires_grad: bool,
) -> EagerInput<ScalarOp> {
    EagerInput {
        key: output.key.clone(),
        trace: output.trace.clone(),
        requires_grad,
        data: arc(value),
    }
}

fn scalar_recorded_graph(
    graph: computegraph::graph::Graph<ScalarOp>,
    input_keys: Vec<ScalarKey>,
) -> RecordedGraph<ScalarOp> {
    let graph = Arc::new(graph);
    let output_keys = graph
        .outputs()
        .iter()
        .map(|output_id| graph.values()[*output_id].key.clone())
        .collect();
    RecordedGraph::new(graph, input_keys, output_keys)
}

fn record_scalar_op(
    recorder: &mut Recorder<TestKeySource>,
    op: ScalarOp,
    inputs: &[EagerInput<ScalarOp>],
    outputs: &[Arc<f64>],
) -> Vec<EagerOutput<ScalarOp>> {
    let graph_input_keys = recorder.fresh_input_keys::<ScalarOp>(inputs.len());
    let graph = RecordedGraph::from_primitive(op, graph_input_keys);
    let retained = retained_outputs(&graph, outputs);
    recorder.record_graph(graph, inputs, outputs, retained)
}

fn record_two_output_op(
    recorder: &mut Recorder<TestKeySource>,
    op: TwoOutputOp,
    inputs: &[EagerInput<TwoOutputOp>],
    outputs: &[Arc<f64>],
) -> Vec<EagerOutput<TwoOutputOp>> {
    let graph_input_keys = recorder.fresh_input_keys::<TwoOutputOp>(inputs.len());
    let graph = RecordedGraph::from_primitive(op, graph_input_keys);
    let retained = retained_outputs(&graph, outputs);
    recorder.record_graph(graph, inputs, outputs, retained)
}

fn retained_outputs<Op: computegraph::GraphOperation>(
    graph: &RecordedGraph<Op>,
    outputs: &[Arc<Op::Operand>],
) -> HashMap<ValueKey<Op>, Arc<Op::Operand>> {
    graph
        .output_keys()
        .iter()
        .cloned()
        .zip(outputs.iter().cloned())
        .collect()
}

struct PartialOutputCallbacks {
    observed_graph_outputs: usize,
    observed_seed_slots: usize,
}

impl BackwardExecutor<TwoOutputOp> for PartialOutputCallbacks {
    fn execute_forward(
        &mut self,
        graph: PrimitiveGraph<'_, TwoOutputOp>,
        initial_data: &HashMap<ValueKey<TwoOutputOp>, Arc<f64>>,
    ) -> HashMap<ValueKey<TwoOutputOp>, Arc<f64>> {
        let graph = graph.as_graph();
        self.observed_graph_outputs = graph.outputs().len();
        initial_data.clone()
    }

    fn run_transposed_linear(
        &mut self,
        linear: &LinearizedGraph<TwoOutputOp>,
        cotangent_out: &[Option<Arc<f64>>],
        _external_data: &HashMap<ValueKey<TwoOutputOp>, Arc<f64>>,
        ctx: &mut <TwoOutputOp as Primitive>::ADContext,
    ) -> tidu::ADRuleResult<Vec<Option<Arc<f64>>>> {
        self.observed_seed_slots = cotangent_out.len();
        let mut builder = TwoOutputBuilder { locals: Vec::new() };
        let cotangent_seed_ids: Vec<Option<LocalValueId>> = cotangent_out
            .iter()
            .map(|maybe_seed| {
                maybe_seed
                    .as_ref()
                    .map(|seed| builder.push_value(seed.clone()))
            })
            .collect();

        try_linear_transpose_with_builder(linear, &mut builder, &cotangent_seed_ids, ctx).map(
            |ids| {
                ids.into_iter()
                    .map(|maybe_id| maybe_id.map(|id| builder.value(id)))
                    .collect()
            },
        )
    }

    fn add_operands(&mut self, a: &Arc<f64>, b: &Arc<f64>) -> Arc<f64> {
        Arc::new(**a + **b)
    }
}

struct ScalarEagerBuilder {
    locals: Vec<Arc<f64>>,
    external_data: HashMap<ValueKey<ScalarOp>, Arc<f64>>,
}

impl ScalarEagerBuilder {
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

    fn resolve_primitive_input(&self, input: &PrimitiveValue<ScalarOp>) -> Arc<f64> {
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

impl PrimitiveBuilder<ScalarOp> for ScalarEagerBuilder {
    fn add_primitive(
        &mut self,
        op: ScalarOp,
        inputs: Vec<PrimitiveValue<ScalarOp>>,
        _mode: OperationRole,
    ) -> Vec<LocalValueId> {
        let resolved_inputs: Vec<Arc<f64>> = inputs
            .iter()
            .map(|input| self.resolve_primitive_input(input))
            .collect();
        let input_refs: Vec<&f64> = resolved_inputs.iter().map(|value| value.as_ref()).collect();
        let outputs = op.eval(&mut (), &input_refs);
        let start = self.locals.len();
        for output in outputs {
            self.locals.push(Arc::new(output));
        }
        let end = self.locals.len();
        (start..end).collect()
    }
}

struct ScalarBackwardCallbacks;

impl BackwardExecutor<ScalarOp> for ScalarBackwardCallbacks {
    fn execute_forward(
        &mut self,
        graph: PrimitiveGraph<'_, ScalarOp>,
        initial_data: &HashMap<ValueKey<ScalarOp>, Arc<f64>>,
    ) -> HashMap<ValueKey<ScalarOp>, Arc<f64>> {
        let mut all_values = initial_data.clone();
        let graph = graph.as_graph();

        for &input_id in graph.inputs() {
            let key = graph.values()[input_id].key.clone();
            all_values.entry(key).or_insert_with(|| arc(0.0));
        }

        for op_node in graph.operations() {
            let resolved_inputs: Vec<Arc<f64>> = op_node
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
                    ValueRef::External(key) => all_values.get(key).cloned().unwrap_or_else(|| {
                        panic!("missing concrete value for external key {:?}", key)
                    }),
                })
                .collect();
            let input_refs: Vec<&f64> =
                resolved_inputs.iter().map(|value| value.as_ref()).collect();
            let outputs = op_node.operation.eval(&mut (), &input_refs);

            for (output_id, output) in op_node.outputs.iter().zip(outputs) {
                let key = graph.values()[*output_id].key.clone();
                all_values.insert(key, Arc::new(output));
            }
        }

        all_values
    }

    fn run_transposed_linear(
        &mut self,
        linear: &LinearizedGraph<ScalarOp>,
        cotangent_out: &[Option<Arc<f64>>],
        external_data: &HashMap<ValueKey<ScalarOp>, Arc<f64>>,
        ctx: &mut <ScalarOp as Primitive>::ADContext,
    ) -> tidu::ADRuleResult<Vec<Option<Arc<f64>>>> {
        let mut builder = ScalarEagerBuilder::new(external_data.clone());
        let cotangent_seed_ids: Vec<Option<LocalValueId>> = cotangent_out
            .iter()
            .map(|maybe_seed| {
                maybe_seed
                    .as_ref()
                    .map(|seed| builder.push_value(seed.clone()))
            })
            .collect();

        try_linear_transpose_with_builder(linear, &mut builder, &cotangent_seed_ids, ctx).map(
            |ids| {
                ids.into_iter()
                    .map(|maybe_id| maybe_id.map(|id| builder.value(id)))
                    .collect()
            },
        )
    }

    fn add_operands(&mut self, a: &Arc<f64>, b: &Arc<f64>) -> Arc<f64> {
        Arc::new(**a + **b)
    }
}

#[test]
fn try_linear_transpose_with_builder_propagates_add_cotangents() {
    let mut builder = GraphBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let y = builder.add_input(sk("y"));
    let sum = builder.add_operation(
        ScalarOp::Add,
        vec![ValueRef::Local(x), ValueRef::Local(y)],
        OperationRole::Primary,
    );
    let sum_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sum[0]]);

    let linear = linearize(
        &resolve(vec![Arc::new(builder.build())]),
        &[sum_key],
        &[sk("x"), sk("y")],
        1,
        &mut (),
        &HashMap::new(),
    );

    let mut builder = ScalarEagerBuilder::new(HashMap::new());
    let seed = builder.push_value(arc(1.0));
    let cotangent_inputs =
        try_linear_transpose_with_builder(&linear, &mut builder, &[Some(seed)], &mut ()).unwrap();

    let values: Vec<f64> = cotangent_inputs
        .into_iter()
        .map(|maybe_id| *builder.value(maybe_id.expect("active cotangent input")))
        .collect();

    assert_eq!(values, vec![1.0, 1.0]);
}

#[test]
fn try_backward_orders_dependencies_before_output() {
    let mut recorder = Recorder::new(TestKeySource::default());
    let inputs = vec![scalar_input("a", 2.0, true), scalar_input("b", 5.0, true)];
    let add_outputs = record_scalar_op(&mut recorder, ScalarOp::Add, &inputs, &[arc(7.0)]);
    let neg_inputs = vec![scalar_input_from_output(&add_outputs[0], 7.0, true)];
    let neg_outputs = record_scalar_op(&mut recorder, ScalarOp::Neg, &neg_inputs, &[arc(-7.0)]);

    let mut callbacks = ScalarBackwardCallbacks;
    let cotangents = eager::try_backward(
        &neg_outputs[0].key,
        neg_outputs[0].trace.as_ref(),
        arc(1.0),
        &mut callbacks,
        &mut (),
    )
    .unwrap();

    assert_eq!(**cotangents.get(&ValueKey::Input(sk("a"))).unwrap(), -1.0);
    assert_eq!(**cotangents.get(&ValueKey::Input(sk("b"))).unwrap(), -1.0);
}

#[test]
fn try_backward_accumulates_x_squared_gradient() {
    let x_grad_key = ValueKey::Input(sk("x"));
    let inputs = vec![
        EagerInput {
            key: x_grad_key.clone(),
            trace: None,
            requires_grad: true,
            data: arc(3.0),
        },
        EagerInput {
            key: x_grad_key.clone(),
            trace: None,
            requires_grad: true,
            data: arc(3.0),
        },
    ];
    let mut recorder = Recorder::new(TestKeySource::default());
    let outputs = record_scalar_op(&mut recorder, ScalarOp::Mul, &inputs, &[arc(9.0)]);
    let mut callbacks = ScalarBackwardCallbacks;
    let cotangents = eager::try_backward(
        &outputs[0].key,
        outputs[0].trace.as_ref(),
        arc(1.0),
        &mut callbacks,
        &mut (),
    )
    .unwrap();

    assert_eq!(**cotangents.get(&x_grad_key).expect("gradient for x"), 6.0);
}

#[test]
fn try_backward_records_multi_op_graph_as_one_node() {
    let mut recorder = Recorder::new(TestKeySource::default());
    let graph_input_keys = recorder.fresh_input_keys::<ScalarOp>(2);
    let mut builder = GraphBuilder::<ScalarOp>::new();
    let x = builder.add_input(graph_input_keys[0].clone());
    let y = builder.add_input(graph_input_keys[1].clone());
    let product = builder.add_operation(
        ScalarOp::Mul,
        vec![ValueRef::Local(x), ValueRef::Local(y)],
        OperationRole::Primary,
    );
    let sum = builder.add_operation(
        ScalarOp::Add,
        vec![ValueRef::Local(product[0]), ValueRef::Local(y)],
        OperationRole::Primary,
    );
    builder.set_outputs(vec![sum[0]]);
    let recorded_graph = scalar_recorded_graph(builder.build(), graph_input_keys);
    let inputs = vec![scalar_input("x", 2.0, true), scalar_input("y", 3.0, true)];
    let outputs = recorder.record_graph(recorded_graph, &inputs, &[arc(9.0)], HashMap::new());

    let mut callbacks = ScalarBackwardCallbacks;
    let cotangents = eager::try_backward(
        &outputs[0].key,
        outputs[0].trace.as_ref(),
        arc(1.0),
        &mut callbacks,
        &mut (),
    )
    .unwrap();

    assert_eq!(**cotangents.get(&ValueKey::Input(sk("x"))).unwrap(), 3.0);
    assert_eq!(**cotangents.get(&ValueKey::Input(sk("y"))).unwrap(), 3.0);
}

#[test]
fn try_backward_linearizes_only_seeded_multi_output_slots() {
    let grad_key = ValueKey::Input(sk("grad_x"));
    let inputs = vec![EagerInput {
        key: grad_key.clone(),
        trace: None,
        requires_grad: true,
        data: arc(3.0),
    }];
    let mut recorder = Recorder::new(TestKeySource::default());
    let outputs = record_two_output_op(
        &mut recorder,
        TwoOutputOp::Split,
        &inputs,
        &[arc(3.0), arc(6.0)],
    );
    let mut callbacks = PartialOutputCallbacks {
        observed_graph_outputs: 0,
        observed_seed_slots: 0,
    };

    let cotangents = eager::try_backward(
        &outputs[1].key,
        outputs[1].trace.as_ref(),
        arc(1.0),
        &mut callbacks,
        &mut (),
    )
    .unwrap();

    assert_eq!(callbacks.observed_graph_outputs, 1);
    assert_eq!(callbacks.observed_seed_slots, 1);
    assert_eq!(**cotangents.get(&grad_key).expect("gradient for x"), 2.0);
}

/// Fan-out test: f(x) = x + x, df/dx = 2.
/// This exercises the cotangent accumulation (Op::add) path in `linear_transpose`.
#[test]
fn try_linear_transpose_with_builder_fan_out_accumulation() {
    // Build linearized graph for x + x: tangent(x) used twice
    let mut builder = GraphBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let sum = builder.add_operation(
        ScalarOp::Add,
        vec![ValueRef::Local(x), ValueRef::Local(x)], // x used twice (fan-out)
        OperationRole::Primary,
    );
    let sum_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sum[0]]);

    let linear = linearize(
        &resolve(vec![Arc::new(builder.build())]),
        &[sum_key],
        &[sk("x")],
        1,
        &mut (),
        &HashMap::new(),
    );

    let mut builder = ScalarEagerBuilder::new(HashMap::new());
    let seed = builder.push_value(arc(1.0));
    let cotangent_inputs =
        try_linear_transpose_with_builder(&linear, &mut builder, &[Some(seed)], &mut ()).unwrap();

    // df/dx = 2 (cotangent accumulated from two paths)
    let dx = *builder.value(cotangent_inputs[0].expect("active"));
    assert!(
        (dx - 2.0).abs() < 1e-12,
        "fan-out gradient: expected 2.0, got {dx}"
    );
}
