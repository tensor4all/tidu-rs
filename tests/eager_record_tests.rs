use std::collections::HashMap;
use std::sync::Arc;

use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::{EvalGraphOp, GraphOp, OpEmitter};
use tidu::eager::{self, BackwardExecutor, Input, KeySource as EagerKeySource, Recorder};
use tidu::emit;
use tidu::{ADKey, DiffPassId, PrimitiveOp};

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

impl GraphOp for RecorderOp {
    type Operand = f64;
    type Context = ();
    type InputKey = Key;

    fn n_inputs(&self) -> usize {
        match self {
            Self::Add | Self::Mul => 2,
            Self::Neg | Self::Split => 1,
            Self::Sum3 => 3,
        }
    }

    fn n_outputs(&self) -> usize {
        match self {
            Self::Split => 2,
            Self::Add | Self::Mul | Self::Neg | Self::Sum3 => 1,
        }
    }
}

impl EvalGraphOp for RecorderOp {
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

impl PrimitiveOp for RecorderOp {
    type ADContext = ();

    fn add() -> Self {
        Self::Add
    }

    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        match self {
            Self::Add => add_tangents(builder, tangent_in),
            Self::Mul => {
                let mut terms = Vec::new();
                if let Some(dx) = tangent_in[0] {
                    let term = builder.add_op(
                        Self::Mul,
                        vec![ValRef::Local(dx), ValRef::External(primal_in[1].clone())],
                        OpMode::Linear {
                            active_mask: vec![true, false],
                        },
                    );
                    terms.push(term[0]);
                }
                if let Some(dy) = tangent_in[1] {
                    let term = builder.add_op(
                        Self::Mul,
                        vec![ValRef::External(primal_in[0].clone()), ValRef::Local(dy)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    terms.push(term[0]);
                }
                sum_terms(builder, terms)
            }
            Self::Neg => tangent_in[0].map_or_else(
                || vec![None],
                |dx| {
                    let out = builder.add_op(
                        Self::Neg,
                        vec![ValRef::Local(dx)],
                        OpMode::Linear {
                            active_mask: vec![true],
                        },
                    );
                    vec![Some(out[0])]
                },
            ),
            Self::Split => {
                let Some(dx) = tangent_in[0] else {
                    return vec![None, None];
                };
                let neg = builder.add_op(
                    Self::Neg,
                    vec![ValRef::Local(dx)],
                    OpMode::Linear {
                        active_mask: vec![true],
                    },
                );
                vec![Some(dx), Some(neg[0])]
            }
            Self::Sum3 => add_tangents(builder, tangent_in),
        }
    }

    fn transpose_rule(
        &self,
        emitter: &mut impl OpEmitter<Self>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<Self>],
        mode: &OpMode,
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        let ct = match cotangent_out[0] {
            Some(ct) => ct,
            None => return vec![None; self.n_inputs()],
        };

        match self {
            Self::Add => vec![Some(ct), Some(ct)],
            Self::Mul => transpose_mul(emitter, inputs, ct, mode),
            Self::Neg => {
                let out = emitter.add_op(
                    Self::Neg,
                    vec![ValRef::Local(ct)],
                    OpMode::Linear {
                        active_mask: vec![true],
                    },
                );
                vec![Some(out[0])]
            }
            Self::Split => panic!("Split is linearized before transpose"),
            Self::Sum3 => vec![Some(ct), Some(ct), Some(ct)],
        }
    }
}

fn add_tangents(
    builder: &mut FragmentBuilder<RecorderOp>,
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    let terms: Vec<_> = tangent_in.iter().filter_map(|id| *id).collect();
    sum_terms(builder, terms)
}

fn sum_terms(
    builder: &mut FragmentBuilder<RecorderOp>,
    terms: Vec<LocalValId>,
) -> Vec<Option<LocalValId>> {
    match terms.as_slice() {
        [] => vec![None],
        [only] => vec![Some(*only)],
        [first, rest @ ..] => {
            let mut acc = *first;
            for term in rest {
                let out = builder.add_op(
                    RecorderOp::Add,
                    vec![ValRef::Local(acc), ValRef::Local(*term)],
                    OpMode::Linear {
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
    emitter: &mut impl OpEmitter<RecorderOp>,
    inputs: &[ValRef<RecorderOp>],
    ct: LocalValId,
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    let active_mask = match mode {
        OpMode::Linear { active_mask } => active_mask,
        OpMode::Primal => return vec![None, None],
    };
    let mut result = vec![None, None];
    if active_mask[0] {
        let out = emitter.add_op(
            RecorderOp::Mul,
            vec![inputs[1].clone(), ValRef::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[0] = Some(out[0]);
    }
    if active_mask[1] {
        let out = emitter.add_op(
            RecorderOp::Mul,
            vec![inputs[0].clone(), ValRef::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(out[0]);
    }
    result
}

#[derive(Default)]
struct KeySource {
    next: usize,
}

impl EagerKeySource<RecorderOp> for KeySource {
    fn fresh_input_key(&mut self) -> Key {
        let key = Key::User(format!("e{}", self.next));
        self.next += 1;
        key
    }
}

fn key(name: &str) -> GlobalValKey<RecorderOp> {
    GlobalValKey::Input(Key::User(name.to_string()))
}

fn eager_value(name: &str, value: f64, requires_grad: bool) -> Input<RecorderOp> {
    Input {
        key: key(name),
        trace: None,
        requires_grad,
        data: Arc::new(value),
    }
}

struct EagerEmitter {
    locals: Vec<Arc<f64>>,
    external_data: HashMap<GlobalValKey<RecorderOp>, Arc<f64>>,
}

impl EagerEmitter {
    fn new(external_data: HashMap<GlobalValKey<RecorderOp>, Arc<f64>>) -> Self {
        Self {
            locals: Vec::new(),
            external_data,
        }
    }

    fn push(&mut self, value: Arc<f64>) -> LocalValId {
        let id = self.locals.len();
        self.locals.push(value);
        id
    }

    fn value(&self, id: LocalValId) -> Arc<f64> {
        self.locals[id].clone()
    }

    fn resolve_input(&self, input: &ValRef<RecorderOp>) -> Arc<f64> {
        match input {
            ValRef::Local(local_id) => self.value(*local_id),
            ValRef::External(key) => self
                .external_data
                .get(key)
                .cloned()
                .unwrap_or_else(|| panic!("missing eager value for {:?}", key)),
        }
    }
}

impl OpEmitter<RecorderOp> for EagerEmitter {
    fn add_op(
        &mut self,
        op: RecorderOp,
        inputs: Vec<ValRef<RecorderOp>>,
        _mode: OpMode,
    ) -> Vec<LocalValId> {
        let resolved: Vec<_> = inputs
            .iter()
            .map(|input| self.resolve_input(input))
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
    last_initial_data: HashMap<GlobalValKey<RecorderOp>, Arc<f64>>,
}

impl BackwardExecutor<RecorderOp> for Callbacks {
    fn execute_forward(
        &mut self,
        fragment: &Fragment<RecorderOp>,
        initial_data: &HashMap<GlobalValKey<RecorderOp>, Arc<f64>>,
    ) -> HashMap<GlobalValKey<RecorderOp>, Arc<f64>> {
        self.last_initial_data = initial_data.clone();
        let mut all_values = initial_data.clone();

        for &input_id in fragment.inputs() {
            let key = fragment.vals()[input_id].key.clone();
            all_values.entry(key).or_insert_with(|| Arc::new(0.0));
        }

        for op_node in fragment.ops() {
            let resolved: Vec<_> = op_node
                .inputs
                .iter()
                .map(|input| match input {
                    ValRef::Local(local_id) => all_values
                        .get(&fragment.vals()[*local_id].key)
                        .cloned()
                        .unwrap_or_else(|| {
                            panic!(
                                "missing concrete value for local key {:?}",
                                fragment.vals()[*local_id].key
                            )
                        }),
                    ValRef::External(key) => all_values
                        .get(key)
                        .cloned()
                        .unwrap_or_else(|| panic!("missing concrete value for {:?}", key)),
                })
                .collect();
            let refs: Vec<_> = resolved.iter().map(|value| value.as_ref()).collect();
            let outputs = op_node.op.eval(&mut (), &refs);
            for (output_id, output) in op_node.outputs.iter().zip(outputs) {
                all_values.insert(fragment.vals()[*output_id].key.clone(), Arc::new(output));
            }
        }

        all_values
    }

    fn execute_transpose(
        &mut self,
        linear: &tidu::LinearFragment<RecorderOp>,
        cotangent_out: &[Option<Arc<f64>>],
        external_data: &HashMap<GlobalValKey<RecorderOp>, Arc<f64>>,
        ctx: &mut (),
    ) -> tidu::ADRuleResult<Vec<Option<Arc<f64>>>> {
        let mut emitter = EagerEmitter::new(external_data.clone());
        let seed_ids: Vec<_> = cotangent_out
            .iter()
            .map(|seed| seed.as_ref().map(|value| emitter.push(value.clone())))
            .collect();
        emit::try_transpose_fragment(linear, &mut emitter, &seed_ids, ctx).map(|ids| {
            ids.into_iter()
                .map(|id| id.map(|id| emitter.value(id)))
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
    let mut recorder = Recorder::new(KeySource::default());
    let outputs = recorder.record(RecorderOp::Mul, &inputs, &[Arc::new(12.0)]);
    let mut callbacks = Callbacks::default();
    let cotangents = eager::try_backward(
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
fn record_eager_nary_op_builds_all_input_edges() {
    let inputs = vec![
        eager_value("a", 1.0, true),
        eager_value("b", 2.0, false),
        eager_value("c", 3.0, true),
    ];
    let mut recorder = Recorder::new(KeySource::default());
    let outputs = recorder.record(RecorderOp::Sum3, &inputs, &[Arc::new(6.0)]);
    let mut callbacks = Callbacks::default();
    let cotangents = eager::try_backward(
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
    let mut recorder = Recorder::new(KeySource::default());
    let outputs = recorder.record(RecorderOp::Split, &inputs, &[Arc::new(3.0), Arc::new(-3.0)]);

    assert_eq!(outputs.len(), 2);
    assert!(outputs[0].trace.is_some());
    assert!(outputs[1].trace.is_some());

    let mut callbacks = Callbacks::default();
    let cotangents = eager::try_backward(
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
                GlobalValKey::Derived { output_slot: 1, .. } if **value == -3.0
            )),
        "saved forward data should include the second derived output"
    );
    assert!(
        callbacks
            .last_initial_data
            .iter()
            .any(|(saved_key, value)| matches!(
                saved_key,
                GlobalValKey::Input(_) if **value == 3.0
            )),
        "saved forward data should include the input alias value"
    );
}
