use std::collections::HashMap;
use std::sync::Arc;

#[allow(dead_code)]
mod common;

use common::{ScalarKey, ScalarOp};
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::{EvalGraphOp, OpEmitter};
use tidu::{
    backward_dag, differentiate, eager_transpose_fragment, topo_sort_grad_dag, BackwardCallbacks,
    GradEdge, GradNode,
};

fn sk(name: &str) -> ScalarKey {
    ScalarKey::User(name.to_string())
}

fn arc(value: f64) -> Arc<f64> {
    Arc::new(value)
}

struct ScalarEagerEmitter {
    locals: Vec<Arc<f64>>,
    external_data: HashMap<GlobalValKey<ScalarOp>, Arc<f64>>,
}

impl ScalarEagerEmitter {
    fn new(external_data: HashMap<GlobalValKey<ScalarOp>, Arc<f64>>) -> Self {
        Self {
            locals: Vec::new(),
            external_data,
        }
    }

    fn push_value(&mut self, value: Arc<f64>) -> LocalValId {
        let id = self.locals.len();
        self.locals.push(value);
        id
    }

    fn value(&self, id: LocalValId) -> Arc<f64> {
        self.locals[id].clone()
    }

    fn resolve_input(&self, input: &ValRef<ScalarOp>) -> Arc<f64> {
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

impl OpEmitter<ScalarOp> for ScalarEagerEmitter {
    fn add_op(
        &mut self,
        op: ScalarOp,
        inputs: Vec<ValRef<ScalarOp>>,
        _mode: OpMode,
    ) -> Vec<LocalValId> {
        let resolved_inputs: Vec<Arc<f64>> = inputs
            .iter()
            .map(|input| self.resolve_input(input))
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

impl BackwardCallbacks<ScalarOp> for ScalarBackwardCallbacks {
    fn execute_forward(
        &mut self,
        fragment: &Fragment<ScalarOp>,
        initial_data: &HashMap<GlobalValKey<ScalarOp>, Arc<f64>>,
    ) -> HashMap<GlobalValKey<ScalarOp>, Arc<f64>> {
        let mut all_values = initial_data.clone();

        for &input_id in fragment.inputs() {
            let key = fragment.vals()[input_id].key.clone();
            all_values.entry(key).or_insert_with(|| arc(0.0));
        }

        for op_node in fragment.ops() {
            let resolved_inputs: Vec<Arc<f64>> = op_node
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
                    ValRef::External(key) => all_values.get(key).cloned().unwrap_or_else(|| {
                        panic!("missing concrete value for external key {:?}", key)
                    }),
                })
                .collect();
            let input_refs: Vec<&f64> =
                resolved_inputs.iter().map(|value| value.as_ref()).collect();
            let outputs = op_node.op.eval(&mut (), &input_refs);

            for (output_id, output) in op_node.outputs.iter().zip(outputs.into_iter()) {
                let key = fragment.vals()[*output_id].key.clone();
                all_values.insert(key, Arc::new(output));
            }
        }

        all_values
    }

    fn eager_transpose(
        &mut self,
        linear: &tidu::LinearFragment<ScalarOp>,
        cotangent_out: &[Option<Arc<f64>>],
        external_data: &HashMap<GlobalValKey<ScalarOp>, Arc<f64>>,
        ctx: &mut <ScalarOp as chainrules::PrimitiveOp>::ADContext,
    ) -> Vec<Option<Arc<f64>>> {
        let mut emitter = ScalarEagerEmitter::new(external_data.clone());
        let cotangent_seed_ids: Vec<Option<LocalValId>> = cotangent_out
            .iter()
            .map(|maybe_seed| {
                maybe_seed
                    .as_ref()
                    .map(|seed| emitter.push_value(seed.clone()))
            })
            .collect();

        eager_transpose_fragment(linear, &mut emitter, &cotangent_seed_ids, ctx)
            .into_iter()
            .map(|maybe_id| maybe_id.map(|id| emitter.value(id)))
            .collect()
    }

    fn add_operands(&mut self, a: &Arc<f64>, b: &Arc<f64>) -> Arc<f64> {
        Arc::new(**a + **b)
    }
}

#[test]
fn eager_transpose_fragment_propagates_add_cotangents() {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let y = builder.add_input(sk("y"));
    let sum = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(x), ValRef::Local(y)],
        OpMode::Primal,
    );
    let sum_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sum[0]]);

    let linear = differentiate(
        &resolve(vec![Arc::new(builder.build())]),
        &[sum_key],
        &[sk("x"), sk("y")],
        1,
        &mut (),
        &HashMap::new(),
    );

    let mut emitter = ScalarEagerEmitter::new(HashMap::new());
    let seed = emitter.push_value(arc(1.0));
    let cotangent_inputs = eager_transpose_fragment(&linear, &mut emitter, &[Some(seed)], &mut ());

    let values: Vec<f64> = cotangent_inputs
        .into_iter()
        .map(|maybe_id| *emitter.value(maybe_id.expect("active cotangent input")))
        .collect();

    assert_eq!(values, vec![1.0, 1.0]);
}

#[test]
fn topo_sort_grad_dag_orders_dependencies_before_output() {
    let leaf = Arc::new(GradNode {
        op: ScalarOp::Add,
        primal_in_keys: vec![GlobalValKey::Input(sk("a")), GlobalValKey::Input(sk("b"))],
        primal_out_keys: vec![GlobalValKey::Input(sk("leaf_out"))],
        saved_data: HashMap::new(),
        input_edges: vec![
            GradEdge {
                node: None,
                key: GlobalValKey::Input(sk("a")),
                requires_grad: true,
            },
            GradEdge {
                node: None,
                key: GlobalValKey::Input(sk("b")),
                requires_grad: true,
            },
        ],
        output_idx: 0,
    });
    let root = Arc::new(GradNode {
        op: ScalarOp::Neg,
        primal_in_keys: vec![GlobalValKey::Input(sk("leaf_out"))],
        primal_out_keys: vec![GlobalValKey::Input(sk("root_out"))],
        saved_data: HashMap::new(),
        input_edges: vec![GradEdge {
            node: Some(leaf.clone()),
            key: GlobalValKey::Input(sk("leaf_out")),
            requires_grad: true,
        }],
        output_idx: 0,
    });

    let sorted = topo_sort_grad_dag(&Some(root));

    assert_eq!(sorted.len(), 2);
    assert!(Arc::ptr_eq(&sorted[0], &leaf));
    assert_eq!(sorted[1].op, ScalarOp::Neg);
}

#[test]
fn backward_dag_accumulates_x_squared_gradient() {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x_left = builder.add_input(sk("x_left"));
    let x_right = builder.add_input(sk("x_right"));
    let y = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x_left), ValRef::Local(x_right)],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();

    let x_grad_key = GlobalValKey::Input(sk("x"));
    let x_left_key = GlobalValKey::Input(sk("x_left"));
    let x_right_key = GlobalValKey::Input(sk("x_right"));
    let node = Arc::new(GradNode {
        op: ScalarOp::Mul,
        primal_in_keys: vec![x_left_key.clone(), x_right_key.clone()],
        primal_out_keys: vec![y_key.clone()],
        saved_data: HashMap::from([
            (x_left_key.clone(), arc(3.0)),
            (x_right_key.clone(), arc(3.0)),
            (y_key.clone(), arc(9.0)),
        ]),
        input_edges: vec![
            GradEdge {
                node: None,
                key: x_grad_key.clone(),
                requires_grad: true,
            },
            GradEdge {
                node: None,
                key: x_grad_key.clone(),
                requires_grad: true,
            },
        ],
        output_idx: 0,
    });

    let sorted = topo_sort_grad_dag(&Some(node));
    let mut callbacks = ScalarBackwardCallbacks;
    let cotangents = backward_dag(&sorted, &y_key, arc(1.0), &mut callbacks, &mut ());

    assert_eq!(**cotangents.get(&x_grad_key).expect("gradient for x"), 6.0);
}

/// Fan-out test: f(x) = x + x, df/dx = 2.
/// This exercises the cotangent accumulation (Op::add) path in eager_transpose_fragment.
#[test]
fn eager_transpose_fragment_fan_out_accumulation() {
    // Build linear fragment for x + x: tangent(x) used twice
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let sum = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(x), ValRef::Local(x)], // x used twice (fan-out)
        OpMode::Primal,
    );
    let sum_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sum[0]]);

    let linear = differentiate(
        &resolve(vec![Arc::new(builder.build())]),
        &[sum_key],
        &[sk("x")],
        1,
        &mut (),
        &HashMap::new(),
    );

    let mut emitter = ScalarEagerEmitter::new(HashMap::new());
    let seed = emitter.push_value(arc(1.0));
    let cotangent_inputs = eager_transpose_fragment(&linear, &mut emitter, &[Some(seed)], &mut ());

    // df/dx = 2 (cotangent accumulated from two paths)
    let dx = *emitter.value(cotangent_inputs[0].expect("active"));
    assert!(
        (dx - 2.0).abs() < 1e-12,
        "fan-out gradient: expected 2.0, got {dx}"
    );
}
