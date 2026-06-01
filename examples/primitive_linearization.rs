use std::collections::HashMap;
use std::sync::Arc;

use computegraph::compile::compile;
use computegraph::graph::{Graph, GraphBuilder};
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use computegraph::{EvaluableGraphOperation, GraphOperation};
use tidu::{
    linear_transpose, linearize, ADKey, DiffPassId, LinearizedGraph, Primitive, PrimitiveBuilder,
    PrimitiveValue,
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

fn sk(name: &str) -> ScalarKey {
    ScalarKey::User(name.to_string())
}

fn input_key(name: &str) -> ValueKey<ScalarOp> {
    ValueKey::Input(sk(name))
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

fn build_x_squared() -> (Arc<Graph<ScalarOp>>, ValueKey<ScalarOp>) {
    let mut builder = GraphBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let y = builder.add_operation(
        ScalarOp::Mul,
        vec![ValueRef::Local(x), ValueRef::Local(x)],
        OperationRole::Primary,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn tangent_input_key(linear: &LinearizedGraph<ScalarOp>, index: usize) -> ValueKey<ScalarOp> {
    let local_id = linear.tangent_inputs()[index].1;
    linear.as_graph().values()[local_id].key.clone()
}

fn tangent_output_key(
    linear: &LinearizedGraph<ScalarOp>,
    index: usize,
) -> Option<ValueKey<ScalarOp>> {
    linear.tangent_outputs()[index].map(|local_id| linear.as_graph().values()[local_id].key.clone())
}

fn evaluate(
    roots: Vec<Arc<Graph<ScalarOp>>>,
    outputs: &[ValueKey<ScalarOp>],
    bindings: &[(ValueKey<ScalarOp>, f64)],
) -> Vec<f64> {
    let view = resolve(roots);
    let graph = materialize_merge(&view, outputs);
    let binding_map: HashMap<_, _> = bindings.iter().cloned().collect();
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
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, got {actual}"
    );
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
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

    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dx_key = tangent_input_key(&linear, 0);
    let primal_and_tangent = evaluate(
        vec![primal.clone(), Arc::new(linear.into_graph())],
        &[y_key, dy_key],
        &[(input_key("x"), 3.0), (dx_key, 1.5)],
    );
    assert_close(primal_and_tangent[0], 9.0);
    assert_close(primal_and_tangent[1], 9.0);

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let cotangent = evaluate(
        vec![primal, Arc::new(transposed.into_graph())],
        &[ct_x_key],
        &[(input_key("x"), 3.0), (ct_y_key, 2.0)],
    );
    assert_close(cotangent[0], 12.0);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}

#[test]
fn example_runs() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
