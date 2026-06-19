//! Gradient of a two-input scalar function, f(x, y) = x * y + x.
//!
//! Demonstrates building a primal graph with `GraphBuilder`, linearizing with
//! respect to multiple inputs, transposing, and reading both gradients. Because
//! `x` feeds two operations, this also exercises cotangent accumulation.

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ScalarOp {
    Add,
    Mul,
}

impl GraphOperation for ScalarOp {
    type Operand = f64;
    type Context = ();
    type InputKey = ScalarKey;

    fn input_count(&self) -> usize {
        2
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
        _primal_outputs: &[ValueKey<Self>],
        tangent_inputs: &[Option<LocalValueId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValueId>> {
        match self {
            Self::Add => {
                let terms: Vec<_> = tangent_inputs.iter().flatten().copied().collect();
                vec![sum_terms(builder, terms)]
            }
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
                vec![sum_terms(builder, terms)]
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
            Self::Mul => {
                let active_mask = match role {
                    OperationRole::Linearized { active_mask } => active_mask,
                    OperationRole::Primary => return vec![None, None],
                };
                let mut result = vec![None, None];
                if active_mask[0] {
                    let out = builder.add_primitive(
                        Self::Mul,
                        vec![inputs[1].clone(), PrimitiveValue::Local(ct)],
                        OperationRole::Linearized {
                            active_mask: vec![false, true],
                        },
                    );
                    result[0] = Some(out[0]);
                }
                if active_mask[1] {
                    let out = builder.add_primitive(
                        Self::Mul,
                        vec![inputs[0].clone(), PrimitiveValue::Local(ct)],
                        OperationRole::Linearized {
                            active_mask: vec![false, true],
                        },
                    );
                    result[1] = Some(out[0]);
                }
                result
            }
        }
    }
}

fn sk(name: &str) -> ScalarKey {
    ScalarKey::User(name.to_string())
}

fn input_key(name: &str) -> ValueKey<ScalarOp> {
    ValueKey::Input(sk(name))
}

fn sum_terms(
    builder: &mut impl PrimitiveBuilder<ScalarOp>,
    terms: Vec<LocalValueId>,
) -> Option<LocalValueId> {
    match terms.as_slice() {
        [] => None,
        [only] => Some(*only),
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
            Some(acc)
        }
    }
}

fn build_xy_plus_x() -> (Arc<Graph<ScalarOp>>, ValueKey<ScalarOp>) {
    let mut builder = GraphBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let y = builder.add_input(sk("y"));
    let xy = builder.add_operation(
        ScalarOp::Mul,
        vec![ValueRef::Local(x), ValueRef::Local(y)],
        OperationRole::Primary,
    );
    let f = builder.add_operation(
        ScalarOp::Add,
        vec![ValueRef::Local(xy[0]), ValueRef::Local(x)],
        OperationRole::Primary,
    );
    let f_key = builder.global_key(f[0]).clone();
    builder.set_outputs(vec![f[0]]);
    (Arc::new(builder.build()), f_key)
}

fn tangent_output_key(
    linear: &LinearizedGraph<ScalarOp>,
    index: usize,
) -> Option<ValueKey<ScalarOp>> {
    linear.tangent_outputs()[index].map(|id| linear.as_graph().values()[id].key.clone())
}

fn tangent_input_key(linear: &LinearizedGraph<ScalarOp>, index: usize) -> ValueKey<ScalarOp> {
    let id = linear.tangent_inputs()[index].1;
    linear.as_graph().values()[id].key.clone()
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

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (primal, f_key) = build_xy_plus_x();

    // Linearize f with respect to both x and y, then transpose to get the VJP.
    let linear = linearize(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&f_key),
        &[sk("x"), sk("y")],
        1,
        &mut (),
        &HashMap::new(),
    );
    let transposed = linear_transpose(&linear, &mut ());

    // The transposed graph takes one cotangent seed (for f) and produces one
    // cotangent per active input, in the order [x, y].
    let ct_f_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent for x");
    let ct_y_key = tangent_output_key(&transposed, 1).expect("active cotangent for y");

    let grads = evaluate(
        vec![primal, Arc::new(transposed.into_graph())],
        &[ct_x_key, ct_y_key],
        &[
            (input_key("x"), 2.0),
            (input_key("y"), 3.0),
            (ct_f_key, 1.0),
        ],
    );

    println!("df/dx = {} (expected 4)", grads[0]);
    println!("df/dy = {} (expected 2)", grads[1]);
    assert!(
        (grads[0] - 4.0).abs() < 1e-12,
        "df/dx expected 4, got {}",
        grads[0]
    );
    assert!(
        (grads[1] - 2.0).abs() < 1e-12,
        "df/dy expected 2, got {}",
        grads[1]
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}

#[test]
fn example_runs() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
