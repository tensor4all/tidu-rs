use std::collections::HashMap;
use std::sync::Arc;

use computegraph::compile::compile;
use computegraph::graph::Graph;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{LocalValueId, OperationRole, ValueKey};
use computegraph::{EvaluableGraphOperation, GraphOperation};
use tidu::LinearizedGraph;
use tidu::{ADKey, DiffPassId, Primitive, PrimitiveBuilder, PrimitiveValue};

use crate::{
    define_ad_key, linearize_add, linearize_exp, linearize_mul, linearize_neg, transpose_add,
    transpose_mul_real, transpose_neg,
};

define_ad_key!(ScalarKey);

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScalarOp {
    Add,
    Mul,
    Exp,
    Neg,
}

impl GraphOperation for ScalarOp {
    type Operand = f64;
    type Context = ();
    type InputKey = ScalarKey;

    fn input_count(&self) -> usize {
        match self {
            ScalarOp::Add | ScalarOp::Mul => 2,
            ScalarOp::Exp | ScalarOp::Neg => 1,
        }
    }

    fn output_count(&self) -> usize {
        1
    }
}

impl EvaluableGraphOperation for ScalarOp {
    fn eval(&self, _ctx: &mut (), inputs: &[&f64]) -> Vec<f64> {
        match self {
            ScalarOp::Add => vec![inputs[0] + inputs[1]],
            ScalarOp::Mul => vec![inputs[0] * inputs[1]],
            ScalarOp::Exp => vec![inputs[0].exp()],
            ScalarOp::Neg => vec![-inputs[0]],
        }
    }
}

impl Primitive for ScalarOp {
    type ADContext = ();

    fn add() -> Self {
        ScalarOp::Add
    }

    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        primal_in: &[ValueKey<Self>],
        primal_out: &[ValueKey<Self>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValueId>> {
        match self {
            ScalarOp::Add => linearize_add!(builder, ScalarOp::Add, tangent_in[0], tangent_in[1]),
            ScalarOp::Mul => {
                linearize_mul!(
                    builder,
                    ScalarOp::Mul,
                    ScalarOp::Add,
                    primal_in,
                    tangent_in[0],
                    tangent_in[1]
                )
            }
            ScalarOp::Exp => linearize_exp!(builder, ScalarOp::Mul, primal_out[0], tangent_in[0]),
            ScalarOp::Neg => linearize_neg!(builder, ScalarOp::Neg, tangent_in[0]),
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
            ScalarOp::Add => transpose_add!(ct),
            ScalarOp::Mul => transpose_mul_real!(builder, ScalarOp::Mul, inputs, ct, role),
            ScalarOp::Exp => panic!("transpose_rule called on primal-only Exp"),
            ScalarOp::Neg => transpose_neg!(builder, ScalarOp::Neg, ct),
        }
    }
}

pub fn evaluate<Op>(
    roots: Vec<Arc<Graph<Op>>>,
    outputs: &[ValueKey<Op>],
    bindings: &[(ValueKey<Op>, Op::Operand)],
) -> Vec<Op::Operand>
where
    Op: Primitive + EvaluableGraphOperation,
    Op::Context: Default,
    Op::InputKey: ADKey,
{
    let view = resolve(roots);
    let graph = materialize_merge(&view, outputs);
    let binding_map: HashMap<_, _> = bindings.iter().cloned().collect();
    let ordered_inputs: Vec<Op::Operand> = graph
        .inputs
        .iter()
        .map(|key| {
            binding_map
                .get(key)
                .cloned()
                .unwrap_or_else(|| panic!("missing value for input key {:?}", key))
        })
        .collect();
    let ordered_refs: Vec<&Op::Operand> = ordered_inputs.iter().collect();
    let program = compile(&graph);
    program.eval(&mut Default::default(), &ordered_refs)
}

pub fn tangent_input_key<Op>(linear: &LinearizedGraph<Op>, index: usize) -> ValueKey<Op>
where
    Op: Primitive,
    Op::InputKey: ADKey,
{
    let local_id = linear.tangent_inputs()[index].1;
    linear.as_graph().values()[local_id].key.clone()
}

pub fn tangent_output_key<Op>(linear: &LinearizedGraph<Op>, index: usize) -> Option<ValueKey<Op>>
where
    Op: Primitive,
    Op::InputKey: ADKey,
{
    linear.tangent_outputs()[index].map(|local_id| linear.as_graph().values()[local_id].key.clone())
}

pub mod assertions;
pub mod key_macros;
pub mod linearize_macros;
pub mod numeric;
pub mod transpose_macros;
