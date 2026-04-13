use std::collections::HashMap;
use std::sync::Arc;

use chainrules::{ADKey, DiffPassId, PrimitiveOp};
use computegraph::compile::compile;
use computegraph::fragment::Fragment;
use computegraph::fragment::FragmentBuilder;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::{EvalGraphOp, GraphOp, OpEmitter};
use tidu::LinearFragment;

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

impl GraphOp for ScalarOp {
    type Operand = f64;
    type Context = ();
    type InputKey = ScalarKey;

    fn n_inputs(&self) -> usize {
        match self {
            ScalarOp::Add | ScalarOp::Mul => 2,
            ScalarOp::Exp | ScalarOp::Neg => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl EvalGraphOp for ScalarOp {
    fn eval(&self, _ctx: &mut (), inputs: &[&f64]) -> Vec<f64> {
        match self {
            ScalarOp::Add => vec![inputs[0] + inputs[1]],
            ScalarOp::Mul => vec![inputs[0] * inputs[1]],
            ScalarOp::Exp => vec![inputs[0].exp()],
            ScalarOp::Neg => vec![-inputs[0]],
        }
    }
}

impl PrimitiveOp for ScalarOp {
    type ADContext = ();

    fn add() -> Self {
        ScalarOp::Add
    }

    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
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
        builder: &mut impl OpEmitter<Self>,
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
            ScalarOp::Add => transpose_add!(ct),
            ScalarOp::Mul => transpose_mul_real!(builder, ScalarOp::Mul, inputs, ct, mode),
            ScalarOp::Exp => panic!("transpose_rule called on primal-only Exp"),
            ScalarOp::Neg => transpose_neg!(builder, ScalarOp::Neg, ct),
        }
    }
}

pub fn evaluate<Op>(
    roots: Vec<Arc<Fragment<Op>>>,
    outputs: &[GlobalValKey<Op>],
    bindings: &[(GlobalValKey<Op>, Op::Operand)],
) -> Vec<Op::Operand>
where
    Op: PrimitiveOp + EvalGraphOp,
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

pub fn tangent_input_key<Op>(linear: &LinearFragment<Op>, index: usize) -> GlobalValKey<Op>
where
    Op: PrimitiveOp,
    Op::InputKey: ADKey,
{
    let local_id = linear.tangent_inputs[index].1;
    linear.fragment.vals()[local_id].key.clone()
}

pub fn tangent_output_key<Op>(linear: &LinearFragment<Op>, index: usize) -> Option<GlobalValKey<Op>>
where
    Op: PrimitiveOp,
    Op::InputKey: ADKey,
{
    linear.tangent_outputs[index].map(|local_id| linear.fragment.vals()[local_id].key.clone())
}

pub mod assertions;
pub mod key_macros;
pub mod linearize_macros;
pub mod numeric;
pub mod transpose_macros;
