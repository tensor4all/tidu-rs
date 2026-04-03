use std::collections::HashMap;
use std::sync::Arc;

use chainrules::{ADKey, DiffPassId, PrimitiveOp};
use computegraph::compile::compile;
use computegraph::fragment::Fragment;
use computegraph::fragment::FragmentBuilder;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::GraphOp;
use tidu::LinearFragment;

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScalarKey {
    User(String),
    Tangent {
        of: Box<ScalarKey>,
        pass: DiffPassId,
    },
}

impl ADKey for ScalarKey {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        ScalarKey::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}

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
    fn add() -> Self {
        ScalarOp::Add
    }

    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
    ) -> Vec<Option<LocalValId>> {
        match self {
            ScalarOp::Add => match (tangent_in[0], tangent_in[1]) {
                (Some(dx), Some(dy)) => {
                    let out = builder.add_op(
                        ScalarOp::Add,
                        vec![ValRef::Local(dx), ValRef::Local(dy)],
                        OpMode::Linear {
                            active_mask: vec![true, true],
                        },
                    );
                    vec![Some(out[0])]
                }
                (Some(dx), None) => vec![Some(dx)],
                (None, Some(dy)) => vec![Some(dy)],
                (None, None) => vec![None],
            },
            ScalarOp::Mul => {
                let mut terms = Vec::new();

                if let Some(dx) = tangent_in[0] {
                    let term = builder.add_op(
                        ScalarOp::Mul,
                        vec![ValRef::Local(dx), ValRef::External(primal_in[1].clone())],
                        OpMode::Linear {
                            active_mask: vec![true, false],
                        },
                    );
                    terms.push(term[0]);
                }

                if let Some(dy) = tangent_in[1] {
                    let term = builder.add_op(
                        ScalarOp::Mul,
                        vec![ValRef::External(primal_in[0].clone()), ValRef::Local(dy)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    terms.push(term[0]);
                }

                match terms.as_slice() {
                    [] => vec![None],
                    [only] => vec![Some(*only)],
                    [lhs, rhs] => {
                        let sum = builder.add_op(
                            ScalarOp::Add,
                            vec![ValRef::Local(*lhs), ValRef::Local(*rhs)],
                            OpMode::Linear {
                                active_mask: vec![true, true],
                            },
                        );
                        vec![Some(sum[0])]
                    }
                    _ => unreachable!("mul linearization creates at most two terms"),
                }
            }
            ScalarOp::Exp => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_op(
                        ScalarOp::Mul,
                        vec![ValRef::External(primal_out[0].clone()), ValRef::Local(dx)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    vec![Some(out[0])]
                }
                None => vec![None],
            },
            ScalarOp::Neg => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_op(
                        ScalarOp::Neg,
                        vec![ValRef::Local(dx)],
                        OpMode::Linear {
                            active_mask: vec![true],
                        },
                    );
                    vec![Some(out[0])]
                }
                None => vec![None],
            },
        }
    }

    fn transpose_rule(
        &self,
        builder: &mut FragmentBuilder<Self>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<Self>],
        mode: &OpMode,
    ) -> Vec<Option<LocalValId>> {
        let ct = match cotangent_out[0] {
            Some(ct) => ct,
            None => return vec![None; self.n_inputs()],
        };

        match self {
            ScalarOp::Add => vec![Some(ct), Some(ct)],
            ScalarOp::Mul => {
                let active_mask = match mode {
                    OpMode::Linear { active_mask } => active_mask,
                    OpMode::Primal => return vec![None, None],
                };

                let mut result = vec![None, None];

                if active_mask[0] {
                    let out = builder.add_op(
                        ScalarOp::Mul,
                        vec![inputs[1].clone(), ValRef::Local(ct)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    result[0] = Some(out[0]);
                }

                if active_mask[1] {
                    let out = builder.add_op(
                        ScalarOp::Mul,
                        vec![inputs[0].clone(), ValRef::Local(ct)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    result[1] = Some(out[0]);
                }

                result
            }
            ScalarOp::Exp => panic!("transpose_rule called on primal-only Exp"),
            ScalarOp::Neg => {
                let out = builder.add_op(
                    ScalarOp::Neg,
                    vec![ValRef::Local(ct)],
                    OpMode::Linear {
                        active_mask: vec![true],
                    },
                );
                vec![Some(out[0])]
            }
        }
    }
}

pub fn evaluate<Op>(
    roots: Vec<Arc<Fragment<Op>>>,
    outputs: &[GlobalValKey<Op>],
    bindings: &[(GlobalValKey<Op>, Op::Operand)],
) -> Vec<Op::Operand>
where
    Op: PrimitiveOp,
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
