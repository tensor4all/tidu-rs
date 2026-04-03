mod common;

use std::sync::Arc;

use chainrules::{ADKey, DiffPassId, PrimitiveOp};
use common::{evaluate, tangent_input_key, tangent_output_key};
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::{GraphOp, Operand};
use ndarray::{ArrayD, Axis, IxDyn};
use tidu::{differentiate, transpose};

const TOL: f64 = 1e-10;
const NUM_TOL: f64 = 1e-5;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum VectorKey {
    User(String),
    Tangent {
        of: Box<VectorKey>,
        pass: DiffPassId,
    },
}

impl ADKey for VectorKey {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        VectorKey::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Tensor(ArrayD<f64>);

impl Operand for Tensor {
    fn zero(shape: &[usize]) -> Self {
        Self(ArrayD::zeros(IxDyn(shape)))
    }

    fn one(shape: &[usize]) -> Self {
        Self(ArrayD::from_elem(IxDyn(shape), 1.0))
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        Self(
            self.0
                .clone()
                .into_shape_with_order(IxDyn(shape))
                .unwrap_or_else(|err| panic!("reshape to {shape:?} failed: {err}")),
        )
    }

    fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Self {
        let src_shape = self.0.shape();
        assert_eq!(
            dims.len(),
            src_shape.len(),
            "broadcast dims {dims:?} must match source rank {}",
            src_shape.len()
        );

        let mut reshape_shape = vec![1; shape.len()];
        for (input_axis, &target_axis) in dims.iter().enumerate() {
            assert!(
                target_axis < shape.len(),
                "broadcast axis {target_axis} out of range for target rank {}",
                shape.len()
            );
            reshape_shape[target_axis] = src_shape[input_axis];
            assert_eq!(
                shape[target_axis], src_shape[input_axis],
                "target axis {target_axis} expected extent {}, got {}",
                src_shape[input_axis], shape[target_axis]
            );
        }

        let reshaped = self
            .0
            .clone()
            .into_shape_with_order(IxDyn(&reshape_shape))
            .unwrap_or_else(|err| {
                panic!(
                    "reshape before broadcast from {:?} to {:?} failed: {err}",
                    src_shape, reshape_shape
                )
            });
        let broadcast = reshaped
            .broadcast(IxDyn(shape))
            .unwrap_or_else(|| panic!("broadcast from {:?} to {:?} failed", reshape_shape, shape));
        Self(broadcast.to_owned())
    }

    fn add(&self, other: &Self) -> Self {
        Self(&self.0 + &other.0)
    }

    fn multiply(&self, other: &Self) -> Self {
        Self(&self.0 * &other.0)
    }

    fn reduce_sum(&self, axes: &[usize]) -> Self {
        let mut result = self.0.clone();
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();
        for &axis in sorted_axes.iter().rev() {
            result = result.sum_axis(Axis(axis)).into_dyn();
        }
        Self(result)
    }

    fn dot_general(
        &self,
        other: &Self,
        lhs_contracting: &[usize],
        rhs_contracting: &[usize],
        lhs_batch: &[usize],
        rhs_batch: &[usize],
    ) -> Self {
        assert!(
            lhs_contracting.is_empty()
                && rhs_contracting.is_empty()
                && lhs_batch.is_empty()
                && rhs_batch.is_empty(),
            "dot_general is only implemented for elementwise multiply in these tests"
        );
        Self(&self.0 * &other.0)
    }

    fn conj(&self) -> Self {
        self.clone()
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum VectorOp {
    Add,
    Mul,
    Exp,
    Neg,
    ReduceSum {
        axes: Vec<usize>,
        input_shape: Vec<usize>,
    },
    BroadcastInDim {
        shape: Vec<usize>,
        dims: Vec<usize>,
    },
}

impl GraphOp for VectorOp {
    type Operand = Tensor;
    type Context = ();
    type InputKey = VectorKey;

    fn n_inputs(&self) -> usize {
        match self {
            VectorOp::Add | VectorOp::Mul => 2,
            VectorOp::Exp
            | VectorOp::Neg
            | VectorOp::ReduceSum { .. }
            | VectorOp::BroadcastInDim { .. } => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn eval(&self, _ctx: &mut (), inputs: &[&Tensor]) -> Vec<Tensor> {
        match self {
            VectorOp::Add => vec![Tensor(&inputs[0].0 + &inputs[1].0)],
            VectorOp::Mul => vec![Tensor(&inputs[0].0 * &inputs[1].0)],
            VectorOp::Exp => vec![Tensor(inputs[0].0.mapv(f64::exp))],
            VectorOp::Neg => vec![Tensor(inputs[0].0.mapv(|x| -x))],
            VectorOp::ReduceSum { axes, .. } => vec![inputs[0].reduce_sum(axes)],
            VectorOp::BroadcastInDim { shape, dims } => {
                vec![inputs[0].broadcast_in_dim(shape, dims)]
            }
        }
    }
}

impl PrimitiveOp for VectorOp {
    fn add() -> Self {
        VectorOp::Add
    }

    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
    ) -> Vec<Option<LocalValId>> {
        match self {
            VectorOp::Add => match (tangent_in[0], tangent_in[1]) {
                (Some(lhs), Some(rhs)) => {
                    let sum = builder.add_op(
                        VectorOp::Add,
                        vec![ValRef::Local(lhs), ValRef::Local(rhs)],
                        OpMode::Linear {
                            active_mask: vec![true, true],
                        },
                    );
                    vec![Some(sum[0])]
                }
                (Some(lhs), None) => vec![Some(lhs)],
                (None, Some(rhs)) => vec![Some(rhs)],
                (None, None) => vec![None],
            },
            VectorOp::Mul => {
                let mut terms = Vec::new();

                if let Some(dlhs) = tangent_in[0] {
                    let term = builder.add_op(
                        VectorOp::Mul,
                        vec![ValRef::Local(dlhs), ValRef::External(primal_in[1].clone())],
                        OpMode::Linear {
                            active_mask: vec![true, false],
                        },
                    );
                    terms.push(term[0]);
                }

                if let Some(drhs) = tangent_in[1] {
                    let term = builder.add_op(
                        VectorOp::Mul,
                        vec![ValRef::External(primal_in[0].clone()), ValRef::Local(drhs)],
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
                            VectorOp::Add,
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
            VectorOp::Exp => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_op(
                        VectorOp::Mul,
                        vec![ValRef::External(primal_out[0].clone()), ValRef::Local(dx)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    vec![Some(out[0])]
                }
                None => vec![None],
            },
            VectorOp::Neg => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_op(
                        VectorOp::Neg,
                        vec![ValRef::Local(dx)],
                        OpMode::Linear {
                            active_mask: vec![true],
                        },
                    );
                    vec![Some(out[0])]
                }
                None => vec![None],
            },
            VectorOp::ReduceSum { axes, input_shape } => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_op(
                        VectorOp::ReduceSum {
                            axes: axes.clone(),
                            input_shape: input_shape.clone(),
                        },
                        vec![ValRef::Local(dx)],
                        OpMode::Linear {
                            active_mask: vec![true],
                        },
                    );
                    vec![Some(out[0])]
                }
                None => vec![None],
            },
            VectorOp::BroadcastInDim { shape, dims } => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_op(
                        VectorOp::BroadcastInDim {
                            shape: shape.clone(),
                            dims: dims.clone(),
                        },
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
            VectorOp::Add => vec![Some(ct), Some(ct)],
            VectorOp::Mul => {
                let active_mask = match mode {
                    OpMode::Linear { active_mask } => active_mask,
                    OpMode::Primal => return vec![None, None],
                };

                let mut result = vec![None, None];

                if active_mask[0] {
                    let out = builder.add_op(
                        VectorOp::Mul,
                        vec![inputs[1].clone(), ValRef::Local(ct)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    result[0] = Some(out[0]);
                }

                if active_mask[1] {
                    let out = builder.add_op(
                        VectorOp::Mul,
                        vec![inputs[0].clone(), ValRef::Local(ct)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    result[1] = Some(out[0]);
                }

                result
            }
            VectorOp::Exp => panic!("transpose_rule called on primal-only Exp"),
            VectorOp::Neg => {
                let out = builder.add_op(
                    VectorOp::Neg,
                    vec![ValRef::Local(ct)],
                    OpMode::Linear {
                        active_mask: vec![true],
                    },
                );
                vec![Some(out[0])]
            }
            VectorOp::ReduceSum { axes, input_shape } => {
                let dims = (0..input_shape.len())
                    .filter(|axis| !axes.contains(axis))
                    .collect();
                let out = builder.add_op(
                    VectorOp::BroadcastInDim {
                        shape: input_shape.clone(),
                        dims,
                    },
                    vec![ValRef::Local(ct)],
                    OpMode::Linear {
                        active_mask: vec![true],
                    },
                );
                vec![Some(out[0])]
            }
            VectorOp::BroadcastInDim { shape, dims } => {
                let axes = (0..shape.len())
                    .filter(|axis| !dims.contains(axis))
                    .collect();
                let out = builder.add_op(
                    VectorOp::ReduceSum {
                        axes,
                        input_shape: shape.clone(),
                    },
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

fn vk(name: &str) -> VectorKey {
    VectorKey::User(name.to_string())
}

fn input_key(name: &str) -> GlobalValKey<VectorOp> {
    GlobalValKey::Input(vk(name))
}

fn scalar(value: f64) -> Tensor {
    Tensor(ArrayD::from_elem(IxDyn(&[]), value))
}

fn vector(values: &[f64]) -> Tensor {
    Tensor(
        ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec())
            .unwrap_or_else(|err| panic!("failed to build vector tensor from {values:?}: {err}")),
    )
}

fn assert_tensor_approx_eq(actual: &Tensor, expected: &Tensor, tol: f64) {
    assert_eq!(
        actual.0.shape(),
        expected.0.shape(),
        "shape mismatch: expected {:?}, got {:?}",
        expected.0.shape(),
        actual.0.shape()
    );

    for (index, (actual_value, expected_value)) in
        actual.0.iter().zip(expected.0.iter()).enumerate()
    {
        let delta = (actual_value - expected_value).abs();
        assert!(
            delta <= tol,
            "entry {index}: expected {expected_value}, got {actual_value}, |delta|={delta}"
        );
    }
}

fn build_exp_ax() -> (Arc<Fragment<VectorOp>>, GlobalValKey<VectorOp>) {
    let mut builder = FragmentBuilder::<VectorOp>::new();
    let x = builder.add_input(vk("x"));
    let a = builder.add_input(vk("a"));
    let ax = builder.add_op(
        VectorOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(a)],
        OpMode::Primal,
    );
    let y = builder.add_op(VectorOp::Exp, vec![ValRef::Local(ax[0])], OpMode::Primal);
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_sum_exp_ax() -> (Arc<Fragment<VectorOp>>, GlobalValKey<VectorOp>) {
    let mut builder = FragmentBuilder::<VectorOp>::new();
    let x = builder.add_input(vk("x"));
    let a = builder.add_input(vk("a"));
    let ax = builder.add_op(
        VectorOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(a)],
        OpMode::Primal,
    );
    let exp_ax = builder.add_op(VectorOp::Exp, vec![ValRef::Local(ax[0])], OpMode::Primal);
    let y = builder.add_op(
        VectorOp::ReduceSum {
            axes: vec![0],
            input_shape: vec![2],
        },
        vec![ValRef::Local(exp_ax[0])],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn finite_difference_sum_exp_ax(
    primal: Arc<Fragment<VectorOp>>,
    y_key: &GlobalValKey<VectorOp>,
) -> Tensor {
    let base = [1.0, 2.0];
    let h = 1e-4;
    let sample = |x: [f64; 2]| {
        evaluate(
            vec![primal.clone()],
            std::slice::from_ref(y_key),
            &[
                (input_key("x"), vector(&x)),
                (input_key("a"), vector(&[2.0, 3.0])),
            ],
        )[0]
        .0[IxDyn(&[])]
    };
    let five_point = |index: usize| {
        let mut displaced = base;
        let mut sample_at = |delta: f64| {
            displaced[index] = base[index] + delta;
            let value = sample(displaced);
            displaced[index] = base[index];
            value
        };
        (-sample_at(2.0 * h) + 8.0 * sample_at(h) - 8.0 * sample_at(-h) + sample_at(-2.0 * h))
            / (12.0 * h)
    };

    vector(&[five_point(0), five_point(1)])
}

#[test]
fn jvp_elementwise_exp_ax() {
    let (primal, y_key) = build_exp_ax();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[vk("x")],
        1,
    );

    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dx_key = tangent_input_key(&linear, 0);
    let result = evaluate(
        vec![primal, Arc::new(linear.fragment)],
        &[dy_key],
        &[
            (input_key("x"), vector(&[1.0, 2.0])),
            (input_key("a"), vector(&[2.0, 3.0])),
            (dx_key, vector(&[1.0, 1.0])),
        ],
    );

    assert_tensor_approx_eq(
        &result[0],
        &vector(&[2.0 * 2.0_f64.exp(), 3.0 * 6.0_f64.exp()]),
        TOL,
    );
}

#[test]
fn vjp_elementwise_exp_ax() {
    let (primal, y_key) = build_exp_ax();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[vk("x")],
        2,
    );
    let transposed = transpose(&linear);

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let result = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[
            (input_key("x"), vector(&[1.0, 2.0])),
            (input_key("a"), vector(&[2.0, 3.0])),
            (ct_y_key, vector(&[1.0, 1.0])),
        ],
    );

    assert_tensor_approx_eq(
        &result[0],
        &vector(&[2.0 * 2.0_f64.exp(), 3.0 * 6.0_f64.exp()]),
        TOL,
    );
}

#[test]
fn jvp_sum_exp_ax() {
    let (primal, y_key) = build_sum_exp_ax();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[vk("x")],
        3,
    );

    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dx_key = tangent_input_key(&linear, 0);
    let result = evaluate(
        vec![primal, Arc::new(linear.fragment)],
        &[dy_key],
        &[
            (input_key("x"), vector(&[1.0, 2.0])),
            (input_key("a"), vector(&[2.0, 3.0])),
            (dx_key, vector(&[1.0, 1.0])),
        ],
    );

    assert_tensor_approx_eq(
        &result[0],
        &scalar(2.0 * 2.0_f64.exp() + 3.0 * 6.0_f64.exp()),
        TOL,
    );
}

#[test]
fn vjp_sum_exp_ax_broadcasts_scalar_cotangent() {
    let (primal, y_key) = build_sum_exp_ax();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[vk("x")],
        4,
    );
    let transposed = transpose(&linear);

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let result = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[
            (input_key("x"), vector(&[1.0, 2.0])),
            (input_key("a"), vector(&[2.0, 3.0])),
            (ct_y_key, scalar(2.0)),
        ],
    );

    assert_tensor_approx_eq(
        &result[0],
        &vector(&[4.0 * 2.0_f64.exp(), 6.0 * 6.0_f64.exp()]),
        TOL,
    );
}

#[test]
fn numerical_gradient_sum_exp_ax_matches_vjp() {
    let (primal, y_key) = build_sum_exp_ax();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[vk("x")],
        5,
    );
    let transposed = transpose(&linear);

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let vjp = evaluate(
        vec![primal.clone(), Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[
            (input_key("x"), vector(&[1.0, 2.0])),
            (input_key("a"), vector(&[2.0, 3.0])),
            (ct_y_key, scalar(1.0)),
        ],
    );
    let numerical = finite_difference_sum_exp_ax(primal, &y_key);

    assert_tensor_approx_eq(&vjp[0], &numerical, NUM_TOL);
}
