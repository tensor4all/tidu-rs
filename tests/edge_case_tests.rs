use std::collections::HashMap;
mod common;

use std::sync::Arc;

use chainrules::{ADKey, DiffPassId, PrimitiveOp};
use common::assertions::{
    assert_ctensor_approx_eq, assert_scalar_approx_eq, assert_tensor_approx_eq,
};
use common::{evaluate, tangent_input_key, tangent_output_key, ScalarKey, ScalarOp};
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::{EvalGraphOp, GraphOp, OpEmitter};
use ndarray::{ArrayD, Axis, IxDyn};
use num_complex::Complex64;
use tidu::{differentiate, transpose};

const TOL: f64 = 1e-10;
const NUM_TOL: f64 = 1e-5;

fn sk(name: &str) -> ScalarKey {
    ScalarKey::User(name.to_string())
}

fn scalar_input_key(name: &str) -> GlobalValKey<ScalarOp> {
    GlobalValKey::Input(sk(name))
}

fn ext_input_key(name: &str) -> GlobalValKey<ExtScalarOp> {
    GlobalValKey::Input(sk(name))
}

fn five_point_derivative(sample: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (-sample(x + 2.0 * h) + 8.0 * sample(x + h) - 8.0 * sample(x - h) + sample(x - 2.0 * h))
        / (12.0 * h)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ExtScalarOp {
    Add,
    Mul,
    Neg,
    SinCos,
}

impl GraphOp for ExtScalarOp {
    type Operand = f64;
    type Context = ();
    type InputKey = ScalarKey;

    fn n_inputs(&self) -> usize {
        match self {
            Self::Add | Self::Mul => 2,
            Self::Neg | Self::SinCos => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        match self {
            Self::SinCos => 2,
            _ => 1,
        }
    }
}

impl EvalGraphOp for ExtScalarOp {
    fn eval(&self, _ctx: &mut (), inputs: &[&f64]) -> Vec<f64> {
        match self {
            Self::Add => vec![inputs[0] + inputs[1]],
            Self::Mul => vec![inputs[0] * inputs[1]],
            Self::Neg => vec![-inputs[0]],
            Self::SinCos => vec![inputs[0].sin(), inputs[0].cos()],
        }
    }
}

impl PrimitiveOp for ExtScalarOp {
    type ADContext = ();

    fn add() -> Self {
        Self::Add
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
            Self::Add => linearize_add!(builder, ExtScalarOp::Add, tangent_in[0], tangent_in[1]),
            Self::Mul => {
                linearize_mul!(
                    builder,
                    ExtScalarOp::Mul,
                    ExtScalarOp::Add,
                    primal_in,
                    tangent_in[0],
                    tangent_in[1]
                )
            }
            Self::Neg => linearize_neg!(builder, ExtScalarOp::Neg, tangent_in[0]),
            Self::SinCos => match tangent_in[0] {
                Some(dx) => {
                    let d_sin = builder.add_op(
                        Self::Mul,
                        vec![ValRef::External(primal_out[1].clone()), ValRef::Local(dx)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    let sin_times_dx = builder.add_op(
                        Self::Mul,
                        vec![ValRef::External(primal_out[0].clone()), ValRef::Local(dx)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    let d_cos = builder.add_op(
                        Self::Neg,
                        vec![ValRef::Local(sin_times_dx[0])],
                        OpMode::Linear {
                            active_mask: vec![true],
                        },
                    );
                    vec![Some(d_sin[0]), Some(d_cos[0])]
                }
                None => vec![None, None],
            },
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
            Self::Add => transpose_add!(ct),
            Self::Mul => transpose_mul_real!(builder, ExtScalarOp::Mul, inputs, ct, mode),
            Self::Neg => transpose_neg!(builder, ExtScalarOp::Neg, ct),
            Self::SinCos => panic!("transpose_rule called on primal-only SinCos"),
        }
    }
}

fn build_sincos_sum() -> (
    Arc<Fragment<ExtScalarOp>>,
    GlobalValKey<ExtScalarOp>,
    GlobalValKey<ExtScalarOp>,
    GlobalValKey<ExtScalarOp>,
) {
    let mut builder = FragmentBuilder::<ExtScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let sin_cos = builder.add_op(ExtScalarOp::SinCos, vec![ValRef::Local(x)], OpMode::Primal);
    let sum = builder.add_op(
        ExtScalarOp::Add,
        vec![ValRef::Local(sin_cos[0]), ValRef::Local(sin_cos[1])],
        OpMode::Primal,
    );
    let sin_key = builder.global_key(sin_cos[0]).clone();
    let cos_key = builder.global_key(sin_cos[1]).clone();
    let sum_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sin_cos[0], sin_cos[1], sum[0]]);
    (Arc::new(builder.build()), sin_key, cos_key, sum_key)
}

fn build_scaled_exp_chain(depth: usize) -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let a = builder.add_input(sk("a"));
    let mut current = x;

    for _ in 0..depth {
        // A small scale keeps ten exp nodes finite in f64 while still creating
        // a deep Exp-heavy graph for the AD transforms.
        let scaled = builder.add_op(
            ScalarOp::Mul,
            vec![ValRef::Local(current), ValRef::Local(a)],
            OpMode::Primal,
        );
        let next = builder.add_op(
            ScalarOp::Exp,
            vec![ValRef::Local(scaled[0])],
            OpMode::Primal,
        );
        current = next[0];
    }

    let y_key = builder.global_key(current).clone();
    builder.set_outputs(vec![current]);
    (Arc::new(builder.build()), y_key)
}

fn scaled_exp_chain_value_and_derivative(x: f64, a: f64, depth: usize) -> (f64, f64) {
    let mut value = x;
    let mut derivative = 1.0;

    for _ in 0..depth {
        value = (a * value).exp();
        derivative *= a * value;
    }

    (value, derivative)
}

fn build_x_cubed() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let x2 = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    let y = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x2[0]), ValRef::Local(x)],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_x_fourth() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let x2_lhs = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    let x2_rhs = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    let y = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x2_lhs[0]), ValRef::Local(x2_rhs[0])],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_exp_x() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let y = builder.add_op(ScalarOp::Exp, vec![ValRef::Local(x)], OpMode::Primal);
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

define_ad_key!(VectorKey);

#[derive(Clone, Debug, PartialEq)]
struct Tensor(ArrayD<f64>);

// Tensor inherent methods (previously from Operand trait, now removed from computegraph)

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum VectorOp {
    Add,
    Mul,
    Exp,
}

impl GraphOp for VectorOp {
    type Operand = Tensor;
    type Context = ();
    type InputKey = VectorKey;

    fn n_inputs(&self) -> usize {
        match self {
            Self::Add | Self::Mul => 2,
            Self::Exp => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl EvalGraphOp for VectorOp {
    fn eval(&self, _ctx: &mut (), inputs: &[&Tensor]) -> Vec<Tensor> {
        match self {
            Self::Add => vec![Tensor(&inputs[0].0 + &inputs[1].0)],
            Self::Mul => vec![Tensor(&inputs[0].0 * &inputs[1].0)],
            Self::Exp => vec![Tensor(inputs[0].0.mapv(f64::exp))],
        }
    }
}

impl PrimitiveOp for VectorOp {
    type ADContext = ();

    fn add() -> Self {
        Self::Add
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
            Self::Add => linearize_add!(builder, VectorOp::Add, tangent_in[0], tangent_in[1]),
            Self::Mul => {
                linearize_mul!(
                    builder,
                    VectorOp::Mul,
                    VectorOp::Add,
                    primal_in,
                    tangent_in[0],
                    tangent_in[1]
                )
            }
            Self::Exp => linearize_exp!(builder, VectorOp::Mul, primal_out[0], tangent_in[0]),
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
            Self::Add => transpose_add!(ct),
            Self::Mul => transpose_mul_real!(builder, VectorOp::Mul, inputs, ct, mode),
            Self::Exp => panic!("transpose_rule called on primal-only Exp"),
        }
    }
}

fn vk(name: &str) -> VectorKey {
    VectorKey::User(name.to_string())
}

fn vector_input_key(name: &str) -> GlobalValKey<VectorOp> {
    GlobalValKey::Input(vk(name))
}

fn vector(values: &[f64]) -> Tensor {
    Tensor(
        ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec())
            .unwrap_or_else(|err| panic!("failed to build vector tensor from {values:?}: {err}")),
    )
}

fn build_vector_x_cubed() -> (Arc<Fragment<VectorOp>>, GlobalValKey<VectorOp>) {
    let mut builder = FragmentBuilder::<VectorOp>::new();
    let x = builder.add_input(vk("x"));
    let x2 = builder.add_op(
        VectorOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    let y = builder.add_op(
        VectorOp::Mul,
        vec![ValRef::Local(x2[0]), ValRef::Local(x)],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_vector_exp_x() -> (Arc<Fragment<VectorOp>>, GlobalValKey<VectorOp>) {
    let mut builder = FragmentBuilder::<VectorOp>::new();
    let x = builder.add_input(vk("x"));
    let y = builder.add_op(VectorOp::Exp, vec![ValRef::Local(x)], OpMode::Primal);
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

define_ad_key!(ComplexVectorKey);

#[derive(Clone, Debug, PartialEq)]
struct CTensor(ArrayD<Complex64>);

impl CTensor {
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

    fn reduce_sum(&self, axes: &[usize]) -> Self {
        let mut result = self.0.clone();
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();
        for &axis in sorted_axes.iter().rev() {
            result = result.sum_axis(Axis(axis)).into_dyn();
        }
        Self(result)
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ComplexVectorOp {
    Add,
    Mul,
    Exp,
    Neg,
    Conj,
    ReduceSum {
        axes: Vec<usize>,
        input_shape: Vec<usize>,
    },
    BroadcastInDim {
        shape: Vec<usize>,
        dims: Vec<usize>,
    },
}

impl GraphOp for ComplexVectorOp {
    type Operand = CTensor;
    type Context = ();
    type InputKey = ComplexVectorKey;

    fn n_inputs(&self) -> usize {
        match self {
            Self::Add | Self::Mul => 2,
            Self::Exp
            | Self::Neg
            | Self::Conj
            | Self::ReduceSum { .. }
            | Self::BroadcastInDim { .. } => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl EvalGraphOp for ComplexVectorOp {
    fn eval(&self, _ctx: &mut (), inputs: &[&CTensor]) -> Vec<CTensor> {
        match self {
            Self::Add => vec![CTensor(&inputs[0].0 + &inputs[1].0)],
            Self::Mul => vec![CTensor(&inputs[0].0 * &inputs[1].0)],
            Self::Exp => vec![CTensor(inputs[0].0.mapv(|z| z.exp()))],
            Self::Neg => vec![CTensor(inputs[0].0.mapv(|z| -z))],
            Self::Conj => vec![CTensor(inputs[0].0.mapv(|z| z.conj()))],
            Self::ReduceSum { axes, .. } => vec![inputs[0].reduce_sum(axes)],
            Self::BroadcastInDim { shape, dims } => {
                vec![inputs[0].broadcast_in_dim(shape, dims)]
            }
        }
    }
}

impl PrimitiveOp for ComplexVectorOp {
    type ADContext = ();

    fn add() -> Self {
        Self::Add
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
            Self::Add => {
                linearize_add!(builder, ComplexVectorOp::Add, tangent_in[0], tangent_in[1])
            }
            Self::Mul => linearize_mul!(
                builder,
                ComplexVectorOp::Mul,
                ComplexVectorOp::Add,
                primal_in,
                tangent_in[0],
                tangent_in[1]
            ),
            Self::Exp => {
                linearize_exp!(builder, ComplexVectorOp::Mul, primal_out[0], tangent_in[0])
            }
            Self::Neg => linearize_neg!(builder, ComplexVectorOp::Neg, tangent_in[0]),
            Self::Conj => linearize_conj!(builder, ComplexVectorOp::Conj, tangent_in[0]),
            Self::ReduceSum { axes, input_shape } => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_op(
                        Self::ReduceSum {
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
            Self::BroadcastInDim { shape, dims } => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_op(
                        Self::BroadcastInDim {
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
            Self::Add => transpose_add!(ct),
            Self::Mul => {
                transpose_mul_complex!(
                    builder,
                    ComplexVectorOp::Mul,
                    ComplexVectorOp::Conj,
                    inputs,
                    ct,
                    mode
                )
            }
            Self::Exp => panic!("transpose_rule called on primal-only Exp"),
            Self::Neg => transpose_neg!(builder, ComplexVectorOp::Neg, ct),
            Self::Conj => transpose_conj!(builder, ComplexVectorOp::Conj, ct),
            Self::ReduceSum { axes, input_shape } => {
                let dims = (0..input_shape.len())
                    .filter(|axis| !axes.contains(axis))
                    .collect();
                let out = builder.add_op(
                    Self::BroadcastInDim {
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
            Self::BroadcastInDim { shape, dims } => {
                let axes = (0..shape.len())
                    .filter(|axis| !dims.contains(axis))
                    .collect();
                let out = builder.add_op(
                    Self::ReduceSum {
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

fn cvk(name: &str) -> ComplexVectorKey {
    ComplexVectorKey::User(name.to_string())
}

fn complex_vector_input_key(name: &str) -> GlobalValKey<ComplexVectorOp> {
    GlobalValKey::Input(cvk(name))
}

fn cscalar(value: Complex64) -> CTensor {
    CTensor(ArrayD::from_elem(IxDyn(&[]), value))
}

fn cvector(values: &[Complex64]) -> CTensor {
    CTensor(
        ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).unwrap_or_else(|err| {
            panic!("failed to build complex vector tensor from {values:?}: {err}")
        }),
    )
}

fn complex_tensor_inner_product(lhs: &CTensor, rhs: &CTensor) -> f64 {
    assert_eq!(
        lhs.0.shape(),
        rhs.0.shape(),
        "shape mismatch in inner product: lhs {:?}, rhs {:?}",
        lhs.0.shape(),
        rhs.0.shape()
    );

    lhs.0
        .iter()
        .zip(rhs.0.iter())
        .map(|(lhs_value, rhs_value)| (lhs_value.conj() * rhs_value).re)
        .sum()
}

fn build_complex_vector_conj() -> (
    Arc<Fragment<ComplexVectorOp>>,
    GlobalValKey<ComplexVectorOp>,
) {
    let mut builder = FragmentBuilder::<ComplexVectorOp>::new();
    let z = builder.add_input(cvk("z"));
    let y = builder.add_op(
        ComplexVectorOp::Conj,
        vec![ValRef::Local(z)],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_complex_vector_sum_abs_squared() -> (
    Arc<Fragment<ComplexVectorOp>>,
    GlobalValKey<ComplexVectorOp>,
) {
    let mut builder = FragmentBuilder::<ComplexVectorOp>::new();
    let z = builder.add_input(cvk("z"));
    let conj_z = builder.add_op(
        ComplexVectorOp::Conj,
        vec![ValRef::Local(z)],
        OpMode::Primal,
    );
    let abs_sq = builder.add_op(
        ComplexVectorOp::Mul,
        vec![ValRef::Local(z), ValRef::Local(conj_z[0])],
        OpMode::Primal,
    );
    let sum = builder.add_op(
        ComplexVectorOp::ReduceSum {
            axes: vec![0],
            input_shape: vec![2],
        },
        vec![ValRef::Local(abs_sq[0])],
        OpMode::Primal,
    );
    let y_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sum[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_complex_vector_abs_squared() -> (
    Arc<Fragment<ComplexVectorOp>>,
    GlobalValKey<ComplexVectorOp>,
) {
    let mut builder = FragmentBuilder::<ComplexVectorOp>::new();
    let z = builder.add_input(cvk("z"));
    let conj_z = builder.add_op(
        ComplexVectorOp::Conj,
        vec![ValRef::Local(z)],
        OpMode::Primal,
    );
    let y = builder.add_op(
        ComplexVectorOp::Mul,
        vec![ValRef::Local(z), ValRef::Local(conj_z[0])],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

#[test]
fn multi_output_sincos_jvp_sum_matches_expected() {
    let (primal, _sin_key, _cos_key, sum_key) = build_sincos_sum();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&sum_key),
        &[sk("x")],
        1001,
        &mut (),
        &HashMap::new(),
    );

    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dx_key = tangent_input_key(&linear, 0);
    let x = 1.0;
    let result = evaluate(
        vec![primal, Arc::new(linear.fragment)],
        &[sum_key, dy_key],
        &[(ext_input_key("x"), x), (dx_key, 1.0)],
    );

    assert_scalar_approx_eq(result[0], x.sin() + x.cos(), TOL);
    assert_scalar_approx_eq(result[1], x.cos() - x.sin(), TOL);
}

#[test]
fn multi_output_sincos_vjp_matches_expected() {
    let (primal, sin_key, cos_key, _sum_key) = build_sincos_sum();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        &[sin_key.clone(), cos_key.clone()],
        &[sk("x")],
        1002,
        &mut (),
        &HashMap::new(),
    );
    let transposed = transpose(&linear, &mut ());

    let ct_sin_key = tangent_input_key(&transposed, 0);
    let ct_cos_key = tangent_input_key(&transposed, 1);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let x = 1.0;
    let result = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[
            (ext_input_key("x"), x),
            (ct_sin_key, 1.0),
            (ct_cos_key, 1.0),
        ],
    );

    assert_scalar_approx_eq(result[0], x.cos() - x.sin(), TOL);
}

#[test]
fn multi_output_sincos_adjoint_consistency() {
    let (primal, sin_key, cos_key, _sum_key) = build_sincos_sum();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        &[sin_key.clone(), cos_key.clone()],
        &[sk("x")],
        1003,
        &mut (),
        &HashMap::new(),
    );
    let dy_sin_key = tangent_output_key(&linear, 0).expect("active tangent output for sin");
    let dy_cos_key = tangent_output_key(&linear, 1).expect("active tangent output for cos");
    let dx_key = tangent_input_key(&linear, 0);
    let transposed = transpose(&linear, &mut ());
    let linear_fragment = Arc::new(linear.fragment);

    let x = 1.0;
    let dx = 0.4;
    let ct_sin = 1.5;
    let ct_cos = -0.25;
    let dy = evaluate(
        vec![primal.clone(), linear_fragment],
        &[dy_sin_key, dy_cos_key],
        &[(ext_input_key("x"), x), (dx_key, dx)],
    );

    let ct_sin_key = tangent_input_key(&transposed, 0);
    let ct_cos_key = tangent_input_key(&transposed, 1);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let ct_x = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[
            (ext_input_key("x"), x),
            (ct_sin_key, ct_sin),
            (ct_cos_key, ct_cos),
        ],
    )[0];

    let lhs = ct_sin * dy[0] + ct_cos * dy[1];
    let rhs = ct_x * dx;
    assert_scalar_approx_eq(ct_x, ct_sin * x.cos() - ct_cos * x.sin(), TOL);
    assert_scalar_approx_eq(lhs, rhs, TOL);
}

#[test]
fn deep_chain_exp_10x() {
    let depth = 10;
    let x = 0.01;
    let a = 0.3;
    let dx = 0.7;
    let ct_y = -0.4;
    let (primal, y_key) = build_scaled_exp_chain(depth);
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        1101,
        &mut (),
        &HashMap::new(),
    );
    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dx_key = tangent_input_key(&linear, 0);
    let transposed = transpose(&linear, &mut ());
    let linear_fragment = Arc::new(linear.fragment);

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let dy = evaluate(
        vec![primal.clone(), linear_fragment],
        &[dy_key],
        &[
            (scalar_input_key("x"), x),
            (scalar_input_key("a"), a),
            (dx_key, dx),
        ],
    )[0];
    let ct_x = evaluate(
        vec![primal.clone(), Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[
            (scalar_input_key("x"), x),
            (scalar_input_key("a"), a),
            (ct_y_key, ct_y),
        ],
    )[0];
    let (_value, derivative) = scaled_exp_chain_value_and_derivative(x, a, depth);
    let numerical = five_point_derivative(
        |point| {
            evaluate(
                vec![primal.clone()],
                std::slice::from_ref(&y_key),
                &[(scalar_input_key("x"), point), (scalar_input_key("a"), a)],
            )[0]
        },
        x,
        1e-4,
    );

    assert_scalar_approx_eq(dy, derivative * dx, TOL);
    assert_scalar_approx_eq(ct_x, derivative * ct_y, TOL);
    assert_scalar_approx_eq(ct_y * dy, ct_x * dx, TOL);
    assert!(
        (numerical - derivative).abs() <= NUM_TOL,
        "expected numerical derivative close to {derivative}, got {numerical}"
    );
}

#[test]
fn third_order_x_cubed() {
    let (primal, y_key) = build_x_cubed();
    let linear_1 = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        1201,
        &mut (),
        &HashMap::new(),
    );
    let dy_key = tangent_output_key(&linear_1, 0).expect("active first-order tangent output");
    let dx1_key = tangent_input_key(&linear_1, 0);
    let linear_1_fragment = Arc::new(linear_1.fragment);

    let linear_2 = differentiate(
        &resolve(vec![primal.clone(), linear_1_fragment.clone()]),
        std::slice::from_ref(&dy_key),
        &[sk("x")],
        1202,
        &mut (),
        &HashMap::new(),
    );
    let d2y_key = tangent_output_key(&linear_2, 0).expect("active second-order tangent output");
    let dx2_key = tangent_input_key(&linear_2, 0);
    let linear_2_fragment = Arc::new(linear_2.fragment);

    let linear_3 = differentiate(
        &resolve(vec![
            primal.clone(),
            linear_1_fragment.clone(),
            linear_2_fragment.clone(),
        ]),
        std::slice::from_ref(&d2y_key),
        &[sk("x")],
        1203,
        &mut (),
        &HashMap::new(),
    );
    let d3y_key = tangent_output_key(&linear_3, 0).expect("active third-order tangent output");
    let dx3_key = tangent_input_key(&linear_3, 0);

    let result = evaluate(
        vec![
            primal,
            linear_1_fragment,
            linear_2_fragment,
            Arc::new(linear_3.fragment),
        ],
        &[d3y_key],
        &[
            (scalar_input_key("x"), 2.0),
            (dx1_key, 1.0),
            (dx2_key, 1.0),
            (dx3_key, 1.0),
        ],
    );

    assert_scalar_approx_eq(result[0], 6.0, TOL);
}

#[test]
fn fourth_order_x_fourth() {
    let (primal, y_key) = build_x_fourth();
    let linear_1 = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        1211,
        &mut (),
        &HashMap::new(),
    );
    let dy_key = tangent_output_key(&linear_1, 0).expect("active first-order tangent output");
    let dx1_key = tangent_input_key(&linear_1, 0);
    let linear_1_fragment = Arc::new(linear_1.fragment);

    let linear_2 = differentiate(
        &resolve(vec![primal.clone(), linear_1_fragment.clone()]),
        std::slice::from_ref(&dy_key),
        &[sk("x")],
        1212,
        &mut (),
        &HashMap::new(),
    );
    let d2y_key = tangent_output_key(&linear_2, 0).expect("active second-order tangent output");
    let dx2_key = tangent_input_key(&linear_2, 0);
    let linear_2_fragment = Arc::new(linear_2.fragment);

    let linear_3 = differentiate(
        &resolve(vec![
            primal.clone(),
            linear_1_fragment.clone(),
            linear_2_fragment.clone(),
        ]),
        std::slice::from_ref(&d2y_key),
        &[sk("x")],
        1213,
        &mut (),
        &HashMap::new(),
    );
    let d3y_key = tangent_output_key(&linear_3, 0).expect("active third-order tangent output");
    let dx3_key = tangent_input_key(&linear_3, 0);
    let linear_3_fragment = Arc::new(linear_3.fragment);

    let linear_4 = differentiate(
        &resolve(vec![
            primal.clone(),
            linear_1_fragment.clone(),
            linear_2_fragment.clone(),
            linear_3_fragment.clone(),
        ]),
        std::slice::from_ref(&d3y_key),
        &[sk("x")],
        1214,
        &mut (),
        &HashMap::new(),
    );
    let d4y_key = tangent_output_key(&linear_4, 0).expect("active fourth-order tangent output");
    let dx4_key = tangent_input_key(&linear_4, 0);

    let result = evaluate(
        vec![
            primal,
            linear_1_fragment,
            linear_2_fragment,
            linear_3_fragment,
            Arc::new(linear_4.fragment),
        ],
        &[d4y_key],
        &[
            (scalar_input_key("x"), 1.0),
            (dx1_key, 1.0),
            (dx2_key, 1.0),
            (dx3_key, 1.0),
            (dx4_key, 1.0),
        ],
    );

    assert_scalar_approx_eq(result[0], 24.0, TOL);
}

#[test]
fn third_order_for_then_f() {
    let (primal, y_key) = build_exp_x();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        1221,
        &mut (),
        &HashMap::new(),
    );
    let transposed = transpose(&linear, &mut ());
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let ct_y_key = tangent_input_key(&transposed, 0);
    let transposed_fragment = Arc::new(transposed.fragment);

    let linear_2 = differentiate(
        &resolve(vec![primal.clone(), transposed_fragment.clone()]),
        std::slice::from_ref(&ct_x_key),
        &[sk("x")],
        1222,
        &mut (),
        &HashMap::new(),
    );
    let d_ct_x_key = tangent_output_key(&linear_2, 0).expect("active forward-over-reverse output");
    let dx2_key = tangent_input_key(&linear_2, 0);
    let linear_2_fragment = Arc::new(linear_2.fragment);

    let linear_3 = differentiate(
        &resolve(vec![
            primal.clone(),
            transposed_fragment.clone(),
            linear_2_fragment.clone(),
        ]),
        std::slice::from_ref(&d_ct_x_key),
        &[sk("x")],
        1223,
        &mut (),
        &HashMap::new(),
    );
    let d2_ct_x_key = tangent_output_key(&linear_3, 0).expect("active third-order output");
    let dx3_key = tangent_input_key(&linear_3, 0);

    let result = evaluate(
        vec![
            primal,
            transposed_fragment,
            linear_2_fragment,
            Arc::new(linear_3.fragment),
        ],
        &[d2_ct_x_key],
        &[
            (scalar_input_key("x"), 1.0),
            (ct_y_key, 1.0),
            (dx2_key, 1.0),
            (dx3_key, 1.0),
        ],
    );

    assert_scalar_approx_eq(result[0], 1.0_f64.exp(), TOL);
}

#[test]
fn fofof_vector_x_cubed() {
    // f(x) = x * x * x elementwise, x = [2.0, 3.0]
    // f'  = 3x^2  -> with dx1=[1,1]: [12, 27]
    // f'' = 6x    -> with dx2=[1,1]: [12, 18]
    // f'''= 6     -> with dx3=[1,1]: [6, 6]
    let (primal, y_key) = build_vector_x_cubed();
    let linear_1 = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[vk("x")],
        1401,
        &mut (),
        &HashMap::new(),
    );
    let dy_key = tangent_output_key(&linear_1, 0).expect("active first-order tangent output");
    let dx1_key = tangent_input_key(&linear_1, 0);
    let linear_1_fragment = Arc::new(linear_1.fragment);

    let linear_2 = differentiate(
        &resolve(vec![primal.clone(), linear_1_fragment.clone()]),
        std::slice::from_ref(&dy_key),
        &[vk("x")],
        1402,
        &mut (),
        &HashMap::new(),
    );
    let d2y_key = tangent_output_key(&linear_2, 0).expect("active second-order tangent output");
    let dx2_key = tangent_input_key(&linear_2, 0);
    let linear_2_fragment = Arc::new(linear_2.fragment);

    let linear_3 = differentiate(
        &resolve(vec![
            primal.clone(),
            linear_1_fragment.clone(),
            linear_2_fragment.clone(),
        ]),
        std::slice::from_ref(&d2y_key),
        &[vk("x")],
        1403,
        &mut (),
        &HashMap::new(),
    );
    let d3y_key = tangent_output_key(&linear_3, 0).expect("active third-order tangent output");
    let dx3_key = tangent_input_key(&linear_3, 0);

    let result = evaluate(
        vec![
            primal,
            linear_1_fragment,
            linear_2_fragment,
            Arc::new(linear_3.fragment),
        ],
        &[d3y_key],
        &[
            (vector_input_key("x"), vector(&[2.0, 3.0])),
            (dx1_key, vector(&[1.0, 1.0])),
            (dx2_key, vector(&[1.0, 1.0])),
            (dx3_key, vector(&[1.0, 1.0])),
        ],
    );

    assert_tensor_approx_eq(&result[0].0, &vector(&[6.0, 6.0]).0, TOL);
}

#[test]
fn fof_vector_adjoint_consistency() {
    // f(x) = exp(x) elementwise, x = [1.0, 2.0]
    // FoF with dx1=[0.3, 0.7], dx2=[0.5, 0.4]:
    // d2(exp(x))*dx1*dx2 = exp(x)*dx1*dx2 = [exp(1)*0.15, exp(2)*0.28]
    // FoR uses ct_y = dx2, then differentiates the cotangent output along dx1.
    let x = vector(&[1.0, 2.0]);
    let dx1 = vector(&[0.3, 0.7]);
    let dx2 = vector(&[0.5, 0.4]);
    let expected = vector(&[1.0_f64.exp() * 0.15, 2.0_f64.exp() * 0.28]);

    let (primal, y_key) = build_vector_exp_x();
    let linear_1 = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[vk("x")],
        1411,
        &mut (),
        &HashMap::new(),
    );
    let dy_key = tangent_output_key(&linear_1, 0).expect("active first-order tangent output");
    let dx1_fof_key = tangent_input_key(&linear_1, 0);
    let transposed = transpose(&linear_1, &mut ());
    let linear_1_fragment = Arc::new(linear_1.fragment);

    let linear_2 = differentiate(
        &resolve(vec![primal.clone(), linear_1_fragment.clone()]),
        std::slice::from_ref(&dy_key),
        &[vk("x")],
        1412,
        &mut (),
        &HashMap::new(),
    );
    let d2y_key = tangent_output_key(&linear_2, 0).expect("active second-order tangent output");
    let dx2_key = tangent_input_key(&linear_2, 0);
    let fof = evaluate(
        vec![
            primal.clone(),
            linear_1_fragment,
            Arc::new(linear_2.fragment),
        ],
        &[d2y_key],
        &[
            (vector_input_key("x"), x.clone()),
            (dx1_fof_key, dx1.clone()),
            (dx2_key, dx2.clone()),
        ],
    )[0]
    .clone();

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let transposed_fragment = Arc::new(transposed.fragment);
    let linear_3 = differentiate(
        &resolve(vec![primal.clone(), transposed_fragment.clone()]),
        std::slice::from_ref(&ct_x_key),
        &[vk("x")],
        1413,
        &mut (),
        &HashMap::new(),
    );
    let d_ct_x_key = tangent_output_key(&linear_3, 0).expect("active forward-over-reverse output");
    let dx1_for_key = tangent_input_key(&linear_3, 0);
    let for_result = evaluate(
        vec![primal, transposed_fragment, Arc::new(linear_3.fragment)],
        &[d_ct_x_key],
        &[
            (vector_input_key("x"), x),
            (ct_y_key, dx2),
            (dx1_for_key, dx1),
        ],
    )[0]
    .clone();

    assert_tensor_approx_eq(&fof.0, &expected.0, TOL);
    assert_tensor_approx_eq(&for_result.0, &expected.0, TOL);
    assert_tensor_approx_eq(&fof.0, &for_result.0, TOL);
}

#[test]
fn complex_vector_jvp_conj_elementwise() {
    let (primal, y_key) = build_complex_vector_conj();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[cvk("z")],
        1301,
        &mut (),
        &HashMap::new(),
    );

    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dz_key = tangent_input_key(&linear, 0);
    let z = cvector(&[Complex64::new(1.0, 1.0), Complex64::new(2.0, 3.0)]);
    let dz = cvector(&[Complex64::new(0.5, 0.3), Complex64::new(0.7, 0.1)]);
    let result = evaluate(
        vec![primal, Arc::new(linear.fragment)],
        &[dy_key],
        &[(complex_vector_input_key("z"), z), (dz_key, dz.clone())],
    );

    assert_ctensor_approx_eq(
        &result[0].0,
        &cvector(&[Complex64::new(0.5, -0.3), Complex64::new(0.7, -0.1)]).0,
        TOL,
    );
}

#[test]
fn complex_vector_vjp_sum_abs_squared() {
    let (primal, y_key) = build_complex_vector_sum_abs_squared();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[cvk("z")],
        1302,
        &mut (),
        &HashMap::new(),
    );
    let transposed = transpose(&linear, &mut ());

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_z_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let z = cvector(&[Complex64::new(1.0, 1.0), Complex64::new(2.0, 3.0)]);
    let result = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_z_key],
        &[
            (complex_vector_input_key("z"), z),
            (ct_y_key, cscalar(Complex64::new(1.0, 0.0))),
        ],
    );

    assert_ctensor_approx_eq(
        &result[0].0,
        &cvector(&[Complex64::new(2.0, 2.0), Complex64::new(4.0, 6.0)]).0,
        TOL,
    );
}

#[test]
fn complex_vector_adjoint_consistency() {
    let (primal, y_key) = build_complex_vector_abs_squared();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[cvk("z")],
        1303,
        &mut (),
        &HashMap::new(),
    );
    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dz_key = tangent_input_key(&linear, 0);
    let transposed = transpose(&linear, &mut ());
    let linear_fragment = Arc::new(linear.fragment);

    let z = cvector(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);
    let dz = cvector(&[Complex64::new(0.2, -0.4), Complex64::new(-0.1, 0.3)]);
    let ct_y = cvector(&[Complex64::new(1.0, -0.5), Complex64::new(-0.25, 0.75)]);
    let dy = evaluate(
        vec![primal.clone(), linear_fragment],
        &[dy_key],
        &[
            (complex_vector_input_key("z"), z.clone()),
            (dz_key, dz.clone()),
        ],
    )[0]
    .clone();

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_z_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let ct_z = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_z_key],
        &[(complex_vector_input_key("z"), z), (ct_y_key, ct_y.clone())],
    )[0]
    .clone();

    assert_scalar_approx_eq(
        complex_tensor_inner_product(&ct_y, &dy),
        complex_tensor_inner_product(&ct_z, &dz),
        TOL,
    );
}
