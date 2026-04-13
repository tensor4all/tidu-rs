use std::collections::HashMap;
mod common;

use std::sync::Arc;

use chainrules::{ADKey, DiffPassId, PrimitiveOp};
use common::assertions::{
    assert_complex_approx_eq, assert_scalar_approx_eq, assert_tensor_approx_eq,
};
use common::{evaluate, tangent_input_key, tangent_output_key, ScalarKey, ScalarOp};
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::{EvalGraphOp, GraphOp, OpEmitter};
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex64;
use tidu::{differentiate, transpose};

const TOL: f64 = 1e-10;

fn sk(name: &str) -> ScalarKey {
    ScalarKey::User(name.to_string())
}

fn scalar_input_key(name: &str) -> GlobalValKey<ScalarOp> {
    GlobalValKey::Input(sk(name))
}

fn build_scalar_exp_ax() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let a = builder.add_input(sk("a"));
    let ax = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(a)],
        OpMode::Primal,
    );
    let y = builder.add_op(ScalarOp::Exp, vec![ValRef::Local(ax[0])], OpMode::Primal);
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_scalar_x_plus_x_times_x() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let sum = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    let y = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(sum[0]), ValRef::Local(x)],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_scalar_inactive_exp_y() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let _x = builder.add_input(sk("x"));
    let y = builder.add_input(sk("y"));
    let exp_y = builder.add_op(ScalarOp::Exp, vec![ValRef::Local(y)], OpMode::Primal);
    let exp_y_key = builder.global_key(exp_y[0]).clone();
    builder.set_outputs(vec![exp_y[0]]);
    (Arc::new(builder.build()), exp_y_key)
}

fn build_scalar_diamond_exp_x_plus_exp_x() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let exp_x = builder.add_op(ScalarOp::Exp, vec![ValRef::Local(x)], OpMode::Primal);
    let y = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(exp_x[0]), ValRef::Local(exp_x[0])],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_scalar_x_times_y() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let y = builder.add_input(sk("y"));
    let product = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(y)],
        OpMode::Primal,
    );
    let product_key = builder.global_key(product[0]).clone();
    builder.set_outputs(vec![product[0]]);
    (Arc::new(builder.build()), product_key)
}

fn build_scalar_identity() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let y_key = builder.global_key(x).clone();
    builder.set_outputs(vec![x]);
    (Arc::new(builder.build()), y_key)
}

fn build_scalar_output_y() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let _x = builder.add_input(sk("x"));
    let y = builder.add_input(sk("y"));
    let y_key = builder.global_key(y).clone();
    builder.set_outputs(vec![y]);
    (Arc::new(builder.build()), y_key)
}

define_ad_key!(ComplexScalarKey);

#[derive(Clone, Debug, PartialEq)]
struct C64(Complex64);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ComplexScalarOp {
    Add,
    Mul,
    Conj,
}

impl GraphOp for ComplexScalarOp {
    type Operand = C64;
    type Context = ();
    type InputKey = ComplexScalarKey;

    fn n_inputs(&self) -> usize {
        match self {
            Self::Add | Self::Mul => 2,
            Self::Conj => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl EvalGraphOp for ComplexScalarOp {
    fn eval(&self, _ctx: &mut (), inputs: &[&C64]) -> Vec<C64> {
        match self {
            Self::Add => vec![C64(inputs[0].0 + inputs[1].0)],
            Self::Mul => vec![C64(inputs[0].0 * inputs[1].0)],
            Self::Conj => vec![C64(inputs[0].0.conj())],
        }
    }
}

impl PrimitiveOp for ComplexScalarOp {
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
            Self::Add => {
                linearize_add!(builder, ComplexScalarOp::Add, tangent_in[0], tangent_in[1])
            }
            Self::Mul => linearize_mul!(
                builder,
                ComplexScalarOp::Mul,
                ComplexScalarOp::Add,
                primal_in,
                tangent_in[0],
                tangent_in[1]
            ),
            Self::Conj => linearize_conj!(builder, ComplexScalarOp::Conj, tangent_in[0]),
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
            Self::Mul => transpose_mul_complex!(
                builder,
                ComplexScalarOp::Mul,
                ComplexScalarOp::Conj,
                inputs,
                ct,
                mode
            ),
            Self::Conj => transpose_conj!(builder, ComplexScalarOp::Conj, ct),
        }
    }
}

fn ck(name: &str) -> ComplexScalarKey {
    ComplexScalarKey::User(name.to_string())
}

fn complex_input_key(name: &str) -> GlobalValKey<ComplexScalarOp> {
    GlobalValKey::Input(ck(name))
}

fn c(re: f64, im: f64) -> C64 {
    C64(Complex64::new(re, im))
}

fn complex_inner_product(lhs: &C64, rhs: &C64) -> f64 {
    (lhs.0.conj() * rhs.0).re
}

fn build_complex_abs_squared() -> (
    Arc<Fragment<ComplexScalarOp>>,
    GlobalValKey<ComplexScalarOp>,
) {
    let mut builder = FragmentBuilder::<ComplexScalarOp>::new();
    let z = builder.add_input(ck("z"));
    let conj_z = builder.add_op(
        ComplexScalarOp::Conj,
        vec![ValRef::Local(z)],
        OpMode::Primal,
    );
    let y = builder.add_op(
        ComplexScalarOp::Mul,
        vec![ValRef::Local(z), ValRef::Local(conj_z[0])],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_complex_z_squared() -> (
    Arc<Fragment<ComplexScalarOp>>,
    GlobalValKey<ComplexScalarOp>,
) {
    let mut builder = FragmentBuilder::<ComplexScalarOp>::new();
    let z = builder.add_input(ck("z"));
    let y = builder.add_op(
        ComplexScalarOp::Mul,
        vec![ValRef::Local(z), ValRef::Local(z)],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

define_ad_key!(VectorKey);

#[derive(Clone, Debug, PartialEq)]
struct Tensor(ArrayD<f64>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum VectorOp {
    Add,
    Mul,
}

impl GraphOp for VectorOp {
    type Operand = Tensor;
    type Context = ();
    type InputKey = VectorKey;

    fn n_inputs(&self) -> usize {
        2
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
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        match self {
            Self::Add => linearize_add!(builder, VectorOp::Add, tangent_in[0], tangent_in[1]),
            Self::Mul => linearize_mul!(
                builder,
                VectorOp::Mul,
                VectorOp::Add,
                primal_in,
                tangent_in[0],
                tangent_in[1]
            ),
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

fn build_vector_x_squared() -> (Arc<Fragment<VectorOp>>, GlobalValKey<VectorOp>) {
    let mut builder = FragmentBuilder::<VectorOp>::new();
    let x = builder.add_input(vk("x"));
    let y = builder.add_op(
        VectorOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

#[test]
fn adjoint_consistency_exp_ax() {
    let (primal, y_key) = build_scalar_exp_ax();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        101,
        &mut (),
        &HashMap::new(),
    );
    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dx_key = tangent_input_key(&linear, 0);
    let transposed = transpose(&linear, &mut ());
    let linear_fragment = Arc::new(linear.fragment);

    let dx = 0.7;
    let ct_y = 0.3;
    let dy = evaluate(
        vec![primal.clone(), linear_fragment],
        &[dy_key],
        &[
            (scalar_input_key("x"), 1.0),
            (scalar_input_key("a"), 2.0),
            (dx_key, dx),
        ],
    )[0];

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let ct_x = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[
            (scalar_input_key("x"), 1.0),
            (scalar_input_key("a"), 2.0),
            (ct_y_key, ct_y),
        ],
    )[0];

    assert_scalar_approx_eq(dy, 2.0 * 2.0_f64.exp() * dx, TOL);
    assert_scalar_approx_eq(ct_x, 2.0 * 2.0_f64.exp() * ct_y, TOL);
    assert_scalar_approx_eq(ct_y * dy, ct_x * dx, TOL);
}

#[test]
fn adjoint_consistency_x_plus_x_times_x() {
    let (primal, y_key) = build_scalar_x_plus_x_times_x();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        102,
        &mut (),
        &HashMap::new(),
    );
    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dx_key = tangent_input_key(&linear, 0);
    let transposed = transpose(&linear, &mut ());
    let linear_fragment = Arc::new(linear.fragment);

    let dx = 0.5;
    let ct_y = 0.7;
    let dy = evaluate(
        vec![primal.clone(), linear_fragment],
        &[dy_key],
        &[(scalar_input_key("x"), 3.0), (dx_key, dx)],
    )[0];

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let ct_x = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[(scalar_input_key("x"), 3.0), (ct_y_key, ct_y)],
    )[0];

    assert_scalar_approx_eq(dy, 6.0, TOL);
    assert_scalar_approx_eq(ct_x, 8.4, TOL);
    assert_scalar_approx_eq(ct_y * dy, ct_x * dx, TOL);
}

#[test]
fn adjoint_consistency_complex() {
    let (primal, y_key) = build_complex_abs_squared();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[ck("z")],
        103,
        &mut (),
        &HashMap::new(),
    );
    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dz_key = tangent_input_key(&linear, 0);
    let transposed = transpose(&linear, &mut ());
    let linear_fragment = Arc::new(linear.fragment);

    let dz = c(0.3, 0.4);
    let ct_y = c(0.5, 0.6);
    let dy = evaluate(
        vec![primal.clone(), linear_fragment],
        &[dy_key],
        &[(complex_input_key("z"), c(1.0, 2.0)), (dz_key, dz.clone())],
    )[0]
    .clone();

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_z_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let ct_z = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_z_key],
        &[
            (complex_input_key("z"), c(1.0, 2.0)),
            (ct_y_key, ct_y.clone()),
        ],
    )[0]
    .clone();

    assert_complex_approx_eq(dy.0, Complex64::new(2.2, 0.0), TOL);
    assert_complex_approx_eq(ct_z.0, Complex64::new(1.0, 2.0), TOL);
    assert_scalar_approx_eq(
        complex_inner_product(&ct_y, &dy),
        complex_inner_product(&ct_z, &dz),
        TOL,
    );
}

#[test]
fn inactive_tangent_returns_none() {
    let (primal, y_key) = build_scalar_inactive_exp_y();
    let linear = differentiate(
        &resolve(vec![primal]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        104,
        &mut (),
        &HashMap::new(),
    );

    assert!(
        tangent_output_key(&linear, 0).is_none(),
        "inactive tangent should stay None"
    );
    assert!(
        linear.fragment.outputs().is_empty(),
        "inactive tangent should not create linear outputs"
    );
}

#[test]
fn diamond_pattern_shared_subexpression() {
    let (primal, y_key) = build_scalar_diamond_exp_x_plus_exp_x();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        105,
        &mut (),
        &HashMap::new(),
    );
    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dx_key = tangent_input_key(&linear, 0);
    let transposed = transpose(&linear, &mut ());
    let linear_fragment = Arc::new(linear.fragment);

    let dy = evaluate(
        vec![primal.clone(), linear_fragment],
        &[dy_key],
        &[(scalar_input_key("x"), 1.0), (dx_key, 1.0)],
    )[0];

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let ct_x = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[(scalar_input_key("x"), 1.0), (ct_y_key, 1.0)],
    )[0];

    let expected = 2.0 * 1.0_f64.exp();
    assert_scalar_approx_eq(dy, expected, TOL);
    assert_scalar_approx_eq(ct_x, expected, TOL);
}

#[test]
fn multi_variable_vjp() {
    let (primal, y_key) = build_scalar_x_times_y();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x"), sk("y")],
        106,
        &mut (),
        &HashMap::new(),
    );
    let transposed = transpose(&linear, &mut ());

    let ct_output_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent for x");
    let ct_y_key = tangent_output_key(&transposed, 1).expect("active cotangent for y");
    let results = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_x_key, ct_y_key],
        &[
            (scalar_input_key("x"), 2.0),
            (scalar_input_key("y"), 3.0),
            (ct_output_key, 1.0),
        ],
    );

    assert_scalar_approx_eq(results[0], 3.0, TOL);
    assert_scalar_approx_eq(results[1], 2.0, TOL);
}

#[test]
fn ror_x_plus_x_times_x() {
    let (primal, y_key) = build_scalar_x_plus_x_times_x();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        107,
        &mut (),
        &HashMap::new(),
    );
    let transposed = transpose(&linear, &mut ());
    let reverse_of_reverse = transpose(&transposed, &mut ());
    let d_ct_x_key = tangent_input_key(&reverse_of_reverse, 0);
    let d_ct_y_key =
        tangent_output_key(&reverse_of_reverse, 0).expect("active reverse-of-reverse output");

    let result = evaluate(
        vec![
            primal,
            Arc::new(transposed.fragment),
            Arc::new(reverse_of_reverse.fragment),
        ],
        &[d_ct_y_key],
        &[(scalar_input_key("x"), 3.0), (d_ct_x_key, 1.0)],
    );

    assert_scalar_approx_eq(result[0], 12.0, TOL);
}

#[test]
fn for_complex_z_squared() {
    let (primal, y_key) = build_complex_z_squared();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[ck("z")],
        108,
        &mut (),
        &HashMap::new(),
    );
    let transposed = transpose(&linear, &mut ());
    let ct_z_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let ct_y_seed_key = tangent_input_key(&transposed, 0);
    let transposed_fragment = Arc::new(transposed.fragment);

    let second_linear = differentiate(
        &resolve(vec![primal.clone(), transposed_fragment.clone()]),
        std::slice::from_ref(&ct_z_key),
        &[ck("z")],
        109,
        &mut (),
        &HashMap::new(),
    );
    let d_ct_z_key =
        tangent_output_key(&second_linear, 0).expect("active forward-over-reverse output");
    let dz_key = tangent_input_key(&second_linear, 0);

    let result = evaluate(
        vec![
            primal,
            transposed_fragment,
            Arc::new(second_linear.fragment),
        ],
        &[d_ct_z_key],
        &[
            (complex_input_key("z"), c(1.0, 2.0)),
            (ct_y_seed_key, c(1.0, 0.0)),
            (dz_key, c(1.0, 0.0)),
        ],
    );

    assert_complex_approx_eq(result[0].0, Complex64::new(2.0, 0.0), TOL);
}

#[test]
fn jvp_identity() {
    let (primal, y_key) = build_scalar_identity();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        110,
        &mut (),
        &HashMap::new(),
    );
    let dy_key = tangent_output_key(&linear, 0).expect("identity should keep tangent active");
    let dx_key = tangent_input_key(&linear, 0);
    let transposed = transpose(&linear, &mut ());
    let linear_fragment = Arc::new(linear.fragment);

    let dy = evaluate(
        vec![primal.clone(), linear_fragment],
        &[dy_key],
        &[(scalar_input_key("x"), 4.0), (dx_key, 1.0)],
    )[0];

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("identity VJP should stay active");
    let ct_x = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[(scalar_input_key("x"), 4.0), (ct_y_key, 1.0)],
    )[0];

    assert_scalar_approx_eq(dy, 1.0, TOL);
    assert_scalar_approx_eq(ct_x, 1.0, TOL);
}

#[test]
fn vjp_constant_output() {
    let (primal, y_key) = build_scalar_output_y();
    let linear = differentiate(
        &resolve(vec![primal]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        111,
        &mut (),
        &HashMap::new(),
    );
    let transposed = transpose(&linear, &mut ());

    assert!(
        tangent_output_key(&linear, 0).is_none(),
        "constant output should have no tangent"
    );
    assert!(
        transposed.tangent_inputs.is_empty(),
        "inactive output should not require cotangent seeds"
    );
    assert!(
        tangent_output_key(&transposed, 0).is_none(),
        "constant output should produce no cotangent for x"
    );
}

#[test]
fn fof_vector_x_squared() {
    let (primal, y_key) = build_vector_x_squared();
    let linear_1 = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[vk("x")],
        112,
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
        113,
        &mut (),
        &HashMap::new(),
    );
    let d2y_key = tangent_output_key(&linear_2, 0).expect("active second-order tangent output");
    let dx2_key = tangent_input_key(&linear_2, 0);

    let result = evaluate(
        vec![primal, linear_1_fragment, Arc::new(linear_2.fragment)],
        &[d2y_key],
        &[
            (vector_input_key("x"), vector(&[2.0, 3.0])),
            (dx1_key, vector(&[1.0, 1.0])),
            (dx2_key, vector(&[1.0, 1.0])),
        ],
    );

    assert_tensor_approx_eq(&result[0].0, &vector(&[2.0, 2.0]).0, TOL);
}
