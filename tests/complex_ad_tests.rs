use std::collections::HashMap;
mod common;

use std::sync::Arc;

use common::assertions::assert_complex_approx_eq;
use common::{evaluate, tangent_input_key, tangent_output_key};
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::{EvalGraphOp, GraphOp};
use num_complex::Complex64;
use tidu::{differentiate, transpose};
use tidu::{ADKey, DiffPassId, Primitive, PrimitiveBuilder, PrimitiveValue};

const TOL: f64 = 1e-10;
const NUM_TOL: f64 = 1e-5;

define_ad_key!(ComplexScalarKey);

#[derive(Clone, Debug, PartialEq)]
struct C64(Complex64);

impl C64 {
    #[allow(dead_code)]
    fn zero() -> Self {
        Self(Complex64::new(0.0, 0.0))
    }

    #[allow(dead_code)]
    fn one() -> Self {
        Self(Complex64::new(1.0, 0.0))
    }

    #[allow(dead_code)]
    fn conj(&self) -> Self {
        Self(self.0.conj())
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ComplexScalarOp {
    Add,
    Mul,
    Exp,
    Neg,
    Conj,
}

impl GraphOp for ComplexScalarOp {
    type Operand = C64;
    type Context = ();
    type InputKey = ComplexScalarKey;

    fn n_inputs(&self) -> usize {
        match self {
            ComplexScalarOp::Add | ComplexScalarOp::Mul => 2,
            ComplexScalarOp::Exp | ComplexScalarOp::Neg | ComplexScalarOp::Conj => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl EvalGraphOp for ComplexScalarOp {
    fn eval(&self, _ctx: &mut (), inputs: &[&C64]) -> Vec<C64> {
        match self {
            ComplexScalarOp::Add => vec![C64(inputs[0].0 + inputs[1].0)],
            ComplexScalarOp::Mul => vec![C64(inputs[0].0 * inputs[1].0)],
            ComplexScalarOp::Exp => vec![C64(inputs[0].0.exp())],
            ComplexScalarOp::Neg => vec![C64(-inputs[0].0)],
            ComplexScalarOp::Conj => vec![C64(inputs[0].0.conj())],
        }
    }
}

impl Primitive for ComplexScalarOp {
    type ADContext = ();

    fn add() -> Self {
        ComplexScalarOp::Add
    }

    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        match self {
            ComplexScalarOp::Add => {
                linearize_add!(builder, ComplexScalarOp::Add, tangent_in[0], tangent_in[1])
            }
            ComplexScalarOp::Mul => linearize_mul!(
                builder,
                ComplexScalarOp::Mul,
                ComplexScalarOp::Add,
                primal_in,
                tangent_in[0],
                tangent_in[1]
            ),
            ComplexScalarOp::Exp => {
                linearize_exp!(builder, ComplexScalarOp::Mul, primal_out[0], tangent_in[0])
            }
            ComplexScalarOp::Neg => linearize_neg!(builder, ComplexScalarOp::Neg, tangent_in[0]),
            ComplexScalarOp::Conj => linearize_conj!(builder, ComplexScalarOp::Conj, tangent_in[0]),
        }
    }

    fn transpose_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[PrimitiveValue<Self>],
        mode: &OpMode,
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        let ct = match cotangent_out[0] {
            Some(ct) => ct,
            None => return vec![None; self.n_inputs()],
        };

        match self {
            ComplexScalarOp::Add => transpose_add!(ct),
            ComplexScalarOp::Mul => transpose_mul_complex!(
                builder,
                ComplexScalarOp::Mul,
                ComplexScalarOp::Conj,
                inputs,
                ct,
                mode
            ),
            ComplexScalarOp::Exp => panic!("transpose_rule called on primal-only Exp"),
            ComplexScalarOp::Neg => transpose_neg!(builder, ComplexScalarOp::Neg, ct),
            ComplexScalarOp::Conj => transpose_conj!(builder, ComplexScalarOp::Conj, ct),
        }
    }
}

fn ck(name: &str) -> ComplexScalarKey {
    ComplexScalarKey::User(name.to_string())
}

fn input_key(name: &str) -> GlobalValKey<ComplexScalarOp> {
    GlobalValKey::Input(ck(name))
}

fn c(re: f64, im: f64) -> C64 {
    C64(Complex64::new(re, im))
}

fn build_conj_z() -> (
    Arc<Fragment<ComplexScalarOp>>,
    GlobalValKey<ComplexScalarOp>,
) {
    let mut builder = FragmentBuilder::<ComplexScalarOp>::new();
    let z = builder.add_input(ck("z"));
    let y = builder.add_op(
        ComplexScalarOp::Conj,
        vec![ValRef::Local(z)],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_z_times_w() -> (
    Arc<Fragment<ComplexScalarOp>>,
    GlobalValKey<ComplexScalarOp>,
) {
    let mut builder = FragmentBuilder::<ComplexScalarOp>::new();
    let z = builder.add_input(ck("z"));
    let w = builder.add_input(ck("w"));
    let y = builder.add_op(
        ComplexScalarOp::Mul,
        vec![ValRef::Local(z), ValRef::Local(w)],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_c_times_z() -> (
    Arc<Fragment<ComplexScalarOp>>,
    GlobalValKey<ComplexScalarOp>,
) {
    let mut builder = FragmentBuilder::<ComplexScalarOp>::new();
    let cst = builder.add_input(ck("c"));
    let z = builder.add_input(ck("z"));
    let y = builder.add_op(
        ComplexScalarOp::Mul,
        vec![ValRef::Local(cst), ValRef::Local(z)],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn build_abs_squared() -> (
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

fn build_exp_z() -> (
    Arc<Fragment<ComplexScalarOp>>,
    GlobalValKey<ComplexScalarOp>,
) {
    let mut builder = FragmentBuilder::<ComplexScalarOp>::new();
    let z = builder.add_input(ck("z"));
    let y = builder.add_op(ComplexScalarOp::Exp, vec![ValRef::Local(z)], OpMode::Primal);
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn finite_difference_loss(
    primal: Arc<Fragment<ComplexScalarOp>>,
    y_key: &GlobalValKey<ComplexScalarOp>,
    z: Complex64,
    seed: Complex64,
) -> Complex64 {
    let h = 1e-4;
    let sample = |point: Complex64| {
        let value = evaluate(
            vec![primal.clone()],
            std::slice::from_ref(y_key),
            &[(input_key("z"), C64(point))],
        )[0]
        .0;
        (seed.conj() * value).re
    };
    let five_point = |mk_point: &dyn Fn(f64) -> Complex64| {
        (-sample(mk_point(2.0 * h)) + 8.0 * sample(mk_point(h)) - 8.0 * sample(mk_point(-h))
            + sample(mk_point(-2.0 * h)))
            / (12.0 * h)
    };

    Complex64::new(
        five_point(&|delta| Complex64::new(z.re + delta, z.im)),
        five_point(&|delta| Complex64::new(z.re, z.im + delta)),
    )
}

#[test]
fn jvp_conj_z() {
    let (primal, y_key) = build_conj_z();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[ck("z")],
        1,
        &mut (),
        &HashMap::new(),
    );

    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dz_key = tangent_input_key(&linear, 0);
    let result = evaluate(
        vec![primal, Arc::new(linear.fragment)],
        &[dy_key],
        &[(input_key("z"), c(2.0, -3.0)), (dz_key, c(1.0, 1.0))],
    );

    assert_complex_approx_eq(result[0].0, Complex64::new(1.0, -1.0), TOL);
}

#[test]
fn vjp_conj_z() {
    let (primal, y_key) = build_conj_z();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[ck("z")],
        2,
        &mut (),
        &HashMap::new(),
    );
    let transposed = transpose(&linear, &mut ());

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_z_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let result = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_z_key],
        &[(input_key("z"), c(-0.5, 0.75)), (ct_y_key, c(1.0, 1.0))],
    );

    assert_complex_approx_eq(result[0].0, Complex64::new(1.0, -1.0), TOL);
}

#[test]
fn jvp_z_times_w() {
    let (primal, y_key) = build_z_times_w();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[ck("z"), ck("w")],
        3,
        &mut (),
        &HashMap::new(),
    );

    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dz_key = tangent_input_key(&linear, 0);
    let dw_key = tangent_input_key(&linear, 1);
    let z = Complex64::new(1.0, 2.0);
    let w = Complex64::new(-3.0, 0.5);
    let dz = Complex64::new(0.5, -1.0);
    let dw = Complex64::new(-2.0, 1.5);
    let result = evaluate(
        vec![primal, Arc::new(linear.fragment)],
        &[dy_key],
        &[
            (input_key("z"), C64(z)),
            (input_key("w"), C64(w)),
            (dz_key, C64(dz)),
            (dw_key, C64(dw)),
        ],
    );

    assert_complex_approx_eq(result[0].0, dz * w + z * dw, TOL);
}

#[test]
fn vjp_c_times_z_uses_conjugated_constant() {
    let (primal, y_key) = build_c_times_z();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[ck("z")],
        4,
        &mut (),
        &HashMap::new(),
    );
    let transposed = transpose(&linear, &mut ());

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_z_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let result = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_z_key],
        &[
            (input_key("c"), c(2.0, 1.0)),
            (input_key("z"), c(1.0, 1.0)),
            (ct_y_key, c(1.0, 0.0)),
        ],
    );

    assert_complex_approx_eq(result[0].0, Complex64::new(2.0, -1.0), TOL);
}

#[test]
fn vjp_abs_squared_returns_two_z() {
    let (primal, y_key) = build_abs_squared();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[ck("z")],
        5,
        &mut (),
        &HashMap::new(),
    );
    let transposed = transpose(&linear, &mut ());

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_z_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let z = Complex64::new(1.0, 2.0);
    let result = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_z_key],
        &[(input_key("z"), C64(z)), (ct_y_key, c(1.0, 0.0))],
    );

    assert_complex_approx_eq(result[0].0, Complex64::new(2.0, 4.0), TOL);
}

#[test]
fn numerical_gradient_exp_z_matches_vjp_for_real_and_imag_losses() {
    let (primal, y_key) = build_exp_z();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[ck("z")],
        6,
        &mut (),
        &HashMap::new(),
    );
    let transposed = transpose(&linear, &mut ());
    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_z_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let transposed_fragment = Arc::new(transposed.fragment);

    let z = Complex64::new(0.4, -0.2);
    for seed in [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)] {
        let vjp = evaluate(
            vec![primal.clone(), transposed_fragment.clone()],
            std::slice::from_ref(&ct_z_key),
            &[(input_key("z"), C64(z)), (ct_y_key.clone(), C64(seed))],
        )[0]
        .0;
        let numerical = finite_difference_loss(primal.clone(), &y_key, z, seed);
        assert!(
            (vjp - numerical).norm() <= NUM_TOL,
            "seed={seed:?}: expected {numerical:?}, got {vjp:?}"
        );
    }
}
