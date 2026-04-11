mod common;

use std::sync::Arc;

use chainrules::{ADKey, DiffPassId, PrimitiveOp};
use common::{evaluate, tangent_input_key, tangent_output_key};
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::{EvalGraphOp, GraphOp};
use num_complex::Complex64;
use tidu::{differentiate, transpose};

const TOL: f64 = 1e-10;
const NUM_TOL: f64 = 1e-5;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ComplexScalarKey {
    User(String),
    Tangent {
        of: Box<ComplexScalarKey>,
        pass: DiffPassId,
    },
}

impl ADKey for ComplexScalarKey {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        ComplexScalarKey::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}

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

impl PrimitiveOp for ComplexScalarOp {
    type ADContext = ();

    fn add() -> Self {
        ComplexScalarOp::Add
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
            ComplexScalarOp::Add => match (tangent_in[0], tangent_in[1]) {
                (Some(lhs), Some(rhs)) => {
                    let sum = builder.add_op(
                        ComplexScalarOp::Add,
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
            ComplexScalarOp::Mul => {
                let mut terms = Vec::new();

                if let Some(dlhs) = tangent_in[0] {
                    let term = builder.add_op(
                        ComplexScalarOp::Mul,
                        vec![ValRef::Local(dlhs), ValRef::External(primal_in[1].clone())],
                        OpMode::Linear {
                            active_mask: vec![true, false],
                        },
                    );
                    terms.push(term[0]);
                }

                if let Some(drhs) = tangent_in[1] {
                    let term = builder.add_op(
                        ComplexScalarOp::Mul,
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
                            ComplexScalarOp::Add,
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
            ComplexScalarOp::Exp => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_op(
                        ComplexScalarOp::Mul,
                        vec![ValRef::External(primal_out[0].clone()), ValRef::Local(dx)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    vec![Some(out[0])]
                }
                None => vec![None],
            },
            ComplexScalarOp::Neg => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_op(
                        ComplexScalarOp::Neg,
                        vec![ValRef::Local(dx)],
                        OpMode::Linear {
                            active_mask: vec![true],
                        },
                    );
                    vec![Some(out[0])]
                }
                None => vec![None],
            },
            ComplexScalarOp::Conj => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_op(
                        ComplexScalarOp::Conj,
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
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        let ct = match cotangent_out[0] {
            Some(ct) => ct,
            None => return vec![None; self.n_inputs()],
        };

        match self {
            ComplexScalarOp::Add => vec![Some(ct), Some(ct)],
            ComplexScalarOp::Mul => {
                let active_mask = match mode {
                    OpMode::Linear { active_mask } => active_mask,
                    OpMode::Primal => return vec![None, None],
                };

                let mut result = vec![None, None];

                if active_mask[0] {
                    let conj_fixed = builder.add_op(
                        ComplexScalarOp::Conj,
                        vec![inputs[1].clone()],
                        OpMode::Linear {
                            active_mask: vec![false],
                        },
                    );
                    let out = builder.add_op(
                        ComplexScalarOp::Mul,
                        vec![ValRef::Local(conj_fixed[0]), ValRef::Local(ct)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    result[0] = Some(out[0]);
                }

                if active_mask[1] {
                    let conj_fixed = builder.add_op(
                        ComplexScalarOp::Conj,
                        vec![inputs[0].clone()],
                        OpMode::Linear {
                            active_mask: vec![false],
                        },
                    );
                    let out = builder.add_op(
                        ComplexScalarOp::Mul,
                        vec![ValRef::Local(conj_fixed[0]), ValRef::Local(ct)],
                        OpMode::Linear {
                            active_mask: vec![false, true],
                        },
                    );
                    result[1] = Some(out[0]);
                }

                result
            }
            ComplexScalarOp::Exp => panic!("transpose_rule called on primal-only Exp"),
            ComplexScalarOp::Neg => {
                let out = builder.add_op(
                    ComplexScalarOp::Neg,
                    vec![ValRef::Local(ct)],
                    OpMode::Linear {
                        active_mask: vec![true],
                    },
                );
                vec![Some(out[0])]
            }
            ComplexScalarOp::Conj => {
                let out = builder.add_op(
                    ComplexScalarOp::Conj,
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

fn ck(name: &str) -> ComplexScalarKey {
    ComplexScalarKey::User(name.to_string())
}

fn input_key(name: &str) -> GlobalValKey<ComplexScalarOp> {
    GlobalValKey::Input(ck(name))
}

fn c(re: f64, im: f64) -> C64 {
    C64(Complex64::new(re, im))
}

fn assert_complex_approx_eq(actual: &C64, expected: Complex64, tol: f64) {
    let delta = actual.0 - expected;
    assert!(
        delta.norm() <= tol,
        "expected {expected:?}, got {:?}, |delta|={}",
        actual.0,
        delta.norm()
    );
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
    );

    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dz_key = tangent_input_key(&linear, 0);
    let result = evaluate(
        vec![primal, Arc::new(linear.fragment)],
        &[dy_key],
        &[(input_key("z"), c(2.0, -3.0)), (dz_key, c(1.0, 1.0))],
    );

    assert_complex_approx_eq(&result[0], Complex64::new(1.0, -1.0), TOL);
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
    );
    let transposed = transpose(&linear, &mut ());

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_z_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let result = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_z_key],
        &[(input_key("z"), c(-0.5, 0.75)), (ct_y_key, c(1.0, 1.0))],
    );

    assert_complex_approx_eq(&result[0], Complex64::new(1.0, -1.0), TOL);
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

    assert_complex_approx_eq(&result[0], dz * w + z * dw, TOL);
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

    assert_complex_approx_eq(&result[0], Complex64::new(2.0, -1.0), TOL);
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

    assert_complex_approx_eq(&result[0], Complex64::new(2.0, 4.0), TOL);
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
