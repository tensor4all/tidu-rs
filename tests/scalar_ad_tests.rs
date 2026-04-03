mod common;

use std::collections::HashMap;
use std::sync::Arc;

use common::{ScalarKey, ScalarOp};
use computegraph::compile::compile;
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use tidu::{differentiate, transpose, LinearFragment};

const TOL: f64 = 1e-10;

fn sk(name: &str) -> ScalarKey {
    ScalarKey::User(name.to_string())
}

fn input_key(name: &str) -> GlobalValKey<ScalarOp> {
    GlobalValKey::Input(sk(name))
}

fn assert_approx_eq(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= TOL,
        "expected {expected}, got {actual}"
    );
}

fn evaluate(
    roots: Vec<Arc<Fragment<ScalarOp>>>,
    outputs: &[GlobalValKey<ScalarOp>],
    bindings: &[(GlobalValKey<ScalarOp>, f64)],
) -> Vec<f64> {
    let view = resolve(roots);
    let graph = materialize_merge(&view, outputs);
    let binding_map: HashMap<_, _> = bindings.iter().cloned().collect();
    let ordered_inputs: Vec<f64> = graph
        .inputs
        .iter()
        .map(|key| {
            *binding_map
                .get(key)
                .unwrap_or_else(|| panic!("missing value for input key {:?}", key))
        })
        .collect();
    let ordered_refs: Vec<&f64> = ordered_inputs.iter().collect();
    let program = compile(&graph);
    program.eval(&mut (), &ordered_refs)
}

fn tangent_input_key(linear: &LinearFragment<ScalarOp>, index: usize) -> GlobalValKey<ScalarOp> {
    let local_id = linear.tangent_inputs[index].1;
    linear.fragment.vals()[local_id].key.clone()
}

fn tangent_output_key(
    linear: &LinearFragment<ScalarOp>,
    index: usize,
) -> Option<GlobalValKey<ScalarOp>> {
    linear.tangent_outputs[index].map(|local_id| linear.fragment.vals()[local_id].key.clone())
}

fn build_x_plus_x() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let sum = builder.add_op(
        ScalarOp::Add,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    let sum_key = builder.global_key(sum[0]).clone();
    builder.set_outputs(vec![sum[0]]);
    (Arc::new(builder.build()), sum_key)
}

fn build_x_times_y() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
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

fn build_exp_ax() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
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

fn build_x_plus_x_times_x() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
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

fn build_x_squared() -> (Arc<Fragment<ScalarOp>>, GlobalValKey<ScalarOp>) {
    let mut builder = FragmentBuilder::<ScalarOp>::new();
    let x = builder.add_input(sk("x"));
    let y = builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    let y_key = builder.global_key(y[0]).clone();
    builder.set_outputs(vec![y[0]]);
    (Arc::new(builder.build()), y_key)
}

fn finite_difference_exp_ax(
    primal: Arc<Fragment<ScalarOp>>,
    y_key: &GlobalValKey<ScalarOp>,
) -> f64 {
    let h = 1e-3;
    let sample = |x: f64| {
        evaluate(
            vec![primal.clone()],
            std::slice::from_ref(y_key),
            &[(input_key("x"), x), (input_key("a"), 2.0)],
        )[0]
    };

    (-sample(1.0 + 2.0 * h) + 8.0 * sample(1.0 + h) - 8.0 * sample(1.0 - h) + sample(1.0 - 2.0 * h))
        / (12.0 * h)
}

#[test]
fn jvp_x_plus_x() {
    let (primal, y_key) = build_x_plus_x();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        1,
    );

    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dx_key = tangent_input_key(&linear, 0);
    let results = evaluate(
        vec![primal, Arc::new(linear.fragment)],
        &[y_key, dy_key],
        &[(input_key("x"), 3.0), (dx_key, 1.0)],
    );

    assert_approx_eq(results[0], 6.0);
    assert_approx_eq(results[1], 2.0);
}

#[test]
fn jvp_x_times_y() {
    let (primal, y_key) = build_x_times_y();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        2,
    );

    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dx_key = tangent_input_key(&linear, 0);
    let results = evaluate(
        vec![primal, Arc::new(linear.fragment)],
        &[y_key, dy_key],
        &[(input_key("x"), 2.0), (input_key("y"), 3.0), (dx_key, 1.0)],
    );

    assert_approx_eq(results[0], 6.0);
    assert_approx_eq(results[1], 3.0);
}

#[test]
fn jvp_exp_ax() {
    let (primal, y_key) = build_exp_ax();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        3,
    );

    let dy_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dx_key = tangent_input_key(&linear, 0);
    let results = evaluate(
        vec![primal, Arc::new(linear.fragment)],
        &[y_key, dy_key],
        &[(input_key("x"), 1.0), (input_key("a"), 2.0), (dx_key, 1.0)],
    );

    assert_approx_eq(results[0], 2.0_f64.exp());
    assert_approx_eq(results[1], 2.0 * 2.0_f64.exp());
}

#[test]
fn vjp_exp_ax() {
    let (primal, y_key) = build_exp_ax();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        4,
    );
    let transposed = transpose(&linear);

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let result = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[
            (input_key("x"), 1.0),
            (input_key("a"), 2.0),
            (ct_y_key, 1.0),
        ],
    );

    assert_approx_eq(result[0], 2.0 * 2.0_f64.exp());
}

#[test]
fn vjp_x_plus_x_times_x() {
    let (primal, y_key) = build_x_plus_x_times_x();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        5,
    );
    let transposed = transpose(&linear);

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let result = evaluate(
        vec![primal, Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[(input_key("x"), 3.0), (ct_y_key, 1.0)],
    );

    assert_approx_eq(result[0], 12.0);
}

#[test]
fn fof_x_squared() {
    let (primal, y_key) = build_x_squared();
    let linear_1 = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        6,
    );
    let dy_key = tangent_output_key(&linear_1, 0).expect("active first-order tangent output");
    let dx1_key = tangent_input_key(&linear_1, 0);
    let linear_1_fragment = Arc::new(linear_1.fragment);

    let linear_2 = differentiate(
        &resolve(vec![primal.clone(), linear_1_fragment.clone()]),
        std::slice::from_ref(&dy_key),
        &[sk("x")],
        7,
    );
    let d2y_key = tangent_output_key(&linear_2, 0).expect("active second-order tangent output");
    let dx2_key = tangent_input_key(&linear_2, 0);

    let result = evaluate(
        vec![primal, linear_1_fragment, Arc::new(linear_2.fragment)],
        &[d2y_key],
        &[(input_key("x"), 3.0), (dx1_key, 1.0), (dx2_key, 1.0)],
    );

    assert_approx_eq(result[0], 2.0);
}

#[test]
fn for_exp_ax() {
    let (primal, y_key) = build_exp_ax();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        8,
    );
    let transposed = transpose(&linear);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let ct_y_seed_key = tangent_input_key(&transposed, 0);
    let transposed_fragment = Arc::new(transposed.fragment);

    let second_linear = differentiate(
        &resolve(vec![primal.clone(), transposed_fragment.clone()]),
        std::slice::from_ref(&ct_x_key),
        &[sk("x")],
        9,
    );
    let d_ct_x_key =
        tangent_output_key(&second_linear, 0).expect("active forward-over-reverse output");
    let dx_key = tangent_input_key(&second_linear, 0);

    let result = evaluate(
        vec![
            primal,
            transposed_fragment,
            Arc::new(second_linear.fragment),
        ],
        &[d_ct_x_key],
        &[
            (input_key("x"), 1.0),
            (input_key("a"), 2.0),
            (ct_y_seed_key, 1.0),
            (dx_key, 1.0),
        ],
    );

    assert_approx_eq(result[0], 4.0 * 2.0_f64.exp());
}

#[test]
fn numerical_gradient_exp_ax() {
    let (primal, y_key) = build_exp_ax();
    let linear = differentiate(
        &resolve(vec![primal.clone()]),
        std::slice::from_ref(&y_key),
        &[sk("x")],
        10,
    );
    let transposed = transpose(&linear);

    let ct_y_key = tangent_input_key(&transposed, 0);
    let ct_x_key = tangent_output_key(&transposed, 0).expect("active cotangent output");
    let vjp = evaluate(
        vec![primal.clone(), Arc::new(transposed.fragment)],
        &[ct_x_key],
        &[
            (input_key("x"), 1.0),
            (input_key("a"), 2.0),
            (ct_y_key, 1.0),
        ],
    )[0];

    let numerical = finite_difference_exp_ax(primal, &y_key);
    assert_approx_eq(vjp, numerical);
}
