use std::collections::HashMap;
use std::sync::Arc;

use computegraph::fragment::FragmentBuilder;
use computegraph::resolve::resolve;
use computegraph::types::{OpMode, ValRef};
use tidu::differentiate;

mod common;

use common::{evaluate, tangent_input_key, tangent_output_key, ScalarKey, ScalarOp};

fn sk(name: &str) -> ScalarKey {
    ScalarKey::User(name.to_string())
}

#[test]
fn differentiate_through_alias() {
    let mut primal_builder = FragmentBuilder::<ScalarOp>::new();
    let x = primal_builder.add_input(sk("x"));
    let y = primal_builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    primal_builder.set_outputs(y.clone());
    let primal = Arc::new(primal_builder.build());
    let y_key = primal.vals()[y[0]].key.clone();

    let mut post_builder = FragmentBuilder::<ScalarOp>::new();
    let y_alias = post_builder.add_input(sk("y_alias"));
    let two = post_builder.add_input(sk("two"));
    let z = post_builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(y_alias), ValRef::Local(two)],
        OpMode::Primal,
    );
    post_builder.set_outputs(z.clone());
    let post = Arc::new(post_builder.build());
    let z_key = post.vals()[z[0]].key.clone();

    let mut aliases = HashMap::new();
    aliases.insert(sk("y_alias"), y_key);

    let linear = differentiate(
        &resolve(vec![post.clone(), primal.clone()]),
        std::slice::from_ref(&z_key),
        &[sk("x")],
        1,
        &mut (),
        &aliases,
    );
    let dz_key = tangent_output_key(&linear, 0).expect("active tangent output");
    let dx_key = tangent_input_key(&linear, 0);

    let results = evaluate(
        vec![post, primal, Arc::new(linear.fragment)],
        &[dz_key],
        &[
            (dx_key, 1.0),
            (computegraph::GlobalValKey::Input(sk("x")), 3.0),
            (computegraph::GlobalValKey::Input(sk("two")), 2.0),
        ],
    );

    assert_eq!(results, vec![12.0]);
}

#[test]
fn differentiate_through_alias_twice() {
    let mut primal_builder = FragmentBuilder::<ScalarOp>::new();
    let x = primal_builder.add_input(sk("x"));
    let y = primal_builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(x), ValRef::Local(x)],
        OpMode::Primal,
    );
    primal_builder.set_outputs(y.clone());
    let primal = Arc::new(primal_builder.build());
    let y_key = primal.vals()[y[0]].key.clone();

    let mut post_builder = FragmentBuilder::<ScalarOp>::new();
    let y_alias = post_builder.add_input(sk("y_alias"));
    let two = post_builder.add_input(sk("two"));
    let z = post_builder.add_op(
        ScalarOp::Mul,
        vec![ValRef::Local(y_alias), ValRef::Local(two)],
        OpMode::Primal,
    );
    post_builder.set_outputs(z.clone());
    let post = Arc::new(post_builder.build());
    let z_key = post.vals()[z[0]].key.clone();

    let mut aliases = HashMap::new();
    aliases.insert(sk("y_alias"), y_key);

    let linear_1 = differentiate(
        &resolve(vec![post.clone(), primal.clone()]),
        std::slice::from_ref(&z_key),
        &[sk("x")],
        1,
        &mut (),
        &aliases,
    );
    let dz_key = tangent_output_key(&linear_1, 0).expect("active first-order tangent output");
    let dx1_key = tangent_input_key(&linear_1, 0);
    let linear_1_fragment = Arc::new(linear_1.fragment);

    let linear_2 = differentiate(
        &resolve(vec![
            post.clone(),
            primal.clone(),
            linear_1_fragment.clone(),
        ]),
        std::slice::from_ref(&dz_key),
        &[sk("x")],
        2,
        &mut (),
        &aliases,
    );
    let d2z_key = tangent_output_key(&linear_2, 0).expect("active second-order tangent output");
    let dx2_key = tangent_input_key(&linear_2, 0);

    let results = evaluate(
        vec![post, primal, linear_1_fragment, Arc::new(linear_2.fragment)],
        &[d2z_key],
        &[
            (computegraph::GlobalValKey::Input(sk("x")), 3.0),
            (computegraph::GlobalValKey::Input(sk("two")), 2.0),
            (dx1_key, 1.0),
            (dx2_key, 1.0),
        ],
    );

    assert_eq!(results, vec![4.0]);
}
