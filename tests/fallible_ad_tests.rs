use std::collections::HashMap;
use std::sync::Arc;

use computegraph::{GraphOperation, LocalValueId, OperationRole};
use tidu::{
    linear_transpose, linear_transpose_with_builder, linearize, LinearizedGraph, PrimitiveBuilder,
    PrimitiveValue,
};
use tidu::{ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, Primitive};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Key {
    Base(&'static str),
    Tangent(Box<Key>, DiffPassId),
}

impl ADKey for Key {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        Key::Tangent(Box::new(self.clone()), pass)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Op {
    Add,
    Missing,
    EmitMissing,
    WrongJvp,
    EmitWrongTranspose,
    WrongTranspose,
}

impl GraphOperation for Op {
    type Operand = f64;
    type Context = ();
    type InputKey = Key;

    fn input_count(&self) -> usize {
        match self {
            Op::Add => 2,
            Op::Missing
            | Op::EmitMissing
            | Op::WrongJvp
            | Op::EmitWrongTranspose
            | Op::WrongTranspose => 1,
        }
    }

    fn output_count(&self) -> usize {
        1
    }
}

impl Primitive for Op {
    type ADContext = ();

    fn add() -> Self {
        Op::Add
    }

    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        _primal_in: &[computegraph::types::ValueKey<Self>],
        _primal_out: &[computegraph::types::ValueKey<Self>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut (),
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        match self {
            Op::Add => Ok(vec![tangent_in[0].or(tangent_in[1])]),
            Op::Missing => Err(ADRuleError::unsupported("Op::Missing", ADRuleKind::Jvp)),
            Op::WrongJvp => Ok(vec![]),
            Op::EmitMissing | Op::EmitWrongTranspose => {
                let Some(dx) = tangent_in[0] else {
                    return Ok(vec![None]);
                };
                let emitted = match self {
                    Op::EmitMissing => Op::Missing,
                    Op::EmitWrongTranspose => Op::WrongTranspose,
                    _ => unreachable!(),
                };
                let out = builder.add_primitive(
                    emitted,
                    vec![PrimitiveValue::Local(dx)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                Ok(vec![Some(out[0])])
            }
            Op::WrongTranspose => Ok(vec![tangent_in[0]]),
        }
    }

    fn transpose_rule(
        &self,
        _builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_out: &[Option<LocalValueId>],
        _inputs: &[PrimitiveValue<Self>],
        _mode: &OperationRole,
        _ctx: &mut (),
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        match self {
            Op::Add => Ok(vec![cotangent_out[0], cotangent_out[0]]),
            Op::Missing => Err(ADRuleError::unsupported(
                "Op::Missing",
                ADRuleKind::Transpose,
            )),
            Op::EmitMissing | Op::EmitWrongTranspose | Op::WrongJvp => Err(
                ADRuleError::unsupported("test primal-only op", ADRuleKind::Transpose),
            ),
            Op::WrongTranspose => Ok(vec![]),
        }
    }
}

#[derive(Default)]
struct TestBuilder {
    next_id: LocalValueId,
}

impl TestBuilder {
    fn add_input(&mut self) -> LocalValueId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

impl PrimitiveBuilder<Op> for TestBuilder {
    fn add_primitive(
        &mut self,
        op: Op,
        _inputs: Vec<PrimitiveValue<Op>>,
        _role: OperationRole,
    ) -> Vec<LocalValueId> {
        (0..op.output_count()).map(|_| self.add_input()).collect()
    }
}

fn linearized_graph_with_missing_op() -> LinearizedGraph<Op> {
    linearized_graph_for(Op::EmitMissing)
}

fn linearized_graph_with_wrong_transpose_op() -> LinearizedGraph<Op> {
    linearized_graph_for(Op::EmitWrongTranspose)
}

fn linearized_graph_for(op: Op) -> LinearizedGraph<Op> {
    let mut builder = computegraph::graph::GraphBuilder::<Op>::new();
    let x = builder.add_input(Key::Base("x"));
    let out = builder.add_operation(
        op,
        vec![computegraph::types::ValueRef::Local(x)],
        OperationRole::Primary,
    );
    builder.set_outputs(out.clone());
    let graph = Arc::new(builder.build());
    let output_key = graph.values()[out[0]].key.clone();
    let view = computegraph::resolve::resolve(vec![graph]);

    linearize(
        &view,
        &[output_key],
        &[Key::Base("x")],
        1,
        &mut (),
        &HashMap::new(),
    )
    .expect("test linearization should build a graph containing the requested transpose rule")
}

#[test]
fn linearize_propagates_jvp_rule_error() {
    let mut builder = computegraph::graph::GraphBuilder::<Op>::new();
    let x = builder.add_input(Key::Base("x"));
    let out = builder.add_operation(
        Op::Missing,
        vec![computegraph::types::ValueRef::Local(x)],
        OperationRole::Primary,
    );
    builder.set_outputs(out.clone());
    let graph = Arc::new(builder.build());
    let output_key = graph.values()[out[0]].key.clone();
    let view = computegraph::resolve::resolve(vec![graph]);
    let mut ctx = ();

    let err = match linearize(
        &view,
        &[output_key],
        &[Key::Base("x")],
        1,
        &mut ctx,
        &HashMap::new(),
    ) {
        Ok(_) => panic!("linearize unexpectedly succeeded"),
        Err(err) => err,
    };

    assert_eq!(err.rule(), ADRuleKind::Jvp);
}

#[test]
fn linearize_reports_wrong_jvp_arity_as_invalid_input() {
    let mut builder = computegraph::graph::GraphBuilder::<Op>::new();
    let x = builder.add_input(Key::Base("x"));
    let out = builder.add_operation(
        Op::WrongJvp,
        vec![computegraph::types::ValueRef::Local(x)],
        OperationRole::Primary,
    );
    builder.set_outputs(out.clone());
    let graph = Arc::new(builder.build());
    let output_key = graph.values()[out[0]].key.clone();
    let view = computegraph::resolve::resolve(vec![graph]);
    let mut ctx = ();

    let err = match linearize(
        &view,
        &[output_key],
        &[Key::Base("x")],
        1,
        &mut ctx,
        &HashMap::new(),
    ) {
        Ok(_) => panic!("linearize unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        ADRuleError::InvalidInput {
            rule: ADRuleKind::Jvp,
            ..
        }
    ));
}

#[test]
fn linear_transpose_propagates_transpose_error() {
    let linear = linearized_graph_with_missing_op();
    let mut ctx = ();

    let err = match linear_transpose(&linear, &mut ctx) {
        Ok(_) => panic!("linear_transpose unexpectedly succeeded"),
        Err(err) => err,
    };

    assert_eq!(err.rule(), ADRuleKind::Transpose);
}

#[test]
fn linear_transpose_reports_wrong_transpose_arity_as_invalid_input() {
    let linear = linearized_graph_with_wrong_transpose_op();
    let mut ctx = ();

    let err = match linear_transpose(&linear, &mut ctx) {
        Ok(_) => panic!("linear_transpose unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        ADRuleError::InvalidInput {
            rule: ADRuleKind::Transpose,
            ..
        }
    ));
}

#[test]
fn linear_transpose_with_builder_propagates_transpose_error() {
    let linear = linearized_graph_with_missing_op();
    let mut builder = TestBuilder::default();
    let seed = builder.add_input();
    let mut ctx = ();

    let err =
        linear_transpose_with_builder(&linear, &mut builder, &[Some(seed)], &mut ctx).unwrap_err();

    assert_eq!(err.rule(), ADRuleKind::Transpose);
}

#[test]
fn linear_transpose_with_builder_reports_wrong_transpose_arity_as_invalid_input() {
    let linear = linearized_graph_with_wrong_transpose_op();
    let mut builder = TestBuilder::default();
    let seed = builder.add_input();
    let mut ctx = ();

    let err =
        linear_transpose_with_builder(&linear, &mut builder, &[Some(seed)], &mut ctx).unwrap_err();

    assert!(matches!(
        err,
        ADRuleError::InvalidInput {
            rule: ADRuleKind::Transpose,
            ..
        }
    ));
}
