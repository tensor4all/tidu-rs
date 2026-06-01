use std::collections::HashMap;
use std::sync::Arc;

use computegraph::{GraphOperation, LocalValueId, OperationRole};
use tidu::{
    linearize, try_linear_transpose, try_linear_transpose_with_builder, try_linearize,
    LinearizedGraph, PrimitiveBuilder, PrimitiveValue,
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
}

impl GraphOperation for Op {
    type Operand = f64;
    type Context = ();
    type InputKey = Key;

    fn input_count(&self) -> usize {
        match self {
            Op::Add => 2,
            Op::Missing | Op::EmitMissing => 1,
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
    ) -> Vec<Option<LocalValueId>> {
        match self {
            Op::Add => vec![tangent_in[0].or(tangent_in[1])],
            Op::Missing => panic!("try_linearize should call try_jvp_rule"),
            Op::EmitMissing => {
                let Some(dx) = tangent_in[0] else {
                    return vec![None];
                };
                let out = builder.add_primitive(
                    Op::Missing,
                    vec![PrimitiveValue::Local(dx)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                vec![Some(out[0])]
            }
        }
    }

    fn try_jvp_rule(
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
            Op::EmitMissing => {
                let Some(dx) = tangent_in[0] else {
                    return Ok(vec![None]);
                };
                let out = builder.add_primitive(
                    Op::Missing,
                    vec![PrimitiveValue::Local(dx)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                Ok(vec![Some(out[0])])
            }
        }
    }

    fn transpose_rule(
        &self,
        _builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_out: &[Option<LocalValueId>],
        _inputs: &[PrimitiveValue<Self>],
        _mode: &OperationRole,
        _ctx: &mut (),
    ) -> Vec<Option<LocalValueId>> {
        match self {
            Op::Add => vec![cotangent_out[0], cotangent_out[0]],
            Op::Missing => {
                panic!("fallible linear_transpose paths should call try_linear_transpose_rule")
            }
            Op::EmitMissing => panic!("EmitMissing is linearized before linear_transpose"),
        }
    }

    fn try_linear_transpose_rule(
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
            Op::EmitMissing => panic!("EmitMissing is linearized before linear_transpose"),
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
    let mut builder = computegraph::graph::GraphBuilder::<Op>::new();
    let x = builder.add_input(Key::Base("x"));
    let out = builder.add_operation(
        Op::EmitMissing,
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
}

#[test]
fn try_linearize_propagates_jvp_rule_error() {
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

    let err = match try_linearize(
        &view,
        &[output_key],
        &[Key::Base("x")],
        1,
        &mut ctx,
        &HashMap::new(),
    ) {
        Ok(_) => panic!("try_linearize unexpectedly succeeded"),
        Err(err) => err,
    };

    assert_eq!(err.rule(), ADRuleKind::Jvp);
}

#[test]
fn try_linear_transpose_propagates_transpose_error() {
    let linear = linearized_graph_with_missing_op();
    let mut ctx = ();

    let err = match try_linear_transpose(&linear, &mut ctx) {
        Ok(_) => panic!("try_linear_transpose unexpectedly succeeded"),
        Err(err) => err,
    };

    assert_eq!(err.rule(), ADRuleKind::Transpose);
}

#[test]
fn try_linear_transpose_with_builder_propagates_transpose_error() {
    let linear = linearized_graph_with_missing_op();
    let mut builder = TestBuilder::default();
    let seed = builder.add_input();
    let mut ctx = ();

    let err = try_linear_transpose_with_builder(&linear, &mut builder, &[Some(seed)], &mut ctx)
        .unwrap_err();

    assert_eq!(err.rule(), ADRuleKind::Transpose);
}
