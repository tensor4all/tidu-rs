use std::collections::HashMap;
use std::sync::Arc;

use computegraph::{GraphOp, LocalValId, OpMode};
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

impl GraphOp for Op {
    type Operand = f64;
    type Context = ();
    type InputKey = Key;

    fn n_inputs(&self) -> usize {
        match self {
            Op::Add => 2,
            Op::Missing | Op::EmitMissing => 1,
        }
    }

    fn n_outputs(&self) -> usize {
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
        _primal_in: &[computegraph::types::GlobalValKey<Self>],
        _primal_out: &[computegraph::types::GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
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
                    OpMode::Linear {
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
        _primal_in: &[computegraph::types::GlobalValKey<Self>],
        _primal_out: &[computegraph::types::GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut (),
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
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
                    OpMode::Linear {
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
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[PrimitiveValue<Self>],
        _mode: &OpMode,
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        match self {
            Op::Add => vec![cotangent_out[0], cotangent_out[0]],
            Op::Missing => panic!("fallible transpose paths should call try_transpose_rule"),
            Op::EmitMissing => panic!("EmitMissing is linearized before transpose"),
        }
    }

    fn try_transpose_rule(
        &self,
        _builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[PrimitiveValue<Self>],
        _mode: &OpMode,
        _ctx: &mut (),
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        match self {
            Op::Add => Ok(vec![cotangent_out[0], cotangent_out[0]]),
            Op::Missing => Err(ADRuleError::unsupported(
                "Op::Missing",
                ADRuleKind::Transpose,
            )),
            Op::EmitMissing => panic!("EmitMissing is linearized before transpose"),
        }
    }
}

fn linearized_graph_with_missing_op() -> LinearizedGraph<Op> {
    let mut builder = computegraph::fragment::FragmentBuilder::<Op>::new();
    let x = builder.add_input(Key::Base("x"));
    let out = builder.add_op(
        Op::EmitMissing,
        vec![computegraph::types::ValRef::Local(x)],
        OpMode::Primal,
    );
    builder.set_outputs(out.clone());
    let fragment = Arc::new(builder.build());
    let output_key = fragment.vals()[out[0]].key.clone();
    let view = computegraph::resolve::resolve(vec![fragment]);

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
    let mut builder = computegraph::fragment::FragmentBuilder::<Op>::new();
    let x = builder.add_input(Key::Base("x"));
    let out = builder.add_op(
        Op::Missing,
        vec![computegraph::types::ValRef::Local(x)],
        OpMode::Primal,
    );
    builder.set_outputs(out.clone());
    let fragment = Arc::new(builder.build());
    let output_key = fragment.vals()[out[0]].key.clone();
    let view = computegraph::resolve::resolve(vec![fragment]);
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
    let mut emitter = computegraph::fragment::FragmentBuilder::<Op>::new();
    let seed = emitter.add_input(Key::Base("ct"));
    let mut ctx = ();

    let err = try_linear_transpose_with_builder(&linear, &mut emitter, &[Some(seed)], &mut ctx)
        .unwrap_err();

    assert_eq!(err.rule(), ADRuleKind::Transpose);
}
