use std::collections::HashMap;
use std::sync::Arc;

use computegraph::fragment::FragmentBuilder;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use computegraph::{GraphOp, LocalValId, OpEmitter};
use tidu::{try_differentiate, try_eager_transpose_fragment, try_transpose, LinearFragment};
use tidu::{ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, PrimitiveOp};

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
}

impl GraphOp for Op {
    type Operand = f64;
    type Context = ();
    type InputKey = Key;

    fn n_inputs(&self) -> usize {
        match self {
            Op::Add => 2,
            Op::Missing => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl PrimitiveOp for Op {
    type ADContext = ();

    fn add() -> Self {
        Op::Add
    }

    fn linearize(
        &self,
        _builder: &mut FragmentBuilder<Self>,
        _primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        match self {
            Op::Add => vec![tangent_in[0].or(tangent_in[1])],
            Op::Missing => panic!("try_differentiate should call try_linearize"),
        }
    }

    fn try_linearize(
        &self,
        _builder: &mut FragmentBuilder<Self>,
        _primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut (),
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        match self {
            Op::Add => Ok(vec![tangent_in[0].or(tangent_in[1])]),
            Op::Missing => Err(ADRuleError::unsupported(
                "Op::Missing",
                ADRuleKind::Linearize,
            )),
        }
    }

    fn transpose_rule(
        &self,
        _emitter: &mut impl OpEmitter<Self>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<Self>],
        _mode: &OpMode,
        _ctx: &mut (),
    ) -> Vec<Option<LocalValId>> {
        match self {
            Op::Add => vec![cotangent_out[0], cotangent_out[0]],
            Op::Missing => panic!("fallible transpose paths should call try_transpose_rule"),
        }
    }

    fn try_transpose_rule(
        &self,
        _emitter: &mut impl OpEmitter<Self>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<Self>],
        _mode: &OpMode,
        _ctx: &mut (),
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        match self {
            Op::Add => Ok(vec![cotangent_out[0], cotangent_out[0]]),
            Op::Missing => Err(ADRuleError::unsupported(
                "Op::Missing",
                ADRuleKind::Transpose,
            )),
        }
    }
}

fn linear_fragment_with_missing_op() -> LinearFragment<Op> {
    let mut builder = FragmentBuilder::<Op>::new();
    let dx = builder.add_input(Key::Base("dx"));
    let out = builder.add_op(
        Op::Missing,
        vec![ValRef::Local(dx)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    );
    builder.set_outputs(out.clone());
    LinearFragment {
        fragment: builder.build(),
        tangent_inputs: vec![(Key::Base("x"), dx)],
        tangent_outputs: vec![Some(out[0])],
    }
}

#[test]
fn try_differentiate_propagates_linearize_error() {
    let mut builder = FragmentBuilder::<Op>::new();
    let x = builder.add_input(Key::Base("x"));
    let out = builder.add_op(Op::Missing, vec![ValRef::Local(x)], OpMode::Primal);
    builder.set_outputs(out.clone());
    let fragment = Arc::new(builder.build());
    let output_key = fragment.vals()[out[0]].key.clone();
    let view = resolve(vec![fragment]);
    let mut ctx = ();

    let err = match try_differentiate(
        &view,
        &[output_key],
        &[Key::Base("x")],
        1,
        &mut ctx,
        &HashMap::new(),
    ) {
        Ok(_) => panic!("try_differentiate unexpectedly succeeded"),
        Err(err) => err,
    };

    assert_eq!(err.rule(), ADRuleKind::Linearize);
}

#[test]
fn try_transpose_propagates_transpose_error() {
    let linear = linear_fragment_with_missing_op();
    let mut ctx = ();

    let err = match try_transpose(&linear, &mut ctx) {
        Ok(_) => panic!("try_transpose unexpectedly succeeded"),
        Err(err) => err,
    };

    assert_eq!(err.rule(), ADRuleKind::Transpose);
}

#[test]
fn try_eager_transpose_fragment_propagates_transpose_error() {
    let linear = linear_fragment_with_missing_op();
    let mut emitter = FragmentBuilder::<Op>::new();
    let seed = emitter.add_input(Key::Base("ct"));
    let mut ctx = ();

    let err =
        try_eager_transpose_fragment(&linear, &mut emitter, &[Some(seed)], &mut ctx).unwrap_err();

    assert_eq!(err.rule(), ADRuleKind::Transpose);
}
