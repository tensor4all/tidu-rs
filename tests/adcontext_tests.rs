#[allow(dead_code)]
mod common;

use std::sync::Arc;

use chainrules::{ADKey, DiffPassId, PrimitiveOp};
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::GraphOp;

define_ad_key!(CtxKey);

#[derive(Default)]
struct CountingContext {
    linearize_count: usize,
    transpose_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CountingOp {
    Add,
    Identity,
}

impl GraphOp for CountingOp {
    type Operand = f64;
    type Context = ();
    type InputKey = CtxKey;

    fn n_inputs(&self) -> usize {
        match self {
            Self::Add => 2,
            Self::Identity => 1,
        }
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl PrimitiveOp for CountingOp {
    type ADContext = CountingContext;

    fn add() -> Self {
        Self::Add
    }

    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        _primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        ctx: &mut CountingContext,
    ) -> Vec<Option<LocalValId>> {
        ctx.linearize_count += 1;

        match self {
            Self::Add => linearize_add!(builder, CountingOp::Add, tangent_in[0], tangent_in[1]),
            Self::Identity => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_op(
                        Self::Identity,
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
        _builder: &mut FragmentBuilder<Self>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<Self>],
        _mode: &OpMode,
        ctx: &mut CountingContext,
    ) -> Vec<Option<LocalValId>> {
        ctx.transpose_count += 1;

        match self {
            Self::Add => match cotangent_out[0] {
                Some(ct) => transpose_add!(ct),
                None => vec![None, None],
            },
            Self::Identity => vec![cotangent_out[0]],
        }
    }
}

fn ck(name: &str) -> CtxKey {
    CtxKey::User(name.to_string())
}

fn build_identity_chain() -> (Arc<Fragment<CountingOp>>, GlobalValKey<CountingOp>) {
    let mut builder = FragmentBuilder::<CountingOp>::new();
    let x = builder.add_input(ck("x"));
    let mid = builder.add_op(CountingOp::Identity, vec![ValRef::Local(x)], OpMode::Primal);
    let out = builder.add_op(
        CountingOp::Identity,
        vec![ValRef::Local(mid[0])],
        OpMode::Primal,
    );
    let out_key = builder.global_key(out[0]).clone();
    builder.set_outputs(vec![out[0]]);
    (Arc::new(builder.build()), out_key)
}

#[test]
fn differentiate_threads_ctx_to_all_ops() {
    let (primal, output_key) = build_identity_chain();
    let view = resolve(vec![primal]);
    let wrt = vec![ck("x")];

    let mut ctx = CountingContext::default();
    let _linear = tidu::differentiate(&view, &[output_key], &wrt, 1, &mut ctx);

    assert_eq!(
        ctx.linearize_count, 2,
        "both Identity ops should be linearized"
    );
}

#[test]
fn transpose_threads_ctx_to_all_ops() {
    let (primal, output_key) = build_identity_chain();
    let view = resolve(vec![primal]);
    let wrt = vec![ck("x")];

    let mut ctx = CountingContext::default();
    let linear = tidu::differentiate(&view, &[output_key], &wrt, 1, &mut ctx);

    ctx.linearize_count = 0;
    ctx.transpose_count = 0;
    let _transposed = tidu::transpose(&linear, &mut ctx);

    assert_eq!(
        ctx.transpose_count, 2,
        "both Identity ops should be transposed"
    );
    assert_eq!(
        ctx.linearize_count, 0,
        "linearize should not be called during transpose"
    );
}
