use std::collections::HashMap;
#[allow(dead_code)]
mod common;

use std::sync::Arc;

use computegraph::graph::{Graph, GraphBuilder};
use computegraph::resolve::resolve;
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use computegraph::GraphOperation;
use tidu::{ADKey, DiffPassId, Primitive, PrimitiveBuilder, PrimitiveValue};

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

impl GraphOperation for CountingOp {
    type Operand = f64;
    type Context = ();
    type InputKey = CtxKey;

    fn input_count(&self) -> usize {
        match self {
            Self::Add => 2,
            Self::Identity => 1,
        }
    }

    fn output_count(&self) -> usize {
        1
    }
}

impl Primitive for CountingOp {
    type ADContext = CountingContext;

    fn add() -> Self {
        Self::Add
    }

    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        _primal_in: &[ValueKey<Self>],
        _primal_out: &[ValueKey<Self>],
        tangent_in: &[Option<LocalValueId>],
        ctx: &mut CountingContext,
    ) -> tidu::ADRuleResult<Vec<Option<LocalValueId>>> {
        ctx.linearize_count += 1;

        match self {
            Self::Add => linearize_add!(builder, CountingOp::Add, tangent_in[0], tangent_in[1]),
            Self::Identity => match tangent_in[0] {
                Some(dx) => {
                    let out = builder.add_primitive(
                        Self::Identity,
                        vec![PrimitiveValue::Local(dx)],
                        OperationRole::Linearized {
                            active_mask: vec![true],
                        },
                    );
                    Ok(vec![Some(out[0])])
                }
                None => Ok(vec![None]),
            },
        }
    }

    fn transpose_rule(
        &self,
        _builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_out: &[Option<LocalValueId>],
        _inputs: &[PrimitiveValue<Self>],
        _mode: &OperationRole,
        ctx: &mut CountingContext,
    ) -> tidu::ADRuleResult<Vec<Option<LocalValueId>>> {
        ctx.transpose_count += 1;

        match self {
            Self::Add => match cotangent_out[0] {
                Some(ct) => transpose_add!(ct),
                None => Ok(vec![None, None]),
            },
            Self::Identity => Ok(vec![cotangent_out[0]]),
        }
    }
}

fn ck(name: &str) -> CtxKey {
    CtxKey::User(name.to_string())
}

fn build_identity_chain() -> (Arc<Graph<CountingOp>>, ValueKey<CountingOp>) {
    let mut builder = GraphBuilder::<CountingOp>::new();
    let x = builder.add_input(ck("x"));
    let mid = builder.add_operation(
        CountingOp::Identity,
        vec![ValueRef::Local(x)],
        OperationRole::Primary,
    );
    let out = builder.add_operation(
        CountingOp::Identity,
        vec![ValueRef::Local(mid[0])],
        OperationRole::Primary,
    );
    let out_key = builder.global_key(out[0]).clone();
    builder.set_outputs(vec![out[0]]);
    (Arc::new(builder.build()), out_key)
}

#[test]
fn linearize_threads_ctx_to_all_ops() {
    let (primal, output_key) = build_identity_chain();
    let view = resolve(vec![primal]);
    let wrt = vec![ck("x")];

    let mut ctx = CountingContext::default();
    let _linear = tidu::linearize(&view, &[output_key], &wrt, 1, &mut ctx, &HashMap::new())
        .expect("identity chain should linearize");

    assert_eq!(
        ctx.linearize_count, 2,
        "both Identity ops should be linearized"
    );
}

#[test]
fn linear_transpose_threads_ctx_to_all_ops() {
    let (primal, output_key) = build_identity_chain();
    let view = resolve(vec![primal]);
    let wrt = vec![ck("x")];

    let mut ctx = CountingContext::default();
    let linear = tidu::linearize(&view, &[output_key], &wrt, 1, &mut ctx, &HashMap::new())
        .expect("identity chain should linearize");

    ctx.linearize_count = 0;
    ctx.transpose_count = 0;
    let _transposed =
        tidu::linear_transpose(&linear, &mut ctx).expect("identity chain should transpose");

    assert_eq!(
        ctx.transpose_count, 2,
        "both Identity ops should be transposed"
    );
    assert_eq!(
        ctx.linearize_count, 0,
        "linearize should not be called during linear_transpose"
    );
}
