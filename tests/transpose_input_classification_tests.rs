use std::collections::HashMap;
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::resolve::resolve;
use computegraph::{GraphOperation, LocalValueId, OperationRole, ValueKey};
use tidu::{
    linear_transpose, linearize, ADKey, ADRuleResult, DiffPassId, Primitive, PrimitiveBuilder,
    PrimitiveTransposeInput, PrimitiveValue,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Key {
    User(u64),
    Tangent { of: Box<Key>, pass: DiffPassId },
}

impl ADKey for Key {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        Self::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Op {
    Add,
    Probe,
    ResidualHelper,
    TangentHelper,
}

impl GraphOperation for Op {
    type Operand = f64;
    type Context = ();
    type InputKey = Key;

    fn input_count(&self) -> usize {
        match self {
            Self::Add => 2,
            Self::Probe | Self::ResidualHelper | Self::TangentHelper => 1,
        }
    }

    fn output_count(&self) -> usize {
        1
    }
}

#[derive(Default)]
struct SeenTransposeInputs {
    add_inputs: Vec<Vec<PrimitiveTransposeInput<Op>>>,
}

impl Primitive for Op {
    type ADContext = SeenTransposeInputs;

    fn add() -> Self {
        Self::Add
    }

    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        primal_in: &[ValueKey<Self>],
        _primal_out: &[ValueKey<Self>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut Self::ADContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        match self {
            Self::Probe => {
                let Some(dx) = tangent_in[0] else {
                    return Ok(vec![None]);
                };
                let residual = builder.add_primitive(
                    Self::ResidualHelper,
                    vec![PrimitiveValue::External(primal_in[0].clone())],
                    OperationRole::Linearized {
                        active_mask: vec![false],
                    },
                )[0];
                let tangent = builder.add_primitive(
                    Self::TangentHelper,
                    vec![PrimitiveValue::Local(dx)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                )[0];
                let mixed = builder.add_primitive(
                    Self::Add,
                    vec![
                        PrimitiveValue::Local(residual),
                        PrimitiveValue::Local(tangent),
                    ],
                    OperationRole::Linearized {
                        active_mask: vec![false, true],
                    },
                )[0];
                Ok(vec![Some(mixed)])
            }
            _ => Ok(vec![tangent_in[0]]),
        }
    }

    fn transpose_rule(
        &self,
        _builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[PrimitiveTransposeInput<Self>],
        mode: &OperationRole,
        ctx: &mut Self::ADContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        if matches!(self, Self::Add) {
            ctx.add_inputs.push(inputs.to_vec());
        }
        let Some(ct) = cotangent_out[0] else {
            return Ok(vec![None; self.input_count()]);
        };
        let active_mask = match mode {
            OperationRole::Linearized { active_mask } => active_mask,
            OperationRole::Primary => return Ok(vec![None; self.input_count()]),
        };
        Ok(active_mask
            .iter()
            .map(|active| active.then_some(ct))
            .collect())
    }
}

#[test]
fn linear_transpose_marks_residual_and_tangent_inputs_distinctly() {
    let mut builder = GraphBuilder::<Op>::new();
    let x_key = Key::User(0);
    let x = builder.add_input(x_key.clone());
    let y = builder.add_operation(
        Op::Probe,
        vec![computegraph::ValueRef::Local(x)],
        OperationRole::Primary,
    )[0];
    builder.set_outputs(vec![y]);
    let graph = Arc::new(builder.build());
    let y_key = graph.values()[y].key.clone();
    let view = resolve(vec![graph]);

    let mut ctx = SeenTransposeInputs::default();
    let linear = linearize(&view, &[y_key], &[x_key], 0, &mut ctx, &HashMap::new()).unwrap();
    let _ = linear_transpose(&linear, &mut ctx).unwrap();

    let add_inputs = ctx
        .add_inputs
        .first()
        .expect("transposed Add rule should have been visited");
    assert!(
        matches!(add_inputs[0], PrimitiveTransposeInput::Residual(_)),
        "inactive residual helper must remain a residual transpose input: {add_inputs:?}"
    );
    assert!(
        matches!(add_inputs[1], PrimitiveTransposeInput::Linear { .. }),
        "active tangent helper must be marked as a linear transpose input: {add_inputs:?}"
    );
}
