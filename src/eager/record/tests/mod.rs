use super::*;
use crate::{DiffPassId, PrimitiveBuilder, PrimitiveTransposeInput};
use computegraph::LocalValueId;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum Key {
    User(&'static str),
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

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum Op {
    Id,
}

impl GraphOperation for Op {
    type Operand = f64;
    type Context = ();
    type InputKey = Key;

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }
}

impl Primitive for Op {
    type ADContext = ();

    fn add() -> Self {
        Self::Id
    }

    fn jvp_rule(
        &self,
        _builder: &mut impl PrimitiveBuilder<Self>,
        _primal_in: &[ValueKey<Self>],
        _primal_out: &[ValueKey<Self>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut (),
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        Ok(tangent_in.to_vec())
    }

    fn transpose_rule(
        &self,
        _builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_out: &[Option<LocalValueId>],
        _inputs: &[PrimitiveTransposeInput<Self>],
        _role: &OperationRole,
        _ctx: &mut (),
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        Ok(cotangent_out.to_vec())
    }
}

#[test]
fn too_many_outputs_error_formats_limit() {
    let err = EagerRecordError::too_many_outputs(u8::MAX as usize + 2);

    assert!(matches!(
        err,
        EagerRecordError::TooManyOutputs {
            actual,
            max: 256
        } if actual == u8::MAX as usize + 2
    ));
    assert_eq!(
        err.to_string(),
        "eager recording supports at most 256 outputs, got 257"
    );
}

#[test]
fn recorded_graph_linearize_rejects_unknown_output_slot() {
    let recorded = RecordedGraph::from_primitive(Op::Id, vec![Key::User("x")])
        .expect("test primitive graph metadata should be valid");

    let err = match recorded.linearize(&[1], &mut ()) {
        Ok(_) => panic!("RecordedGraph::linearize unexpectedly accepted bad output slot"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        ADRuleError::InvalidInput {
            op,
            rule: ADRuleKind::Jvp,
            message
        } if op == "tidu::eager::RecordedGraph"
            && message.contains("requested output slot 1")
    ));
}
