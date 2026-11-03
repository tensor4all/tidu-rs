use super::*;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TestOp;

impl GraphOperation for TestOp {
    type Operand = f64;
    type Context = ();
    type InputKey = &'static str;

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }
}

fn key(name: &'static str) -> ValueKey<TestOp> {
    ValueKey::Input(name)
}

#[test]
fn key_is_available_for_residual_and_linear_inputs() {
    let residual = PrimitiveTransposeInput::<TestOp>::Residual(key("r"));
    assert_eq!(residual.key(), &key("r"));

    let linear = PrimitiveTransposeInput::Linear {
        key: key("l"),
        primal: None,
    };
    assert_eq!(linear.key(), &key("l"));
}

#[test]
fn primal_is_none_for_residual_and_unknown_linear_inputs() {
    let residual = PrimitiveTransposeInput::<TestOp>::Residual(key("r"));
    assert_eq!(residual.primal(), None);

    let unknown = PrimitiveTransposeInput::Linear {
        key: key("l"),
        primal: None,
    };
    assert_eq!(unknown.primal(), None);
}

#[test]
fn primal_reports_the_tracked_primal_for_linear_inputs() {
    let linear = PrimitiveTransposeInput::Linear {
        key: key("l"),
        primal: Some(key("p")),
    };
    assert_eq!(linear.primal(), Some(&key("p")));
}

#[test]
fn as_residual_value_only_accepts_residual_inputs() {
    let residual = PrimitiveTransposeInput::<TestOp>::Residual(key("r"));
    assert_eq!(
        residual.as_residual_value(),
        Some(PrimitiveValue::External(key("r")))
    );

    let linear = PrimitiveTransposeInput::Linear {
        key: key("l"),
        primal: Some(key("p")),
    };
    assert_eq!(linear.as_residual_value(), None);
}
