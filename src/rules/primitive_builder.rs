use computegraph::{GraphOperation, LocalValueId, OperationRole, ValueKey, ValueRef};

/// Reference to a value available to a primitive AD rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveValue<Op: GraphOperation> {
    /// Value produced inside the graph being built.
    Local(LocalValueId),
    /// Value from the source primitive computation graph.
    External(computegraph::ValueKey<Op>),
}

/// Input reference passed to a primitive transpose rule.
///
/// Linearized graphs contain both residual values and values that belong to
/// the tangent flow. A transpose rule may freely use residual values as
/// ordinary graph operands, but tangent-flow values must not be retained as
/// tensor-valued operands in the transposed graph. When a tangent-flow value is
/// the tangent of a known primal value, `primal` carries that counterpart for
/// metadata or runtime-shape use by downstream primitive sets.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveTransposeInput<Op: GraphOperation> {
    /// Value that is independent of the differentiated tangent flow.
    Residual(ValueKey<Op>),
    /// Value on the differentiated tangent flow.
    Linear {
        /// Key of the linearized graph value.
        key: ValueKey<Op>,
        /// Primal value whose tangent this value represents, when known.
        primal: Option<ValueKey<Op>>,
    },
}

impl<Op: GraphOperation> PrimitiveTransposeInput<Op> {
    /// Return the key of the linearized graph value.
    pub fn key(&self) -> &ValueKey<Op> {
        match self {
            Self::Residual(key) | Self::Linear { key, .. } => key,
        }
    }

    /// Return the corresponding primal key for a linear input, when known.
    pub fn primal(&self) -> Option<&ValueKey<Op>> {
        match self {
            Self::Residual(_) => None,
            Self::Linear { primal, .. } => primal.as_ref(),
        }
    }

    /// Return this input as a value usable by rule builders when it is residual.
    pub fn as_residual_value(&self) -> Option<PrimitiveValue<Op>> {
        match self {
            Self::Residual(key) => Some(PrimitiveValue::External(key.clone())),
            Self::Linear { .. } => None,
        }
    }
}

impl<Op: GraphOperation> From<PrimitiveValue<Op>> for ValueRef<Op> {
    fn from(value: PrimitiveValue<Op>) -> Self {
        match value {
            PrimitiveValue::Local(id) => ValueRef::Local(id),
            PrimitiveValue::External(key) => ValueRef::External(key),
        }
    }
}

impl<Op: GraphOperation> From<ValueRef<Op>> for PrimitiveValue<Op> {
    fn from(value: ValueRef<Op>) -> Self {
        match value {
            ValueRef::Local(id) => PrimitiveValue::Local(id),
            ValueRef::External(key) => PrimitiveValue::External(key),
        }
    }
}

/// Builder used by primitive JVP and transpose rules to append primitive applications.
pub trait PrimitiveBuilder<Op: GraphOperation> {
    /// Add one primitive application and return local ids for its outputs.
    fn add_primitive(
        &mut self,
        op: Op,
        inputs: Vec<PrimitiveValue<Op>>,
        role: OperationRole,
    ) -> Vec<LocalValueId>;
}

#[cfg(test)]
mod tests {
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
}
