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
