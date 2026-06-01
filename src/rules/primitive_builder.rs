use computegraph::graph::GraphBuilder;
use computegraph::{GraphOperation, LocalValueId, OperationRole, ValueRef};

/// Reference to a value available to a primitive AD rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveValue<Op: GraphOperation> {
    /// Value produced inside the graph being built.
    Local(LocalValueId),
    /// Value from the source primitive computation graph.
    External(computegraph::ValueKey<Op>),
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

pub(crate) struct GraphPrimitiveBuilder<'a, Op: GraphOperation> {
    inner: &'a mut GraphBuilder<Op>,
}

impl<'a, Op: GraphOperation> GraphPrimitiveBuilder<'a, Op> {
    pub(crate) fn new(inner: &'a mut GraphBuilder<Op>) -> Self {
        Self { inner }
    }
}

impl<Op: GraphOperation> PrimitiveBuilder<Op> for GraphPrimitiveBuilder<'_, Op> {
    fn add_primitive(
        &mut self,
        op: Op,
        inputs: Vec<PrimitiveValue<Op>>,
        role: OperationRole,
    ) -> Vec<LocalValueId> {
        let inputs = inputs.into_iter().map(ValueRef::from).collect();
        self.inner.add_operation(op, inputs, role)
    }
}
