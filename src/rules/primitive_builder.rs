use computegraph::fragment::FragmentBuilder;
use computegraph::{GraphOp, LocalValId, OpMode, ValRef};

/// Reference to a value available to a primitive AD rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveValue<Op: GraphOp> {
    /// Value produced inside the graph being built.
    Local(LocalValId),
    /// Value from the source primitive computation graph.
    External(computegraph::GlobalValKey<Op>),
}

impl<Op: GraphOp> From<PrimitiveValue<Op>> for ValRef<Op> {
    fn from(value: PrimitiveValue<Op>) -> Self {
        match value {
            PrimitiveValue::Local(id) => ValRef::Local(id),
            PrimitiveValue::External(key) => ValRef::External(key),
        }
    }
}

impl<Op: GraphOp> From<ValRef<Op>> for PrimitiveValue<Op> {
    fn from(value: ValRef<Op>) -> Self {
        match value {
            ValRef::Local(id) => PrimitiveValue::Local(id),
            ValRef::External(key) => PrimitiveValue::External(key),
        }
    }
}

/// Builder used by primitive JVP and transpose rules to append primitive applications.
pub trait PrimitiveBuilder<Op: GraphOp> {
    /// Add one primitive application and return local ids for its outputs.
    fn add_primitive(
        &mut self,
        op: Op,
        inputs: Vec<PrimitiveValue<Op>>,
        mode: OpMode,
    ) -> Vec<LocalValId>;
}

pub(crate) struct FragmentPrimitiveBuilder<'a, Op: GraphOp> {
    inner: &'a mut FragmentBuilder<Op>,
}

impl<'a, Op: GraphOp> FragmentPrimitiveBuilder<'a, Op> {
    pub(crate) fn new(inner: &'a mut FragmentBuilder<Op>) -> Self {
        Self { inner }
    }
}

impl<Op: GraphOp> PrimitiveBuilder<Op> for FragmentPrimitiveBuilder<'_, Op> {
    fn add_primitive(
        &mut self,
        op: Op,
        inputs: Vec<PrimitiveValue<Op>>,
        mode: OpMode,
    ) -> Vec<LocalValId> {
        let inputs = inputs.into_iter().map(ValRef::from).collect();
        self.inner.add_op(op, inputs, mode)
    }
}
