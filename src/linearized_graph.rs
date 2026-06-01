use computegraph::fragment::Fragment;
use computegraph::{GraphOp, LocalValId};

/// Graph produced by linearizing a primitive computation graph.
pub struct LinearizedGraph<Op: GraphOp> {
    graph: Fragment<Op>,
    tangent_inputs: Vec<(Op::InputKey, LocalValId)>,
    tangent_outputs: Vec<Option<LocalValId>>,
}

impl<Op: GraphOp> LinearizedGraph<Op> {
    pub(crate) fn from_parts(
        graph: Fragment<Op>,
        tangent_inputs: Vec<(Op::InputKey, LocalValId)>,
        tangent_outputs: Vec<Option<LocalValId>>,
    ) -> Self {
        Self {
            graph,
            tangent_inputs,
            tangent_outputs,
        }
    }

    /// Borrow the lower-level graph representation.
    pub fn as_graph(&self) -> &Fragment<Op> {
        &self.graph
    }

    /// Consume this value and return the lower-level graph representation.
    pub fn into_graph(self) -> Fragment<Op> {
        self.graph
    }

    /// Tangent input keys and local value ids.
    pub fn tangent_inputs(&self) -> &[(Op::InputKey, LocalValId)] {
        &self.tangent_inputs
    }

    /// Tangent outputs aligned with requested primal outputs.
    pub fn tangent_outputs(&self) -> &[Option<LocalValId>] {
        &self.tangent_outputs
    }
}
