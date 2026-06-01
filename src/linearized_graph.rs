use computegraph::graph::Graph;
use computegraph::{GraphOperation, LocalValueId};

/// Graph produced by linearizing a primitive computation graph.
pub struct LinearizedGraph<Op: GraphOperation> {
    graph: Graph<Op>,
    tangent_inputs: Vec<(Op::InputKey, LocalValueId)>,
    tangent_outputs: Vec<Option<LocalValueId>>,
}

impl<Op: GraphOperation> LinearizedGraph<Op> {
    pub(crate) fn from_parts(
        graph: Graph<Op>,
        tangent_inputs: Vec<(Op::InputKey, LocalValueId)>,
        tangent_outputs: Vec<Option<LocalValueId>>,
    ) -> Self {
        Self {
            graph,
            tangent_inputs,
            tangent_outputs,
        }
    }

    /// Borrow the lower-level graph representation.
    pub fn as_graph(&self) -> &Graph<Op> {
        &self.graph
    }

    /// Consume this value and return the lower-level graph representation.
    pub fn into_graph(self) -> Graph<Op> {
        self.graph
    }

    /// Tangent input keys and local value ids.
    pub fn tangent_inputs(&self) -> &[(Op::InputKey, LocalValueId)] {
        &self.tangent_inputs
    }

    /// Tangent outputs aligned with requested primal outputs.
    pub fn tangent_outputs(&self) -> &[Option<LocalValueId>] {
        &self.tangent_outputs
    }
}
