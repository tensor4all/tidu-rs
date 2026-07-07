use std::sync::Arc;

use computegraph::graph::Graph;
use computegraph::{GraphOperation, LocalValueId, ValueKey};

/// Graph produced by linearizing a primitive computation graph.
pub struct LinearizedGraph<Op: GraphOperation> {
    linear: Graph<Op>,
    residual: Arc<Graph<Op>>,
    tangent_inputs: Vec<(Op::InputKey, LocalValueId)>,
    tangent_outputs: Vec<Option<LocalValueId>>,
    linear_primals: Vec<Option<ValueKey<Op>>>,
}

impl<Op: GraphOperation> LinearizedGraph<Op> {
    pub(crate) fn from_parts(
        linear: Graph<Op>,
        residual: Arc<Graph<Op>>,
        tangent_inputs: Vec<(Op::InputKey, LocalValueId)>,
        tangent_outputs: Vec<Option<LocalValueId>>,
        linear_primals: Vec<Option<ValueKey<Op>>>,
    ) -> Self {
        debug_assert_eq!(linear.values().len(), linear_primals.len());
        Self {
            linear,
            residual,
            tangent_inputs,
            tangent_outputs,
            linear_primals,
        }
    }

    /// Borrow the strictly-linear graph representation.
    pub fn as_graph(&self) -> &Graph<Op> {
        &self.linear
    }

    /// Borrow the residual graph referenced by the linear graph.
    pub fn residual_graph(&self) -> &Graph<Op> {
        &self.residual
    }

    /// Consume this value and return the strictly-linear graph representation.
    pub fn into_graph(self) -> Graph<Op> {
        self.linear
    }

    /// Consume this value and return both graph parts.
    pub fn into_graphs(self) -> (Graph<Op>, Arc<Graph<Op>>) {
        (self.linear, self.residual)
    }

    /// Tangent input keys and local value ids.
    pub fn tangent_inputs(&self) -> &[(Op::InputKey, LocalValueId)] {
        &self.tangent_inputs
    }

    /// Tangent outputs aligned with requested primal outputs.
    pub fn tangent_outputs(&self) -> &[Option<LocalValueId>] {
        &self.tangent_outputs
    }

    pub(crate) fn linear_primal(&self, id: LocalValueId) -> Option<&ValueKey<Op>> {
        self.linear_primals.get(id).and_then(Option::as_ref)
    }
}
