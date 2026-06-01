use computegraph::graph::Graph;
use computegraph::GraphOperation;

/// Borrowed primitive computation graph passed to downstream executors.
pub struct PrimitiveGraph<'a, Op: GraphOperation> {
    graph: &'a Graph<Op>,
}

impl<'a, Op: GraphOperation> PrimitiveGraph<'a, Op> {
    pub(crate) fn new(graph: &'a Graph<Op>) -> Self {
        Self { graph }
    }

    /// Borrow the lower-level graph representation.
    pub fn as_graph(&self) -> &Graph<Op> {
        self.graph
    }
}
