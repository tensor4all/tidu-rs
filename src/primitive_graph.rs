use computegraph::fragment::Fragment;
use computegraph::GraphOp;

/// Borrowed primitive computation graph passed to downstream executors.
pub struct PrimitiveGraph<'a, Op: GraphOp> {
    graph: &'a Fragment<Op>,
}

impl<'a, Op: GraphOp> PrimitiveGraph<'a, Op> {
    pub(crate) fn new(graph: &'a Fragment<Op>) -> Self {
        Self { graph }
    }

    /// Borrow the lower-level graph representation.
    pub fn as_graph(&self) -> &Fragment<Op> {
        self.graph
    }
}
