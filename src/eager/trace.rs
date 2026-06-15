use std::collections::HashMap;
use std::sync::Arc;

use computegraph::{GraphOperation, ValueKey};

use super::record::RecordedGraph;

/// Opaque handle to an eager reverse-mode trace node.
///
/// Downstream eager values store this handle next to their concrete data and
/// pass it back to [`crate::eager::try_backward`]. The node and edge layout is
/// intentionally private.
pub struct Trace<Op: GraphOperation> {
    node: Arc<TraceNode<Op>>,
}

impl<Op: GraphOperation> Clone for Trace<Op> {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
        }
    }
}

impl<Op: GraphOperation> Trace<Op> {
    pub(crate) fn new(node: Arc<TraceNode<Op>>) -> Self {
        Self { node }
    }

    pub(crate) fn node(&self) -> &Arc<TraceNode<Op>> {
        &self.node
    }

    /// Saved concrete primal values used as initial data during backward replay.
    pub fn saved_values(&self) -> &HashMap<ValueKey<Op>, Arc<Op::Operand>> {
        self.node.saved_data()
    }
}

pub(crate) struct TraceNode<Op: GraphOperation> {
    computation: RecordedGraph<Op>,
    primal_out_keys: Vec<ValueKey<Op>>,
    saved_data: HashMap<ValueKey<Op>, Arc<Op::Operand>>,
    input_edges: Vec<TraceEdge<Op>>,
}

impl<Op: GraphOperation> TraceNode<Op> {
    pub(crate) fn new(
        computation: RecordedGraph<Op>,
        primal_out_keys: Vec<ValueKey<Op>>,
        saved_data: HashMap<ValueKey<Op>, Arc<Op::Operand>>,
        input_edges: Vec<TraceEdge<Op>>,
    ) -> Self {
        assert_eq!(
            primal_out_keys.len(),
            computation.output_keys().len(),
            "trace node expected {} primal output keys, got {}",
            computation.output_keys().len(),
            primal_out_keys.len()
        );
        assert_eq!(
            input_edges.len(),
            computation.input_keys().len(),
            "trace node expected {} input edges, got {}",
            computation.input_keys().len(),
            input_edges.len()
        );

        Self {
            computation,
            primal_out_keys,
            saved_data,
            input_edges,
        }
    }

    pub(crate) fn computation(&self) -> &RecordedGraph<Op> {
        &self.computation
    }

    pub(crate) fn primal_out_keys(&self) -> &[ValueKey<Op>] {
        &self.primal_out_keys
    }

    pub(crate) fn saved_data(&self) -> &HashMap<ValueKey<Op>, Arc<Op::Operand>> {
        &self.saved_data
    }

    pub(crate) fn input_edges(&self) -> &[TraceEdge<Op>] {
        &self.input_edges
    }
}

pub(crate) struct TraceEdge<Op: GraphOperation> {
    pub(crate) node: Option<Arc<TraceNode<Op>>>,
    pub(crate) key: ValueKey<Op>,
    pub(crate) requires_grad: bool,
}

impl<Op: GraphOperation> TraceEdge<Op> {
    pub(crate) fn new(
        node: Option<Arc<TraceNode<Op>>>,
        key: ValueKey<Op>,
        requires_grad: bool,
    ) -> Self {
        Self {
            node,
            key,
            requires_grad,
        }
    }
}
