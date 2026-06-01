use std::collections::HashMap;
use std::sync::Arc;

use computegraph::{GraphOperation, ValueKey};

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
    operation: Op,
    primal_in_keys: Vec<ValueKey<Op>>,
    primal_out_keys: Vec<ValueKey<Op>>,
    saved_data: HashMap<ValueKey<Op>, Arc<Op::Operand>>,
    input_edges: Vec<TraceEdge<Op>>,
}

impl<Op: GraphOperation> TraceNode<Op> {
    pub(crate) fn new(
        operation: Op,
        primal_in_keys: Vec<ValueKey<Op>>,
        primal_out_keys: Vec<ValueKey<Op>>,
        saved_data: HashMap<ValueKey<Op>, Arc<Op::Operand>>,
        input_edges: Vec<TraceEdge<Op>>,
    ) -> Self {
        assert_eq!(
            primal_in_keys.len(),
            operation.input_count(),
            "trace node for {:?} expected {} primal input keys, got {}",
            operation,
            operation.input_count(),
            primal_in_keys.len()
        );
        assert_eq!(
            primal_out_keys.len(),
            operation.output_count(),
            "trace node for {:?} expected {} primal output keys, got {}",
            operation,
            operation.output_count(),
            primal_out_keys.len()
        );
        assert_eq!(
            input_edges.len(),
            operation.input_count(),
            "trace node for {:?} expected {} input edges, got {}",
            operation,
            operation.input_count(),
            input_edges.len()
        );
        assert!(
            primal_in_keys
                .iter()
                .all(|key| matches!(key, ValueKey::Input(_))),
            "trace node for {:?} requires ValueKey::Input aliases in primal_in_keys",
            operation
        );

        Self {
            operation,
            primal_in_keys,
            primal_out_keys,
            saved_data,
            input_edges,
        }
    }

    pub(crate) fn operation(&self) -> &Op {
        &self.operation
    }

    pub(crate) fn primal_in_keys(&self) -> &[ValueKey<Op>] {
        &self.primal_in_keys
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
