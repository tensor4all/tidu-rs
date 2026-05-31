use std::collections::HashMap;
use std::sync::Arc;

use computegraph::{GlobalValKey, GraphOp};

/// Opaque handle to an eager reverse-mode trace node.
///
/// Downstream eager values store this handle next to their concrete data and
/// pass it back to [`crate::eager::try_backward`]. The node and edge layout is
/// intentionally private.
pub struct Trace<Op: GraphOp> {
    node: Arc<TraceNode<Op>>,
}

impl<Op: GraphOp> Clone for Trace<Op> {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
        }
    }
}

impl<Op: GraphOp> Trace<Op> {
    pub(crate) fn new(node: Arc<TraceNode<Op>>) -> Self {
        Self { node }
    }

    pub(crate) fn node(&self) -> &Arc<TraceNode<Op>> {
        &self.node
    }

    /// Saved concrete primal values used as initial data during backward replay.
    pub fn saved_values(&self) -> &HashMap<GlobalValKey<Op>, Arc<Op::Operand>> {
        self.node.saved_data()
    }
}

pub(crate) struct TraceNode<Op: GraphOp> {
    op: Op,
    primal_in_keys: Vec<GlobalValKey<Op>>,
    primal_out_keys: Vec<GlobalValKey<Op>>,
    saved_data: HashMap<GlobalValKey<Op>, Arc<Op::Operand>>,
    input_edges: Vec<TraceEdge<Op>>,
}

impl<Op: GraphOp> TraceNode<Op> {
    pub(crate) fn new(
        op: Op,
        primal_in_keys: Vec<GlobalValKey<Op>>,
        primal_out_keys: Vec<GlobalValKey<Op>>,
        saved_data: HashMap<GlobalValKey<Op>, Arc<Op::Operand>>,
        input_edges: Vec<TraceEdge<Op>>,
    ) -> Self {
        assert_eq!(
            primal_in_keys.len(),
            op.n_inputs(),
            "trace node for {:?} expected {} primal input keys, got {}",
            op,
            op.n_inputs(),
            primal_in_keys.len()
        );
        assert_eq!(
            primal_out_keys.len(),
            op.n_outputs(),
            "trace node for {:?} expected {} primal output keys, got {}",
            op,
            op.n_outputs(),
            primal_out_keys.len()
        );
        assert_eq!(
            input_edges.len(),
            op.n_inputs(),
            "trace node for {:?} expected {} input edges, got {}",
            op,
            op.n_inputs(),
            input_edges.len()
        );
        assert!(
            primal_in_keys
                .iter()
                .all(|key| matches!(key, GlobalValKey::Input(_))),
            "trace node for {:?} requires GlobalValKey::Input aliases in primal_in_keys",
            op
        );

        Self {
            op,
            primal_in_keys,
            primal_out_keys,
            saved_data,
            input_edges,
        }
    }

    pub(crate) fn op(&self) -> &Op {
        &self.op
    }

    pub(crate) fn primal_in_keys(&self) -> &[GlobalValKey<Op>] {
        &self.primal_in_keys
    }

    pub(crate) fn primal_out_keys(&self) -> &[GlobalValKey<Op>] {
        &self.primal_out_keys
    }

    pub(crate) fn saved_data(&self) -> &HashMap<GlobalValKey<Op>, Arc<Op::Operand>> {
        &self.saved_data
    }

    pub(crate) fn input_edges(&self) -> &[TraceEdge<Op>] {
        &self.input_edges
    }
}

pub(crate) struct TraceEdge<Op: GraphOp> {
    pub(crate) node: Option<Arc<TraceNode<Op>>>,
    pub(crate) key: GlobalValKey<Op>,
    pub(crate) requires_grad: bool,
}

impl<Op: GraphOp> TraceEdge<Op> {
    pub(crate) fn new(
        node: Option<Arc<TraceNode<Op>>>,
        key: GlobalValKey<Op>,
        requires_grad: bool,
    ) -> Self {
        Self {
            node,
            key,
            requires_grad,
        }
    }
}
