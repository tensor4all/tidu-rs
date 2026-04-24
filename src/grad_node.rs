use std::collections::HashMap;
use std::sync::Arc;

use computegraph::{GlobalValKey, GraphOp};

/// Backward computation node for eager reverse-mode AD.
///
/// A `GradNode` records one primal operation, the stable input aliases used to
/// replay that operation during backward, the user-visible output keys that can
/// receive cotangent seeds, and the edges to parent eager values.
///
/// # Examples
///
/// ```ignore
/// use tidu::{GradEdge, GradNode};
///
/// let node = GradNode::new(
///     op,
///     input_aliases,
///     output_keys,
///     saved_forward_values,
///     vec![GradEdge::new(parent_node, input_key, true)],
/// );
/// ```
pub struct GradNode<Op: GraphOp> {
    op: Op,
    primal_in_keys: Vec<GlobalValKey<Op>>,
    primal_out_keys: Vec<GlobalValKey<Op>>,
    saved_data: HashMap<GlobalValKey<Op>, Arc<Op::Operand>>,
    input_edges: Vec<GradEdge<Op>>,
}

impl<Op: GraphOp> GradNode<Op> {
    /// Create a grad node and validate the shape of its eager AD metadata.
    ///
    /// `primal_in_keys` must contain `GlobalValKey::Input` aliases. The eager
    /// backward path linearizes one operation at a time and rebuilds those
    /// aliases as fragment inputs.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let node = tidu::GradNode::new(
    ///     op,
    ///     input_aliases,
    ///     output_keys,
    ///     saved_data,
    ///     input_edges,
    /// );
    /// ```
    pub fn new(
        op: Op,
        primal_in_keys: Vec<GlobalValKey<Op>>,
        primal_out_keys: Vec<GlobalValKey<Op>>,
        saved_data: HashMap<GlobalValKey<Op>, Arc<Op::Operand>>,
        input_edges: Vec<GradEdge<Op>>,
    ) -> Self {
        assert_eq!(
            primal_in_keys.len(),
            op.n_inputs(),
            "grad node for {:?} expected {} primal input keys, got {}",
            op,
            op.n_inputs(),
            primal_in_keys.len()
        );
        assert_eq!(
            primal_out_keys.len(),
            op.n_outputs(),
            "grad node for {:?} expected {} primal output keys, got {}",
            op,
            op.n_outputs(),
            primal_out_keys.len()
        );
        assert_eq!(
            input_edges.len(),
            op.n_inputs(),
            "grad node for {:?} expected {} input edges, got {}",
            op,
            op.n_inputs(),
            input_edges.len()
        );
        assert!(
            primal_in_keys
                .iter()
                .all(|key| matches!(key, GlobalValKey::Input(_))),
            "grad node for {:?} requires GlobalValKey::Input aliases in primal_in_keys",
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

    /// The primal operation recorded by this node.
    pub fn op(&self) -> &Op {
        &self.op
    }

    /// Stable input aliases used for single-op backward replay.
    pub fn primal_in_keys(&self) -> &[GlobalValKey<Op>] {
        &self.primal_in_keys
    }

    /// User-visible output keys, one per primal output slot.
    pub fn primal_out_keys(&self) -> &[GlobalValKey<Op>] {
        &self.primal_out_keys
    }

    /// Saved concrete primal input and derived output values.
    pub fn saved_data(&self) -> &HashMap<GlobalValKey<Op>, Arc<Op::Operand>> {
        &self.saved_data
    }

    /// Edges to the eager values that provided this node's inputs.
    pub fn input_edges(&self) -> &[GradEdge<Op>] {
        &self.input_edges
    }
}

/// Edge from a grad node to one of its primal inputs.
///
/// `node` points to the parent operation that produced the input. `None`
/// denotes a leaf eager value. `key` is the cotangent accumulation target for
/// that input.
///
/// # Examples
///
/// ```ignore
/// let edge = tidu::GradEdge::new(parent_node, input_key, requires_grad);
/// ```
pub struct GradEdge<Op: GraphOp> {
    /// Parent grad node. `None` denotes a leaf input.
    pub node: Option<Arc<GradNode<Op>>>,
    /// Gradient accumulation target for this input.
    pub key: GlobalValKey<Op>,
    /// Whether the input participates in gradient propagation.
    pub requires_grad: bool,
}

impl<Op: GraphOp> GradEdge<Op> {
    /// Create an eager backward edge.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let edge = tidu::GradEdge::new(parent_node, input_key, true);
    /// ```
    pub fn new(
        node: Option<Arc<GradNode<Op>>>,
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
