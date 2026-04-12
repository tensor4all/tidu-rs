use std::collections::HashMap;
use std::sync::Arc;

use computegraph::{GlobalValKey, GraphOp};

/// Backward computation node for eager AD.
///
/// The graph is an acyclic `Arc` DAG with edges from a node to the nodes that
/// produced its inputs.
pub struct GradNode<Op: GraphOp> {
    /// The primal operation associated with this node.
    pub op: Op,
    /// Primal input keys for this operation, one entry per input position.
    pub primal_in_keys: Vec<GlobalValKey<Op>>,
    /// Primal output keys for this operation, one entry per output position.
    pub primal_out_keys: Vec<GlobalValKey<Op>>,
    /// Saved concrete values keyed by primal value identity.
    pub saved_data: HashMap<GlobalValKey<Op>, Arc<Op::Operand>>,
    /// Edges to the grad nodes that produced this node's inputs.
    pub input_edges: Vec<GradEdge<Op>>,
    /// Which output of the primal op this owning value corresponds to.
    pub output_idx: usize,
}

/// Edge from a grad node to one of its inputs.
pub struct GradEdge<Op: GraphOp> {
    /// Parent grad node. `None` denotes a leaf input.
    pub node: Option<Arc<GradNode<Op>>>,
    /// Gradient accumulation target for this input.
    pub key: GlobalValKey<Op>,
    /// Whether the input participates in gradient propagation.
    pub requires_grad: bool,
}
