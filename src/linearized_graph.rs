use computegraph::graph::Graph;
use computegraph::{GraphOperation, LocalValueId, ValueKey};

/// Classification of a value inside a linearized graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LinearizedValueKind<Op: GraphOperation> {
    /// Value independent of differentiated tangent flow.
    Residual,
    /// Value on differentiated tangent flow.
    Linear {
        /// Primal value whose tangent this value represents, when known.
        primal: Option<ValueKey<Op>>,
    },
}

/// Graph produced by linearizing a primitive computation graph.
pub struct LinearizedGraph<Op: GraphOperation> {
    graph: Graph<Op>,
    tangent_inputs: Vec<(Op::InputKey, LocalValueId)>,
    tangent_outputs: Vec<Option<LocalValueId>>,
    value_kinds: Vec<LinearizedValueKind<Op>>,
}

impl<Op: GraphOperation> LinearizedGraph<Op> {
    pub(crate) fn from_parts(
        graph: Graph<Op>,
        tangent_inputs: Vec<(Op::InputKey, LocalValueId)>,
        tangent_outputs: Vec<Option<LocalValueId>>,
        value_kinds: Vec<LinearizedValueKind<Op>>,
    ) -> Self {
        debug_assert_eq!(graph.values().len(), value_kinds.len());
        Self {
            graph,
            tangent_inputs,
            tangent_outputs,
            value_kinds,
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

    /// Return the classification for a local value in the linearized graph.
    pub fn value_kind(&self, id: LocalValueId) -> &LinearizedValueKind<Op> {
        &self.value_kinds[id]
    }
}
