use std::collections::HashMap;
use std::sync::Arc;

use computegraph::{GraphOperation, ValueKey};

use super::record::{EagerRecordError, EagerRecordResult, RecordedGraph};

/// Opaque handle to an eager reverse-mode trace node.
///
/// Downstream eager values store this handle next to their concrete data and
/// pass it back to [`crate::eager::backward`]. The node and edge layout is
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
    ) -> EagerRecordResult<Self> {
        if primal_out_keys.len() != computation.output_keys().len() {
            return Err(EagerRecordError::count_mismatch(
                "TraceNode primal output keys",
                computation.output_keys().len(),
                primal_out_keys.len(),
            ));
        }
        if input_edges.len() != computation.input_keys().len() {
            return Err(EagerRecordError::count_mismatch(
                "TraceNode input edges",
                computation.input_keys().len(),
                input_edges.len(),
            ));
        }

        Ok(Self {
            computation,
            primal_out_keys,
            saved_data,
            input_edges,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use computegraph::GraphOperation;

    #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    enum Op {
        Id,
    }

    impl GraphOperation for Op {
        type Operand = f64;
        type Context = ();
        type InputKey = &'static str;

        fn input_count(&self) -> usize {
            1
        }

        fn output_count(&self) -> usize {
            1
        }
    }

    fn recorded_id() -> RecordedGraph<Op> {
        RecordedGraph::from_primitive(Op::Id, vec!["x"])
            .expect("test primitive graph metadata should be valid")
    }

    #[test]
    fn trace_node_rejects_mismatched_output_keys() {
        let err = match TraceNode::new(
            recorded_id(),
            Vec::new(),
            HashMap::new(),
            vec![TraceEdge::new(None, ValueKey::Input("x"), true)],
        ) {
            Ok(_) => panic!("TraceNode::new unexpectedly accepted mismatched outputs"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            EagerRecordError::CountMismatch {
                field: "TraceNode primal output keys",
                expected: 1,
                actual: 0
            }
        ));
    }

    #[test]
    fn trace_node_rejects_mismatched_input_edges() {
        let output_key = recorded_id().output_keys()[0].clone();
        let err = match TraceNode::new(recorded_id(), vec![output_key], HashMap::new(), Vec::new())
        {
            Ok(_) => panic!("TraceNode::new unexpectedly accepted mismatched input edges"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            EagerRecordError::CountMismatch {
                field: "TraceNode input edges",
                expected: 1,
                actual: 0
            }
        ));
    }
}
