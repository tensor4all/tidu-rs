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
    let err = match TraceNode::new(recorded_id(), vec![output_key], HashMap::new(), Vec::new()) {
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
