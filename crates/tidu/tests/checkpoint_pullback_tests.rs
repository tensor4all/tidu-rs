use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tidu::{AdResult, CheckpointRecipe, NodeId, ReplayResult, ReverseRule, Tape};

struct SquareRule {
    input: NodeId,
    x: f64,
}

impl ReverseRule<f64> for SquareRule {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        Ok(vec![(self.input, 2.0 * self.x * *cotangent)])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input]
    }
}

struct ReplayCountingSquareRecipe {
    input: NodeId,
    counter: Arc<AtomicUsize>,
}

impl ReplayCountingSquareRecipe {
    fn new(input: NodeId, counter: Arc<AtomicUsize>) -> Self {
        Self { input, counter }
    }
}

impl CheckpointRecipe<f64> for ReplayCountingSquareRecipe {
    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input]
    }

    fn replay(&self, inputs: &[&f64]) -> AdResult<ReplayResult<f64>> {
        self.counter.fetch_add(1, Ordering::SeqCst);
        let x = *inputs[0];
        Ok(ReplayResult {
            output_primal: x * x,
            rule: Box::new(SquareRule {
                input: self.input,
                x,
            }),
        })
    }
}

#[test]
fn checkpointed_pullback_matches_materialized_pullback() {
    let tape_materialized = Tape::<f64>::new();
    let x_materialized = tape_materialized.leaf(3.0);
    let y_materialized = tape_materialized.record_op(
        9.0,
        Box::new(SquareRule {
            input: x_materialized.node_id().unwrap(),
            x: 3.0,
        }),
        None,
    );
    let materialized_grads = tape_materialized.pullback(&y_materialized).unwrap();

    let tape_checkpointed = Tape::<f64>::new();
    let x_checkpointed = tape_checkpointed.leaf(3.0);
    let replay_counter = Arc::new(AtomicUsize::new(0));
    let y_checkpointed = tape_checkpointed.record_checkpointed_op(
        9.0,
        Box::new(ReplayCountingSquareRecipe::new(
            x_checkpointed.node_id().unwrap(),
            replay_counter,
        )),
        None,
    );
    let checkpointed_grads = tape_checkpointed.pullback(&y_checkpointed).unwrap();

    assert_eq!(
        materialized_grads.get(x_materialized.node_id().unwrap()),
        checkpointed_grads.get(x_checkpointed.node_id().unwrap())
    );
}

#[test]
fn checkpointed_pullback_replays_lazily() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(4.0);
    let replay_counter = Arc::new(AtomicUsize::new(0));
    let y = tape.record_checkpointed_op(
        16.0,
        Box::new(ReplayCountingSquareRecipe::new(
            x.node_id().unwrap(),
            replay_counter.clone(),
        )),
        None,
    );

    assert_eq!(replay_counter.load(Ordering::SeqCst), 0);
    let _ = tape.pullback(&y).unwrap();
    assert_eq!(replay_counter.load(Ordering::SeqCst), 1);
}

#[test]
fn checkpointed_pullback_does_not_persist_replayed_state_on_tape() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(5.0);
    let replay_counter = Arc::new(AtomicUsize::new(0));
    let y = tape.record_checkpointed_op(
        25.0,
        Box::new(ReplayCountingSquareRecipe::new(
            x.node_id().unwrap(),
            replay_counter.clone(),
        )),
        None,
    );

    let _ = tape.pullback(&y).unwrap();
    let _ = tape.pullback(&y).unwrap();

    assert_eq!(replay_counter.load(Ordering::SeqCst), 2);
}
