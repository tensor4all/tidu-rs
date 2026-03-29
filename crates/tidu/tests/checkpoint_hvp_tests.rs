use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tidu::{AdResult, CheckpointRecipe, HvpResult, NodeId, ReplayResult, ReverseRule, Tape};

struct SquareRuleHvp {
    input: NodeId,
    x: f64,
}

impl ReverseRule<f64> for SquareRuleHvp {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        Ok(vec![(self.input, 2.0 * self.x * *cotangent)])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input]
    }

    fn forward_tangents<'t>(
        &self,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t f64>,
    ) -> AdResult<Option<f64>>
    where
        f64: 't,
    {
        let dx = input_tangents(self.input).copied().unwrap_or(0.0);
        Ok(Some(2.0 * self.x * dx))
    }

    fn pullback_with_tangents<'t>(
        &self,
        cotangent: &f64,
        cotangent_tangent: &f64,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t f64>,
    ) -> AdResult<Vec<(NodeId, f64, f64)>>
    where
        f64: 't,
    {
        let dx = input_tangents(self.input).copied().unwrap_or(0.0);
        Ok(vec![(
            self.input,
            2.0 * self.x * *cotangent,
            2.0 * dx * *cotangent + 2.0 * self.x * *cotangent_tangent,
        )])
    }
}

struct ReplayCountingSquareRecipeHvp {
    input: NodeId,
    counter: Arc<AtomicUsize>,
}

impl ReplayCountingSquareRecipeHvp {
    fn new(input: NodeId, counter: Arc<AtomicUsize>) -> Self {
        Self { input, counter }
    }
}

impl CheckpointRecipe<f64> for ReplayCountingSquareRecipeHvp {
    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input]
    }

    fn replay(&self, inputs: &[&f64]) -> AdResult<ReplayResult<f64>> {
        self.counter.fetch_add(1, Ordering::SeqCst);
        let x = *inputs[0];
        Ok(ReplayResult {
            output_primal: x * x,
            rule: Box::new(SquareRuleHvp {
                input: self.input,
                x,
            }),
        })
    }
}

#[test]
fn checkpointed_hvp_matches_materialized_hvp() {
    let mut leaf_tangents = HashMap::new();

    let tape_materialized = Tape::<f64>::new();
    let x_materialized = tape_materialized.leaf(3.0);
    leaf_tangents.insert(x_materialized.node_id().unwrap(), 1.0);
    let y_materialized = tape_materialized.record_op(
        9.0,
        Box::new(SquareRuleHvp {
            input: x_materialized.node_id().unwrap(),
            x: 3.0,
        }),
        None,
    );
    let materialized_hvp: HvpResult<f64> = tape_materialized
        .hvp(&y_materialized, &leaf_tangents)
        .unwrap();

    let mut checkpoint_leaf_tangents = HashMap::new();
    let tape_checkpointed = Tape::<f64>::new();
    let x_checkpointed = tape_checkpointed.leaf(3.0);
    checkpoint_leaf_tangents.insert(x_checkpointed.node_id().unwrap(), 1.0);
    let replay_counter = Arc::new(AtomicUsize::new(0));
    let y_checkpointed = tape_checkpointed.record_checkpointed_op(
        9.0,
        Box::new(ReplayCountingSquareRecipeHvp::new(
            x_checkpointed.node_id().unwrap(),
            replay_counter,
        )),
        None,
    );
    let checkpointed_hvp: HvpResult<f64> = tape_checkpointed
        .hvp(&y_checkpointed, &checkpoint_leaf_tangents)
        .unwrap();

    assert_eq!(
        materialized_hvp
            .gradients
            .get(x_materialized.node_id().unwrap()),
        checkpointed_hvp
            .gradients
            .get(x_checkpointed.node_id().unwrap())
    );
    assert_eq!(
        materialized_hvp.hvp.get(x_materialized.node_id().unwrap()),
        checkpointed_hvp.hvp.get(x_checkpointed.node_id().unwrap())
    );
}

#[test]
fn checkpointed_hvp_replays_each_phase_independently() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    let replay_counter = Arc::new(AtomicUsize::new(0));
    let y = tape.record_checkpointed_op(
        9.0,
        Box::new(ReplayCountingSquareRecipeHvp::new(
            x.node_id().unwrap(),
            replay_counter.clone(),
        )),
        None,
    );
    let z = tape.record_op(
        81.0,
        Box::new(SquareRuleHvp {
            input: y.node_id().unwrap(),
            x: 9.0,
        }),
        None,
    );
    let mut leaf_tangents = HashMap::new();
    leaf_tangents.insert(x.node_id().unwrap(), 1.0);

    let _ = tape.hvp(&z, &leaf_tangents).unwrap();

    assert_eq!(replay_counter.load(Ordering::SeqCst), 2);
}
