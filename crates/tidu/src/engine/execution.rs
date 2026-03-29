use std::collections::HashMap;

use crate::engine::{AutogradGraph, NodeExec};
use crate::{AdResult, AutodiffError, Differentiable, NodeId, ReverseRule};

pub(crate) struct ReplayExecution<'g, V: Differentiable> {
    graph: &'g AutogradGraph<V>,
    replayed_primals: HashMap<NodeId, V>,
    replayed_rules: HashMap<NodeId, Box<dyn ReverseRule<V>>>,
}

impl<'g, V: Differentiable> ReplayExecution<'g, V> {
    pub(crate) fn new(graph: &'g AutogradGraph<V>) -> Self {
        Self {
            graph,
            replayed_primals: HashMap::new(),
            replayed_rules: HashMap::new(),
        }
    }

    pub(crate) fn rule(&mut self, node: NodeId) -> AdResult<Option<&dyn ReverseRule<V>>> {
        match self.graph.node(node)?.exec() {
            NodeExec::Leaf | NodeExec::Placeholder => Ok(None),
            NodeExec::Materialized(rule) => Ok(Some(rule.as_ref())),
            NodeExec::Replayable(_) => {
                self.ensure_replayed(node)?;
                Ok(Some(
                    self.replayed_rules
                        .get(&node)
                        .expect("replayed rule missing after ensure_replayed")
                        .as_ref(),
                ))
            }
        }
    }

    fn ensure_primal(&mut self, node: NodeId) -> AdResult<()> {
        if self.peek_primal(node).is_some() {
            return Ok(());
        }
        self.ensure_replayed(node)
    }

    fn ensure_replayed(&mut self, node: NodeId) -> AdResult<()> {
        if self.replayed_rules.contains_key(&node) {
            return Ok(());
        }
        if self.graph.node(node)?.retained_primal().is_some()
            && self.graph.node(node)?.replay_recipe().is_none()
        {
            return Ok(());
        }
        let input_ids = self
            .graph
            .node(node)?
            .replay_recipe()
            .ok_or_else(|| {
                AutodiffError::InvalidArgument(format!(
                    "node {} is not replayable and has no retained primal",
                    node.index()
                ))
            })?
            .inputs();
        for input in &input_ids {
            self.ensure_primal(*input)?;
        }
        let input_primal_refs = input_ids
            .iter()
            .map(|input| {
                self.peek_primal(*input).ok_or_else(|| {
                    AutodiffError::InvalidArgument(format!(
                        "missing replay input primal for node {}",
                        input.index()
                    ))
                })
            })
            .collect::<AdResult<Vec<_>>>()?;
        let replayed = self
            .graph
            .node(node)?
            .replay_recipe()
            .expect("recipe disappeared while replaying")
            .replay(&input_primal_refs)?;
        self.replayed_primals.insert(node, replayed.output_primal);
        self.replayed_rules.insert(node, replayed.rule);
        Ok(())
    }

    fn peek_primal(&self, node: NodeId) -> Option<&V> {
        self.graph
            .node(node)
            .ok()
            .and_then(|entry| entry.retained_primal())
            .or_else(|| self.replayed_primals.get(&node))
    }
}

pub(crate) struct ForwardTangentExecution<'g, V>
where
    V: Differentiable,
    V::Tangent: Clone + Differentiable<Tangent = V::Tangent>,
{
    replay: ReplayExecution<'g, V>,
    leaf_tangents: &'g HashMap<NodeId, V::Tangent>,
    tangents: HashMap<NodeId, Option<V::Tangent>>,
}

impl<'g, V> ForwardTangentExecution<'g, V>
where
    V: Differentiable,
    V::Tangent: Clone + Differentiable<Tangent = V::Tangent>,
{
    pub(crate) fn new(
        graph: &'g AutogradGraph<V>,
        leaf_tangents: &'g HashMap<NodeId, V::Tangent>,
    ) -> Self {
        Self {
            replay: ReplayExecution::new(graph),
            leaf_tangents,
            tangents: HashMap::new(),
        }
    }

    pub(crate) fn tangent(&mut self, node: NodeId) -> AdResult<Option<V::Tangent>> {
        if let Some(tangent) = self.tangents.get(&node) {
            return Ok(tangent.clone());
        }

        let computed = match self.replay.graph.node(node)?.exec() {
            NodeExec::Leaf => self.leaf_tangents.get(&node).cloned(),
            NodeExec::Placeholder => None,
            NodeExec::Materialized(_) | NodeExec::Replayable(_) => {
                let input_tangents = self
                    .replay
                    .graph
                    .node(node)?
                    .inputs()
                    .iter()
                    .map(|input| Ok((*input, self.tangent(*input)?)))
                    .collect::<AdResult<Vec<_>>>()?;
                let input_tangents_fn = |query: NodeId| -> Option<&V::Tangent> {
                    input_tangents
                        .iter()
                        .find(|(node_id, _)| *node_id == query)
                        .and_then(|(_, tangent)| tangent.as_ref())
                };
                self.replay
                    .rule(node)?
                    .ok_or_else(|| {
                        AutodiffError::InvalidArgument(format!(
                            "missing rule while computing tangent for node {}",
                            node.index()
                        ))
                    })?
                    .forward_tangents(&input_tangents_fn)?
            }
        };

        self.tangents.insert(node, computed.clone());
        Ok(computed)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::{ForwardTangentExecution, ReplayExecution};
    use crate::engine::AutogradGraph;
    use crate::{AdResult, CheckpointRecipe, NodeId, ReplayResult, ReverseRule};

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
    fn replay_execution_replays_nested_checkpoint_inputs_once_per_execution() {
        let graph = AutogradGraph::<f64>::new();
        let mut guard = graph.lock().unwrap();
        let x = guard.record_leaf(2.0);
        let y_counter = Arc::new(AtomicUsize::new(0));
        let z_counter = Arc::new(AtomicUsize::new(0));
        let y = guard.record_checkpointed_op(Box::new(ReplayCountingSquareRecipe::new(
            x,
            y_counter.clone(),
        )));
        let z = guard.record_checkpointed_op(Box::new(ReplayCountingSquareRecipe::new(
            y,
            z_counter.clone(),
        )));

        let mut execution = ReplayExecution::new(&guard);
        execution.ensure_replayed(x).unwrap();
        assert!(execution.rule(x).unwrap().is_none());

        assert!(execution.rule(z).unwrap().is_some());
        assert_eq!(y_counter.load(Ordering::SeqCst), 1);
        assert_eq!(z_counter.load(Ordering::SeqCst), 1);
        assert_eq!(execution.peek_primal(y), Some(&4.0));

        assert!(execution.rule(z).unwrap().is_some());
        assert_eq!(y_counter.load(Ordering::SeqCst), 1);
        assert_eq!(z_counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn forward_tangent_execution_caches_results_and_handles_placeholders() {
        let graph = AutogradGraph::<f64>::new();
        let mut guard = graph.lock().unwrap();
        let x = guard.record_leaf(3.0);
        let placeholder = guard.record_placeholder(0.0);
        let y_counter = Arc::new(AtomicUsize::new(0));
        let y = guard.record_checkpointed_op(Box::new(ReplayCountingSquareRecipeHvp::new(
            x,
            y_counter.clone(),
        )));
        let z = guard.record_op(81.0, Box::new(SquareRuleHvp { input: y, x: 9.0 }));
        let mut leaf_tangents = HashMap::new();
        leaf_tangents.insert(x, 1.0);

        let mut execution = ForwardTangentExecution::new(&guard, &leaf_tangents);
        assert_eq!(execution.tangent(placeholder).unwrap(), None);
        assert_eq!(execution.tangent(y).unwrap(), Some(6.0));
        assert_eq!(execution.tangent(y).unwrap(), Some(6.0));
        assert_eq!(execution.tangent(z).unwrap(), Some(108.0));
        assert_eq!(y_counter.load(Ordering::SeqCst), 1);
    }
}
