use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::reverse_graph::{
    accumulate_leaf_grad, leaf_key, node_key, ReverseEdge, ReverseInput, ReverseNode,
};
use crate::{AdResult, AutodiffError, Differentiable};

pub(crate) enum TaskInput<V: Differentiable> {
    Leaf(crate::reverse_graph::LeafHandle<V>),
    Edge { node_id: usize, output_slot: usize },
}

impl<V: Differentiable> Clone for TaskInput<V> {
    fn clone(&self) -> Self {
        match self {
            Self::Leaf(handle) => Self::Leaf(handle.clone()),
            Self::Edge {
                node_id,
                output_slot,
            } => Self::Edge {
                node_id: *node_id,
                output_slot: *output_slot,
            },
        }
    }
}

pub(crate) struct TaskNode<V: Differentiable> {
    pub(crate) node: Arc<ReverseNode<V>>,
    pub(crate) parents: Vec<Option<TaskInput<V>>>,
    pub(crate) grad_outputs: Vec<Option<V::Tangent>>,
    pub(crate) live_output_slots: Vec<bool>,
    pub(crate) remaining_contributions: Vec<usize>,
    pub(crate) pending_output_slots: usize,
    pub(crate) enqueued: bool,
}

pub(crate) struct GraphTask<V: Differentiable> {
    pub(crate) nodes: Vec<TaskNode<V>>,
    pub(crate) node_ids: HashMap<usize, usize>,
    pub(crate) ready: VecDeque<usize>,
    pub(crate) edge_queries: HashMap<(usize, usize), Vec<usize>>,
    pub(crate) leaf_queries: HashMap<usize, Vec<usize>>,
    pub(crate) query_grads: Vec<Option<V::Tangent>>,
}

impl<V: Differentiable> GraphTask<V> {
    pub(crate) fn from_root(root: &ReverseEdge<V>, seed: V::Tangent) -> AdResult<Self> {
        let mut task = Self {
            nodes: Vec::new(),
            node_ids: HashMap::new(),
            ready: VecDeque::new(),
            edge_queries: HashMap::new(),
            leaf_queries: HashMap::new(),
            query_grads: Vec::new(),
        };
        task.discover(&root.node);
        task.finalize_parents_and_contributions(root);
        task.seed_root(root, seed)?;
        Ok(task)
    }

    fn discover(&mut self, node: &Arc<ReverseNode<V>>) {
        let key = node_key(node);
        if self.node_ids.contains_key(&key) {
            return;
        }
        let task_node_id = self.nodes.len();
        self.node_ids.insert(key, task_node_id);
        self.nodes.push(TaskNode {
            node: node.clone(),
            parents: Vec::new(),
            grad_outputs: (0..node.output_count).map(|_| None).collect(),
            live_output_slots: vec![false; node.output_count],
            remaining_contributions: vec![0; node.output_count],
            pending_output_slots: 0,
            enqueued: false,
        });
        for parent in &node.parents {
            if let Some(ReverseInput::Edge(edge)) = parent {
                self.discover(&edge.node);
            }
        }
    }

    fn finalize_parents_and_contributions(&mut self, root: &ReverseEdge<V>) {
        for task_node_id in 0..self.nodes.len() {
            let parents = self.nodes[task_node_id].node.parents.clone();
            let mut normalized_parents = Vec::with_capacity(parents.len());
            for parent in parents {
                let normalized = match parent {
                    Some(ReverseInput::Leaf(handle)) => Some(TaskInput::Leaf(handle)),
                    Some(ReverseInput::Edge(edge)) => {
                        let parent_id = self.node_ids[&node_key(&edge.node)];
                        self.nodes[parent_id].live_output_slots[edge.output_slot] = true;
                        self.nodes[parent_id].remaining_contributions[edge.output_slot] += 1;
                        Some(TaskInput::Edge {
                            node_id: parent_id,
                            output_slot: edge.output_slot,
                        })
                    }
                    None => None,
                };
                normalized_parents.push(normalized);
            }
            self.nodes[task_node_id].parents = normalized_parents;
        }

        let root_id = self.node_ids[&node_key(&root.node)];
        self.nodes[root_id].live_output_slots[root.output_slot] = true;
        self.nodes[root_id].remaining_contributions[root.output_slot] += 1;

        for node in &mut self.nodes {
            node.pending_output_slots = node
                .live_output_slots
                .iter()
                .zip(&node.remaining_contributions)
                .filter(|(live, count)| **live && **count > 0)
                .count();
        }
    }

    fn seed_root(&mut self, root: &ReverseEdge<V>, seed: V::Tangent) -> AdResult<()> {
        let root_id = self.node_ids[&node_key(&root.node)];
        self.accumulate_slot(root_id, root.output_slot, seed)
    }

    pub(crate) fn register_queries(&mut self, wrt: &[Option<ReverseInput<V>>]) {
        self.query_grads = (0..wrt.len()).map(|_| None).collect();
        for (index, target) in wrt.iter().enumerate() {
            match target {
                Some(ReverseInput::Leaf(handle)) => {
                    self.leaf_queries
                        .entry(leaf_key(handle))
                        .or_default()
                        .push(index);
                }
                Some(ReverseInput::Edge(edge)) => {
                    if let Some(&node_id) = self.node_ids.get(&node_key(&edge.node)) {
                        self.edge_queries
                            .entry((node_id, edge.output_slot))
                            .or_default()
                            .push(index);
                    }
                }
                None => {}
            }
        }
    }

    fn accumulate_query_slots(&mut self, query_slots: &[usize], grad: &V::Tangent)
    where
        V::Tangent: Clone,
    {
        for &query_slot in query_slots {
            let slot = &mut self.query_grads[query_slot];
            match slot.take() {
                Some(existing) => *slot = Some(V::accumulate_tangent(existing, grad)),
                None => *slot = Some(grad.clone()),
            }
        }
    }

    fn accumulate_slot(
        &mut self,
        task_node_id: usize,
        output_slot: usize,
        grad: V::Tangent,
    ) -> AdResult<()> {
        let Some(output_len) = self
            .nodes
            .get(task_node_id)
            .map(|node| node.grad_outputs.len())
        else {
            return Err(AutodiffError::MissingNode);
        };
        if output_slot >= output_len {
            return Err(AutodiffError::InvalidArgument(format!(
                "output slot {output_slot} is out of bounds for task node {task_node_id}"
            )));
        }
        if let Some(query_slots) = self.edge_queries.get(&(task_node_id, output_slot)).cloned() {
            self.accumulate_query_slots(&query_slots, &grad);
        }
        let node = &mut self.nodes[task_node_id];
        let slot = &mut node.grad_outputs[output_slot];
        match slot.take() {
            Some(existing) => *slot = Some(V::accumulate_tangent(existing, &grad)),
            None => *slot = Some(grad),
        }
        if node.remaining_contributions[output_slot] == 0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "received too many cotangent contributions for task node {task_node_id} output slot {output_slot}"
            )));
        }
        node.remaining_contributions[output_slot] -= 1;
        if node.live_output_slots[output_slot] && node.remaining_contributions[output_slot] == 0 {
            debug_assert!(node.pending_output_slots > 0);
            node.pending_output_slots -= 1;
        }
        if node.pending_output_slots == 0 && !node.enqueued {
            node.enqueued = true;
            self.ready.push_back(task_node_id);
        }
        Ok(())
    }

    pub(crate) fn run(mut self, accumulate_leafs: bool) -> AdResult<Vec<Option<V::Tangent>>>
    where
        V::Tangent: Clone,
    {
        while let Some(task_node_id) = self.ready.pop_front() {
            let grads = {
                let node = &mut self.nodes[task_node_id];
                node.enqueued = false;
                let grad_outputs = std::mem::take(&mut node.grad_outputs);
                node.node.vjp(&grad_outputs)?
            };

            if grads.len() != self.nodes[task_node_id].parents.len() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "linearized vjp returned {} gradients for {} inputs",
                    grads.len(),
                    self.nodes[task_node_id].parents.len()
                )));
            }

            let parents = self.nodes[task_node_id].parents.clone();
            for (parent, grad) in parents.into_iter().zip(grads.into_iter()) {
                let Some(grad) = grad else {
                    continue;
                };
                match parent {
                    Some(TaskInput::Leaf(handle)) => {
                        if let Some(query_slots) =
                            self.leaf_queries.get(&leaf_key(&handle)).cloned()
                        {
                            self.accumulate_query_slots(&query_slots, &grad);
                        }
                        if accumulate_leafs {
                            accumulate_leaf_grad::<V>(&handle, grad);
                        }
                    }
                    Some(TaskInput::Edge {
                        node_id,
                        output_slot,
                    }) => self.accumulate_slot(node_id, output_slot, grad)?,
                    None => {}
                }
            }
        }
        Ok(self.query_grads)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::GraphTask;
    use crate::linearized::LinearizedOp;
    use crate::reverse_graph::{
        leaf_handle, leaf_key, ReverseEdge, ReverseInput, ReverseNode, StoredNodeLinearization,
    };
    use crate::{AdResult, AutodiffError};

    struct FixedLinearized {
        grads: Vec<Option<f64>>,
    }

    impl LinearizedOp<f64> for FixedLinearized {
        fn jvp(&self, _input_tangents: &[Option<f64>]) -> AdResult<Vec<Option<f64>>> {
            Ok(vec![None; self.grads.len()])
        }

        fn vjp(
            &self,
            _output_cotangents: &[Option<f64>],
            _input_grad_mask: &[bool],
        ) -> AdResult<Vec<Option<f64>>> {
            Ok(self.grads.clone())
        }
    }

    fn retained_node(
        parents: Vec<Option<ReverseInput<f64>>>,
        output_count: usize,
        input_grad_mask: Vec<bool>,
        grads: Vec<Option<f64>>,
    ) -> Arc<ReverseNode<f64>> {
        Arc::new(ReverseNode::new(
            parents,
            output_count,
            input_grad_mask,
            StoredNodeLinearization::retained(FixedLinearized { grads }),
        ))
    }

    #[test]
    fn discover_deduplicates_shared_parent_nodes() {
        let leaf = leaf_handle::<f64>();
        let shared = retained_node(
            vec![Some(ReverseInput::Leaf(leaf))],
            1,
            vec![true],
            vec![Some(1.0)],
        );
        let root = ReverseEdge {
            node: retained_node(
                vec![
                    Some(ReverseInput::Edge(ReverseEdge {
                        node: shared.clone(),
                        output_slot: 0,
                    })),
                    Some(ReverseInput::Edge(ReverseEdge {
                        node: shared,
                        output_slot: 0,
                    })),
                ],
                1,
                vec![true, true],
                vec![Some(1.0), Some(1.0)],
            ),
            output_slot: 0,
        };

        let task = GraphTask::from_root(&root, 1.0).unwrap();
        assert_eq!(task.nodes.len(), 2);
    }

    #[test]
    fn register_queries_tracks_leaf_and_edge_targets_and_ignores_missing_ones() {
        let leaf = leaf_handle::<f64>();
        let root = ReverseEdge {
            node: retained_node(
                vec![Some(ReverseInput::Leaf(leaf.clone()))],
                1,
                vec![true],
                vec![Some(1.0)],
            ),
            output_slot: 0,
        };
        let disconnected = ReverseEdge {
            node: retained_node(
                vec![Some(ReverseInput::Leaf(leaf_handle::<f64>()))],
                1,
                vec![true],
                vec![Some(1.0)],
            ),
            output_slot: 0,
        };

        let mut task = GraphTask::from_root(&root, 1.0).unwrap();
        task.register_queries(&[
            Some(ReverseInput::Leaf(leaf.clone())),
            Some(ReverseInput::Edge(root.clone())),
            Some(ReverseInput::Edge(disconnected)),
            None,
        ]);

        assert_eq!(task.query_grads, vec![None, None, None, None]);
        assert_eq!(task.leaf_queries.get(&leaf_key(&leaf)), Some(&vec![0]));
        assert_eq!(task.edge_queries.get(&(0, 0)), Some(&vec![1]));
        assert!(!task.edge_queries.values().any(|slots| slots.contains(&2)));
    }

    #[test]
    fn accumulate_slot_validates_indices_accumulates_queries_and_rejects_extra_cotangents() {
        let root = ReverseEdge {
            node: retained_node(vec![None], 1, vec![false], vec![None]),
            output_slot: 0,
        };
        let mut task = GraphTask::from_root(&root, 1.0).unwrap();

        assert!(matches!(
            task.accumulate_slot(99, 0, 1.0),
            Err(AutodiffError::MissingNode)
        ));
        assert!(matches!(
            task.accumulate_slot(0, 99, 1.0),
            Err(AutodiffError::InvalidArgument(_))
        ));

        task.ready.clear();
        task.nodes[0].enqueued = false;
        task.nodes[0].grad_outputs[0] = Some(0.5);
        task.nodes[0].remaining_contributions[0] = 1;
        task.nodes[0].pending_output_slots = 1;
        task.query_grads = vec![Some(1.0), None];
        task.edge_queries.insert((0, 0), vec![0, 1]);

        task.accumulate_slot(0, 0, 1.0).unwrap();
        assert_eq!(task.query_grads, vec![Some(2.0), Some(1.0)]);
        assert_eq!(task.nodes[0].grad_outputs[0], Some(1.5));
        assert_eq!(task.ready.pop_front(), Some(0));

        assert!(matches!(
            task.accumulate_slot(0, 0, 1.0),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn run_rejects_linearized_vjp_arity_mismatches() {
        let root = ReverseEdge {
            node: retained_node(vec![], 1, vec![], vec![Some(1.0)]),
            output_slot: 0,
        };

        let err = GraphTask::from_root(&root, 1.0)
            .unwrap()
            .run(false)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        assert!(err
            .to_string()
            .contains("linearized vjp returned 1 gradients for 0 inputs"));
    }
}
