use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::{AdResult, AutodiffError, Differentiable, Gradients, HvpResult, NodeId, ReverseRule};

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

struct GraphNode<V: Differentiable> {
    rule: Option<Box<dyn ReverseRule<V>>>,
    is_leaf: bool,
}

pub(crate) struct AutogradGraph<V: Differentiable> {
    id: u64,
    graph_alive: bool,
    nodes: Vec<GraphNode<V>>,
}

impl<V: Differentiable> AutogradGraph<V> {
    pub(crate) fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            graph_alive: true,
            nodes: Vec::new(),
        }))
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn ensure_alive(&self) -> AdResult<()> {
        if self.graph_alive {
            Ok(())
        } else {
            Err(AutodiffError::GraphFreed)
        }
    }

    pub(crate) fn free_graph(&mut self) {
        self.graph_alive = false;
    }

    pub(crate) fn record_leaf(&mut self) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        self.nodes.push(GraphNode {
            rule: None,
            is_leaf: true,
        });
        self.graph_alive = true;
        id
    }

    pub(crate) fn record_op(&mut self, rule: Box<dyn ReverseRule<V>>) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        self.nodes.push(GraphNode {
            rule: Some(rule),
            is_leaf: false,
        });
        self.graph_alive = true;
        id
    }

    pub(crate) fn record_placeholder(&mut self) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        self.nodes.push(GraphNode {
            rule: None,
            is_leaf: false,
        });
        self.graph_alive = true;
        id
    }

    pub(crate) fn has_node(&self, node: NodeId) -> bool {
        node.index() < self.nodes.len()
    }

    pub(crate) fn attach_rule(
        &mut self,
        node: NodeId,
        rule: Box<dyn ReverseRule<V>>,
    ) -> AdResult<()> {
        let Some(entry) = self.nodes.get_mut(node.index()) else {
            return Err(AutodiffError::MissingNode);
        };
        entry.rule = Some(rule);
        entry.is_leaf = false;
        Ok(())
    }

    pub(crate) fn compute_cotangents(
        &self,
        output_node: NodeId,
        seed: V::Tangent,
    ) -> AdResult<Vec<Option<V::Tangent>>> {
        let n = self.nodes.len();
        if output_node.index() >= n {
            return Err(AutodiffError::MissingNode);
        }

        let mut cotangents = vec![None; n];
        cotangents[output_node.index()] = Some(seed);

        for i in (0..=output_node.index()).rev() {
            let Some(rule) = self.nodes[i].rule.as_ref() else {
                continue;
            };
            let Some(cot) = cotangents[i].take() else {
                continue;
            };
            let input_grads = rule.pullback(&cot)?;
            for (node_id, grad) in input_grads {
                let idx = node_id.index();
                match cotangents[idx].take() {
                    Some(existing) => {
                        cotangents[idx] = Some(V::accumulate_tangent(existing, &grad))
                    }
                    None => cotangents[idx] = Some(grad),
                }
            }
        }

        Ok(cotangents)
    }

    pub(crate) fn pullback_from(
        &self,
        output_node: NodeId,
        seed: V::Tangent,
    ) -> AdResult<Gradients<V>> {
        let mut cotangents = self.compute_cotangents(output_node, seed)?;
        let mut gradients = Gradients::new();
        for (i, cot) in cotangents.iter_mut().enumerate() {
            if !self.nodes[i].is_leaf {
                continue;
            }
            if let Some(value) = cot.take() {
                gradients.push_entry(NodeId::new(i), value);
            }
        }
        Ok(gradients)
    }

    pub(crate) fn compute_cotangents_with_tangents(
        &self,
        output_node: NodeId,
        seed: V::Tangent,
        seed_tangent: V::Tangent,
    ) -> AdResult<(Vec<Option<V::Tangent>>, Vec<Option<V::Tangent>>)>
    where
        V::Tangent: Clone + Differentiable<Tangent = V::Tangent>,
    {
        let n = self.nodes.len();
        if output_node.index() >= n {
            return Err(AutodiffError::MissingNode);
        }

        let mut cotangents = vec![None; n];
        let mut cot_tangents = vec![None; n];
        cotangents[output_node.index()] = Some(seed);
        cot_tangents[output_node.index()] = Some(seed_tangent);

        for i in (0..=output_node.index()).rev() {
            let Some(rule) = self.nodes[i].rule.as_ref() else {
                continue;
            };
            let Some(cot) = cotangents[i].take() else {
                continue;
            };
            let cot_tan = cot_tangents[i].take().unwrap_or_else(|| cot.zero_tangent());
            let input_grads = rule.pullback_with_tangents(&cot, &cot_tan)?;
            for (node_id, grad, grad_tan) in input_grads {
                let idx = node_id.index();
                match cotangents[idx].take() {
                    Some(existing) => {
                        cotangents[idx] = Some(V::accumulate_tangent(existing, &grad))
                    }
                    None => cotangents[idx] = Some(grad),
                }
                match cot_tangents[idx].take() {
                    Some(existing) => {
                        cot_tangents[idx] = Some(V::accumulate_tangent(existing, &grad_tan))
                    }
                    None => cot_tangents[idx] = Some(grad_tan),
                }
            }
        }

        Ok((cotangents, cot_tangents))
    }

    pub(crate) fn hvp_from(
        &self,
        output_node: NodeId,
        seed: V::Tangent,
        seed_tangent: V::Tangent,
    ) -> AdResult<HvpResult<V>>
    where
        V::Tangent: Clone + Differentiable<Tangent = V::Tangent>,
    {
        let (mut cotangents, mut cot_tangents) =
            self.compute_cotangents_with_tangents(output_node, seed, seed_tangent)?;
        let mut gradients = Gradients::new();
        let mut hvp = Gradients::new();
        for i in 0..self.nodes.len() {
            if !self.nodes[i].is_leaf {
                continue;
            }
            if let Some(value) = cotangents[i].take() {
                gradients.push_entry(NodeId::new(i), value);
            }
            if let Some(value) = cot_tangents[i].take() {
                hvp.push_entry(NodeId::new(i), value);
            }
        }
        Ok(HvpResult { gradients, hvp })
    }
}
