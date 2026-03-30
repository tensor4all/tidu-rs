use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use chainrules_core::NodeId;

use crate::engine::{EngineRule, Gradients, HvpResult, OutputRef};
use crate::{AdResult, AutodiffError, Differentiable};

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

type TangentSlots<V> = Vec<Option<<V as Differentiable>::Tangent>>;
type NodeTangentSlots<V> = Vec<TangentSlots<V>>;

struct GraphNode<V: Differentiable> {
    rule: Option<Box<dyn EngineRule<V>>>,
    output_count: usize,
    is_leaf: bool,
    grad: Option<V::Tangent>,
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
            output_count: 1,
            is_leaf: true,
            grad: None,
        });
        self.graph_alive = true;
        id
    }

    pub(crate) fn record_op(&mut self, rule: Box<dyn EngineRule<V>>) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        let output_count = rule.output_count();
        debug_assert!(output_count > 0);
        self.nodes.push(GraphNode {
            rule: Some(rule),
            output_count,
            is_leaf: false,
            grad: None,
        });
        self.graph_alive = true;
        id
    }

    pub(crate) fn record_placeholder(&mut self) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        self.nodes.push(GraphNode {
            rule: None,
            output_count: 1,
            is_leaf: false,
            grad: None,
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
        rule: Box<dyn EngineRule<V>>,
    ) -> AdResult<()> {
        let Some(entry) = self.nodes.get_mut(node.index()) else {
            return Err(AutodiffError::MissingNode);
        };
        entry.output_count = rule.output_count();
        entry.rule = Some(rule);
        entry.is_leaf = false;
        Ok(())
    }

    pub(crate) fn leaf_grad(&self, node: NodeId) -> AdResult<Option<V::Tangent>>
    where
        V::Tangent: Clone,
    {
        let Some(entry) = self.nodes.get(node.index()) else {
            return Err(AutodiffError::MissingNode);
        };
        if !entry.is_leaf {
            return Ok(None);
        }
        Ok(entry.grad.clone())
    }

    pub(crate) fn zero_leaf_grad(&mut self, node: NodeId) -> AdResult<()> {
        let Some(entry) = self.nodes.get_mut(node.index()) else {
            return Err(AutodiffError::MissingNode);
        };
        if entry.is_leaf {
            entry.grad = None;
        }
        Ok(())
    }

    pub(crate) fn accumulate_leaf_gradients(&mut self, gradients: &Gradients<V>) -> AdResult<()>
    where
        V::Tangent: Clone,
    {
        for (node, grad) in gradients.entries() {
            let Some(entry) = self.nodes.get_mut(node.index()) else {
                return Err(AutodiffError::MissingNode);
            };
            if !entry.is_leaf {
                continue;
            }
            entry.grad = match entry.grad.take() {
                Some(existing) => Some(V::accumulate_tangent(existing, grad)),
                None => Some(grad.clone()),
            };
        }
        Ok(())
    }

    fn validate_output_ref(&self, output_ref: OutputRef) -> AdResult<()> {
        let Some(node) = self.nodes.get(output_ref.node_id.index()) else {
            return Err(AutodiffError::MissingNode);
        };
        if output_ref.output_slot >= node.output_count {
            return Err(AutodiffError::InvalidArgument(format!(
                "output slot {} is out of bounds for node {} with {} outputs",
                output_ref.output_slot,
                output_ref.node_id.index(),
                node.output_count
            )));
        }
        Ok(())
    }

    fn empty_slots(&self) -> NodeTangentSlots<V> {
        self.nodes
            .iter()
            .map(|node| vec![None; node.output_count])
            .collect()
    }

    fn accumulate_slot(
        &self,
        slots: &mut NodeTangentSlots<V>,
        output_ref: OutputRef,
        grad: V::Tangent,
    ) -> AdResult<()> {
        self.validate_output_ref(output_ref)?;
        let node_slots = &mut slots[output_ref.node_id.index()];
        let slot = &mut node_slots[output_ref.output_slot];
        match slot.take() {
            Some(existing) => *slot = Some(V::accumulate_tangent(existing, &grad)),
            None => *slot = Some(grad),
        }
        Ok(())
    }

    pub(crate) fn compute_cotangents(
        &self,
        output_ref: OutputRef,
        seed: V::Tangent,
    ) -> AdResult<NodeTangentSlots<V>> {
        self.validate_output_ref(output_ref)?;

        let mut cotangents = self.empty_slots();
        cotangents[output_ref.node_id.index()][output_ref.output_slot] = Some(seed);

        for i in (0..=output_ref.node_id.index()).rev() {
            let Some(rule) = self.nodes[i].rule.as_ref() else {
                continue;
            };
            if cotangents[i].iter().all(Option::is_none) {
                continue;
            }
            let grad_outputs = std::mem::take(&mut cotangents[i]);
            let input_grads = rule.pullback(&grad_outputs)?;
            for (input_ref, grad) in input_grads {
                self.accumulate_slot(&mut cotangents, input_ref, grad)?;
            }
        }

        Ok(cotangents)
    }

    pub(crate) fn pullback_from(
        &self,
        output_ref: OutputRef,
        seed: V::Tangent,
    ) -> AdResult<Gradients<V>> {
        let mut cotangents = self.compute_cotangents(output_ref, seed)?;
        let mut gradients = Gradients::new();
        for (index, node_slots) in cotangents.iter_mut().enumerate() {
            if !self.nodes[index].is_leaf {
                continue;
            }
            if let Some(value) = node_slots[0].take() {
                gradients.push_entry(NodeId::new(index), value);
            }
        }
        Ok(gradients)
    }

    pub(crate) fn compute_cotangents_with_tangents(
        &self,
        output_ref: OutputRef,
        seed: V::Tangent,
        seed_tangent: V::Tangent,
        tangents: &NodeTangentSlots<V>,
    ) -> AdResult<(NodeTangentSlots<V>, NodeTangentSlots<V>)>
    where
        V::Tangent: Clone + Differentiable<Tangent = V::Tangent>,
    {
        self.validate_output_ref(output_ref)?;

        let mut cotangents = self.empty_slots();
        let mut cot_tangents = self.empty_slots();
        cotangents[output_ref.node_id.index()][output_ref.output_slot] = Some(seed);
        cot_tangents[output_ref.node_id.index()][output_ref.output_slot] = Some(seed_tangent);

        for i in (0..=output_ref.node_id.index()).rev() {
            let Some(rule) = self.nodes[i].rule.as_ref() else {
                continue;
            };
            if cotangents[i].iter().all(Option::is_none) {
                continue;
            }
            let grad_outputs = std::mem::take(&mut cotangents[i]);
            let grad_output_tangents = std::mem::take(&mut cot_tangents[i]);
            let input_tangents_fn = |input_ref: OutputRef| -> Option<&V::Tangent> {
                tangents
                    .get(input_ref.node_id.index())
                    .and_then(|node_slots| node_slots.get(input_ref.output_slot))
                    .and_then(|tangent| tangent.as_ref())
            };
            let input_grads = rule.pullback_with_tangents(
                &grad_outputs,
                &grad_output_tangents,
                &input_tangents_fn,
            )?;
            for (input_ref, grad, grad_tangent) in input_grads {
                self.accumulate_slot(&mut cotangents, input_ref, grad)?;
                self.accumulate_slot(&mut cot_tangents, input_ref, grad_tangent)?;
            }
        }

        Ok((cotangents, cot_tangents))
    }

    /// Two-phase HVP: forward tangent propagation then reverse pass.
    pub(crate) fn hvp_from(
        &self,
        output_ref: OutputRef,
        seed: V::Tangent,
        seed_tangent: V::Tangent,
        leaf_tangents: &std::collections::HashMap<NodeId, V::Tangent>,
    ) -> AdResult<HvpResult<V>>
    where
        V::Tangent: Clone + Differentiable<Tangent = V::Tangent>,
    {
        self.validate_output_ref(output_ref)?;

        let mut tangents = self.empty_slots();
        for i in 0..=output_ref.node_id.index() {
            let node = &self.nodes[i];
            if node.is_leaf {
                tangents[i][0] = leaf_tangents.get(&NodeId::new(i)).cloned();
                continue;
            }
            let Some(rule) = node.rule.as_ref() else {
                continue;
            };
            let tangents_fn = |input_ref: OutputRef| -> Option<&V::Tangent> {
                tangents
                    .get(input_ref.node_id.index())
                    .and_then(|node_slots| node_slots.get(input_ref.output_slot))
                    .and_then(|tangent| tangent.as_ref())
            };
            let output_tangents = rule.forward_tangents(&tangents_fn)?;
            if output_tangents.len() != node.output_count {
                return Err(AutodiffError::InvalidArgument(format!(
                    "rule for node {} returned {} forward tangent slots for {} outputs",
                    i,
                    output_tangents.len(),
                    node.output_count
                )));
            }
            tangents[i] = output_tangents;
        }

        let (mut cotangents, mut cot_tangents) =
            self.compute_cotangents_with_tangents(output_ref, seed, seed_tangent, &tangents)?;

        let mut gradients = Gradients::new();
        let mut hvp = Gradients::new();
        for i in 0..self.nodes.len() {
            if !self.nodes[i].is_leaf {
                continue;
            }
            if let Some(value) = cotangents[i][0].take() {
                gradients.push_entry(NodeId::new(i), value);
            }
            if let Some(value) = cot_tangents[i][0].take() {
                hvp.push_entry(NodeId::new(i), value);
            }
        }
        Ok(HvpResult { gradients, hvp })
    }
}
