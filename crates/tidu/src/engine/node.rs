use std::sync::Arc;

use crate::{AdResult, AutodiffError, Differentiable, NodeId, ReverseRule};

use super::replay::CheckpointRecipe;

pub(crate) struct Node<V: Differentiable> {
    inputs: Vec<NodeId>,
    exec: NodeExec<V>,
    primal: PrimalStorage<V>,
}

pub(crate) enum NodeExec<V: Differentiable> {
    Leaf,
    Materialized(Box<dyn ReverseRule<V>>),
    Replayable(Box<dyn CheckpointRecipe<V>>),
    Placeholder,
}

pub(crate) enum PrimalStorage<V> {
    Retained(Arc<V>),
    Evicted,
}

impl<V: Differentiable> Node<V> {
    pub(crate) fn leaf(primal: Arc<V>) -> Self {
        Self {
            inputs: Vec::new(),
            exec: NodeExec::Leaf,
            primal: PrimalStorage::Retained(primal),
        }
    }

    pub(crate) fn materialized(primal: Arc<V>, rule: Box<dyn ReverseRule<V>>) -> Self {
        let inputs = rule.inputs();
        Self {
            inputs,
            exec: NodeExec::Materialized(rule),
            primal: PrimalStorage::Retained(primal),
        }
    }

    pub(crate) fn replayable(recipe: Box<dyn CheckpointRecipe<V>>) -> Self {
        let inputs = recipe.inputs();
        Self {
            inputs,
            exec: NodeExec::Replayable(recipe),
            primal: PrimalStorage::Evicted,
        }
    }

    pub(crate) fn placeholder(primal: Arc<V>) -> Self {
        Self {
            inputs: Vec::new(),
            exec: NodeExec::Placeholder,
            primal: PrimalStorage::Retained(primal),
        }
    }

    pub(crate) fn is_leaf(&self) -> bool {
        matches!(self.exec, NodeExec::Leaf)
    }

    pub(crate) fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    pub(crate) fn exec(&self) -> &NodeExec<V> {
        &self.exec
    }

    pub(crate) fn retained_primal(&self) -> Option<&V> {
        match &self.primal {
            PrimalStorage::Retained(value) => Some(value.as_ref()),
            PrimalStorage::Evicted => None,
        }
    }

    pub(crate) fn retained_primal_shared(&self) -> Option<Arc<V>> {
        match &self.primal {
            PrimalStorage::Retained(value) => Some(Arc::clone(value)),
            PrimalStorage::Evicted => None,
        }
    }

    pub(crate) fn replay_recipe(&self) -> Option<&dyn CheckpointRecipe<V>> {
        match &self.exec {
            NodeExec::Replayable(recipe) => Some(recipe.as_ref()),
            _ => None,
        }
    }

    pub(crate) fn attach_rule(&mut self, rule: Box<dyn ReverseRule<V>>) -> AdResult<()> {
        match self.exec {
            NodeExec::Leaf => Err(AutodiffError::InvalidArgument(
                "attach_rule() requires a placeholder node, got a leaf".into(),
            )),
            NodeExec::Replayable(_) => Err(AutodiffError::InvalidArgument(
                "attach_rule() cannot replace a checkpointed node".into(),
            )),
            NodeExec::Placeholder | NodeExec::Materialized(_) => {
                self.inputs = rule.inputs();
                self.exec = NodeExec::Materialized(rule);
                Ok(())
            }
        }
    }
}
