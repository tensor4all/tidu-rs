use std::sync::Arc;

use crate::checkpoint::{current_ad_policy, storage_decision, CheckpointClass, StorageDecision};
use crate::reverse_graph::{ReverseEdge, ReverseNode, StoredNodeLinearization};
use crate::{AdResult, AutodiffError, Differentiable, Value};

/// AD-role metadata for one input or output slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotSchema {
    pub differentiable: bool,
    pub auxiliary: bool,
}

impl SlotSchema {
    fn validate(self, kind: &str, index: usize) -> AdResult<Self> {
        if self.auxiliary && self.differentiable {
            return Err(AutodiffError::InvalidArgument(format!(
                "{kind} schema slot {index} cannot be auxiliary and differentiable at the same time"
            )));
        }
        Ok(self)
    }
}

/// Runtime schema for op inputs or outputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    pub slots: Vec<SlotSchema>,
}

impl Schema {
    pub(crate) fn validate_len(&self, kind: &str, expected_len: usize) -> AdResult<()> {
        if self.slots.len() != expected_len {
            return Err(AutodiffError::InvalidArgument(format!(
                "{kind} schema returned {} slots for {expected_len} values",
                self.slots.len()
            )));
        }
        for (index, slot) in self.slots.iter().copied().enumerate() {
            slot.validate(kind, index)?;
        }
        Ok(())
    }
}

pub trait LinearizableOp<V: Differentiable + Send + Sync + 'static>: Send + Sync + 'static {
    type Linearized: LinearizedOp<V> + Send + Sync + 'static;

    fn primal(&self, inputs: &[&V]) -> AdResult<Vec<V>>;
    fn input_schema(&self, inputs: &[&V]) -> AdResult<Schema>;
    fn output_schema(&self, inputs: &[&V], outputs: &[V]) -> AdResult<Schema>;
    fn linearize(&self, inputs: &[&V], outputs: &[V]) -> AdResult<Self::Linearized>;

    #[doc(hidden)]
    fn checkpoint_class(&self) -> CheckpointClass {
        CheckpointClass::CheapReplay
    }

    fn apply(&self, inputs: &[&Value<V>]) -> AdResult<Vec<Value<V>>>
    where
        Self: Sized + Clone,
    {
        let primals: Vec<&V> = inputs.iter().map(|input| input.primal()).collect();
        let input_schema = self.input_schema(&primals)?;
        input_schema.validate_len("input", inputs.len())?;

        let outputs = self.primal(&primals)?;
        let output_schema = self.output_schema(&primals, &outputs)?;
        output_schema.validate_len("output", outputs.len())?;

        let input_grad_mask: Vec<bool> = inputs
            .iter()
            .zip(&input_schema.slots)
            .map(|(input, slot)| input.requires_grad() && slot.differentiable)
            .collect();

        let differentiable_output_slots: Vec<usize> = output_schema
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| slot.differentiable.then_some(index))
            .collect();

        if !input_grad_mask.iter().any(|needed| *needed) || differentiable_output_slots.is_empty() {
            return Ok(outputs.into_iter().map(Value::new).collect());
        }

        let input_nodes = inputs
            .iter()
            .zip(&input_schema.slots)
            .map(|(input, slot)| {
                if slot.differentiable {
                    Ok(input.reverse_input())
                } else {
                    Ok(None)
                }
            })
            .collect::<AdResult<Vec<_>>>()?;

        let linearized = self.linearize(&primals, &outputs)?;
        let stored_linearization =
            match storage_decision(current_ad_policy(), self.checkpoint_class()) {
                StorageDecision::Retain | StorageDecision::Replay => {
                    StoredNodeLinearization::retained(linearized)
                }
            };

        let output_count = output_schema.slots.len();
        let node = Arc::new(ReverseNode::new(
            input_nodes,
            output_count,
            input_grad_mask,
            stored_linearization,
        ));

        Ok(outputs
            .into_iter()
            .enumerate()
            .map(|(index, output)| {
                if output_schema.slots[index].differentiable {
                    Value::from_reverse_edge(
                        output,
                        ReverseEdge {
                            node: node.clone(),
                            output_slot: index,
                        },
                    )
                } else {
                    Value::new(output)
                }
            })
            .collect())
    }

    fn apply_one(&self, inputs: &[&Value<V>]) -> AdResult<Value<V>>
    where
        Self: Sized + Clone,
    {
        let mut outputs = self.apply(inputs)?;
        if outputs.len() != 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "LinearizableOp::apply_one expected exactly 1 output, got {}",
                outputs.len()
            )));
        }
        Ok(outputs.remove(0))
    }
}

pub trait LinearizedOp<V: Differentiable + Send + Sync + 'static>: Send + Sync + 'static {
    fn jvp(&self, input_tangents: &[Option<V::Tangent>]) -> AdResult<Vec<Option<V::Tangent>>>;

    fn vjp(
        &self,
        output_cotangents: &[Option<V::Tangent>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<V::Tangent>>>;
}
