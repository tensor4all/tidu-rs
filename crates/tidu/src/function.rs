use std::marker::PhantomData;
use std::sync::Arc;

use crate::reverse_graph::{ReverseEdge, ReverseNode, ReverseRule};
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
    fn validate_len(&self, kind: &str, expected_len: usize) -> AdResult<()> {
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

/// High-level custom autograd op trait.
pub trait Op<V: Differentiable + Send + Sync + 'static>: Send + Sync + 'static {
    type SavedBackward: Send + Sync + 'static;
    type SavedJvp: Send + Sync + 'static;

    /// Compute the primal outputs of the operation.
    fn primal(&self, inputs: &[&V]) -> AdResult<Vec<V>>;

    /// Declare which inputs are differentiable for this call.
    fn input_schema(&self, inputs: &[&V]) -> AdResult<Schema>;

    /// Declare which outputs are differentiable after seeing the primal outputs.
    fn output_schema(&self, inputs: &[&V], outputs: &[V]) -> AdResult<Schema>;

    /// Capture reverse-mode state only when gradients are required.
    fn save_for_backward(&self, inputs: &[&V], outputs: &[V]) -> AdResult<Self::SavedBackward>;

    /// Capture forward-mode state only when JVP is requested.
    fn save_for_jvp(&self, inputs: &[&V], outputs: &[V]) -> AdResult<Self::SavedJvp>;

    /// Whether undefined cotangents should be materialized to zero.
    fn materialize_grads(&self) -> bool {
        false
    }

    /// Compute the reverse-mode pullback in input order.
    fn backward(
        &self,
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<V::Tangent>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<V::Tangent>>>;

    /// Compute the forward-mode pushforward in output order.
    fn jvp(
        &self,
        saved: &Self::SavedJvp,
        tangents: &[Option<V::Tangent>],
    ) -> AdResult<Vec<Option<V::Tangent>>>;

    /// Apply the operation to tracked inputs.
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

        let saved = self.save_for_backward(&primals, &outputs)?;
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

        let output_count = output_schema.slots.len();
        let node = Arc::new(ReverseNode::new(
            input_nodes,
            output_count,
            Box::new(OpRule::<Self, V> {
                op: self.clone(),
                saved,
                input_grad_mask,
                materialize_grads: self.materialize_grads(),
                _marker: PhantomData,
            }),
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

    /// Apply the operation and require exactly one output.
    fn apply_one(&self, inputs: &[&Value<V>]) -> AdResult<Value<V>>
    where
        Self: Sized + Clone,
    {
        let mut outputs = self.apply(inputs)?;
        if outputs.len() != 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Op::apply_one expected exactly 1 output, got {}",
                outputs.len()
            )));
        }
        Ok(outputs.remove(0))
    }
}

struct OpRule<O, V>
where
    O: Op<V>,
    V: Differentiable + Send + Sync + 'static,
{
    op: O,
    saved: O::SavedBackward,
    input_grad_mask: Vec<bool>,
    materialize_grads: bool,
    _marker: PhantomData<V>,
}

impl<O, V> ReverseRule<V> for OpRule<O, V>
where
    O: Op<V>,
    V: Differentiable + Send + Sync + 'static,
{
    fn pullback(&self, grad_outputs: &[Option<V::Tangent>]) -> AdResult<Vec<Option<V::Tangent>>> {
        let _ = self.materialize_grads;
        self.op
            .backward(&self.saved, grad_outputs, &self.input_grad_mask)
    }
}
