use chainrules_core::{NodeId, ReverseRule};

use crate::{AdResult, AutodiffError, Differentiable};

/// Identifies one output slot of a graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OutputRef {
    pub(crate) node_id: NodeId,
    pub(crate) output_slot: usize,
}

impl OutputRef {
    pub(crate) fn new(node_id: NodeId, output_slot: usize) -> Self {
        Self {
            node_id,
            output_slot,
        }
    }
}

/// Internal engine-facing rule abstraction that can represent multi-output ops.
pub(crate) trait EngineRule<V: Differentiable>: Send + Sync {
    fn output_count(&self) -> usize;

    fn pullback(&self, cotangents: &[Option<V::Tangent>])
        -> AdResult<Vec<(OutputRef, V::Tangent)>>;

    fn forward_tangents<'t>(
        &self,
        input_tangents: &dyn Fn(OutputRef) -> Option<&'t V::Tangent>,
    ) -> AdResult<Vec<Option<V::Tangent>>>
    where
        V::Tangent: 't + Differentiable<Tangent = V::Tangent>,
    {
        let _ = input_tangents;
        Err(AutodiffError::HvpNotSupported)
    }

    fn pullback_with_tangents<'t>(
        &self,
        cotangents: &[Option<V::Tangent>],
        cotangent_tangents: &[Option<V::Tangent>],
        input_tangents: &dyn Fn(OutputRef) -> Option<&'t V::Tangent>,
    ) -> AdResult<Vec<(OutputRef, V::Tangent, V::Tangent)>>
    where
        V::Tangent: 't + Differentiable<Tangent = V::Tangent>,
    {
        let _ = (cotangents, cotangent_tangents, input_tangents);
        Err(AutodiffError::HvpNotSupported)
    }
}

pub(crate) struct ReverseRuleAdapter<V: Differentiable> {
    rule: Box<dyn ReverseRule<V>>,
}

impl<V: Differentiable> ReverseRuleAdapter<V> {
    pub(crate) fn new(rule: Box<dyn ReverseRule<V>>) -> Self {
        Self { rule }
    }
}

impl<V: Differentiable> EngineRule<V> for ReverseRuleAdapter<V> {
    fn output_count(&self) -> usize {
        1
    }

    fn pullback(
        &self,
        cotangents: &[Option<V::Tangent>],
    ) -> AdResult<Vec<(OutputRef, V::Tangent)>> {
        if cotangents.len() != 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "single-output rule received {} cotangent slots",
                cotangents.len()
            )));
        }
        let Some(cotangent) = cotangents[0].as_ref() else {
            return Ok(Vec::new());
        };
        Ok(self
            .rule
            .pullback(cotangent)?
            .into_iter()
            .map(|(node_id, tangent)| (OutputRef::new(node_id, 0), tangent))
            .collect())
    }

    fn forward_tangents<'t>(
        &self,
        input_tangents: &dyn Fn(OutputRef) -> Option<&'t V::Tangent>,
    ) -> AdResult<Vec<Option<V::Tangent>>>
    where
        V::Tangent: 't + Differentiable<Tangent = V::Tangent>,
    {
        Ok(vec![self.rule.forward_tangents(&|node_id| {
            input_tangents(OutputRef::new(node_id, 0))
        })?])
    }

    fn pullback_with_tangents<'t>(
        &self,
        cotangents: &[Option<V::Tangent>],
        cotangent_tangents: &[Option<V::Tangent>],
        input_tangents: &dyn Fn(OutputRef) -> Option<&'t V::Tangent>,
    ) -> AdResult<Vec<(OutputRef, V::Tangent, V::Tangent)>>
    where
        V::Tangent: 't + Differentiable<Tangent = V::Tangent>,
    {
        if cotangents.len() != 1 || cotangent_tangents.len() != 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "single-output rule received {} cotangent slots and {} cotangent tangent slots",
                cotangents.len(),
                cotangent_tangents.len()
            )));
        }
        let Some(cotangent) = cotangents[0].as_ref() else {
            return Ok(Vec::new());
        };
        let cotangent_tangent = cotangent_tangents[0]
            .as_ref()
            .cloned()
            .unwrap_or_else(|| cotangent.zero_tangent());
        Ok(self
            .rule
            .pullback_with_tangents(cotangent, &cotangent_tangent, &|node_id| {
                input_tangents(OutputRef::new(node_id, 0))
            })?
            .into_iter()
            .map(|(node_id, tangent, tangent_tangent)| {
                (OutputRef::new(node_id, 0), tangent, tangent_tangent)
            })
            .collect())
    }
}
