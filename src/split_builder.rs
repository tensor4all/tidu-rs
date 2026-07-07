use std::sync::Arc;

use crate::{ADRuleError, ADRuleKind, PrimitiveBuilder, PrimitiveValue};
use computegraph::graph::{Graph, GraphBuilder};
use computegraph::{GraphOperation, LocalValueId, OperationRole, ValueKey, ValueRef};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplitValue {
    Residual(LocalValueId),
    Linear(LocalValueId),
}

pub(crate) struct SplitGraphBuilder<Op: GraphOperation> {
    residual: GraphBuilder<Op>,
    linear: GraphBuilder<Op>,
    unified_values: Vec<SplitValue>,
    linear_primals: Vec<Option<ValueKey<Op>>>,
    first_error: Option<ADRuleError>,
}

impl<Op: GraphOperation> SplitGraphBuilder<Op> {
    pub(crate) fn new() -> Self {
        Self {
            residual: GraphBuilder::new(),
            linear: GraphBuilder::new(),
            unified_values: Vec::new(),
            linear_primals: Vec::new(),
            first_error: None,
        }
    }

    pub(crate) fn add_linear_input(
        &mut self,
        key: Op::InputKey,
        primal: Option<ValueKey<Op>>,
    ) -> (LocalValueId, LocalValueId) {
        let local_id = self.linear.add_input(key);
        self.push_linear_value(local_id, primal);
        let unified_id = self.unified_values.len() - 1;
        (unified_id, local_id)
    }

    pub(crate) fn linear_local_id(&self, unified_id: LocalValueId) -> Option<LocalValueId> {
        match self.unified_values.get(unified_id).copied() {
            Some(SplitValue::Linear(local_id)) => Some(local_id),
            _ => None,
        }
    }

    pub(crate) fn set_linear_primal(
        &mut self,
        unified_id: LocalValueId,
        primal: ValueKey<Op>,
    ) -> Result<(), ADRuleError> {
        let Some(SplitValue::Linear(local_id)) = self.unified_values.get(unified_id).copied()
        else {
            return Err(ADRuleError::invalid_input(
                "tidu::linearize",
                ADRuleKind::Jvp,
                "JVP rule returned a residual value as an active tangent output",
            ));
        };
        if let Some(slot) = self.linear_primals.get_mut(local_id) {
            *slot = Some(primal);
        }
        Ok(())
    }

    pub(crate) fn take_error(&mut self) -> Option<ADRuleError> {
        self.first_error.take()
    }

    pub(crate) fn finish(
        mut self,
        linear_outputs: Vec<LocalValueId>,
    ) -> (Graph<Op>, Arc<Graph<Op>>, Vec<Option<ValueKey<Op>>>) {
        if !linear_outputs.is_empty() {
            self.linear.set_outputs(linear_outputs);
        }
        let residual = Arc::new(self.residual.build());
        self.linear.add_parent(Arc::clone(&residual));
        (self.linear.build(), residual, self.linear_primals)
    }

    fn push_residual_value(&mut self, local_id: LocalValueId) {
        self.unified_values.push(SplitValue::Residual(local_id));
    }

    fn push_linear_value(&mut self, local_id: LocalValueId, primal: Option<ValueKey<Op>>) {
        debug_assert_eq!(local_id, self.linear_primals.len());
        self.unified_values.push(SplitValue::Linear(local_id));
        self.linear_primals.push(primal);
    }

    fn is_linear_value(&self, input: &PrimitiveValue<Op>) -> bool {
        match input {
            PrimitiveValue::Local(local_id) => {
                matches!(
                    self.unified_values.get(*local_id),
                    Some(SplitValue::Linear(_))
                )
            }
            PrimitiveValue::External(_) => false,
        }
    }

    fn record_inactive_tangent_error(&mut self, op: &Op, input_index: usize) {
        if self.first_error.is_none() {
            self.first_error = Some(ADRuleError::invalid_input(
                format!("{op:?}"),
                ADRuleKind::Jvp,
                format!(
                    "linearized operation used tangent-flow input {input_index} in an inactive position"
                ),
            ));
        }
    }

    fn validate_linearity(&mut self, op: &Op, inputs: &[PrimitiveValue<Op>], role: &OperationRole) {
        let OperationRole::Linearized { active_mask } = role else {
            return;
        };
        for (input_index, input) in inputs.iter().enumerate() {
            if self.is_linear_value(input)
                && !active_mask.get(input_index).copied().unwrap_or(false)
            {
                self.record_inactive_tangent_error(op, input_index);
            }
        }
    }

    fn input_for_residual_graph(&self, input: PrimitiveValue<Op>) -> ValueRef<Op> {
        match input {
            PrimitiveValue::Local(local_id) => match self.unified_values[local_id] {
                SplitValue::Residual(real_id) => ValueRef::Local(real_id),
                SplitValue::Linear(real_id) => {
                    ValueRef::External(self.linear.global_key(real_id).clone())
                }
            },
            PrimitiveValue::External(key) => ValueRef::External(key),
        }
    }

    fn input_for_linear_graph(&self, input: PrimitiveValue<Op>) -> ValueRef<Op> {
        match input {
            PrimitiveValue::Local(local_id) => match self.unified_values[local_id] {
                SplitValue::Residual(real_id) => {
                    ValueRef::External(self.residual.global_key(real_id).clone())
                }
                SplitValue::Linear(real_id) => ValueRef::Local(real_id),
            },
            PrimitiveValue::External(key) => ValueRef::External(key),
        }
    }
}

impl<Op: GraphOperation> PrimitiveBuilder<Op> for SplitGraphBuilder<Op> {
    fn add_primitive(
        &mut self,
        op: Op,
        inputs: Vec<PrimitiveValue<Op>>,
        role: OperationRole,
    ) -> Vec<LocalValueId> {
        self.validate_linearity(&op, &inputs, &role);
        let target_linear = inputs.iter().any(|input| self.is_linear_value(input));
        if target_linear {
            let graph_inputs = inputs
                .into_iter()
                .map(|input| self.input_for_linear_graph(input))
                .collect();
            let outputs = self.linear.add_operation(op, graph_inputs, role);
            for &output_id in &outputs {
                self.push_linear_value(output_id, None);
            }
            return (self.unified_values.len() - outputs.len()..self.unified_values.len())
                .collect();
        }

        let graph_inputs = inputs
            .into_iter()
            .map(|input| self.input_for_residual_graph(input))
            .collect();
        let outputs = self.residual.add_operation(op, graph_inputs, role);
        for &output_id in &outputs {
            self.push_residual_value(output_id);
        }
        (self.unified_values.len() - outputs.len()..self.unified_values.len()).collect()
    }
}
