use std::collections::HashMap;

use crate::{
    ADKey, ADRuleError, ADRuleKind, ADRuleResult, LinearizedValueKind, Primitive, PrimitiveBuilder,
    PrimitiveTransposeInput, PrimitiveValue,
};
use computegraph::graph::GraphBuilder;
use computegraph::{LocalValueId, OperationRole, ValueKey, ValueRef};

use crate::LinearizedGraph;

/// Transpose a linearized graph, reversing linear flow.
///
/// Fan-out accumulation is emitted explicitly with [`crate::Primitive::add`];
/// no duplication primitive is assumed by the graph transform.
///
/// # Examples
///
/// ```ignore
/// let mut ctx = ();
/// let transposed = tidu::linear_transpose(&linear, &mut ctx)?;
/// assert_eq!(transposed.tangent_outputs().len(), linear.tangent_inputs().len());
/// # Ok::<(), tidu::ADRuleError>(())
/// ```
pub fn linear_transpose<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
    ctx: &mut Op::ADContext,
) -> ADRuleResult<LinearizedGraph<Op>>
where
    Op::InputKey: ADKey,
{
    let mut builder = GraphBuilder::<Op>::new();
    let mut cotangent_env: HashMap<ValueKey<Op>, LocalValueId> = HashMap::new();
    let mut cotangent_seed_inputs = Vec::new();
    let graph = linear.as_graph();
    let mut value_kinds = Vec::new();

    for (index, maybe_tangent_output) in linear.tangent_outputs().iter().enumerate() {
        let tangent_output_id = match maybe_tangent_output {
            Some(tangent_output_id) => tangent_output_id,
            None => continue,
        };

        let source_key = graph.values()[*tangent_output_id].key.clone();
        let seed_key = cotangent_seed_key(linear, index)?;
        let seed_id = builder.add_input(seed_key.clone());
        debug_assert_eq!(seed_id, value_kinds.len());
        value_kinds.push(LinearizedValueKind::Linear { primal: None });
        cotangent_env.insert(source_key, seed_id);
        cotangent_seed_inputs.push((seed_key, seed_id));
    }

    for op_node in graph.operations().iter().rev() {
        let cotangent_out: Vec<Option<LocalValueId>> = op_node
            .outputs
            .iter()
            .map(|output_id| cotangent_env.get(&graph.values()[*output_id].key).copied())
            .collect();
        if cotangent_out.iter().all(Option::is_none) {
            continue;
        }

        let rule_inputs: Vec<PrimitiveTransposeInput<Op>> = op_node
            .inputs
            .iter()
            .map(|input| transpose_input_for(linear, input))
            .collect();

        let mut primitive_builder = TrackingPrimitiveBuilder::new(&mut builder, &mut value_kinds);
        let cotangent_in = op_node.operation.transpose_rule(
            &mut primitive_builder,
            &cotangent_out,
            &rule_inputs,
            &op_node.role,
            ctx,
        )?;
        if cotangent_in.len() != rule_inputs.len() {
            return Err(ADRuleError::invalid_input(
                format!("{:?}", op_node.operation),
                ADRuleKind::Transpose,
                format!(
                    "rule returned {} cotangents for {} inputs",
                    cotangent_in.len(),
                    rule_inputs.len()
                ),
            ));
        }

        for (input, maybe_cotangent) in rule_inputs.iter().zip(cotangent_in) {
            let cotangent_id = match maybe_cotangent {
                Some(cotangent_id) => cotangent_id,
                None => continue,
            };
            let input_key = input.key().clone();

            match cotangent_env.get(&input_key).copied() {
                Some(existing_id) => {
                    let mut primitive_builder =
                        TrackingPrimitiveBuilder::new(&mut builder, &mut value_kinds);
                    let sum = primitive_builder.add_primitive(
                        Op::add(),
                        vec![
                            PrimitiveValue::Local(existing_id),
                            PrimitiveValue::Local(cotangent_id),
                        ],
                        OperationRole::Linearized {
                            active_mask: vec![true, true],
                        },
                    );
                    cotangent_env.insert(input_key, sum[0]);
                }
                None => {
                    cotangent_env.insert(input_key, cotangent_id);
                }
            }
        }
    }

    let tangent_outputs: Vec<Option<LocalValueId>> = linear
        .tangent_inputs()
        .iter()
        .map(|(_, tangent_input_id)| {
            let tangent_input_key = &graph.values()[*tangent_input_id].key;
            cotangent_env.get(tangent_input_key).copied()
        })
        .collect();
    let active_outputs: Vec<LocalValueId> = tangent_outputs.iter().filter_map(|id| *id).collect();
    if !active_outputs.is_empty() {
        builder.set_outputs(active_outputs);
    }

    Ok(LinearizedGraph::from_parts(
        builder.build(),
        cotangent_seed_inputs,
        tangent_outputs,
        value_kinds,
    ))
}

/// Execute the transpose of a linearized graph using a caller-provided builder.
pub fn linear_transpose_with_builder<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
    builder: &mut impl PrimitiveBuilder<Op>,
    cotangent_seeds: &[Option<LocalValueId>],
    ctx: &mut Op::ADContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>>
where
    Op::InputKey: ADKey,
{
    let mut cotangent_env: HashMap<ValueKey<Op>, LocalValueId> = HashMap::new();
    let graph = linear.as_graph();

    for (index, maybe_tangent_output) in linear.tangent_outputs().iter().enumerate() {
        if let (Some(output_id), Some(Some(seed_id))) =
            (maybe_tangent_output, cotangent_seeds.get(index))
        {
            let key = graph.values()[*output_id].key.clone();
            cotangent_env.insert(key, *seed_id);
        }
    }

    for op_node in graph.operations().iter().rev() {
        let cotangent_out: Vec<Option<LocalValueId>> = op_node
            .outputs
            .iter()
            .map(|output_id| cotangent_env.get(&graph.values()[*output_id].key).copied())
            .collect();
        if cotangent_out.iter().all(Option::is_none) {
            continue;
        }

        let rule_inputs: Vec<PrimitiveTransposeInput<Op>> = op_node
            .inputs
            .iter()
            .map(|input| transpose_input_for(linear, input))
            .collect();

        let cotangent_in = op_node.operation.transpose_rule(
            builder,
            &cotangent_out,
            &rule_inputs,
            &op_node.role,
            ctx,
        )?;
        if cotangent_in.len() != rule_inputs.len() {
            return Err(ADRuleError::invalid_input(
                format!("{:?}", op_node.operation),
                ADRuleKind::Transpose,
                format!(
                    "rule returned {} cotangents for {} inputs",
                    cotangent_in.len(),
                    rule_inputs.len()
                ),
            ));
        }

        for (input, maybe_cotangent) in rule_inputs.iter().zip(cotangent_in) {
            let cotangent_id = match maybe_cotangent {
                Some(cotangent_id) => cotangent_id,
                None => continue,
            };
            let input_key = input.key().clone();

            match cotangent_env.get(&input_key).copied() {
                Some(existing_id) => {
                    let sum = builder.add_primitive(
                        Op::add(),
                        vec![
                            PrimitiveValue::Local(existing_id),
                            PrimitiveValue::Local(cotangent_id),
                        ],
                        OperationRole::Linearized {
                            active_mask: vec![true, true],
                        },
                    );
                    cotangent_env.insert(input_key, sum[0]);
                }
                None => {
                    cotangent_env.insert(input_key, cotangent_id);
                }
            }
        }
    }

    Ok(linear
        .tangent_inputs()
        .iter()
        .map(|(_, tangent_input_id)| {
            let tangent_input_key = &graph.values()[*tangent_input_id].key;
            cotangent_env.get(tangent_input_key).copied()
        })
        .collect())
}

fn transpose_input_for<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
    input: &ValueRef<Op>,
) -> PrimitiveTransposeInput<Op>
where
    Op::InputKey: ADKey,
{
    let graph = linear.as_graph();
    match input {
        ValueRef::Local(local_id) => {
            let key = graph.values()[*local_id].key.clone();
            match linear.value_kind(*local_id) {
                LinearizedValueKind::Residual => PrimitiveTransposeInput::Residual(key),
                LinearizedValueKind::Linear { primal } => PrimitiveTransposeInput::Linear {
                    key,
                    primal: primal.clone(),
                },
            }
        }
        ValueRef::External(key) => PrimitiveTransposeInput::Residual(key.clone()),
    }
}

struct TrackingPrimitiveBuilder<'a, Op: computegraph::GraphOperation> {
    inner: &'a mut GraphBuilder<Op>,
    value_kinds: &'a mut Vec<LinearizedValueKind<Op>>,
}

impl<'a, Op: computegraph::GraphOperation> TrackingPrimitiveBuilder<'a, Op> {
    fn new(
        inner: &'a mut GraphBuilder<Op>,
        value_kinds: &'a mut Vec<LinearizedValueKind<Op>>,
    ) -> Self {
        Self { inner, value_kinds }
    }

    fn output_kind(
        &self,
        inputs: &[PrimitiveValue<Op>],
        role: &OperationRole,
    ) -> LinearizedValueKind<Op> {
        let OperationRole::Linearized { active_mask } = role else {
            return LinearizedValueKind::Residual;
        };

        if inputs.iter().enumerate().any(|(index, input)| {
            active_mask.get(index).copied().unwrap_or(false) && self.is_linear_input(input)
        }) {
            LinearizedValueKind::Linear { primal: None }
        } else {
            LinearizedValueKind::Residual
        }
    }

    fn is_linear_input(&self, input: &PrimitiveValue<Op>) -> bool {
        match input {
            PrimitiveValue::Local(id) => {
                matches!(self.value_kinds[*id], LinearizedValueKind::Linear { .. })
            }
            PrimitiveValue::External(_) => true,
        }
    }
}

impl<Op: computegraph::GraphOperation> PrimitiveBuilder<Op> for TrackingPrimitiveBuilder<'_, Op> {
    fn add_primitive(
        &mut self,
        op: Op,
        inputs: Vec<PrimitiveValue<Op>>,
        role: OperationRole,
    ) -> Vec<LocalValueId> {
        let output_kind = self.output_kind(&inputs, &role);
        let inputs = inputs.into_iter().map(ValueRef::from).collect();
        let outputs = self.inner.add_operation(op, inputs, role);
        for output_id in &outputs {
            debug_assert_eq!(*output_id, self.value_kinds.len());
            self.value_kinds.push(output_kind.clone());
        }
        outputs
    }
}

fn cotangent_seed_key<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
    index: usize,
) -> ADRuleResult<Op::InputKey>
where
    Op::InputKey: ADKey,
{
    if linear.tangent_inputs().is_empty() {
        return Err(ADRuleError::invalid_input(
            "tidu::linear_transpose",
            ADRuleKind::Transpose,
            "active tangent outputs require at least one tangent input to derive seed keys",
        ));
    }

    let base_slot = index.min(linear.tangent_inputs().len() - 1);
    let base_key = &linear.tangent_inputs()[base_slot].0;
    Ok(base_key.tangent_of(u64::MAX - index as u64))
}
