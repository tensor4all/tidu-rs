use std::collections::HashMap;

use crate::{
    ADKey, ADRuleError, ADRuleKind, ADRuleResult, Primitive, PrimitiveBuilder,
    PrimitiveTransposeInput, PrimitiveValue,
};
use computegraph::{LocalValueId, OperationRole, ValueKey, ValueRef};

use crate::split_builder::SplitGraphBuilder;
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
    let mut builder = SplitGraphBuilder::<Op>::new();
    let mut cotangent_env: HashMap<ValueKey<Op>, LocalValueId> = HashMap::new();
    let mut cotangent_seed_inputs = Vec::new();
    let graph = linear.as_graph();

    for (index, maybe_tangent_output) in linear.tangent_outputs().iter().enumerate() {
        let tangent_output_id = match maybe_tangent_output {
            Some(tangent_output_id) => tangent_output_id,
            None => continue,
        };

        let source_key = graph.values()[*tangent_output_id].key.clone();
        let seed_key = cotangent_seed_key(linear, index)?;
        let (seed_unified_id, seed_linear_id) = builder.add_linear_input(seed_key.clone(), None);
        cotangent_env.insert(source_key, seed_unified_id);
        cotangent_seed_inputs.push((seed_key, seed_linear_id));
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
            &mut builder,
            &cotangent_out,
            &rule_inputs,
            &op_node.role,
            ctx,
        )?;
        if let Some(err) = builder.take_error() {
            return Err(err);
        }
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

    let tangent_outputs: Vec<Option<LocalValueId>> = linear
        .tangent_inputs()
        .iter()
        .map(|(_, tangent_input_id)| {
            let tangent_input_key = &graph.values()[*tangent_input_id].key;
            cotangent_env
                .get(tangent_input_key)
                .and_then(|unified_id| builder.linear_local_id(*unified_id))
        })
        .collect();
    let active_outputs: Vec<LocalValueId> = tangent_outputs.iter().filter_map(|id| *id).collect();
    let (linear_graph, residual_graph, linear_primals) = builder.finish(active_outputs);

    Ok(LinearizedGraph::from_parts(
        linear_graph,
        residual_graph,
        cotangent_seed_inputs,
        tangent_outputs,
        linear_primals,
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
            PrimitiveTransposeInput::Linear {
                key,
                primal: linear.linear_primal(*local_id).cloned(),
            }
        }
        ValueRef::External(key) => PrimitiveTransposeInput::Residual(key.clone()),
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
