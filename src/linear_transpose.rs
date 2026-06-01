use std::collections::HashMap;

use crate::rules::GraphPrimitiveBuilder;
use crate::{ADKey, ADRuleResult, Primitive, PrimitiveBuilder, PrimitiveValue};
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
/// let transposed = tidu::linear_transpose(&linear, &mut ctx);
/// assert_eq!(transposed.tangent_outputs().len(), linear.tangent_inputs().len());
/// ```
pub fn linear_transpose<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
    ctx: &mut Op::ADContext,
) -> LinearizedGraph<Op>
where
    Op::InputKey: ADKey,
{
    match try_linear_transpose(linear, ctx) {
        Ok(transposed) => transposed,
        Err(err) => panic!("{err}"),
    }
}

/// Fallible form of [`linear_transpose`].
///
/// This returns [`crate::ADRuleError`] when a primitive cannot emit a
/// transpose rule.
pub fn try_linear_transpose<Op: Primitive>(
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

    for (index, maybe_tangent_output) in linear.tangent_outputs().iter().enumerate() {
        let Some(tangent_output_id) = maybe_tangent_output else {
            continue;
        };

        let source_key = graph.values()[*tangent_output_id].key.clone();
        let seed_key = cotangent_seed_key(linear, index);
        let seed_id = builder.add_input(seed_key.clone());
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

        let rule_inputs: Vec<PrimitiveValue<Op>> = op_node
            .inputs
            .iter()
            .map(|input| match input {
                ValueRef::Local(local_id) => {
                    PrimitiveValue::External(graph.values()[*local_id].key.clone())
                }
                ValueRef::External(key) => PrimitiveValue::External(key.clone()),
            })
            .collect();

        let mut primitive_builder = GraphPrimitiveBuilder::new(&mut builder);
        let cotangent_in = op_node.operation.try_linear_transpose_rule(
            &mut primitive_builder,
            &cotangent_out,
            &rule_inputs,
            &op_node.role,
            ctx,
        )?;
        assert_eq!(
            cotangent_in.len(),
            rule_inputs.len(),
            "transpose_rule for {:?} returned {} cotangents for {} inputs",
            op_node.operation,
            cotangent_in.len(),
            rule_inputs.len()
        );

        for (input, maybe_cotangent) in rule_inputs.iter().zip(cotangent_in) {
            let Some(cotangent_id) = maybe_cotangent else {
                continue;
            };
            let input_key = match input {
                PrimitiveValue::Local(_) => {
                    unreachable!("rule inputs are normalized to external refs")
                }
                PrimitiveValue::External(key) => key.clone(),
            };

            match cotangent_env.get(&input_key).copied() {
                Some(existing_id) => {
                    let mut primitive_builder = GraphPrimitiveBuilder::new(&mut builder);
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
    ))
}

/// Execute the transpose of a linearized graph using a caller-provided builder.
pub fn try_linear_transpose_with_builder<Op: Primitive>(
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

        let rule_inputs: Vec<PrimitiveValue<Op>> = op_node
            .inputs
            .iter()
            .map(|input| match input {
                ValueRef::Local(local_id) => {
                    PrimitiveValue::External(graph.values()[*local_id].key.clone())
                }
                ValueRef::External(key) => PrimitiveValue::External(key.clone()),
            })
            .collect();

        let cotangent_in = op_node.operation.try_linear_transpose_rule(
            builder,
            &cotangent_out,
            &rule_inputs,
            &op_node.role,
            ctx,
        )?;
        assert_eq!(
            cotangent_in.len(),
            rule_inputs.len(),
            "transpose_rule for {:?} returned {} cotangents for {} inputs",
            op_node.operation,
            cotangent_in.len(),
            rule_inputs.len()
        );

        for (input, maybe_cotangent) in rule_inputs.iter().zip(cotangent_in) {
            let Some(cotangent_id) = maybe_cotangent else {
                continue;
            };
            let input_key = match input {
                PrimitiveValue::Local(_) => {
                    unreachable!("rule inputs are normalized to external refs")
                }
                PrimitiveValue::External(key) => key.clone(),
            };

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

fn cotangent_seed_key<Op: Primitive>(linear: &LinearizedGraph<Op>, index: usize) -> Op::InputKey
where
    Op::InputKey: ADKey,
{
    assert!(
        !linear.tangent_inputs().is_empty(),
        "active tangent outputs require at least one tangent input to derive seed keys"
    );

    let base_slot = index.min(linear.tangent_inputs().len() - 1);
    let base_key = &linear.tangent_inputs()[base_slot].0;
    base_key.tangent_of(u64::MAX - index as u64)
}
