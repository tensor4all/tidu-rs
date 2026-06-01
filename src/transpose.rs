use std::collections::HashMap;

use crate::rules::FragmentPrimitiveBuilder;
use crate::{ADKey, ADRuleResult, Primitive, PrimitiveBuilder, PrimitiveValue};
use computegraph::fragment::FragmentBuilder;
use computegraph::{GlobalValKey, LocalValId, OpMode, ValRef};

use crate::LinearizedGraph;

/// Transpose a linear fragment, reversing linear flow.
///
/// Fan-out accumulation is emitted explicitly with [`crate::Primitive::add`];
/// no duplication primitive is assumed by the graph transform.
///
/// # Examples
///
/// ```ignore
/// let mut ctx = ();
/// let transposed = tidu::transpose(&linear_fragment, &mut ctx);
/// assert_eq!(transposed.tangent_outputs().len(), linear_fragment.tangent_inputs().len());
/// ```
pub fn transpose<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
    ctx: &mut Op::ADContext,
) -> LinearizedGraph<Op>
where
    Op::InputKey: ADKey,
{
    match try_transpose(linear, ctx) {
        Ok(transposed) => transposed,
        Err(err) => panic!("{err}"),
    }
}

/// Fallible form of [`transpose`].
///
/// This returns [`crate::ADRuleError`] when a primitive cannot emit a
/// transpose rule.
pub fn try_transpose<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
    ctx: &mut Op::ADContext,
) -> ADRuleResult<LinearizedGraph<Op>>
where
    Op::InputKey: ADKey,
{
    let mut builder = FragmentBuilder::<Op>::new();
    let mut cotangent_env: HashMap<GlobalValKey<Op>, LocalValId> = HashMap::new();
    let mut cotangent_seed_inputs = Vec::new();
    let graph = linear.as_graph();

    for (index, maybe_tangent_output) in linear.tangent_outputs().iter().enumerate() {
        let Some(tangent_output_id) = maybe_tangent_output else {
            continue;
        };

        let source_key = graph.vals()[*tangent_output_id].key.clone();
        let seed_key = cotangent_seed_key(linear, index);
        let seed_id = builder.add_input(seed_key.clone());
        cotangent_env.insert(source_key, seed_id);
        cotangent_seed_inputs.push((seed_key, seed_id));
    }

    for op_node in graph.ops().iter().rev() {
        let cotangent_out: Vec<Option<LocalValId>> = op_node
            .outputs
            .iter()
            .map(|output_id| cotangent_env.get(&graph.vals()[*output_id].key).copied())
            .collect();
        if cotangent_out.iter().all(Option::is_none) {
            continue;
        }

        let rule_inputs: Vec<PrimitiveValue<Op>> = op_node
            .inputs
            .iter()
            .map(|input| match input {
                ValRef::Local(local_id) => {
                    PrimitiveValue::External(graph.vals()[*local_id].key.clone())
                }
                ValRef::External(key) => PrimitiveValue::External(key.clone()),
            })
            .collect();

        let mut primitive_builder = FragmentPrimitiveBuilder::new(&mut builder);
        let cotangent_in = op_node.op.try_transpose_rule(
            &mut primitive_builder,
            &cotangent_out,
            &rule_inputs,
            &op_node.mode,
            ctx,
        )?;
        assert_eq!(
            cotangent_in.len(),
            rule_inputs.len(),
            "transpose_rule for {:?} returned {} cotangents for {} inputs",
            op_node.op,
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
                    let mut primitive_builder = FragmentPrimitiveBuilder::new(&mut builder);
                    let sum = primitive_builder.add_primitive(
                        Op::add(),
                        vec![
                            PrimitiveValue::Local(existing_id),
                            PrimitiveValue::Local(cotangent_id),
                        ],
                        OpMode::Linear {
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

    let tangent_outputs: Vec<Option<LocalValId>> = linear
        .tangent_inputs()
        .iter()
        .map(|(_, tangent_input_id)| {
            let tangent_input_key = &graph.vals()[*tangent_input_id].key;
            cotangent_env.get(tangent_input_key).copied()
        })
        .collect();
    let active_outputs: Vec<LocalValId> = tangent_outputs.iter().filter_map(|id| *id).collect();
    if !active_outputs.is_empty() {
        builder.set_outputs(active_outputs);
    }

    Ok(LinearizedGraph::from_parts(
        builder.build(),
        cotangent_seed_inputs,
        tangent_outputs,
    ))
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
