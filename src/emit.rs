use std::collections::HashMap;

use crate::{ADRuleResult, Primitive, PrimitiveBuilder, PrimitiveValue};
use computegraph::{GlobalValKey, LocalValId, OpMode, ValRef};

use crate::LinearizedGraph;

/// Execute the transpose of a linear fragment using a caller-provided emitter.
///
/// This mirrors [`crate::try_transpose`] but leaves concrete execution and
/// value storage to the downstream [`PrimitiveBuilder`].
pub fn try_transpose_fragment<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
    builder: &mut impl PrimitiveBuilder<Op>,
    cotangent_seeds: &[Option<LocalValId>],
    ctx: &mut Op::ADContext,
) -> ADRuleResult<Vec<Option<LocalValId>>>
where
    Op::InputKey: crate::ADKey,
{
    let mut cotangent_env: HashMap<GlobalValKey<Op>, LocalValId> = HashMap::new();
    let graph = linear.as_graph();

    for (index, maybe_tangent_output) in linear.tangent_outputs().iter().enumerate() {
        if let (Some(output_id), Some(Some(seed_id))) =
            (maybe_tangent_output, cotangent_seeds.get(index))
        {
            let key = graph.vals()[*output_id].key.clone();
            cotangent_env.insert(key, *seed_id);
        }
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

        let cotangent_in = op_node.op.try_transpose_rule(
            builder,
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
                    let sum = builder.add_primitive(
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

    Ok(linear
        .tangent_inputs()
        .iter()
        .map(|(_, tangent_input_id)| {
            let tangent_input_key = &graph.vals()[*tangent_input_id].key;
            cotangent_env.get(tangent_input_key).copied()
        })
        .collect())
}
