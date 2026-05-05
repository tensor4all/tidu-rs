use std::collections::HashMap;

use chainrules::{ADRuleResult, PrimitiveOp};
use computegraph::{GlobalValKey, LocalValId, OpEmitter, OpMode, ValRef};

use crate::LinearFragment;

/// Execute the transpose of a linear fragment using an eager emitter.
///
/// This mirrors [`crate::transpose`] but leaves execution strategy to the
/// caller-provided [`computegraph::OpEmitter`].
pub fn eager_transpose_fragment<Op: PrimitiveOp>(
    linear: &LinearFragment<Op>,
    emitter: &mut impl OpEmitter<Op>,
    cotangent_seeds: &[Option<LocalValId>],
    ctx: &mut Op::ADContext,
) -> Vec<Option<LocalValId>>
where
    Op::InputKey: chainrules::ADKey,
{
    match try_eager_transpose_fragment(linear, emitter, cotangent_seeds, ctx) {
        Ok(cotangents) => cotangents,
        Err(err) => panic!("{err}"),
    }
}

/// Fallible form of [`eager_transpose_fragment`].
pub fn try_eager_transpose_fragment<Op: PrimitiveOp>(
    linear: &LinearFragment<Op>,
    emitter: &mut impl OpEmitter<Op>,
    cotangent_seeds: &[Option<LocalValId>],
    ctx: &mut Op::ADContext,
) -> ADRuleResult<Vec<Option<LocalValId>>>
where
    Op::InputKey: chainrules::ADKey,
{
    let mut cotangent_env: HashMap<GlobalValKey<Op>, LocalValId> = HashMap::new();

    for (index, maybe_tangent_output) in linear.tangent_outputs.iter().enumerate() {
        if let (Some(output_id), Some(Some(seed_id))) =
            (maybe_tangent_output, cotangent_seeds.get(index))
        {
            let key = linear.fragment.vals()[*output_id].key.clone();
            cotangent_env.insert(key, *seed_id);
        }
    }

    for op_node in linear.fragment.ops().iter().rev() {
        let cotangent_out: Vec<Option<LocalValId>> = op_node
            .outputs
            .iter()
            .map(|output_id| {
                cotangent_env
                    .get(&linear.fragment.vals()[*output_id].key)
                    .copied()
            })
            .collect();
        if cotangent_out.iter().all(Option::is_none) {
            continue;
        }

        let rule_inputs: Vec<ValRef<Op>> = op_node
            .inputs
            .iter()
            .map(|input| match input {
                ValRef::Local(local_id) => {
                    ValRef::External(linear.fragment.vals()[*local_id].key.clone())
                }
                ValRef::External(key) => ValRef::External(key.clone()),
            })
            .collect();

        let cotangent_in = op_node.op.try_transpose_rule(
            emitter,
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

        for (input, maybe_cotangent) in rule_inputs.iter().zip(cotangent_in.into_iter()) {
            let Some(cotangent_id) = maybe_cotangent else {
                continue;
            };
            let input_key = match input {
                ValRef::Local(_) => unreachable!("rule inputs are normalized to external refs"),
                ValRef::External(key) => key.clone(),
            };

            match cotangent_env.get(&input_key).copied() {
                Some(existing_id) => {
                    let sum = emitter.add_op(
                        Op::add(),
                        vec![ValRef::Local(existing_id), ValRef::Local(cotangent_id)],
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
        .tangent_inputs
        .iter()
        .map(|(_, tangent_input_id)| {
            let tangent_input_key = &linear.fragment.vals()[*tangent_input_id].key;
            cotangent_env.get(tangent_input_key).copied()
        })
        .collect())
}
