use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::rules::GraphPrimitiveBuilder;
use crate::{ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, Primitive};
use computegraph::graph::GraphBuilder;
use computegraph::resolve::{ResolvedView, ValueDef};
use computegraph::{GraphOperation, LocalValueId, OperationKey, ValueKey};

use crate::LinearizedGraph;

/// Linearize a resolved computation graph, producing a linear graph.
///
/// The transform walks the reachable DAG from `outputs` in dependency-first
/// order and delegates primitive-specific JVP generation to
/// [`crate::Primitive::jvp_rule`].
///
/// # Examples
///
/// ```ignore
/// use computegraph::resolve::resolve;
/// use tidu::linearize;
///
/// let view = resolve(vec![primal_graph]);
/// let mut ctx = ();
/// let aliases = std::collections::HashMap::new();
/// let linear = linearize(&view, &[output_key], &[input_key], 1, &mut ctx, &aliases)?;
/// assert_eq!(linear.tangent_outputs().len(), 1);
/// # Ok::<(), crate::ADRuleError>(())
/// ```
pub fn linearize<Op: Primitive>(
    view: &ResolvedView<Op>,
    outputs: &[ValueKey<Op>],
    wrt: &[Op::InputKey],
    pass: DiffPassId,
    ctx: &mut Op::ADContext,
    aliases: &HashMap<Op::InputKey, ValueKey<Op>>,
) -> ADRuleResult<LinearizedGraph<Op>>
where
    Op::InputKey: ADKey,
{
    let mut builder = GraphBuilder::<Op>::new();
    let topo_keys = topological_order(view, outputs, aliases);
    let mut tangent_env: HashMap<ValueKey<Op>, Option<LocalValueId>> = HashMap::new();
    let mut processed_ops = HashSet::new();

    let mut tangent_inputs = Vec::with_capacity(wrt.len());
    for wrt_key in wrt {
        let tangent_key = wrt_key.tangent_of(pass);
        let tangent_id = builder.add_input(tangent_key);
        tangent_env.insert(ValueKey::Input(wrt_key.clone()), Some(tangent_id));
        tangent_inputs.push((wrt_key.clone(), tangent_id));
    }

    for key in topo_keys {
        if tangent_env.contains_key(&key) {
            continue;
        }

        let Some(val_def) = view.resolve_value(&key) else {
            continue;
        };

        match val_def {
            ValueDef::Input { key: ref input_key } => {
                if let Some(aliased_key) = aliases.get(input_key) {
                    let aliased_tangent = tangent_env.get(aliased_key).copied().flatten();
                    tangent_env.insert(key, aliased_tangent);
                } else {
                    tangent_env.insert(key, None);
                }
            }
            ValueDef::Produced {
                operation,
                input_keys,
                role,
                ..
            } => {
                let global_op_key =
                    OperationKey::new(operation.clone(), input_keys.clone(), role.clone());
                if !processed_ops.insert(global_op_key.clone()) {
                    continue;
                }

                let tangent_in: Vec<Option<LocalValueId>> = input_keys
                    .iter()
                    .map(|input_key| tangent_env.get(input_key).copied().flatten())
                    .collect();
                let output_keys = output_keys(&global_op_key, operation.output_count());

                if tangent_in.iter().all(Option::is_none) {
                    for output_key in output_keys {
                        tangent_env.insert(output_key, None);
                    }
                    continue;
                }

                let mut primitive_builder = GraphPrimitiveBuilder::new(&mut builder);
                let tangent_out = operation.jvp_rule(
                    &mut primitive_builder,
                    &input_keys,
                    &output_keys,
                    &tangent_in,
                    ctx,
                )?;
                if tangent_out.len() != output_keys.len() {
                    return Err(ADRuleError::invalid_input(
                        format!("{operation:?}"),
                        ADRuleKind::Jvp,
                        format!(
                            "rule returned {} tangents for {} outputs",
                            tangent_out.len(),
                            output_keys.len()
                        ),
                    ));
                }

                for (output_key, tangent_output) in output_keys.into_iter().zip(tangent_out) {
                    tangent_env.insert(output_key, tangent_output);
                }
            }
        }
    }

    let tangent_outputs: Vec<Option<LocalValueId>> = outputs
        .iter()
        .map(|key| tangent_env.get(key).copied().flatten())
        .collect();
    let active_outputs: Vec<LocalValueId> = tangent_outputs.iter().filter_map(|id| *id).collect();
    if !active_outputs.is_empty() {
        builder.set_outputs(active_outputs);
    }

    Ok(LinearizedGraph::from_parts(
        builder.build(),
        tangent_inputs,
        tangent_outputs,
    ))
}

fn output_keys<Op: GraphOperation>(
    op_key: &OperationKey<Op>,
    output_count: usize,
) -> Vec<ValueKey<Op>> {
    let op_key = Arc::new(op_key.clone());
    (0..output_count)
        .map(|output_slot| ValueKey::Derived {
            operation: Arc::clone(&op_key),
            output_slot: output_slot as u8,
        })
        .collect()
}

fn topological_order<Op: GraphOperation>(
    view: &ResolvedView<Op>,
    outputs: &[ValueKey<Op>],
    aliases: &HashMap<Op::InputKey, ValueKey<Op>>,
) -> Vec<ValueKey<Op>> {
    fn visit<Op: GraphOperation>(
        key: &ValueKey<Op>,
        view: &ResolvedView<Op>,
        aliases: &HashMap<Op::InputKey, ValueKey<Op>>,
        visited: &mut HashSet<ValueKey<Op>>,
        order: &mut Vec<ValueKey<Op>>,
    ) {
        if !visited.insert(key.clone()) {
            return;
        }

        match view.resolve_value(key) {
            Some(ValueDef::Produced { input_keys, .. }) => {
                for input_key in input_keys {
                    visit(&input_key, view, aliases, visited, order);
                }
            }
            Some(ValueDef::Input { key: input_key }) => {
                if let Some(aliased_key) = aliases.get(&input_key) {
                    visit(aliased_key, view, aliases, visited, order);
                }
            }
            None => {}
        }

        order.push(key.clone());
    }

    let mut visited = HashSet::new();
    let mut order = Vec::new();
    for output_key in outputs {
        visit(output_key, view, aliases, &mut visited, &mut order);
    }
    order
}
