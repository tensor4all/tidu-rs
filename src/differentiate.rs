use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::rules::FragmentPrimitiveBuilder;
use crate::{ADKey, ADRuleResult, DiffPassId, Primitive};
use computegraph::fragment::FragmentBuilder;
use computegraph::resolve::{ResolvedView, ValDef};
use computegraph::{GlobalOpKey, GlobalValKey, GraphOp, LocalValId};

use crate::LinearFragment;

/// Differentiate a resolved computation graph, producing a linear fragment.
///
/// The transform walks the reachable DAG from `outputs` in dependency-first
/// order and delegates primitive-specific JVP generation to
/// [`crate::Primitive::try_jvp_rule`].
///
/// # Examples
///
/// ```ignore
/// use computegraph::resolve::resolve;
/// use tidu::try_differentiate;
///
/// let view = resolve(vec![primal_fragment]);
/// let mut ctx = ();
/// let aliases = std::collections::HashMap::new();
/// let linear = try_differentiate(&view, &[output_key], &[input_key], 1, &mut ctx, &aliases)?;
/// assert_eq!(linear.tangent_outputs.len(), 1);
/// # Ok::<(), crate::ADRuleError>(())
/// ```
pub fn differentiate<Op: Primitive>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
    wrt: &[Op::InputKey],
    pass: DiffPassId,
    ctx: &mut Op::ADContext,
    aliases: &HashMap<Op::InputKey, GlobalValKey<Op>>,
) -> LinearFragment<Op>
where
    Op::InputKey: ADKey,
{
    match try_differentiate(view, outputs, wrt, pass, ctx, aliases) {
        Ok(linear) => linear,
        Err(err) => panic!("{err}"),
    }
}

/// Fallible form of [`differentiate`].
///
/// This returns [`crate::ADRuleError`] when a primitive cannot emit a JVP
/// rule, allowing downstream frontends to surface missing extension rules as
/// normal errors instead of panics.
pub fn try_differentiate<Op: Primitive>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
    wrt: &[Op::InputKey],
    pass: DiffPassId,
    ctx: &mut Op::ADContext,
    aliases: &HashMap<Op::InputKey, GlobalValKey<Op>>,
) -> ADRuleResult<LinearFragment<Op>>
where
    Op::InputKey: ADKey,
{
    let mut builder = FragmentBuilder::<Op>::new();
    let topo_keys = topological_order(view, outputs, aliases);
    let mut tangent_env: HashMap<GlobalValKey<Op>, Option<LocalValId>> = HashMap::new();
    let mut processed_ops = HashSet::new();

    let mut tangent_inputs = Vec::with_capacity(wrt.len());
    for wrt_key in wrt {
        let tangent_key = wrt_key.tangent_of(pass);
        let tangent_id = builder.add_input(tangent_key);
        tangent_env.insert(GlobalValKey::Input(wrt_key.clone()), Some(tangent_id));
        tangent_inputs.push((wrt_key.clone(), tangent_id));
    }

    for key in topo_keys {
        if tangent_env.contains_key(&key) {
            continue;
        }

        let Some(val_def) = view.resolve_val(&key) else {
            continue;
        };

        match val_def {
            ValDef::Input { key: ref input_key } => {
                if let Some(aliased_key) = aliases.get(input_key) {
                    let aliased_tangent = tangent_env.get(aliased_key).copied().flatten();
                    tangent_env.insert(key, aliased_tangent);
                } else {
                    tangent_env.insert(key, None);
                }
            }
            ValDef::Produced {
                op,
                input_keys,
                mode,
                ..
            } => {
                let global_op_key = GlobalOpKey::new(op.clone(), input_keys.clone(), mode.clone());
                if !processed_ops.insert(global_op_key.clone()) {
                    continue;
                }

                let tangent_in: Vec<Option<LocalValId>> = input_keys
                    .iter()
                    .map(|input_key| tangent_env.get(input_key).copied().flatten())
                    .collect();
                let output_keys = output_keys(&global_op_key, op.n_outputs());

                if tangent_in.iter().all(Option::is_none) {
                    for output_key in output_keys {
                        tangent_env.insert(output_key, None);
                    }
                    continue;
                }

                let mut primitive_builder = FragmentPrimitiveBuilder::new(&mut builder);
                let tangent_out = op.try_jvp_rule(
                    &mut primitive_builder,
                    &input_keys,
                    &output_keys,
                    &tangent_in,
                    ctx,
                )?;
                assert_eq!(
                    tangent_out.len(),
                    output_keys.len(),
                    "jvp_rule for {:?} returned {} tangents for {} outputs",
                    op,
                    tangent_out.len(),
                    output_keys.len()
                );

                for (output_key, tangent_output) in output_keys.into_iter().zip(tangent_out) {
                    tangent_env.insert(output_key, tangent_output);
                }
            }
        }
    }

    let tangent_outputs: Vec<Option<LocalValId>> = outputs
        .iter()
        .map(|key| tangent_env.get(key).copied().flatten())
        .collect();
    let active_outputs: Vec<LocalValId> = tangent_outputs.iter().filter_map(|id| *id).collect();
    if !active_outputs.is_empty() {
        builder.set_outputs(active_outputs);
    }

    Ok(LinearFragment {
        fragment: builder.build(),
        tangent_inputs,
        tangent_outputs,
    })
}

fn output_keys<Op: GraphOp>(op_key: &GlobalOpKey<Op>, n_outputs: usize) -> Vec<GlobalValKey<Op>> {
    let op_key = Arc::new(op_key.clone());
    (0..n_outputs)
        .map(|output_slot| GlobalValKey::Derived {
            op: Arc::clone(&op_key),
            output_slot: output_slot as u8,
        })
        .collect()
}

fn topological_order<Op: GraphOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
    aliases: &HashMap<Op::InputKey, GlobalValKey<Op>>,
) -> Vec<GlobalValKey<Op>> {
    fn visit<Op: GraphOp>(
        key: &GlobalValKey<Op>,
        view: &ResolvedView<Op>,
        aliases: &HashMap<Op::InputKey, GlobalValKey<Op>>,
        visited: &mut HashSet<GlobalValKey<Op>>,
        order: &mut Vec<GlobalValKey<Op>>,
    ) {
        if !visited.insert(key.clone()) {
            return;
        }

        match view.resolve_val(key) {
            Some(ValDef::Produced { input_keys, .. }) => {
                for input_key in input_keys {
                    visit(&input_key, view, aliases, visited, order);
                }
            }
            Some(ValDef::Input { key: input_key }) => {
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
