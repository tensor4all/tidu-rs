use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{ADKey, ADRuleResult, Primitive};
use computegraph::fragment::FragmentBuilder;
use computegraph::resolve::resolve;
use computegraph::{GlobalValKey, GraphOp, OpMode, ValRef};

use crate::{LinearizedGraph, PrimitiveGraph};

use super::trace::{Trace, TraceNode};

/// Downstream execution hooks for eager backward.
pub trait BackwardExecutor<Op: Primitive>
where
    Op::InputKey: ADKey,
{
    /// Replay a primitive graph and return any concrete values needed by
    /// transpose execution.
    fn execute_forward(
        &mut self,
        graph: PrimitiveGraph<'_, Op>,
        initial_data: &HashMap<GlobalValKey<Op>, Arc<Op::Operand>>,
    ) -> HashMap<GlobalValKey<Op>, Arc<Op::Operand>>;

    /// Run a transposed linear graph with concrete cotangent seeds.
    fn run_transposed_linear(
        &mut self,
        linear: &LinearizedGraph<Op>,
        cotangent_out: &[Option<Arc<Op::Operand>>],
        external_data: &HashMap<GlobalValKey<Op>, Arc<Op::Operand>>,
        ctx: &mut Op::ADContext,
    ) -> ADRuleResult<Vec<Option<Arc<Op::Operand>>>>;

    /// Add two concrete operands for cotangent accumulation.
    fn add_operands(&mut self, a: &Arc<Op::Operand>, b: &Arc<Op::Operand>) -> Arc<Op::Operand>;
}

/// Execute reverse-mode AD over an eager trace.
pub fn try_backward<Op: Primitive>(
    output_key: &GlobalValKey<Op>,
    output_trace: Option<&Trace<Op>>,
    seed: Arc<Op::Operand>,
    executor: &mut impl BackwardExecutor<Op>,
    ctx: &mut Op::ADContext,
) -> ADRuleResult<HashMap<GlobalValKey<Op>, Arc<Op::Operand>>>
where
    Op::InputKey: ADKey,
{
    let sorted_nodes = topo_sort_trace(output_trace);
    let mut cotangents: HashMap<GlobalValKey<Op>, Arc<Op::Operand>> = HashMap::new();
    cotangents.insert(output_key.clone(), seed);

    for node in sorted_nodes.iter().rev() {
        let cotangent_out: Vec<Option<Arc<Op::Operand>>> = node
            .primal_out_keys()
            .iter()
            .map(|key| cotangents.get(key).cloned())
            .collect();
        if cotangent_out.iter().all(Option::is_none) {
            continue;
        }

        let mut active_output_slots = Vec::new();
        let mut active_cotangent_out = Vec::new();
        for (slot, maybe_cotangent) in cotangent_out.into_iter().enumerate() {
            if let Some(cotangent) = maybe_cotangent {
                active_output_slots.push(slot);
                active_cotangent_out.push(Some(cotangent));
            }
        }

        let linear = try_build_single_op_linear(node, &active_output_slots, ctx)?;
        let replay_graph = PrimitiveGraph::new(linear.as_graph());
        let all_values = executor.execute_forward(replay_graph, node.saved_data());
        let cotangent_in =
            executor.run_transposed_linear(&linear, &active_cotangent_out, &all_values, ctx)?;

        for (edge, maybe_cotangent) in node.input_edges().iter().zip(cotangent_in) {
            let Some(cotangent) = maybe_cotangent else {
                continue;
            };
            if !edge.requires_grad {
                continue;
            }

            let accumulated = match cotangents.remove(&edge.key) {
                Some(existing) => executor.add_operands(&existing, &cotangent),
                None => cotangent,
            };
            cotangents.insert(edge.key.clone(), accumulated);
        }
    }

    Ok(cotangents)
}

fn topo_sort_trace<Op: GraphOp>(output_trace: Option<&Trace<Op>>) -> Vec<Arc<TraceNode<Op>>> {
    fn visit<Op: GraphOp>(
        node: &Arc<TraceNode<Op>>,
        visited: &mut HashSet<*const TraceNode<Op>>,
        order: &mut Vec<Arc<TraceNode<Op>>>,
    ) {
        let ptr = Arc::as_ptr(node);
        if !visited.insert(ptr) {
            return;
        }

        for edge in node.input_edges() {
            if let Some(parent) = &edge.node {
                visit(parent, visited, order);
            }
        }

        order.push(node.clone());
    }

    let mut visited = HashSet::new();
    let mut order = Vec::new();
    if let Some(trace) = output_trace {
        visit(trace.node(), &mut visited, &mut order);
    }
    order
}

fn try_build_single_op_linear<Op: Primitive>(
    node: &TraceNode<Op>,
    output_slots: &[usize],
    ctx: &mut Op::ADContext,
) -> ADRuleResult<LinearizedGraph<Op>>
where
    Op::InputKey: ADKey,
{
    let mut builder = FragmentBuilder::new();

    let input_local_ids: Vec<_> = node
        .primal_in_keys()
        .iter()
        .map(|key| match key {
            GlobalValKey::Input(input_key) => builder.add_input(input_key.clone()),
            GlobalValKey::Derived { .. } => {
                panic!(
                    "build_single_op_linear requires GlobalValKey::Input aliases in node.primal_in_keys"
                )
            }
        })
        .collect();

    let outputs = builder.add_op(
        node.op().clone(),
        input_local_ids
            .iter()
            .map(|local_id| ValRef::Local(*local_id))
            .collect(),
        OpMode::Primal,
    );
    let selected_outputs: Vec<_> = output_slots
        .iter()
        .map(|&slot| {
            *outputs.get(slot).unwrap_or_else(|| {
                panic!(
                    "build_single_op_linear got output slot {slot} for {:?}, \
                     which has only {} outputs",
                    node.op(),
                    outputs.len()
                )
            })
        })
        .collect();
    builder.set_outputs(selected_outputs.clone());

    let fragment = Arc::new(builder.build());
    let view = resolve(vec![fragment.clone()]);
    let output_keys: Vec<_> = selected_outputs
        .iter()
        .map(|output_id| fragment.vals()[*output_id].key.clone())
        .collect();
    let wrt_keys: Vec<_> = node
        .primal_in_keys()
        .iter()
        .map(|key| match key {
            GlobalValKey::Input(input_key) => input_key.clone(),
            GlobalValKey::Derived { .. } => {
                panic!(
                    "build_single_op_linear requires GlobalValKey::Input aliases in node.primal_in_keys"
                )
            }
        })
        .collect();
    let aliases = HashMap::new();

    crate::try_linearize(&view, &output_keys, &wrt_keys, 0, ctx, &aliases)
}
