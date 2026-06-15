use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{ADKey, ADRuleResult, Primitive};
use computegraph::{GraphOperation, ValueKey};

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
        initial_data: &HashMap<ValueKey<Op>, Arc<Op::Operand>>,
    ) -> HashMap<ValueKey<Op>, Arc<Op::Operand>>;

    /// Run a transposed linear graph with concrete cotangent seeds.
    fn run_transposed_linear(
        &mut self,
        linear: &LinearizedGraph<Op>,
        cotangent_out: &[Option<Arc<Op::Operand>>],
        external_data: &HashMap<ValueKey<Op>, Arc<Op::Operand>>,
        ctx: &mut Op::ADContext,
    ) -> ADRuleResult<Vec<Option<Arc<Op::Operand>>>>;

    /// Add two concrete operands for cotangent accumulation.
    fn add_operands(&mut self, a: &Arc<Op::Operand>, b: &Arc<Op::Operand>) -> Arc<Op::Operand>;
}

/// Execute reverse-mode AD over an eager trace.
pub fn try_backward<Op: Primitive>(
    output_key: &ValueKey<Op>,
    output_trace: Option<&Trace<Op>>,
    seed: Arc<Op::Operand>,
    executor: &mut impl BackwardExecutor<Op>,
    ctx: &mut Op::ADContext,
) -> ADRuleResult<HashMap<ValueKey<Op>, Arc<Op::Operand>>>
where
    Op::InputKey: ADKey,
{
    let sorted_nodes = topo_sort_trace(output_trace);
    let mut cotangents: HashMap<ValueKey<Op>, Arc<Op::Operand>> = HashMap::new();
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

        let linear = node
            .computation()
            .try_linearize(&active_output_slots, ctx)?;
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

fn topo_sort_trace<Op: GraphOperation>(
    output_trace: Option<&Trace<Op>>,
) -> Vec<Arc<TraceNode<Op>>> {
    fn visit<Op: GraphOperation>(
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
