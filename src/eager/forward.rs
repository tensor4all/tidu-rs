use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{ADKey, ADRuleResult, Primitive};
use computegraph::{GraphOperation, ValueKey};

use crate::LinearizedGraph;

use super::record::RecordedGraph;
use super::trace::{Trace, TraceNode};

/// Tangent seed aligned with one linearized graph input.
pub type ForwardTangentInput<Op> = (
    <Op as GraphOperation>::InputKey,
    Option<Arc<<Op as GraphOperation>::Operand>>,
);

/// Downstream execution hooks for eager forward-mode AD.
pub trait ForwardExecutor<Op: Primitive>
where
    Op::InputKey: ADKey,
{
    /// Linearize one recorded eager graph node for forward-mode propagation.
    ///
    /// The default implementation calls [`RecordedGraph::linearize`]. Runtime
    /// owners may override this to memoize per-node transform results without
    /// taking trace traversal ownership away from tidu.
    fn linearize_recorded_graph(
        &mut self,
        graph: &RecordedGraph<Op>,
        output_slots: &[usize],
        ctx: &mut Op::ADContext,
    ) -> ADRuleResult<Arc<LinearizedGraph<Op>>> {
        graph.linearize(output_slots, ctx).map(Arc::new)
    }

    /// Run a linearized graph with concrete tangent seeds.
    fn run_linearized_forward(
        &mut self,
        linear: &LinearizedGraph<Op>,
        tangent_in: &[ForwardTangentInput<Op>],
        external_data: &HashMap<ValueKey<Op>, Arc<Op::Operand>>,
        ctx: &mut Op::ADContext,
    ) -> ADRuleResult<Vec<Option<Arc<Op::Operand>>>>;

    /// Add two concrete operands for tangent accumulation.
    fn add_operands(&mut self, a: &Arc<Op::Operand>, b: &Arc<Op::Operand>) -> Arc<Op::Operand>;
}

/// Execute forward-mode AD over an eager trace for one output value.
pub fn try_forward<Op: Primitive>(
    output_key: &ValueKey<Op>,
    output_trace: Option<&Trace<Op>>,
    tangent_seeds: &HashMap<ValueKey<Op>, Arc<Op::Operand>>,
    executor: &mut impl ForwardExecutor<Op>,
    ctx: &mut Op::ADContext,
) -> ADRuleResult<Option<Arc<Op::Operand>>>
where
    Op::InputKey: ADKey,
{
    let sorted_nodes = topo_sort_trace(output_trace);
    let mut tangents = tangent_seeds.clone();

    for node in &sorted_nodes {
        let mut graph_input_tangents: HashMap<Op::InputKey, Option<Arc<Op::Operand>>> =
            HashMap::new();
        let mut has_active_input = false;
        for (input_key, edge) in node
            .computation()
            .input_keys()
            .iter()
            .zip(node.input_edges().iter())
        {
            let tangent = tangents.get(&edge.key).cloned();
            if tangent.is_some() {
                has_active_input = true;
            }
            graph_input_tangents.insert(input_key.clone(), tangent);
        }
        if !has_active_input {
            continue;
        }

        let output_slots: Vec<usize> = (0..node.primal_out_keys().len()).collect();
        let linear = executor.linearize_recorded_graph(node.computation(), &output_slots, ctx)?;
        let tangent_in: Vec<_> = linear
            .tangent_inputs()
            .iter()
            .map(|(key, _)| {
                (
                    key.clone(),
                    graph_input_tangents.get(key).cloned().unwrap_or(None),
                )
            })
            .collect();
        let tangent_out = executor.run_linearized_forward(
            linear.as_ref(),
            &tangent_in,
            node.saved_data(),
            ctx,
        )?;
        assert_eq!(
            tangent_out.len(),
            output_slots.len(),
            "eager forward executor returned {} tangents for {} output slots",
            tangent_out.len(),
            output_slots.len()
        );

        for (slot, maybe_tangent) in output_slots.into_iter().zip(tangent_out) {
            let tangent = match maybe_tangent {
                Some(tangent) => tangent,
                None => continue,
            };
            let key = node.primal_out_keys()[slot].clone();
            let accumulated = match tangents.remove(&key) {
                Some(existing) => executor.add_operands(&existing, &tangent),
                None => tangent,
            };
            tangents.insert(key, accumulated);
        }
    }

    Ok(tangents.get(output_key).cloned())
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
