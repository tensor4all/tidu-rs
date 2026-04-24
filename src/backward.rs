use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chainrules::{ADKey, PrimitiveOp};
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::resolve::resolve;
use computegraph::{GlobalValKey, GraphOp, OpMode, ValRef};

use crate::grad_node::GradNode;
use crate::LinearFragment;

/// Caller-provided execution hooks for eager backward.
pub trait BackwardCallbacks<Op: PrimitiveOp>
where
    Op::InputKey: ADKey,
{
    /// Execute a linear fragment forward and return any concrete values needed
    /// by eager transpose.
    fn execute_forward(
        &mut self,
        fragment: &Fragment<Op>,
        initial_data: &HashMap<GlobalValKey<Op>, Arc<Op::Operand>>,
    ) -> HashMap<GlobalValKey<Op>, Arc<Op::Operand>>;

    /// Execute transpose eagerly for a linear fragment with concrete seeds.
    fn eager_transpose(
        &mut self,
        linear: &LinearFragment<Op>,
        cotangent_out: &[Option<Arc<Op::Operand>>],
        external_data: &HashMap<GlobalValKey<Op>, Arc<Op::Operand>>,
        ctx: &mut Op::ADContext,
    ) -> Vec<Option<Arc<Op::Operand>>>;

    /// Add two concrete operands for cotangent accumulation.
    fn add_operands(&mut self, a: &Arc<Op::Operand>, b: &Arc<Op::Operand>) -> Arc<Op::Operand>;
}

/// Topologically sort the reachable grad DAG in dependency-first order.
pub fn topo_sort_grad_dag<Op: GraphOp>(
    output_node: &Option<Arc<GradNode<Op>>>,
) -> Vec<Arc<GradNode<Op>>> {
    fn visit<Op: GraphOp>(
        node: &Arc<GradNode<Op>>,
        visited: &mut HashSet<*const GradNode<Op>>,
        order: &mut Vec<Arc<GradNode<Op>>>,
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
    if let Some(node) = output_node {
        visit(node, &mut visited, &mut order);
    }
    order
}

/// Execute reverse-mode AD over a grad DAG.
pub fn backward_dag<Op: PrimitiveOp>(
    sorted_nodes: &[Arc<GradNode<Op>>],
    output_key: &GlobalValKey<Op>,
    seed: Arc<Op::Operand>,
    callbacks: &mut impl BackwardCallbacks<Op>,
    ctx: &mut Op::ADContext,
) -> HashMap<GlobalValKey<Op>, Arc<Op::Operand>>
where
    Op::InputKey: ADKey,
{
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

        let linear = build_single_op_linear(node, ctx);
        let all_values = callbacks.execute_forward(&linear.fragment, node.saved_data());
        let cotangent_in = callbacks.eager_transpose(&linear, &cotangent_out, &all_values, ctx);

        for (edge, maybe_cotangent) in node.input_edges().iter().zip(cotangent_in.into_iter()) {
            let Some(cotangent) = maybe_cotangent else {
                continue;
            };
            if !edge.requires_grad {
                continue;
            }

            let accumulated = match cotangents.remove(&edge.key) {
                Some(existing) => callbacks.add_operands(&existing, &cotangent),
                None => cotangent,
            };
            cotangents.insert(edge.key.clone(), accumulated);
        }
    }

    cotangents
}

fn build_single_op_linear<Op: PrimitiveOp>(
    node: &GradNode<Op>,
    ctx: &mut Op::ADContext,
) -> LinearFragment<Op>
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
    builder.set_outputs(outputs.clone());

    let fragment = Arc::new(builder.build());
    let view = resolve(vec![fragment.clone()]);
    let output_keys: Vec<_> = outputs
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

    crate::differentiate(&view, &output_keys, &wrt_keys, 0, ctx, &aliases)
}
