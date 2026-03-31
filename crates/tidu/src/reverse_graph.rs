use std::collections::HashSet;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::graph_task::GraphTask;
use crate::linearized::LinearizedOp;
use crate::{AdResult, Differentiable};

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

pub(crate) struct LeafState<V: Differentiable> {
    grad: Option<V::Tangent>,
}

pub(crate) type LeafHandle<V> = Arc<Mutex<LeafState<V>>>;

pub(crate) struct ReverseEdge<V: Differentiable> {
    pub(crate) node: Arc<ReverseNode<V>>,
    pub(crate) output_slot: usize,
}

pub(crate) enum ReverseInput<V: Differentiable> {
    Leaf(LeafHandle<V>),
    Edge(ReverseEdge<V>),
}

impl<V: Differentiable> Clone for ReverseEdge<V> {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            output_slot: self.output_slot,
        }
    }
}

impl<V: Differentiable> Clone for ReverseInput<V> {
    fn clone(&self) -> Self {
        match self {
            Self::Leaf(handle) => Self::Leaf(handle.clone()),
            Self::Edge(edge) => Self::Edge(edge.clone()),
        }
    }
}

pub(crate) struct ReverseNode<V: Differentiable> {
    pub(crate) parents: Vec<Option<ReverseInput<V>>>,
    pub(crate) output_count: usize,
    input_grad_mask: Vec<bool>,
    linearization: StoredNodeLinearization<V>,
}

impl<V: Differentiable> ReverseNode<V> {
    pub(crate) fn new(
        parents: Vec<Option<ReverseInput<V>>>,
        output_count: usize,
        input_grad_mask: Vec<bool>,
        linearization: StoredNodeLinearization<V>,
    ) -> Self {
        Self {
            parents,
            output_count,
            input_grad_mask,
            linearization,
        }
    }

    pub(crate) fn vjp(
        &self,
        output_cotangents: &[Option<V::Tangent>],
    ) -> AdResult<Vec<Option<V::Tangent>>> {
        self.linearization
            .vjp(output_cotangents, &self.input_grad_mask)
    }
}

pub(crate) trait StoredLinearization<V: Differentiable>: Send + Sync {
    fn vjp(
        &self,
        output_cotangents: &[Option<V::Tangent>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<V::Tangent>>>;
}

impl<V, L> StoredLinearization<V> for L
where
    V: Differentiable + Send + Sync + 'static,
    L: LinearizedOp<V> + Send + Sync + 'static,
{
    fn vjp(
        &self,
        output_cotangents: &[Option<V::Tangent>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<V::Tangent>>> {
        LinearizedOp::vjp(self, output_cotangents, input_grad_mask)
    }
}

pub(crate) enum StoredNodeLinearization<V: Differentiable> {
    Retained(Box<dyn StoredLinearization<V>>),
}

impl<V: Differentiable> StoredNodeLinearization<V> {
    pub(crate) fn retained<L>(linearized: L) -> Self
    where
        V: Send + Sync + 'static,
        L: LinearizedOp<V> + Send + Sync + 'static,
    {
        Self::Retained(Box::new(linearized))
    }

    pub(crate) fn vjp(
        &self,
        output_cotangents: &[Option<V::Tangent>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<V::Tangent>>> {
        match self {
            Self::Retained(linearization) => linearization.vjp(output_cotangents, input_grad_mask),
        }
    }
}

fn new_leaf_handle<V: Differentiable>() -> LeafHandle<V> {
    Arc::new(Mutex::new(LeafState { grad: None }))
}

pub(crate) fn leaf_handle<V: Differentiable>() -> LeafHandle<V> {
    new_leaf_handle()
}

pub(crate) fn leaf_grad<V>(handle: &LeafHandle<V>) -> Option<V::Tangent>
where
    V: Differentiable,
    V::Tangent: Clone,
{
    lock_unpoisoned(handle).grad.clone()
}

pub(crate) fn zero_leaf_grad<V: Differentiable>(handle: &LeafHandle<V>) {
    lock_unpoisoned(handle).grad = None;
}

pub(crate) fn accumulate_leaf_grad<V>(handle: &LeafHandle<V>, grad: V::Tangent)
where
    V: Differentiable,
    V::Tangent: Clone,
{
    let mut leaf = lock_unpoisoned(handle);
    leaf.grad = match leaf.grad.take() {
        Some(existing) => Some(V::accumulate_tangent(existing, &grad)),
        None => Some(grad),
    };
}

pub(crate) fn node_key<V: Differentiable>(node: &Arc<ReverseNode<V>>) -> usize {
    Arc::as_ptr(node) as *const () as usize
}

pub(crate) fn leaf_key<V: Differentiable>(handle: &LeafHandle<V>) -> usize {
    Arc::as_ptr(handle) as *const () as usize
}

pub(crate) fn backward_from<V>(input: ReverseInput<V>, seed: V::Tangent) -> AdResult<()>
where
    V: Differentiable,
    V::Tangent: Clone,
{
    match input {
        ReverseInput::Leaf(handle) => {
            accumulate_leaf_grad::<V>(&handle, seed);
            Ok(())
        }
        ReverseInput::Edge(edge) => {
            GraphTask::from_root(&edge, seed)?.run(true)?;
            Ok(())
        }
    }
}

pub(crate) fn grad_wrt<V>(
    input: ReverseInput<V>,
    seed: V::Tangent,
    wrt: &[Option<ReverseInput<V>>],
) -> AdResult<Vec<Option<V::Tangent>>>
where
    V: Differentiable,
    V::Tangent: Clone,
{
    match input {
        ReverseInput::Leaf(handle) => {
            let mut grads: Vec<Option<V::Tangent>> = (0..wrt.len()).map(|_| None).collect();
            for (index, target) in wrt.iter().enumerate() {
                if let Some(ReverseInput::Leaf(query_handle)) = target {
                    if leaf_key(query_handle) == leaf_key(&handle) {
                        grads[index] = Some(seed.clone());
                    }
                }
            }
            Ok(grads)
        }
        ReverseInput::Edge(edge) => {
            let mut task = GraphTask::from_root(&edge, seed)?;
            task.register_queries(wrt);
            task.run(false)
        }
    }
}

fn collect_graph_keys<V: Differentiable>(
    input: &ReverseInput<V>,
    node_keys: &mut HashSet<usize>,
    leaf_keys: &mut HashSet<usize>,
) {
    match input {
        ReverseInput::Leaf(handle) => {
            leaf_keys.insert(leaf_key(handle));
        }
        ReverseInput::Edge(edge) => collect_graph_node_keys(&edge.node, node_keys, leaf_keys),
    }
}

fn collect_graph_node_keys<V: Differentiable>(
    node: &Arc<ReverseNode<V>>,
    node_keys: &mut HashSet<usize>,
    leaf_keys: &mut HashSet<usize>,
) {
    let key = node_key(node);
    if !node_keys.insert(key) {
        return;
    }
    for parent in node.parents.iter().flatten() {
        match parent {
            ReverseInput::Leaf(handle) => {
                leaf_keys.insert(leaf_key(handle));
            }
            ReverseInput::Edge(edge) => collect_graph_node_keys(&edge.node, node_keys, leaf_keys),
        }
    }
}

fn intersects_graph_keys<V: Differentiable>(
    input: &ReverseInput<V>,
    node_keys: &HashSet<usize>,
    leaf_keys: &HashSet<usize>,
) -> bool {
    match input {
        ReverseInput::Leaf(handle) => leaf_keys.contains(&leaf_key(handle)),
        ReverseInput::Edge(edge) => intersects_graph_node_keys(&edge.node, node_keys, leaf_keys),
    }
}

fn intersects_graph_node_keys<V: Differentiable>(
    node: &Arc<ReverseNode<V>>,
    node_keys: &HashSet<usize>,
    leaf_keys: &HashSet<usize>,
) -> bool {
    let key = node_key(node);
    if node_keys.contains(&key) {
        return true;
    }
    for parent in node.parents.iter().flatten() {
        match parent {
            ReverseInput::Leaf(handle) => {
                if leaf_keys.contains(&leaf_key(handle)) {
                    return true;
                }
            }
            ReverseInput::Edge(edge) => {
                if intersects_graph_node_keys(&edge.node, node_keys, leaf_keys) {
                    return true;
                }
            }
        }
    }
    false
}

pub(crate) fn shares_graph<V: Differentiable>(
    lhs: &ReverseInput<V>,
    rhs: &ReverseInput<V>,
) -> bool {
    let mut lhs_node_keys = HashSet::new();
    let mut lhs_leaf_keys = HashSet::new();
    collect_graph_keys(lhs, &mut lhs_node_keys, &mut lhs_leaf_keys);
    intersects_graph_keys(rhs, &lhs_node_keys, &lhs_leaf_keys)
}
