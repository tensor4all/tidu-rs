use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::{AdResult, AutodiffError, Differentiable};

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

pub(crate) trait ReverseRule<V: Differentiable>: Send + Sync {
    fn pullback(&self, grad_outputs: &[Option<V::Tangent>]) -> AdResult<Vec<Option<V::Tangent>>>;
}

pub(crate) struct ReverseNode<V: Differentiable> {
    pub(crate) parents: Vec<Option<ReverseInput<V>>>,
    pub(crate) output_count: usize,
    pub(crate) rule: Box<dyn ReverseRule<V>>,
}

impl<V: Differentiable> ReverseNode<V> {
    pub(crate) fn new(
        parents: Vec<Option<ReverseInput<V>>>,
        output_count: usize,
        rule: Box<dyn ReverseRule<V>>,
    ) -> Self {
        Self {
            parents,
            output_count,
            rule,
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

fn accumulate_leaf_grad<V>(handle: &LeafHandle<V>, grad: V::Tangent)
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

fn node_key<V: Differentiable>(node: &Arc<ReverseNode<V>>) -> usize {
    Arc::as_ptr(node) as *const () as usize
}

fn leaf_key<V: Differentiable>(handle: &LeafHandle<V>) -> usize {
    Arc::as_ptr(handle) as *const () as usize
}

enum TaskInput<V: Differentiable> {
    Leaf(LeafHandle<V>),
    Edge { node_id: usize, output_slot: usize },
}

impl<V: Differentiable> Clone for TaskInput<V> {
    fn clone(&self) -> Self {
        match self {
            Self::Leaf(handle) => Self::Leaf(handle.clone()),
            Self::Edge {
                node_id,
                output_slot,
            } => Self::Edge {
                node_id: *node_id,
                output_slot: *output_slot,
            },
        }
    }
}

struct TaskNode<V: Differentiable> {
    node: Arc<ReverseNode<V>>,
    parents: Vec<Option<TaskInput<V>>>,
    grad_outputs: Vec<Option<V::Tangent>>,
    live_output_slots: Vec<bool>,
    remaining_contributions: Vec<usize>,
    pending_output_slots: usize,
    enqueued: bool,
}

struct GraphTask<V: Differentiable> {
    nodes: Vec<TaskNode<V>>,
    node_ids: HashMap<usize, usize>,
    ready: VecDeque<usize>,
    edge_queries: HashMap<(usize, usize), Vec<usize>>,
    leaf_queries: HashMap<usize, Vec<usize>>,
    query_grads: Vec<Option<V::Tangent>>,
}

impl<V: Differentiable> GraphTask<V> {
    fn from_root(root: &ReverseEdge<V>, seed: V::Tangent) -> AdResult<Self> {
        let mut task = Self {
            nodes: Vec::new(),
            node_ids: HashMap::new(),
            ready: VecDeque::new(),
            edge_queries: HashMap::new(),
            leaf_queries: HashMap::new(),
            query_grads: Vec::new(),
        };
        task.discover(&root.node);
        task.finalize_parents_and_contributions(root);
        task.seed_root(root, seed)?;
        Ok(task)
    }

    fn discover(&mut self, node: &Arc<ReverseNode<V>>) {
        let key = node_key(node);
        if self.node_ids.contains_key(&key) {
            return;
        }
        let task_node_id = self.nodes.len();
        self.node_ids.insert(key, task_node_id);
        self.nodes.push(TaskNode {
            node: node.clone(),
            parents: Vec::new(),
            grad_outputs: (0..node.output_count).map(|_| None).collect(),
            live_output_slots: vec![false; node.output_count],
            remaining_contributions: vec![0; node.output_count],
            pending_output_slots: 0,
            enqueued: false,
        });
        for parent in &node.parents {
            if let Some(ReverseInput::Edge(edge)) = parent {
                self.discover(&edge.node);
            }
        }
    }

    fn finalize_parents_and_contributions(&mut self, root: &ReverseEdge<V>) {
        for task_node_id in 0..self.nodes.len() {
            let parents = self.nodes[task_node_id].node.parents.clone();
            let mut normalized_parents = Vec::with_capacity(parents.len());
            for parent in parents {
                let normalized = match parent {
                    Some(ReverseInput::Leaf(handle)) => Some(TaskInput::Leaf(handle)),
                    Some(ReverseInput::Edge(edge)) => {
                        let parent_id = self.node_ids[&node_key(&edge.node)];
                        self.nodes[parent_id].live_output_slots[edge.output_slot] = true;
                        self.nodes[parent_id].remaining_contributions[edge.output_slot] += 1;
                        Some(TaskInput::Edge {
                            node_id: parent_id,
                            output_slot: edge.output_slot,
                        })
                    }
                    None => None,
                };
                normalized_parents.push(normalized);
            }
            self.nodes[task_node_id].parents = normalized_parents;
        }

        let root_id = self.node_ids[&node_key(&root.node)];
        self.nodes[root_id].live_output_slots[root.output_slot] = true;
        self.nodes[root_id].remaining_contributions[root.output_slot] += 1;

        for node in &mut self.nodes {
            node.pending_output_slots = node
                .live_output_slots
                .iter()
                .zip(&node.remaining_contributions)
                .filter(|(live, count)| **live && **count > 0)
                .count();
        }
    }

    fn seed_root(&mut self, root: &ReverseEdge<V>, seed: V::Tangent) -> AdResult<()> {
        let root_id = self.node_ids[&node_key(&root.node)];
        self.accumulate_slot(root_id, root.output_slot, seed)
    }

    fn register_queries(&mut self, wrt: &[Option<ReverseInput<V>>]) {
        self.query_grads = (0..wrt.len()).map(|_| None).collect();
        for (index, target) in wrt.iter().enumerate() {
            match target {
                Some(ReverseInput::Leaf(handle)) => {
                    self.leaf_queries
                        .entry(leaf_key(handle))
                        .or_default()
                        .push(index);
                }
                Some(ReverseInput::Edge(edge)) => {
                    if let Some(&node_id) = self.node_ids.get(&node_key(&edge.node)) {
                        self.edge_queries
                            .entry((node_id, edge.output_slot))
                            .or_default()
                            .push(index);
                    }
                }
                None => {}
            }
        }
    }

    fn accumulate_query_slots(&mut self, query_slots: &[usize], grad: &V::Tangent)
    where
        V::Tangent: Clone,
    {
        for &query_slot in query_slots {
            let slot = &mut self.query_grads[query_slot];
            match slot.take() {
                Some(existing) => *slot = Some(V::accumulate_tangent(existing, grad)),
                None => *slot = Some(grad.clone()),
            }
        }
    }

    fn accumulate_slot(
        &mut self,
        task_node_id: usize,
        output_slot: usize,
        grad: V::Tangent,
    ) -> AdResult<()> {
        let Some(output_len) = self
            .nodes
            .get(task_node_id)
            .map(|node| node.grad_outputs.len())
        else {
            return Err(AutodiffError::MissingNode);
        };
        if output_slot >= output_len {
            return Err(AutodiffError::InvalidArgument(format!(
                "output slot {output_slot} is out of bounds for task node {task_node_id}"
            )));
        }
        if let Some(query_slots) = self.edge_queries.get(&(task_node_id, output_slot)).cloned() {
            self.accumulate_query_slots(&query_slots, &grad);
        }
        let node = &mut self.nodes[task_node_id];
        let slot = &mut node.grad_outputs[output_slot];
        match slot.take() {
            Some(existing) => *slot = Some(V::accumulate_tangent(existing, &grad)),
            None => *slot = Some(grad),
        }
        if node.remaining_contributions[output_slot] == 0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "received too many cotangent contributions for task node {task_node_id} output slot {output_slot}"
            )));
        }
        node.remaining_contributions[output_slot] -= 1;
        if node.live_output_slots[output_slot] && node.remaining_contributions[output_slot] == 0 {
            debug_assert!(node.pending_output_slots > 0);
            node.pending_output_slots -= 1;
        }
        if node.pending_output_slots == 0 && !node.enqueued {
            node.enqueued = true;
            self.ready.push_back(task_node_id);
        }
        Ok(())
    }

    fn run(mut self, accumulate_leafs: bool) -> AdResult<Vec<Option<V::Tangent>>>
    where
        V::Tangent: Clone,
    {
        while let Some(task_node_id) = self.ready.pop_front() {
            let grads = {
                let node = &mut self.nodes[task_node_id];
                node.enqueued = false;
                let grad_outputs = std::mem::take(&mut node.grad_outputs);
                node.node.rule.pullback(&grad_outputs)?
            };

            if grads.len() != self.nodes[task_node_id].parents.len() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "reverse rule returned {} gradients for {} inputs",
                    grads.len(),
                    self.nodes[task_node_id].parents.len()
                )));
            }

            let parents = self.nodes[task_node_id].parents.clone();
            for (parent, grad) in parents.into_iter().zip(grads.into_iter()) {
                let Some(grad) = grad else {
                    continue;
                };
                match parent {
                    Some(TaskInput::Leaf(handle)) => {
                        if let Some(query_slots) =
                            self.leaf_queries.get(&leaf_key(&handle)).cloned()
                        {
                            self.accumulate_query_slots(&query_slots, &grad);
                        }
                        if accumulate_leafs {
                            accumulate_leaf_grad::<V>(&handle, grad);
                        }
                    }
                    Some(TaskInput::Edge {
                        node_id,
                        output_slot,
                    }) => self.accumulate_slot(node_id, output_slot, grad)?,
                    None => {}
                }
            }
        }
        Ok(self.query_grads)
    }
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
