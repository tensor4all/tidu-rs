use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::engine::{AutogradGraph, CheckpointRecipe, Gradients, TrackedValue};
use crate::{AdResult, AutodiffError, Differentiable, HvpResult, NodeId, ReverseRule};

/// Reverse-mode AD tape.
///
/// The tape records operations performed on [`TrackedValue`] values and
/// enables gradient computation via [`Tape::pullback`] or HVP via
/// [`Tape::hvp`].
///
/// Use [`Tape::record_op`] when you want to retain the materialized reverse
/// rule on the tape. Use [`Tape::record_checkpointed_op`] when you want to
/// trade memory for replay: the tape stores a lightweight
/// [`CheckpointRecipe`](crate::CheckpointRecipe) and rebuilds the reverse rule
/// lazily during pullback or HVP.
///
/// `Tape` is cheaply cloneable (internally reference-counted). Multiple
/// clones refer to the same underlying autograd graph.
///
/// # Examples
///
/// ```rust
/// use chainrules::powf_rrule;
/// use tidu::{AdResult, NodeId, ReverseRule, Tape};
///
/// struct PowfRule {
///     input: NodeId,
///     x: f64,
///     exponent: f64,
/// }
///
/// impl ReverseRule<f64> for PowfRule {
///     fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
///         Ok(vec![(self.input, powf_rrule(self.x, self.exponent, *cotangent))])
///     }
///
///     fn inputs(&self) -> Vec<NodeId> {
///         vec![self.input]
///     }
/// }
///
/// let tape = Tape::<f64>::new();
/// let x = tape.leaf(2.0);
/// let y = tape.record_op(
///     8.0,
///     Box::new(PowfRule {
///         input: x.node_id().unwrap(),
///         x: 2.0,
///         exponent: 3.0,
///     }),
///     None,
/// );
/// let grads = tape.pullback(&y).unwrap();
/// assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 12.0);
/// ```
pub struct Tape<V: Differentiable> {
    inner: Arc<Mutex<AutogradGraph<V>>>,
}

impl<V: Differentiable> Tape<V> {
    fn lock_graph(&self) -> MutexGuard<'_, AutogradGraph<V>> {
        match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Creates a new empty tape.
    pub fn new() -> Self {
        Self {
            inner: AutogradGraph::new(),
        }
    }

    /// Returns `true` if `self` and `other` are the same tape.
    pub fn same_tape(&self, other: &Tape<V>) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Returns a stable process-local identifier for this tape.
    pub fn id(&self) -> usize {
        self.lock_graph().id() as usize
    }

    /// Returns the current number of nodes recorded on this tape.
    pub fn node_count(&self) -> usize {
        self.lock_graph().node_count()
    }

    /// Creates a leaf (input variable) on this tape.
    ///
    /// Leaves are the starting points of the computation graph. After
    /// pullback, their gradients can be retrieved from [`Gradients::get`]
    /// using the returned value's [`TrackedValue::node_id`].
    pub fn leaf(&self, value: V) -> TrackedValue<V> {
        let primal = Arc::new(value);
        let node_id = self.lock_graph().record_leaf_shared(Arc::clone(&primal));
        TrackedValue::attached_shared(primal, node_id, self.clone(), None)
    }

    /// Creates a leaf with a tangent direction for HVP computation.
    ///
    /// The `tangent` specifies the direction vector **v** in the
    /// Hessian-vector product H·v. See [`Tape::hvp`] for the full workflow.
    pub fn leaf_with_tangent(&self, value: V, tangent: V::Tangent) -> AdResult<TrackedValue<V>> {
        let primal = Arc::new(value);
        let node_id = self.lock_graph().record_leaf_shared(Arc::clone(&primal));
        Ok(TrackedValue::attached_shared(
            primal,
            node_id,
            self.clone(),
            Some(tangent),
        ))
    }

    /// Records a placeholder node on the tape without a reverse rule.
    ///
    /// Use this for **two-phase recording**: first call `placeholder` to
    /// reserve a node (e.g. when the rule needs the output's own `NodeId`),
    /// then call [`Tape::attach_rule`] to supply the reverse rule later.
    pub fn placeholder(&self, value: V, tangent: Option<V::Tangent>) -> TrackedValue<V> {
        let primal = Arc::new(value);
        let node_id = self
            .lock_graph()
            .record_placeholder_shared(Arc::clone(&primal));
        TrackedValue::attached_shared(primal, node_id, self.clone(), tangent)
    }

    /// Reconstructs a [`TrackedValue`] handle for a node already on this tape.
    ///
    /// Useful when you have a `NodeId` (e.g. serialised or passed across an
    /// API boundary) and need to re-wrap it with its primal value and tape
    /// reference. When the node still retains its primal on the tape, that
    /// shared primal is reused and `value` is ignored. If the node's retained
    /// primal has been evicted, `value` becomes the fallback forward value for
    /// the returned handle. Returns an error if `node_id` is not present on
    /// this tape.
    pub fn tracked_existing(
        &self,
        node_id: NodeId,
        value: V,
        tangent: Option<V::Tangent>,
    ) -> AdResult<TrackedValue<V>> {
        let guard = self.lock_graph();
        if !guard.has_node(node_id) {
            return Err(AutodiffError::InvalidArgument(format!(
                "node {} is not present on this tape",
                node_id.index()
            )));
        }
        let retained = guard.node(node_id)?.retained_primal_shared();
        drop(guard);
        Ok(match retained {
            Some(primal) => TrackedValue::attached_shared(primal, node_id, self.clone(), tangent),
            None => TrackedValue::attached_owned(value, node_id, self.clone(), tangent),
        })
    }

    /// Records an operation on the tape, returning a tracked output.
    ///
    /// - `output_value`: the pre-computed forward result of the operation.
    /// - `rule`: the reverse rule used during pullback.
    /// - `output_tangent`: optional tangent of the output, only needed for
    ///   HVP (forward-over-reverse) computation. Pass `None` for standard
    ///   gradient computation.
    pub fn record_op(
        &self,
        output_value: V,
        rule: Box<dyn ReverseRule<V>>,
        output_tangent: Option<V::Tangent>,
    ) -> TrackedValue<V> {
        let primal = Arc::new(output_value);
        let node_id = self
            .lock_graph()
            .record_op_shared(Arc::clone(&primal), rule);
        TrackedValue::attached_shared(primal, node_id, self.clone(), output_tangent)
    }

    /// Records a checkpointed operation on the tape.
    ///
    /// The output value is known at forward time, but the reverse rule is
    /// rebuilt later from the recipe when a future execution context reaches
    /// this node. The returned [`TrackedValue`] keeps the forward primal even
    /// though the tape may evict the graph-side retained primal for replay.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let y = tape.record_checkpointed_op(output, Box::new(recipe), None);
    /// ```
    pub fn record_checkpointed_op(
        &self,
        output_value: V,
        recipe: Box<dyn CheckpointRecipe<V>>,
        output_tangent: Option<V::Tangent>,
    ) -> TrackedValue<V> {
        let node_id = self.lock_graph().record_checkpointed_op(recipe);
        TrackedValue::attached_owned(output_value, node_id, self.clone(), output_tangent)
    }

    /// Attaches or replaces the reverse rule for an existing node.
    ///
    /// Typically used after [`Tape::placeholder`] to complete a two-phase
    /// recording.
    pub fn attach_rule(&self, node_id: NodeId, rule: Box<dyn ReverseRule<V>>) -> AdResult<()> {
        self.lock_graph().attach_rule(node_id, rule)
    }

    /// Runs reverse-mode pullback from a scalar loss value.
    ///
    /// The loss must satisfy `num_elements() == 1`; for non-scalar outputs use
    /// [`pullback_with_seed`](Self::pullback_with_seed) with an explicit
    /// cotangent seed instead.
    ///
    /// Only leaf-node gradients are stored in the returned [`Gradients`].
    pub fn pullback(&self, loss: &TrackedValue<V>) -> AdResult<Gradients<V>> {
        let n = loss.value().num_elements();
        if n != 1 {
            return Err(AutodiffError::NonScalarLoss { num_elements: n });
        }
        self.pullback_with_seed(loss, loss.value().seed_cotangent())
    }

    /// Runs reverse-mode pullback from an arbitrary output cotangent seed.
    ///
    /// Use this instead of [`pullback`](Self::pullback) when the output is
    /// non-scalar or when you need a custom seed direction.
    ///
    /// Only leaf-node gradients are stored in the returned [`Gradients`].
    pub fn pullback_with_seed(
        &self,
        output: &TrackedValue<V>,
        seed: V::Tangent,
    ) -> AdResult<Gradients<V>> {
        let output_node = output.node_id.ok_or(AutodiffError::MissingNode)?;
        let guard = self.lock_graph();
        guard.ensure_alive()?;
        guard.pullback_from(output_node, seed)
    }

    /// Computes gradient **and** Hessian-vector product via forward-over-reverse.
    ///
    /// Requires:
    /// - A **scalar** loss (`num_elements() == 1`).
    /// - `leaf_tangents` maps leaf [`NodeId`] values to tangent directions
    ///   **v** for the Hessian-vector product H·v. Leaves not present in
    ///   the map are treated as having zero tangent.
    /// - Each rule must implement [`ReverseRule::forward_tangents`] and
    ///   [`ReverseRule::pullback_with_tangents`]
    ///   (the defaults return `Err(HvpNotSupported)`).
    ///
    /// Returns an [`HvpResult`] containing both the gradient and the HVP.
    pub fn hvp(
        &self,
        loss: &TrackedValue<V>,
        leaf_tangents: &HashMap<NodeId, V::Tangent>,
    ) -> AdResult<HvpResult<V>>
    where
        V::Tangent: Differentiable<Tangent = V::Tangent>,
    {
        let loss_node = loss.node_id.ok_or(AutodiffError::MissingNode)?;
        let n = loss.value().num_elements();
        if n != 1 {
            return Err(AutodiffError::NonScalarLoss { num_elements: n });
        }
        let guard = self.lock_graph();
        guard.ensure_alive()?;
        guard.hvp_from(
            loss_node,
            loss.value().seed_cotangent(),
            loss.value().zero_tangent(),
            leaf_tangents,
        )
    }

    /// Marks the computation graph as freed, releasing stored rules.
    ///
    /// After this call, [`pullback`](Self::pullback) and [`hvp`](Self::hvp)
    /// will return `Err(GraphFreed)` until a new leaf or operation is
    /// recorded. Creating new leaves or operations automatically revives
    /// the graph.
    pub fn free_graph(&self) {
        self.lock_graph().free_graph();
    }
}

impl<V: Differentiable> Clone for Tape<V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<V: Differentiable> Default for Tape<V> {
    fn default() -> Self {
        Self::new()
    }
}
