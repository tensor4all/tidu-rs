use std::sync::Arc;

use crate::engine::Tape;
use crate::{Differentiable, NodeId};

pub(crate) enum TrackedPrimal<V> {
    Owned(V),
    Shared(Arc<V>),
}

impl<V> TrackedPrimal<V> {
    fn as_ref(&self) -> &V {
        match self {
            Self::Owned(value) => value,
            Self::Shared(value) => value.as_ref(),
        }
    }

    fn into_owned(self) -> V
    where
        V: Clone,
    {
        match self {
            Self::Owned(value) => value,
            Self::Shared(value) => value.as_ref().clone(),
        }
    }
}

/// A value connected to a [`Tape`] for reverse-mode AD.
///
/// Every `TrackedValue` holds:
/// - a **primal view** of the value (`V`), owned or shared,
/// - an optional **node ID** linking it to the computation graph,
/// - a reference to the **tape** that owns the graph.
///
/// You obtain a `TrackedValue` from [`Tape::leaf`] (input variables) or
/// [`Tape::record_op`] / [`Tape::record_checkpointed_op`] (operation outputs).
/// You can also create a **detached** value with [`TrackedValue::new`] — it
/// carries no graph connection and will not participate in gradient
/// computation.
///
/// # Examples
///
/// ```rust
/// use tidu::Tape;
///
/// let tape = Tape::<f64>::new();
/// let x = tape.leaf(3.0);
/// assert!(x.requires_grad());
/// assert_eq!(*x.value(), 3.0);
/// ```
pub struct TrackedValue<V: Differentiable> {
    pub(crate) primal: TrackedPrimal<V>,
    pub(crate) node_id: Option<NodeId>,
    pub(crate) tape: Option<Tape<V>>,
    pub(crate) requires_grad: bool,
    pub(crate) tangent: Option<V::Tangent>,
}

impl<V: Differentiable> TrackedValue<V> {
    /// Creates a **detached** tracked value (no tape, `requires_grad = false`).
    ///
    /// Use this when you need a `TrackedValue` that does not participate in
    /// gradient computation — for example, as a constant input to an
    /// operation.
    pub fn new(value: V) -> Self {
        Self {
            primal: TrackedPrimal::Owned(value),
            node_id: None,
            tape: None,
            requires_grad: false,
            tangent: None,
        }
    }

    pub(crate) fn attached_shared(
        primal: Arc<V>,
        node_id: NodeId,
        tape: Tape<V>,
        tangent: Option<V::Tangent>,
    ) -> Self {
        Self {
            primal: TrackedPrimal::Shared(primal),
            node_id: Some(node_id),
            tape: Some(tape),
            requires_grad: true,
            tangent,
        }
    }

    pub(crate) fn attached_owned(
        primal: V,
        node_id: NodeId,
        tape: Tape<V>,
        tangent: Option<V::Tangent>,
    ) -> Self {
        Self {
            primal: TrackedPrimal::Owned(primal),
            node_id: Some(node_id),
            tape: Some(tape),
            requires_grad: true,
            tangent,
        }
    }

    /// Returns the underlying value.
    ///
    /// Attached values may share their primal with the tape's retained node
    /// storage, but this still yields a plain `&V` view.
    pub fn value(&self) -> &V {
        self.primal.as_ref()
    }

    /// Consumes and returns the underlying value.
    ///
    /// When the primal is shared with the tape, this clones `V`.
    pub fn into_value(self) -> V
    where
        V: Clone,
    {
        self.primal.into_owned()
    }

    /// Returns whether this value participates in gradient propagation.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Returns the graph node ID, or `None` if this value is detached.
    ///
    /// The `NodeId` is used to look up gradients in [`crate::Gradients::get`].
    pub fn node_id(&self) -> Option<NodeId> {
        self.node_id
    }

    /// Returns the tangent for HVP computation, or `None` if not set.
    ///
    /// Tangents are set via [`Tape::leaf_with_tangent`] and specify the
    /// direction vector **v** in the Hessian-vector product H·v.
    pub fn tangent(&self) -> Option<&V::Tangent> {
        self.tangent.as_ref()
    }

    /// Returns whether this tracked value has a tangent for HVP.
    pub fn has_tangent(&self) -> bool {
        self.tangent.is_some()
    }

    /// Returns a reference to the tape this value is connected to, if any.
    pub fn tape(&self) -> Option<&Tape<V>> {
        self.tape.as_ref()
    }

    /// Consumes and returns a detached copy that does not require gradients.
    ///
    /// The returned value keeps its primal but drops the tape connection,
    /// node ID, and tangent. If the primal is shared with the tape, detaching
    /// clones it into owned storage. Use this to stop gradient flow through a
    /// value (analogous to PyTorch's `Tensor.detach()`).
    pub fn detach(self) -> Self
    where
        V: Clone,
    {
        Self::new(self.primal.into_owned())
    }
}
