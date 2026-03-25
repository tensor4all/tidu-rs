use std::marker::PhantomData;

use crate::{AdResult, AutodiffError, Differentiable, NodeId, TrackedValue};

/// Accumulated gradients indexed by [`NodeId`].
///
/// # Examples
///
/// ```
/// use tidu::{Gradients, NodeId};
///
/// let mut grads = Gradients::<f64>::new();
/// grads.accumulate(NodeId::new(0), 3.0).unwrap();
/// assert_eq!(*grads.get(NodeId::new(0)).unwrap(), 3.0);
/// ```
pub struct Gradients<V: Differentiable> {
    entries: Vec<(NodeId, V::Tangent)>,
}

impl<V: Differentiable> Gradients<V> {
    /// Creates an empty gradient container.
    ///
    /// # Examples
    ///
    /// ```
    /// use tidu::Gradients;
    /// let grads = Gradients::<f64>::new();
    /// assert!(grads.entries().is_empty());
    /// ```
    pub fn new() -> Self {
        Self { entries: vec![] }
    }

    pub(crate) fn push_entry(&mut self, node: NodeId, grad: V::Tangent) {
        self.entries.push((node, grad));
    }

    /// Returns the gradient for `node`, if present.
    ///
    /// # Examples
    ///
    /// ```
    /// use tidu::{Gradients, NodeId};
    ///
    /// let mut grads = Gradients::<f64>::new();
    /// grads.accumulate(NodeId::new(0), 5.0).unwrap();
    /// assert_eq!(*grads.get(NodeId::new(0)).unwrap(), 5.0);
    /// assert!(grads.get(NodeId::new(1)).is_none());
    /// ```
    pub fn get(&self, node: NodeId) -> Option<&V::Tangent> {
        self.entries
            .iter()
            .find(|(id, _)| *id == node)
            .map(|(_, grad)| grad)
    }

    /// Inserts or accumulates a gradient for `node`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tidu::{Gradients, NodeId};
    ///
    /// let mut grads = Gradients::<f64>::new();
    /// grads.accumulate(NodeId::new(0), 2.0).unwrap();
    /// grads.accumulate(NodeId::new(0), 3.0).unwrap();
    /// assert_eq!(*grads.get(NodeId::new(0)).unwrap(), 5.0);
    /// ```
    pub fn accumulate(&mut self, node: NodeId, grad: V::Tangent) -> AdResult<()> {
        if let Some(entry) = self.entries.iter_mut().find(|(id, _)| *id == node) {
            let existing = entry.1.clone();
            entry.1 = V::accumulate_tangent(existing, &grad);
        } else {
            self.entries.push((node, grad));
        }
        Ok(())
    }

    /// Returns all `(node, grad)` entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tidu::{Gradients, NodeId};
    ///
    /// let mut grads = Gradients::<f64>::new();
    /// grads.accumulate(NodeId::new(0), 1.0).unwrap();
    /// assert_eq!(grads.entries().len(), 1);
    /// ```
    pub fn entries(&self) -> &[(NodeId, V::Tangent)] {
        &self.entries
    }
}

impl<V: Differentiable> Default for Gradients<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Differentiable> std::fmt::Debug for Gradients<V>
where
    V::Tangent: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gradients")
            .field("entries", &self.entries)
            .finish()
    }
}

/// Compiled pullback execution plan.
///
/// # Examples
///
/// ```ignore
/// let plan = tidu::PullbackPlan::<MyType>::build(&loss).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PullbackPlan<V: Differentiable> {
    loss: NodeId,
    _marker: PhantomData<V>,
}

impl<V: Differentiable> PullbackPlan<V> {
    /// Builds a pullback plan from a loss value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tidu::{PullbackPlan, Tape};
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(2.0);
    /// let plan = PullbackPlan::build(&x).unwrap();
    /// assert_eq!(plan.loss_node().index(), 0);
    /// ```
    pub fn build(loss: &TrackedValue<V>) -> AdResult<Self> {
        let node_id = loss.node_id.ok_or(AutodiffError::MissingNode)?;
        Ok(Self {
            loss: node_id,
            _marker: PhantomData,
        })
    }

    /// Executes the pre-built pullback plan.
    ///
    /// # Examples
    ///
    /// ```
    /// use tidu::{PullbackPlan, Tape};
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(2.0);
    /// let plan = PullbackPlan::build(&x).unwrap();
    /// let grads = plan.execute(&x).unwrap();
    /// assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
    /// ```
    pub fn execute(&self, loss: &TrackedValue<V>) -> AdResult<Gradients<V>> {
        let tape = loss.tape.as_ref().ok_or(AutodiffError::MissingNode)?;
        tape.pullback(loss)
    }

    /// Returns loss node ID for this plan.
    ///
    /// # Examples
    ///
    /// ```
    /// use tidu::{NodeId, PullbackPlan};
    /// let _id_fn: fn(&PullbackPlan<f64>) -> NodeId = PullbackPlan::loss_node;
    /// ```
    pub fn loss_node(&self) -> NodeId {
        self.loss
    }
}

/// Result of a forward-over-reverse HVP computation.
///
/// Contains both the standard gradient and the Hessian-vector
/// product H*v, where v is the tangent direction set on leaf values
/// via [`crate::Tape::leaf_with_tangent`].
///
/// # Examples
///
/// ```rust
/// use tidu::{AdResult, HvpResult, NodeId, ReverseRule, Tape};
///
/// struct SquareRuleHvp {
///     input: NodeId,
///     x: f64,
///     dx: f64,
/// }
///
/// impl ReverseRule<f64> for SquareRuleHvp {
///     fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
///         Ok(vec![(self.input, 2.0 * self.x * *cotangent)])
///     }
///
///     fn inputs(&self) -> Vec<NodeId> {
///         vec![self.input]
///     }
///
///     fn pullback_with_tangents(
///         &self,
///         cotangent: &f64,
///         cotangent_tangent: &f64,
///     ) -> AdResult<Vec<(NodeId, f64, f64)>> {
///         Ok(vec![(
///             self.input,
///             2.0 * self.x * *cotangent,
///             2.0 * self.dx * *cotangent + 2.0 * self.x * *cotangent_tangent,
///         )])
///     }
/// }
///
/// let tape = Tape::<f64>::new();
/// let x = tape.leaf_with_tangent(3.0, 1.0).unwrap();
/// let y = tape.record_op(
///     9.0,
///     Box::new(SquareRuleHvp {
///         input: x.node_id().unwrap(),
///         x: 3.0,
///         dx: 1.0,
///     }),
///     None,
/// );
/// let result: HvpResult<f64> = tape.hvp(&y).unwrap();
/// assert_eq!(*result.gradients.get(x.node_id().unwrap()).unwrap(), 6.0);
/// assert_eq!(*result.hvp.get(x.node_id().unwrap()).unwrap(), 2.0);
/// ```
pub struct HvpResult<V: Differentiable> {
    /// Gradients.
    pub gradients: Gradients<V>,
    /// Hessian-vector product: H*v.
    pub hvp: Gradients<V>,
}
