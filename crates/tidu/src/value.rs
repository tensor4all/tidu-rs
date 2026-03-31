use std::sync::{Arc, Mutex};

use crate::reverse_graph::{
    backward_from, grad_wrt, leaf_grad, leaf_handle, shares_graph, zero_leaf_grad, LeafHandle,
    ReverseEdge, ReverseInput,
};
use crate::{AdResult, AutodiffError, Differentiable};

enum ReverseHandle<V: Differentiable> {
    None,
    Leaf(LeafHandle<V>),
    Edge(ReverseEdge<V>),
}

struct ReverseState<V: Differentiable> {
    requires_grad: bool,
    handle: ReverseHandle<V>,
}

/// Public value handle for reverse-mode AD.
///
/// `Value` exposes a torch-like surface while keeping graph ownership hidden.
/// Internally it carries either a leaf gradient sink or an edge into a reverse
/// graph.
pub struct Value<V: Differentiable> {
    primal: Arc<V>,
    reverse: Mutex<ReverseState<V>>,
}

impl<V: Differentiable + 'static> Value<V> {
    /// Create a detached value.
    pub fn new(primal: V) -> Self {
        Self {
            primal: Arc::new(primal),
            reverse: Mutex::new(ReverseState {
                requires_grad: false,
                handle: ReverseHandle::None,
            }),
        }
    }

    pub(crate) fn from_reverse_edge(primal: V, edge: ReverseEdge<V>) -> Self {
        Self {
            primal: Arc::new(primal),
            reverse: Mutex::new(ReverseState {
                requires_grad: true,
                handle: ReverseHandle::Edge(edge),
            }),
        }
    }

    /// Borrow the primal value.
    pub fn primal(&self) -> &V {
        self.primal.as_ref()
    }

    pub(crate) fn shared_primal(&self) -> Arc<V> {
        self.primal.clone()
    }

    /// Return whether this value participates in reverse-mode AD.
    pub fn requires_grad(&self) -> bool {
        self.reverse
            .lock()
            .expect("reverse state poisoned")
            .requires_grad
    }

    /// Enable or disable gradient tracking.
    pub fn requires_grad_(self, enabled: bool) -> Self {
        {
            let mut reverse = self.reverse.lock().expect("reverse state poisoned");
            reverse.requires_grad = enabled;
            reverse.handle = if enabled {
                match &reverse.handle {
                    ReverseHandle::Leaf(existing) => ReverseHandle::Leaf(existing.clone()),
                    ReverseHandle::Edge(existing) => ReverseHandle::Edge(existing.clone()),
                    ReverseHandle::None => ReverseHandle::Leaf(leaf_handle()),
                }
            } else {
                ReverseHandle::None
            };
        }
        self
    }

    pub(crate) fn reverse_input(&self) -> Option<ReverseInput<V>> {
        let mut reverse = self.reverse.lock().expect("reverse state poisoned");
        if !reverse.requires_grad {
            return None;
        }
        let input = match &reverse.handle {
            ReverseHandle::Leaf(handle) => ReverseInput::Leaf(handle.clone()),
            ReverseHandle::Edge(edge) => ReverseInput::Edge(edge.clone()),
            ReverseHandle::None => {
                let handle = leaf_handle();
                reverse.handle = ReverseHandle::Leaf(handle.clone());
                ReverseInput::Leaf(handle)
            }
        };
        Some(input)
    }

    /// Read the accumulated leaf gradient, if available.
    pub fn grad(&self) -> AdResult<Option<V::Tangent>>
    where
        V::Tangent: Clone,
    {
        let reverse = self.reverse.lock().expect("reverse state poisoned");
        if !reverse.requires_grad {
            return Ok(None);
        }
        match &reverse.handle {
            ReverseHandle::Leaf(handle) => Ok(leaf_grad::<V>(handle)),
            ReverseHandle::Edge(_) | ReverseHandle::None => Ok(None),
        }
    }

    /// Clear the accumulated leaf gradient.
    pub fn zero_grad(&self) -> AdResult<()> {
        let reverse = self.reverse.lock().expect("reverse state poisoned");
        if !reverse.requires_grad {
            return Ok(());
        }
        if let ReverseHandle::Leaf(handle) = &reverse.handle {
            zero_leaf_grad::<V>(handle);
        }
        Ok(())
    }

    /// Run reverse-mode backward with the default scalar cotangent seed.
    pub fn backward(&self) -> AdResult<()>
    where
        V::Tangent: Clone,
    {
        let n = self.primal.as_ref().num_elements();
        if n != 1 {
            return Err(AutodiffError::NonScalarLoss { num_elements: n });
        }
        self.backward_with_seed(self.primal.as_ref().seed_cotangent())
    }

    /// Run reverse-mode backward with an explicit cotangent seed.
    pub fn backward_with_seed(&self, seed: V::Tangent) -> AdResult<()>
    where
        V::Tangent: Clone,
    {
        let input = self.reverse_input().ok_or(AutodiffError::MissingNode)?;
        backward_from(input, seed)
    }

    /// Compute functional gradients with respect to the requested inputs.
    pub fn grad_wrt_with_seed(
        &self,
        seed: V::Tangent,
        wrt: &[&Self],
    ) -> AdResult<Vec<Option<V::Tangent>>>
    where
        V::Tangent: Clone,
    {
        let input = self.reverse_input().ok_or(AutodiffError::MissingNode)?;
        let wrt_inputs = wrt
            .iter()
            .map(|value| value.reverse_input())
            .collect::<Vec<_>>();
        grad_wrt(input, seed, &wrt_inputs)
    }

    /// Return whether `self` and `other` share any reachable reverse graph.
    pub fn shares_reverse_graph(&self, other: &Self) -> bool {
        match (self.reverse_input(), other.reverse_input()) {
            (Some(lhs), Some(rhs)) => shares_graph(&lhs, &rhs),
            (None, None) => true,
            _ => false,
        }
    }
}
