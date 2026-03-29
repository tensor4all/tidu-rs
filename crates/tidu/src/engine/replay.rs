use crate::{AdResult, Differentiable, NodeId, ReverseRule};

/// Replay result for a checkpointed operation.
///
/// The replay rebuilds both the output primal and the materialized reverse rule
/// needed for pullback or HVP.
///
/// # Examples
///
/// ```ignore
/// let replayed = ReplayResult {
///     output_primal,
///     rule,
/// };
/// ```
pub struct ReplayResult<V: Differentiable> {
    pub output_primal: V,
    pub rule: Box<dyn ReverseRule<V>>,
}

/// Lightweight recipe that can rebuild a checkpointed operation on demand.
///
/// Recipes hold static parameters and input dependencies, but not the heavy
/// backward/HVP state itself. The runtime gathers the direct input primals and
/// calls [`CheckpointRecipe::replay`] when execution reaches the node.
///
/// # Examples
///
/// ```ignore
/// struct SquareRecipe {
///     input: NodeId,
/// }
///
/// impl CheckpointRecipe<f64> for SquareRecipe {
///     fn inputs(&self) -> Vec<NodeId> {
///         vec![self.input]
///     }
///
///     fn replay(&self, inputs: &[&f64]) -> AdResult<ReplayResult<f64>> {
///         let x = *inputs[0];
///         # todo!()
///     }
/// }
/// ```
pub trait CheckpointRecipe<V: Differentiable>: Send + Sync {
    /// Returns the direct input nodes required to replay this operation.
    fn inputs(&self) -> Vec<NodeId>;

    /// Rebuilds the output primal and reverse rule from direct input primals.
    fn replay(&self, inputs: &[&V]) -> AdResult<ReplayResult<V>>;
}
