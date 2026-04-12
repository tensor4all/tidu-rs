use computegraph::fragment::Fragment;
use computegraph::{GraphOp, LocalValId};

/// A linear fragment produced by [`crate::differentiate`] or [`crate::transpose`].
///
/// # Examples
///
/// ```ignore
/// use computegraph::resolve::resolve;
/// use std::collections::HashMap;
/// use tidu::differentiate;
///
/// let view = resolve(vec![primal_fragment]);
/// let mut ctx = ();
/// let linear = differentiate(
///     &view,
///     &[output_key],
///     &[input_key],
///     1,
///     &mut ctx,
///     &HashMap::new(),
/// );
/// assert_eq!(linear.tangent_inputs.len(), 1);
/// ```
pub struct LinearFragment<Op: GraphOp> {
    /// The fragment containing linear ops.
    pub fragment: Fragment<Op>,
    /// `(primal_input_key, tangent_local_val_id)` pairs.
    pub tangent_inputs: Vec<(Op::InputKey, LocalValId)>,
    /// Tangent outputs, aligned with the requested outputs of the source transform.
    /// `None` means the corresponding output is inactive.
    pub tangent_outputs: Vec<Option<LocalValId>>,
}
