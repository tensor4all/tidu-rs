//! AD graph transforms for the tensor4all v2 stack.
//!
//! This crate provides two graph-to-graph transforms:
//! [`differentiate`] for forward linearization (JVP) and [`transpose`] for
//! reverse linear flow over a linear fragment.
//! Fallible variants (`try_differentiate`, `try_transpose`, and
//! `try_backward_dag`) propagate [`ADRuleError`] for missing
//! primitive or extension AD rules.
//! It also provides eager reverse-mode AD helpers: [`record_eager_op`] builds
//! [`GradNode`] metadata around concrete frontend execution, and
//! [`backward_dag`] replays recorded nodes through caller-provided
//! [`BackwardCallbacks`].
//!
//! # Examples
//!
//! ```ignore
//! use computegraph::resolve::resolve;
//! use tidu::{try_differentiate, try_transpose};
//!
//! let view = resolve(vec![primal_fragment]);
//! let mut ctx = ();
//! let aliases = std::collections::HashMap::new();
//! let linear = try_differentiate(&view, &[output_key], &[input_key], 1, &mut ctx, &aliases)?;
//! let _transposed = try_transpose(&linear, &mut ctx)?;
//! # Ok::<(), tidu::ADRuleError>(())
//! ```

pub mod backward;
mod differentiate;
mod eager_record;
pub mod eager_transpose;
pub mod grad_node;
mod linear_fragment;
pub mod rules;
mod transpose;

pub use backward::{backward_dag, topo_sort_grad_dag, try_backward_dag, BackwardCallbacks};
pub use differentiate::{differentiate, try_differentiate};
pub use eager_record::{
    derived_output_key, record_eager_op, saved_forward_values, EagerKeySource, EagerOutput,
    EagerValue,
};
pub use eager_transpose::{eager_transpose_fragment, try_eager_transpose_fragment};
pub use grad_node::{GradEdge, GradNode};
pub use linear_fragment::LinearFragment;
pub use rules::{ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, PrimitiveOp};
pub use transpose::{transpose, try_transpose};
