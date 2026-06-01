//! AD graph transforms for the tensor4all v2 stack.
//!
//! This crate provides two graph-to-graph transforms:
//! [`differentiate`] for forward linearization (JVP) and [`transpose`] for
//! reverse linear flow over a linear fragment.
//! Fallible variants (`try_differentiate`, `try_transpose`, and
//! `eager::try_backward`) propagate [`ADRuleError`] for missing primitive or
//! extension AD rules.
//! It also provides a small [`eager`] module for downstream frontends that want
//! to record PyTorch-style eager reverse-mode traces over `Primitive` values.
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

mod differentiate;
pub mod eager;
pub mod emit;
mod linear_fragment;
pub mod rules;
mod transpose;

pub use differentiate::{differentiate, try_differentiate};
pub use linear_fragment::LinearFragment;
pub use rules::{
    ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, Primitive, PrimitiveBuilder,
    PrimitiveValue,
};
pub use transpose::{transpose, try_transpose};
