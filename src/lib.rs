//! AD graph transforms for the tensor4all v2 stack.
//!
//! This crate provides two graph-to-graph transforms:
//! [`linearize`] for forward linearization (JVP) and [`linear_transpose`] for
//! reverse linear flow over a linearized graph.
//! Fallible variants (`try_linearize`, `try_linear_transpose`, and
//! `eager::try_backward`) propagate [`ADRuleError`] for missing primitive or
//! extension AD rules.
//! It also provides a small [`eager`] module for downstream frontends that want
//! to record PyTorch-style eager reverse-mode traces over `Primitive` values.
//!
//! # Examples
//!
//! ```ignore
//! use computegraph::resolve::resolve;
//! use tidu::{try_linear_transpose, try_linearize};
//!
//! let view = resolve(vec![primal_fragment]);
//! let mut ctx = ();
//! let aliases = std::collections::HashMap::new();
//! let linear = try_linearize(&view, &[output_key], &[input_key], 1, &mut ctx, &aliases)?;
//! let _transposed = try_linear_transpose(&linear, &mut ctx)?;
//! # Ok::<(), tidu::ADRuleError>(())
//! ```

pub mod eager;
mod linear_transpose;
mod linearize;
mod linearized_graph;
pub mod rules;

pub use linear_transpose::{
    linear_transpose, try_linear_transpose, try_linear_transpose_with_builder,
};
pub use linearize::{linearize, try_linearize};
pub use linearized_graph::LinearizedGraph;
pub use rules::{
    ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, Primitive, PrimitiveBuilder,
    PrimitiveValue,
};
