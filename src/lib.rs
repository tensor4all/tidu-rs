//! Automatic-differentiation transforms for primitive computation graphs.
//!
//! `tidu` is for downstream crates that define primitive operations, local AD
//! rules, graph runtimes, or eager tensor frontends. It does not define tensor
//! operations itself. Instead, downstream primitive sets implement [`Primitive`],
//! then call the graph transforms here to build new primitive computation
//! graphs.
//!
//! The main transforms are:
//!
//! - [`linearize`] / [`try_linearize`], which build a graph for a
//!   Jacobian-vector product (JVP) of selected outputs with respect to selected
//!   inputs.
//! - [`linear_transpose`] / [`try_linear_transpose`], which transpose a
//!   linearized graph so cotangents can flow backward through the corresponding
//!   linear map.
//! - [`eager::try_backward`], which supports downstream eager frontends that
//!   record graph invocations and want a reverse-mode `backward()` workflow.
//!
//! Fallible variants (`try_linearize`, `try_linear_transpose`, and
//! `eager::try_backward`) propagate [`ADRuleError`] for missing primitive or
//! extension AD rules.
//!
//! See the repository `docs/` tree for the terminology guide, tutorials, and
//! implementer guides.
//!
//! # Examples
//!
//! ```ignore
//! use computegraph::resolve::resolve;
//! use tidu::{try_linear_transpose, try_linearize};
//!
//! let view = resolve(vec![source_graph]);
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
mod primitive_graph;
pub mod rules;

pub use linear_transpose::{
    linear_transpose, try_linear_transpose, try_linear_transpose_with_builder,
};
pub use linearize::{linearize, try_linearize};
pub use linearized_graph::LinearizedGraph;
pub use primitive_graph::PrimitiveGraph;
pub use rules::{
    ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, Primitive, PrimitiveBuilder,
    PrimitiveValue,
};
