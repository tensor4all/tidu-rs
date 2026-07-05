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
//! - [`linearize`], which builds a graph for a Jacobian-vector product (JVP)
//!   of selected outputs with respect to selected inputs.
//! - [`linear_transpose`], which transposes a linearized graph so cotangents
//!   can flow backward through the corresponding linear map.
//! - [`eager::backward`], which supports downstream eager frontends that
//!   record graph invocations and want a reverse-mode `backward()` workflow.
//!
//! These transforms propagate [`ADRuleError`] for missing primitive or
//! extension AD rules.
//!
//! See the repository `docs/` tree for the terminology guide, tutorials, and
//! implementer guides.
//!
//! # Examples
//!
//! ```ignore
//! use computegraph::resolve::resolve;
//! use tidu::{linear_transpose, linearize};
//!
//! let view = resolve(vec![source_graph]);
//! let mut ctx = ();
//! let aliases = std::collections::HashMap::new();
//! let linear = linearize(&view, &[output_key], &[input_key], 1, &mut ctx, &aliases)?;
//! let _transposed = linear_transpose(&linear, &mut ctx)?;
//! # Ok::<(), tidu::ADRuleError>(())
//! ```

pub mod eager;
mod linear_transpose;
mod linearize;
mod linearized_graph;
mod primitive_graph;
pub mod rules;

pub use linear_transpose::{linear_transpose, linear_transpose_with_builder};
pub use linearize::linearize;
pub use linearized_graph::LinearizedGraph;
pub use primitive_graph::PrimitiveGraph;
pub use rules::{
    ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, Primitive, PrimitiveBuilder,
    PrimitiveValue,
};
