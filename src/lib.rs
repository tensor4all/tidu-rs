//! AD graph transforms for the tensor4all v2 stack.
//!
//! This crate provides two graph-to-graph transforms:
//! [`differentiate`] for forward linearization (JVP) and [`transpose`] for
//! reverse linear flow over a linear fragment.
//!
//! # Examples
//!
//! ```ignore
//! use computegraph::resolve::resolve;
//! use tidu::{differentiate, transpose};
//!
//! let view = resolve(vec![primal_fragment]);
//! let mut ctx = ();
//! let linear = differentiate(&view, &[output_key], &[input_key], 1, &mut ctx);
//! let _transposed = transpose(&linear, &mut ctx);
//! ```

pub mod backward;
mod differentiate;
pub mod eager_transpose;
pub mod grad_node;
mod linear_fragment;
mod transpose;

pub use backward::{backward_dag, topo_sort_grad_dag, BackwardCallbacks};
pub use differentiate::differentiate;
pub use eager_transpose::eager_transpose_fragment;
pub use grad_node::{GradEdge, GradNode};
pub use linear_fragment::LinearFragment;
pub use transpose::transpose;
