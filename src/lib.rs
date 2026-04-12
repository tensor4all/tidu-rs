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
//! use std::collections::HashMap;
//! use tidu::{differentiate, transpose};
//!
//! let view = resolve(vec![primal_fragment]);
//! let mut ctx = ();
//! let linear = differentiate(
//!     &view,
//!     &[output_key],
//!     &[input_key],
//!     1,
//!     &mut ctx,
//!     &HashMap::new(),
//! );
//! let _transposed = transpose(&linear, &mut ctx);
//! ```

mod differentiate;
mod linear_fragment;
mod transpose;

pub use differentiate::differentiate;
pub use linear_fragment::LinearFragment;
pub use transpose::transpose;
