//! Torch-like public autograd API with a generic linearize-first core.
//!
//! `tidu` provides a value-centered public API for reverse-mode AD built around
//! a first-order `linearize` step. The engine stays generic over the
//! differentiable value type, so the same runtime can power scalar examples,
//! tensor engines, or downstream custom value types.
//!
//! The normal public surface is:
//! - [`Value`] for reverse-mode leaves and outputs,
//! - [`LinearizableOp`] for custom high-level operations,
//! - [`LinearizedOp`] for local `jvp`/`vjp` access,
//! - [`CheckpointMode`], [`AdExecutionPolicy`], and [`with_ad_policy`] for
//!   checkpoint policy scopes,
//! - [`CheckpointHint`] for advanced retain-vs-replay hints on custom ops.
//!
//! **Companion crate:** The doc examples below import scalar rule helpers
//! (e.g. `powf_rrule`) from the
//! [`chainrules`](https://github.com/tensor4all/chainrules-rs) crate.
//! Add it alongside `tidu` in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! tidu       = { git = "https://github.com/tensor4all/tidu-rs" }
//! chainrules = { git = "https://github.com/tensor4all/chainrules-rs" }
//! ```
//!
//! ## Table of Contents
//! - [Value-Centered Reverse Mode](#value-centered-reverse-mode)
//! - [Local Directional Derivatives](#local-directional-derivatives)
//! - [Checkpoint Policy](#checkpoint-policy)
//! - [Custom Value Type](#custom-value-type)
//!
//! ## Value-Centered Reverse Mode
//! ```rust
//! use tidu::{LinearizableOp, LinearizedOp, Schema, SlotSchema, Value};
//!
//! #[derive(Clone, Copy)]
//! struct Cube;
//!
//! struct CubeLinearized {
//!     x: f64,
//! }
//!
//! impl LinearizedOp<f64> for CubeLinearized {
//!     fn jvp(&self, input_tangents: &[Option<f64>]) -> tidu::AdResult<Vec<Option<f64>>> {
//!         Ok(vec![input_tangents[0].map(|dx| 3.0 * self.x * self.x * dx)])
//!     }
//!
//!     fn vjp(
//!         &self,
//!         output_cotangents: &[Option<f64>],
//!         input_grad_mask: &[bool],
//!     ) -> tidu::AdResult<Vec<Option<f64>>> {
//!         assert_eq!(input_grad_mask, &[true]);
//!         let grad_out = output_cotangents[0].unwrap_or(0.0);
//!         Ok(vec![Some(3.0 * self.x * self.x * grad_out)])
//!     }
//! }
//!
//! impl LinearizableOp<f64> for Cube {
//!     type Linearized = CubeLinearized;
//!
//!     fn primal(&self, inputs: &[&f64]) -> tidu::AdResult<Vec<f64>> {
//!         Ok(vec![*inputs[0] * *inputs[0] * *inputs[0]])
//!     }
//!
//!     fn input_schema(&self, _inputs: &[&f64]) -> tidu::AdResult<Schema> {
//!         Ok(Schema {
//!             slots: vec![SlotSchema {
//!                 differentiable: true,
//!                 auxiliary: false,
//!             }],
//!         })
//!     }
//!
//!     fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Schema> {
//!         Ok(Schema {
//!             slots: vec![SlotSchema {
//!                 differentiable: true,
//!                 auxiliary: false,
//!             }],
//!         })
//!     }
//!
//!     fn linearize(
//!         &self,
//!         inputs: &[&f64],
//!         _outputs: &[f64],
//!     ) -> tidu::AdResult<Self::Linearized> {
//!         Ok(CubeLinearized { x: *inputs[0] })
//!     }
//! }
//!
//! let x = Value::new(2.0).requires_grad_(true);
//! let y = Cube.apply_one(&[&x]).unwrap();
//! y.backward().unwrap();
//! assert_eq!(x.grad().unwrap().unwrap(), 12.0);
//! ```
//!
//! ## Local Directional Derivatives
//!
//! ```rust
//! use tidu::{LinearizableOp, LinearizedOp, Schema, SlotSchema};
//!
//! #[derive(Clone, Copy)]
//! struct Square;
//!
//! struct SquareLinearized {
//!     x: f64,
//! }
//!
//! impl LinearizedOp<f64> for SquareLinearized {
//!     fn jvp(&self, input_tangents: &[Option<f64>]) -> tidu::AdResult<Vec<Option<f64>>> {
//!         Ok(vec![input_tangents[0].map(|dx| 2.0 * self.x * dx)])
//!     }
//!
//!     fn vjp(
//!         &self,
//!         output_cotangents: &[Option<f64>],
//!         input_grad_mask: &[bool],
//!     ) -> tidu::AdResult<Vec<Option<f64>>> {
//!         assert_eq!(input_grad_mask, &[true]);
//!         let grad_out = output_cotangents[0].unwrap_or(0.0);
//!         Ok(vec![Some(2.0 * self.x * grad_out)])
//!     }
//! }
//!
//! impl LinearizableOp<f64> for Square {
//!     type Linearized = SquareLinearized;
//!
//!     fn primal(&self, inputs: &[&f64]) -> tidu::AdResult<Vec<f64>> {
//!         Ok(vec![*inputs[0] * *inputs[0]])
//!     }
//!
//!     fn input_schema(&self, _inputs: &[&f64]) -> tidu::AdResult<Schema> {
//!         Ok(Schema {
//!             slots: vec![SlotSchema {
//!                 differentiable: true,
//!                 auxiliary: false,
//!             }],
//!         })
//!     }
//!
//!     fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Schema> {
//!         Ok(Schema {
//!             slots: vec![SlotSchema {
//!                 differentiable: true,
//!                 auxiliary: false,
//!             }],
//!         })
//!     }
//!
//!     fn linearize(
//!         &self,
//!         inputs: &[&f64],
//!         _outputs: &[f64],
//!     ) -> tidu::AdResult<Self::Linearized> {
//!         Ok(SquareLinearized { x: *inputs[0] })
//!     }
//! }
//!
//! let lin = Square.linearize(&[&3.0], &[9.0]).unwrap();
//! assert_eq!(lin.jvp(&[Some(1.0)]).unwrap(), vec![Some(6.0)]);
//! ```
//!
//! ## Checkpoint Policy
//!
//! ```rust
//! use tidu::{AdExecutionPolicy, CheckpointMode, with_ad_policy};
//!
//! let policy = AdExecutionPolicy {
//!     checkpoint_mode: CheckpointMode::Conservative,
//! };
//!
//! with_ad_policy(policy, || -> tidu::AdResult<()> {
//!     // Record and differentiate values inside this scope.
//!     Ok(())
//! })
//! .unwrap();
//! ```
//!
//! ## Custom Value Type
//!
//! `tidu` stays generic over any type implementing [`Differentiable`]. Custom
//! values participate through the same [`Value`] and [`LinearizableOp`] surface,
//! while reverse-mode seeding still happens through [`Value::backward`] or
//! [`Value::backward_with_seed`] depending on the output shape.

pub use chainrules_core::{AdResult, AutodiffError, Differentiable};

mod checkpoint;
mod graph_task;
mod linearized;
mod reverse_graph;
mod value;

pub use checkpoint::{with_ad_policy, AdExecutionPolicy, CheckpointHint, CheckpointMode};
pub use linearized::{LinearizableOp, LinearizedOp, Schema, SlotSchema};
pub use value::Value;
