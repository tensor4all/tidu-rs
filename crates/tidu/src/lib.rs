//! Torch-like public autograd API with a generic internal AD engine.
//!
//! `tidu` provides a value-centered public API for reverse-mode AD together
//! with dual-number forward mode. The engine stays generic over the
//! differentiable value type, so the same runtime can power scalar examples,
//! tensor engines, or downstream custom value types.
//!
//! The normal public surface is:
//! - [`Value`] for reverse-mode leaves and outputs,
//! - [`Op`] for custom high-level operations,
//! - [`DualValue`] for forward-mode JVP-style computations.
//!
//! Low-level tape/rule APIs are still available under [`expert`], but they are
//! intended for advanced engine work rather than normal downstream usage.
//!
//! **Companion crate:** The doc examples below import scalar rule helpers
//! (e.g. `powf_rrule`, `powf_frule`) from the
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
//! - [Scalar Forward Mode](#scalar-forward-mode)
//! - [Custom Value Type](#custom-value-type)
//! - [Expert API](#expert-api)
//!
//! ## Value-Centered Reverse Mode
//! ```rust
//! use tidu::{Op, Schema, SlotSchema, Value};
//!
//! #[derive(Clone, Copy)]
//! struct Cube;
//!
//! impl Op<f64> for Cube {
//!     type SavedBackward = f64;
//!     type SavedJvp = ();
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
//!     fn save_for_backward(&self, inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Self::SavedBackward> {
//!         Ok(*inputs[0])
//!     }
//!
//!     fn save_for_jvp(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Self::SavedJvp> {
//!         Ok(())
//!     }
//!
//!     fn backward(
//!         &self,
//!         saved: &Self::SavedBackward,
//!         grad_outputs: &[Option<f64>],
//!         input_grad_mask: &[bool],
//!     ) -> tidu::AdResult<Vec<Option<f64>>> {
//!         assert_eq!(input_grad_mask, &[true]);
//!         let grad_out = grad_outputs[0].unwrap_or(0.0);
//!         Ok(vec![Some(3.0 * *saved * *saved * grad_out)])
//!     }
//!
//!     fn jvp(
//!         &self,
//!         _saved: &Self::SavedJvp,
//!         tangents: &[Option<f64>],
//!     ) -> tidu::AdResult<Vec<Option<f64>>> {
//!         Ok(vec![tangents[0].map(|dx| 12.0 * dx)])
//!     }
//! }
//!
//! let x = Value::new(2.0).requires_grad_(true);
//! let y = Cube.apply_one(&[&x]).unwrap();
//! y.backward().unwrap();
//! assert_eq!(x.grad().unwrap().unwrap(), 12.0);
//! ```
//!
//! ## Scalar Forward Mode
//!
//! Forward mode propagates tangents (directional derivatives) alongside
//! the primal computation. Use [`DualValue`] to pair a value with its
//! tangent, then pass them through `_frule` helpers from `chainrules`.
//!
//! ```rust
//! use chainrules::powf_frule;
//! use tidu::DualValue;
//!
//! // Create x = 2.0 with tangent dx = 1.0 (i.e. d/dx).
//! let x = DualValue::with_tangent(2.0_f64, 1.0_f64).unwrap();
//!
//! // Compute y = x^3 and its derivative dy = 3*x^2*dx = 12.0.
//! let (y, dy) = powf_frule(*x.primal(), 3.0, *x.tangent().unwrap());
//! assert_eq!(y, 8.0);
//! assert_eq!(dy, 12.0);
//! ```
//!
//! ## Custom Value Type
//!
//! `tidu` stays generic over any type implementing [`Differentiable`]. This
//! example defines a simple `Vec2` and uses the high-level [`Op`] API.
//!
//! **Note:** [`Value::backward`] still requires a scalar loss
//! (`num_elements() == 1`). For non-scalar outputs, use
//! [`Value::backward_with_seed`] and supply an explicit cotangent seed.
//!
//! ```rust
//! use tidu::{Differentiable, Op, Schema, SlotSchema, Value};
//!
//! #[derive(Clone, Copy, Debug, PartialEq)]
//! struct Vec2([f64; 2]);
//!
//! impl Differentiable for Vec2 {
//!     type Tangent = Self;
//!
//!     fn zero_tangent(&self) -> Self::Tangent {
//!         Self([0.0, 0.0])
//!     }
//!
//!     fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
//!         Self([a.0[0] + b.0[0], a.0[1] + b.0[1]])
//!     }
//!
//!     fn num_elements(&self) -> usize {
//!         2
//!     }
//!
//!     fn seed_cotangent(&self) -> Self::Tangent {
//!         Self([1.0, 1.0])
//!     }
//! }
//!
//! #[derive(Clone, Copy)]
//! struct ScaleByTwo;
//!
//! impl Op<Vec2> for ScaleByTwo {
//!     type SavedBackward = ();
//!     type SavedJvp = ();
//!
//!     fn primal(&self, inputs: &[&Vec2]) -> tidu::AdResult<Vec<Vec2>> {
//!         let x = inputs[0];
//!         Ok(vec![Vec2([2.0 * x.0[0], 2.0 * x.0[1]])])
//!     }
//!
//!     fn input_schema(&self, _inputs: &[&Vec2]) -> tidu::AdResult<Schema> {
//!         Ok(Schema {
//!             slots: vec![SlotSchema {
//!                 differentiable: true,
//!                 auxiliary: false,
//!             }],
//!         })
//!     }
//!
//!     fn output_schema(&self, _inputs: &[&Vec2], _outputs: &[Vec2]) -> tidu::AdResult<Schema> {
//!         Ok(Schema {
//!             slots: vec![SlotSchema {
//!                 differentiable: true,
//!                 auxiliary: false,
//!             }],
//!         })
//!     }
//!
//!     fn save_for_backward(&self, _inputs: &[&Vec2], _outputs: &[Vec2]) -> tidu::AdResult<Self::SavedBackward> {
//!         Ok(())
//!     }
//!
//!     fn save_for_jvp(&self, _inputs: &[&Vec2], _outputs: &[Vec2]) -> tidu::AdResult<Self::SavedJvp> {
//!         Ok(())
//!     }
//!
//!     fn backward(
//!         &self,
//!         _saved: &Self::SavedBackward,
//!         grad_outputs: &[Option<Vec2>],
//!         input_grad_mask: &[bool],
//!     ) -> tidu::AdResult<Vec<Option<Vec2>>> {
//!         assert_eq!(input_grad_mask, &[true]);
//!         let grad_out = grad_outputs[0].unwrap();
//!         Ok(vec![Some(Vec2([
//!             2.0 * grad_out.0[0],
//!             2.0 * grad_out.0[1],
//!         ]))])
//!     }
//!
//!     fn jvp(
//!         &self,
//!         _saved: &Self::SavedJvp,
//!         tangents: &[Option<Vec2>],
//!     ) -> tidu::AdResult<Vec<Option<Vec2>>> {
//!         Ok(vec![tangents[0].map(|dx| Vec2([2.0 * dx.0[0], 2.0 * dx.0[1]]))])
//!     }
//! }
//!
//! let x = Value::new(Vec2([3.0, -1.0])).requires_grad_(true);
//! let y = ScaleByTwo.apply_one(&[&x]).unwrap();
//! y.backward_with_seed(Vec2([1.0, -1.0])).unwrap();
//! assert_eq!(x.grad().unwrap().unwrap(), Vec2([2.0, -2.0]));
//! ```
//!
//! ## Expert API
//!
//! The low-level tape-centered runtime remains available under [`expert`] for
//! advanced use cases such as custom HVP rules, graph debugging, and two-phase
//! recording.
//!
//! ```rust
//! use chainrules::powf_rrule;
//! use tidu::{AdResult, expert::{NodeId, ReverseRule, Tape}};
//!
//! struct PowfRule {
//!     input: NodeId,
//!     x: f64,
//!     exponent: f64,
//! }
//!
//! impl ReverseRule<f64> for PowfRule {
//!     fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
//!         Ok(vec![(self.input, powf_rrule(self.x, self.exponent, *cotangent))])
//!     }
//!
//!     fn inputs(&self) -> Vec<NodeId> {
//!         vec![self.input]
//!     }
//! }
//!
//! let tape = Tape::<f64>::new();
//! let x = tape.leaf(2.0);
//! let y = tape.record_op(
//!     8.0,
//!     Box::new(PowfRule {
//!         input: x.node_id().unwrap(),
//!         x: 2.0,
//!         exponent: 3.0,
//!     }),
//!     None,
//! );
//! let grads = tape.pullback(&y).unwrap();
//! assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 12.0);
//! ```

pub use chainrules_core::{AdResult, AutodiffError, Differentiable};

mod engine;
pub mod expert;
mod function;
mod reverse_graph;
mod value;

pub use engine::DualValue;
pub use function::{Op, Schema, SlotSchema};
pub use value::Value;
