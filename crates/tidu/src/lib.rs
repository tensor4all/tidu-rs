//! Tape-based reverse-mode and dual-number forward-mode AD engine.
//!
//! `tidu` provides the execution runtime for `chainrules_core` traits. It is
//! generic over the differentiable value type, so the same tape can power
//! scalar examples, tensor engines, or downstream custom value types.
//!
//! ## How it works
//!
//! Unlike eager-mode AD frameworks (e.g. PyTorch autograd), `tidu` is a
//! **low-level AD engine**: you compute forward values yourself and register
//! reverse rules on the tape. The tape then runs reverse-mode pullback to
//! accumulate gradients. This design keeps the engine generic — the same tape
//! works for scalars, tensors, or any custom type that implements
//! [`Differentiable`].
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
//! - [Scalar Reverse Mode](#scalar-reverse-mode)
//! - [Scalar Forward Mode](#scalar-forward-mode)
//! - [Scalar Hessian-Vector Product](#scalar-hessian-vector-product)
//! - [Custom Value Type](#custom-value-type)
//!
//! ## Scalar Reverse Mode
//! ```rust
//! use chainrules::powf_rrule;
//! use tidu::{AdResult, NodeId, ReverseRule, Tape};
//!
//! // Define a reverse rule for f(x) = x^exponent.
//! // The rule stores the values it needs for the backward pass.
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
//! // Create a tape and register x = 2.0 as a leaf (input requiring gradient).
//! let tape = Tape::<f64>::new();
//! let x = tape.leaf(2.0);
//!
//! // Record y = x^3 = 8.0 with its reverse rule.
//! // The first argument is the pre-computed forward value.
//! // The third argument is an optional output tangent (only needed for HVP).
//! let y = tape.record_op(
//!     8.0, // forward value: 2.0^3.0
//!     Box::new(PowfRule {
//!         input: x.node_id().unwrap(),
//!         x: 2.0,
//!         exponent: 3.0,
//!     }),
//!     None, // no output tangent (only needed for HVP)
//! );
//!
//! // Run reverse-mode pullback: dy/dx = 3 * 2^2 = 12.
//! let grads = tape.pullback(&y).unwrap();
//! assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 12.0);
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
//! ## Scalar Hessian-Vector Product
//!
//! A Hessian-vector product (HVP) computes **H*v** -- the product of the
//! Hessian of a scalar function with a tangent direction **v** -- without
//! materialising the full Hessian matrix. `tidu` achieves this via
//! forward-over-reverse mode.
//!
//! To enable HVP, implement [`ReverseRule::forward_tangents`] and
//! [`ReverseRule::pullback_with_tangents`] on your rule. The default
//! implementations return `Err(HvpNotSupported)`, so they are only
//! required when you need second-order derivatives.
//!
//! Tangent directions are passed as a `HashMap<NodeId, V::Tangent>` to
//! [`Tape::hvp`] rather than being stored on leaves or rules.
//!
//! ```rust
//! use std::collections::HashMap;
//! use tidu::{AdResult, HvpResult, NodeId, ReverseRule, Tape};
//!
//! struct SquareRuleHvp {
//!     input: NodeId,
//!     x: f64,
//! }
//!
//! impl ReverseRule<f64> for SquareRuleHvp {
//!     fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
//!         Ok(vec![(self.input, 2.0 * self.x * *cotangent)])
//!     }
//!
//!     fn inputs(&self) -> Vec<NodeId> {
//!         vec![self.input]
//!     }
//!
//!     // Forward tangent propagation: d(x^2) = 2*x*dx.
//!     fn forward_tangents<'t>(
//!         &self,
//!         input_tangents: &dyn Fn(NodeId) -> Option<&'t f64>,
//!     ) -> AdResult<Option<f64>>
//!     where
//!         f64: 't,
//!     {
//!         let dx = input_tangents(self.input).copied().unwrap_or(0.0);
//!         Ok(Some(2.0 * self.x * dx))
//!     }
//!
//!     // Forward-over-reverse: differentiates the pullback itself.
//!     // `input_tangents` provides the forward tangent for each input node.
//!     fn pullback_with_tangents<'t>(
//!         &self,
//!         cotangent: &f64,
//!         cotangent_tangent: &f64,
//!         input_tangents: &dyn Fn(NodeId) -> Option<&'t f64>,
//!     ) -> AdResult<Vec<(NodeId, f64, f64)>>
//!     where
//!         f64: 't,
//!     {
//!         let dx = input_tangents(self.input).copied().unwrap_or(0.0);
//!         Ok(vec![(
//!             self.input,
//!             2.0 * self.x * *cotangent,
//!             2.0 * dx * *cotangent + 2.0 * self.x * *cotangent_tangent,
//!         )])
//!     }
//! }
//!
//! let tape = Tape::<f64>::new();
//! let x = tape.leaf(3.0);
//! let y = tape.record_op(
//!     9.0, // forward value: 3.0^2
//!     Box::new(SquareRuleHvp {
//!         input: x.node_id().unwrap(),
//!         x: 3.0,
//!     }),
//!     None,
//! );
//! // Pass tangent direction v = 1.0 via HashMap.
//! let mut leaf_tangents = HashMap::new();
//! leaf_tangents.insert(x.node_id().unwrap(), 1.0);
//! let result: HvpResult<f64> = tape.hvp(&y, &leaf_tangents).unwrap();
//! // Gradient: d(x^2)/dx at x=3 = 6.0
//! assert_eq!(*result.gradients.get(x.node_id().unwrap()).unwrap(), 6.0);
//! // HVP: H*v = d^2(x^2)/dx^2 * 1.0 = 2.0
//! assert_eq!(*result.hvp.get(x.node_id().unwrap()).unwrap(), 2.0);
//! ```
//!
//! ## Custom Value Type
//!
//! The tape is generic over any type implementing [`Differentiable`]. This
//! example defines a simple `Vec2` and differentiates through it.
//!
//! **Note:** [`Tape::pullback`] requires a scalar loss (`num_elements() == 1`).
//! For non-scalar outputs, use [`Tape::pullback_with_seed`] and supply an
//! explicit cotangent seed.
//!
//! ```rust
//! use tidu::{AdResult, Differentiable, NodeId, ReverseRule, Tape};
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
//! struct ScaleByTwoRule {
//!     input: NodeId,
//! }
//!
//! impl ReverseRule<Vec2> for ScaleByTwoRule {
//!     fn pullback(&self, cotangent: &Vec2) -> AdResult<Vec<(NodeId, Vec2)>> {
//!         Ok(vec![(
//!             self.input,
//!             Vec2([2.0 * cotangent.0[0], 2.0 * cotangent.0[1]]),
//!         )])
//!     }
//!
//!     fn inputs(&self) -> Vec<NodeId> {
//!         vec![self.input]
//!     }
//! }
//!
//! let tape = Tape::<Vec2>::new();
//! let x = tape.leaf(Vec2([3.0, -1.0]));
//! let y = tape.record_op(
//!     Vec2([6.0, -2.0]),
//!     Box::new(ScaleByTwoRule {
//!         input: x.node_id().unwrap(),
//!     }),
//!     None, // no output tangent (only needed for HVP)
//! );
//! // Use pullback_with_seed because Vec2 is non-scalar (num_elements = 2).
//! // pullback() would return Err(NonScalarLoss).
//! let grads = tape.pullback_with_seed(&y, Vec2([1.0, -1.0])).unwrap();
//! assert_eq!(
//!     *grads.get(x.node_id().unwrap()).unwrap(),
//!     Vec2([2.0, -2.0]),
//! );
//! ```

// Re-export all core traits so downstream can depend on just `tidu`.
pub use chainrules_core::*;

mod engine;

pub use engine::{DualValue, Gradients, HvpResult, PullbackPlan, Tape, TrackedValue};
