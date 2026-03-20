//! Tape-based reverse-mode and dual-number forward-mode AD engine.
//!
//! `tidu` provides the execution runtime for `chainrules_core` traits. It is
//! generic over the differentiable value type, so the same tape can power
//! scalar examples, tensor engines, or downstream custom value types.
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
//!
//! ## Scalar Forward Mode
//! ```rust
//! use chainrules::powf_frule;
//! use tidu::DualValue;
//!
//! let x = DualValue::with_tangent(2.0_f64, 1.0_f64).unwrap();
//! let (y, dy) = powf_frule(*x.primal(), 3.0, *x.tangent().unwrap());
//! assert_eq!(y, 8.0);
//! assert_eq!(dy, 12.0);
//! ```
//!
//! ## Scalar Hessian-Vector Product
//!
//! This example uses a `tidu`-specific `ReverseRule` with
//! `pullback_with_tangents`, because `chainrules` exposes scalar `frule` and
//! `rrule` helpers rather than a ready-made rule object.
//!
//! ```rust
//! use tidu::{AdResult, HvpResult, NodeId, ReverseRule, Tape};
//!
//! struct SquareRuleHvp {
//!     input: NodeId,
//!     x: f64,
//!     dx: f64,
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
//!     fn pullback_with_tangents(
//!         &self,
//!         cotangent: &f64,
//!         cotangent_tangent: &f64,
//!     ) -> AdResult<Vec<(NodeId, f64, f64)>> {
//!         Ok(vec![(
//!             self.input,
//!             2.0 * self.x * *cotangent,
//!             2.0 * self.dx * *cotangent + 2.0 * self.x * *cotangent_tangent,
//!         )])
//!     }
//! }
//!
//! let tape = Tape::<f64>::new();
//! let x = tape.leaf_with_tangent(3.0, 1.0).unwrap();
//! let y = tape.record_op(
//!     9.0,
//!     Box::new(SquareRuleHvp {
//!         input: x.node_id().unwrap(),
//!         x: 3.0,
//!         dx: 1.0,
//!     }),
//!     None,
//! );
//! let result: HvpResult<f64> = tape.hvp(&y).unwrap();
//! assert_eq!(*result.gradients.get(x.node_id().unwrap()).unwrap(), 6.0);
//! assert_eq!(*result.hvp.get(x.node_id().unwrap()).unwrap(), 2.0);
//! ```
//!
//! ## Custom Value Type
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
//!     None,
//! );
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
