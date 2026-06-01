//! Primitive AD rule contract consumed by `tidu`.
//!
//! This module defines the graph-level rule surface used by
//! [`crate::try_linearize`], [`crate::try_linear_transpose`], and eager transpose
//! helpers. It is intentionally narrower than Julia ChainRules: downstream
//! primitive sets implement JVP and transpose emission for
//! `computegraph` graph primitives.

mod ad_key;
mod ad_rule_error;
mod primitive_builder;
mod primitive_op;

pub use ad_key::{ADKey, DiffPassId};
pub use ad_rule_error::{ADRuleError, ADRuleKind, ADRuleResult};
pub(crate) use primitive_builder::FragmentPrimitiveBuilder;
pub use primitive_builder::{PrimitiveBuilder, PrimitiveValue};
pub use primitive_op::Primitive;
