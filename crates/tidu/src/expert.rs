//! Expert-facing low-level graph and rule APIs.
//!
//! Normal users should prefer [`crate::Value`] and [`crate::Op`].
//! This module keeps the tape-centered runtime available for engine work,
//! advanced debugging, and compatibility shims during the redesign.

pub use chainrules_core::{
    ForwardRule, NodeId, PullbackEntry, PullbackWithTangentsEntry, ReverseRule, SavePolicy,
};

pub use crate::engine::{Gradients, HvpResult, PullbackPlan, Tape, TrackedValue};
