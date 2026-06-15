//! Generic eager reverse-mode recording for `Primitive` frontends.

mod backward;
mod record;
mod trace;

pub use backward::{try_backward, BackwardExecutor};
pub use record::{EagerInput, EagerOutput, KeySource, RecordedGraph, Recorder};
pub use trace::Trace;
