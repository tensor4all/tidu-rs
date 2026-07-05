//! Generic eager reverse-mode recording for `Primitive` frontends.

mod backward;
mod forward;
mod record;
mod trace;

pub use backward::{try_backward, BackwardExecutor};
pub use forward::{try_forward, ForwardExecutor, ForwardTangentInput};
pub use record::{EagerInput, EagerOutput, KeySource, RecordedGraph, Recorder};
pub use trace::Trace;
