//! Generic eager reverse-mode recording for `PrimitiveOp` frontends.

mod backward;
mod record;
mod trace;

pub use backward::{try_backward, BackwardExecutor};
pub use record::{Input, KeySource, Output, Recorder};
pub use trace::Trace;
