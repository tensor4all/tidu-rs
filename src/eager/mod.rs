//! Generic eager reverse-mode recording for `Primitive` frontends.

mod backward;
mod record;
mod trace;

pub use backward::{backward, BackwardExecutor};
pub use record::{
    EagerInput, EagerOutput, EagerRecordError, EagerRecordResult, KeySource, RecordedGraph,
    Recorder,
};
pub use trace::Trace;
