use std::cell::RefCell;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointMode {
    Off,
    Conservative,
    Aggressive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdExecutionPolicy {
    pub checkpoint_mode: CheckpointMode,
}

impl Default for AdExecutionPolicy {
    fn default() -> Self {
        Self {
            checkpoint_mode: CheckpointMode::Off,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StorageDecision {
    Retain,
    Replay,
}

/// Public hint used by [`crate::LinearizableOp::checkpoint_hint`] to guide
/// retain-vs-replay policy decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointHint {
    CheapReplay,
    ExpensiveReplay,
    MustRetain,
}

thread_local! {
    static POLICY_STACK: RefCell<Vec<AdExecutionPolicy>> =
        RefCell::new(vec![AdExecutionPolicy::default()]);
}

struct PolicyScopeGuard;

impl PolicyScopeGuard {
    fn push(policy: AdExecutionPolicy) -> Self {
        POLICY_STACK.with(|stack| stack.borrow_mut().push(policy));
        Self
    }
}

impl Drop for PolicyScopeGuard {
    fn drop(&mut self) {
        POLICY_STACK.with(|stack| {
            let popped = stack.borrow_mut().pop();
            debug_assert!(popped.is_some());
        });
    }
}

pub fn with_ad_policy<R>(policy: AdExecutionPolicy, f: impl FnOnce() -> R) -> R {
    let _guard = PolicyScopeGuard::push(policy);
    f()
}

pub(crate) fn current_ad_policy() -> AdExecutionPolicy {
    POLICY_STACK.with(|stack| stack.borrow().last().copied().unwrap_or_default())
}

pub(crate) fn storage_decision(
    policy: AdExecutionPolicy,
    checkpoint_hint: CheckpointHint,
) -> StorageDecision {
    match (policy.checkpoint_mode, checkpoint_hint) {
        (_, CheckpointHint::MustRetain) => StorageDecision::Retain,
        (CheckpointMode::Off, _) => StorageDecision::Retain,
        (CheckpointMode::Conservative, CheckpointHint::CheapReplay) => StorageDecision::Replay,
        (CheckpointMode::Conservative, CheckpointHint::ExpensiveReplay) => StorageDecision::Retain,
        (CheckpointMode::Aggressive, _) => StorageDecision::Replay,
    }
}
