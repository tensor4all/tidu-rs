use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use tidu::{
    with_ad_policy, AdExecutionPolicy, CheckpointHint, CheckpointMode, LinearizableOp,
    LinearizedOp, Schema, SlotSchema, Value,
};

#[derive(Clone)]
struct LoggingOp {
    slope: f64,
    hint: CheckpointHint,
    events: Arc<Mutex<Vec<&'static str>>>,
}

struct LoggingLinearized {
    slope: f64,
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl LoggingOp {
    fn new(slope: f64, hint: CheckpointHint) -> Self {
        Self {
            slope,
            hint,
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn snapshot(&self) -> Vec<&'static str> {
        self.events.lock().expect("event log poisoned").clone()
    }

    fn record(&self, event: &'static str) {
        self.events.lock().expect("event log poisoned").push(event);
    }
}

impl LoggingLinearized {
    fn record(&self, event: &'static str) {
        self.events.lock().expect("event log poisoned").push(event);
    }
}

impl LinearizedOp<f64> for LoggingLinearized {
    fn jvp(&self, input_tangents: &[Option<f64>]) -> tidu::AdResult<Vec<Option<f64>>> {
        self.record("jvp");
        Ok(vec![input_tangents[0].map(|dx| self.slope * dx)])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<f64>],
        input_grad_mask: &[bool],
    ) -> tidu::AdResult<Vec<Option<f64>>> {
        self.record("vjp");
        assert_eq!(input_grad_mask, &[true]);
        let grad_out = output_cotangents[0].unwrap_or(0.0);
        Ok(vec![Some(self.slope * grad_out)])
    }
}

impl LinearizableOp<f64> for LoggingOp {
    type Linearized = LoggingLinearized;

    fn primal(&self, inputs: &[&f64]) -> tidu::AdResult<Vec<f64>> {
        self.record("primal");
        Ok(vec![self.slope * *inputs[0]])
    }

    fn input_schema(&self, _inputs: &[&f64]) -> tidu::AdResult<Schema> {
        Ok(scalar_schema())
    }

    fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Schema> {
        Ok(scalar_schema())
    }

    fn linearize(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Self::Linearized> {
        self.record("linearize");
        Ok(LoggingLinearized {
            slope: self.slope,
            events: self.events.clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        self.hint
    }
}

fn scalar_schema() -> Schema {
    Schema {
        slots: vec![SlotSchema {
            differentiable: true,
            auxiliary: false,
        }],
    }
}

fn run_backward(op: &LoggingOp, mode: Option<CheckpointMode>) -> tidu::AdResult<Vec<&'static str>> {
    let run = || -> tidu::AdResult<()> {
        let x = Value::new(3.0_f64).with_requires_grad(true);
        let y = op.apply_one(&[&x])?;
        y.backward()?;
        assert_eq!(x.grad()?, Some(op.slope));
        Ok(())
    };

    match mode {
        Some(mode) => with_ad_policy(
            AdExecutionPolicy {
                checkpoint_mode: mode,
            },
            run,
        )?,
        None => run()?,
    }

    Ok(op.snapshot())
}

#[test]
fn cheap_replay_policy_switches_between_retain_and_replay() -> tidu::AdResult<()> {
    let retain = LoggingOp::new(1.0, CheckpointHint::CheapReplay);
    assert_eq!(
        run_backward(&retain, Some(CheckpointMode::Off))?,
        vec!["primal", "linearize", "vjp"]
    );

    let replay = LoggingOp::new(1.0, CheckpointHint::CheapReplay);
    assert_eq!(
        run_backward(&replay, Some(CheckpointMode::Conservative))?,
        vec!["primal", "primal", "linearize", "vjp"]
    );

    Ok(())
}

#[test]
fn expensive_and_must_retain_hints_route_to_distinct_storage_modes() -> tidu::AdResult<()> {
    let expensive_conservative = LoggingOp::new(2.0, CheckpointHint::ExpensiveReplay);
    assert_eq!(
        run_backward(&expensive_conservative, Some(CheckpointMode::Conservative))?,
        vec!["primal", "linearize", "vjp"]
    );

    let expensive_aggressive = LoggingOp::new(2.0, CheckpointHint::ExpensiveReplay);
    assert_eq!(
        run_backward(&expensive_aggressive, Some(CheckpointMode::Aggressive))?,
        vec!["primal", "primal", "linearize", "vjp"]
    );

    let must_retain = LoggingOp::new(3.0, CheckpointHint::MustRetain);
    assert_eq!(
        run_backward(&must_retain, Some(CheckpointMode::Aggressive))?,
        vec!["primal", "linearize", "vjp"]
    );

    Ok(())
}

#[test]
fn with_ad_policy_restores_default_policy_after_panic() -> tidu::AdResult<()> {
    let panic_result = catch_unwind(AssertUnwindSafe(|| {
        with_ad_policy(
            AdExecutionPolicy {
                checkpoint_mode: CheckpointMode::Conservative,
            },
            || -> tidu::AdResult<()> {
                panic!("forced panic inside policy scope");
            },
        )
    }));
    assert!(panic_result.is_err());

    let default_policy = LoggingOp::new(4.0, CheckpointHint::CheapReplay);
    assert_eq!(
        run_backward(&default_policy, None)?,
        vec!["primal", "linearize", "vjp"]
    );
    Ok(())
}
