use std::sync::atomic::{AtomicUsize, Ordering};

use tidu::{
    with_ad_policy, AdExecutionPolicy, CheckpointMode, LinearizableOp, LinearizedOp, Schema,
    SlotSchema, Value,
};

static RETAIN_LINEARIZE_COUNT: AtomicUsize = AtomicUsize::new(0);
static REPLAY_LINEARIZE_COUNT: AtomicUsize = AtomicUsize::new(0);
static REPLAY_VJP_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
struct CheapReplay;

#[derive(Clone, Copy)]
struct ExpensiveReplay;

#[derive(Clone, Copy)]
struct MustRetain;

struct CountingLinearized {
    slope: f64,
    replay_vjp_counter: &'static AtomicUsize,
}

impl LinearizedOp<f64> for CountingLinearized {
    fn jvp(&self, input_tangents: &[Option<f64>]) -> tidu::AdResult<Vec<Option<f64>>> {
        Ok(vec![input_tangents[0].map(|dx| self.slope * dx)])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<f64>],
        input_grad_mask: &[bool],
    ) -> tidu::AdResult<Vec<Option<f64>>> {
        self.replay_vjp_counter.fetch_add(1, Ordering::SeqCst);
        assert_eq!(input_grad_mask, &[true]);
        let grad_out = output_cotangents[0].unwrap_or(0.0);
        Ok(vec![Some(self.slope * grad_out)])
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

fn run_with_policy<O>(op: O, mode: CheckpointMode) -> tidu::AdResult<f64>
where
    O: LinearizableOp<f64> + Clone,
{
    let policy = AdExecutionPolicy {
        checkpoint_mode: mode,
    };
    with_ad_policy(policy, || {
        let x = Value::new(3.0_f64).requires_grad_(true);
        let y = op.apply_one(&[&x])?;
        y.backward()?;
        x.grad()?.ok_or(tidu::AutodiffError::MissingNode)
    })
}

impl LinearizableOp<f64> for CheapReplay {
    type Linearized = CountingLinearized;

    fn primal(&self, inputs: &[&f64]) -> tidu::AdResult<Vec<f64>> {
        Ok(vec![*inputs[0] + 1.0])
    }

    fn input_schema(&self, _inputs: &[&f64]) -> tidu::AdResult<Schema> {
        Ok(scalar_schema())
    }

    fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Schema> {
        Ok(scalar_schema())
    }

    fn linearize(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Self::Linearized> {
        REPLAY_LINEARIZE_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(CountingLinearized {
            slope: 1.0,
            replay_vjp_counter: &REPLAY_VJP_COUNT,
        })
    }
}

impl LinearizableOp<f64> for ExpensiveReplay {
    type Linearized = CountingLinearized;

    fn primal(&self, inputs: &[&f64]) -> tidu::AdResult<Vec<f64>> {
        Ok(vec![2.0 * *inputs[0]])
    }

    fn input_schema(&self, _inputs: &[&f64]) -> tidu::AdResult<Schema> {
        Ok(scalar_schema())
    }

    fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Schema> {
        Ok(scalar_schema())
    }

    fn linearize(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Self::Linearized> {
        RETAIN_LINEARIZE_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(CountingLinearized {
            slope: 2.0,
            replay_vjp_counter: &REPLAY_VJP_COUNT,
        })
    }
}

impl LinearizableOp<f64> for MustRetain {
    type Linearized = CountingLinearized;

    fn primal(&self, inputs: &[&f64]) -> tidu::AdResult<Vec<f64>> {
        Ok(vec![3.0 * *inputs[0]])
    }

    fn input_schema(&self, _inputs: &[&f64]) -> tidu::AdResult<Schema> {
        Ok(scalar_schema())
    }

    fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Schema> {
        Ok(scalar_schema())
    }

    fn linearize(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Self::Linearized> {
        RETAIN_LINEARIZE_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(CountingLinearized {
            slope: 3.0,
            replay_vjp_counter: &REPLAY_VJP_COUNT,
        })
    }
}

#[test]
fn checkpoint_policy_controls_retain_vs_replay() -> tidu::AdResult<()> {
    RETAIN_LINEARIZE_COUNT.store(0, Ordering::SeqCst);
    REPLAY_LINEARIZE_COUNT.store(0, Ordering::SeqCst);
    REPLAY_VJP_COUNT.store(0, Ordering::SeqCst);

    assert_eq!(run_with_policy(CheapReplay, CheckpointMode::Off)?, 1.0);
    assert_eq!(
        run_with_policy(CheapReplay, CheckpointMode::Conservative)?,
        1.0
    );
    assert_eq!(
        run_with_policy(CheapReplay, CheckpointMode::Aggressive)?,
        1.0
    );

    assert_eq!(run_with_policy(ExpensiveReplay, CheckpointMode::Off)?, 2.0);
    assert_eq!(
        run_with_policy(ExpensiveReplay, CheckpointMode::Conservative)?,
        2.0
    );
    assert_eq!(
        run_with_policy(ExpensiveReplay, CheckpointMode::Aggressive)?,
        2.0
    );

    assert_eq!(run_with_policy(MustRetain, CheckpointMode::Off)?, 3.0);
    assert_eq!(
        run_with_policy(MustRetain, CheckpointMode::Conservative)?,
        3.0
    );
    assert_eq!(
        run_with_policy(MustRetain, CheckpointMode::Aggressive)?,
        3.0
    );

    assert_eq!(RETAIN_LINEARIZE_COUNT.load(Ordering::SeqCst), 6);
    assert_eq!(REPLAY_LINEARIZE_COUNT.load(Ordering::SeqCst), 3);
    assert_eq!(REPLAY_VJP_COUNT.load(Ordering::SeqCst), 9);
    Ok(())
}
