use std::sync::atomic::{AtomicUsize, Ordering};

use tidu::{AdResult, Op, Schema, SlotSchema, Value};

#[derive(Clone, Copy)]
struct Square;

impl Op<f64> for Square {
    type SavedBackward = f64;
    type SavedJvp = ();

    fn primal(&self, inputs: &[&f64]) -> AdResult<Vec<f64>> {
        Ok(vec![*inputs[0] * *inputs[0]])
    }

    fn input_schema(&self, _inputs: &[&f64]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn save_for_backward(
        &self,
        inputs: &[&f64],
        _outputs: &[f64],
    ) -> AdResult<Self::SavedBackward> {
        Ok(*inputs[0])
    }

    fn save_for_jvp(&self, _inputs: &[&f64], _outputs: &[f64]) -> AdResult<Self::SavedJvp> {
        Ok(())
    }

    fn backward(
        &self,
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<f64>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<f64>>> {
        assert_eq!(input_grad_mask, &[true]);
        let grad_out = grad_outputs[0].unwrap_or(0.0);
        Ok(vec![Some(2.0 * *saved * grad_out)])
    }

    fn jvp(&self, _saved: &Self::SavedJvp, tangents: &[Option<f64>]) -> AdResult<Vec<Option<f64>>> {
        Ok(vec![tangents[0].map(|dx| 2.0 * dx)])
    }
}

static SAVE_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
struct Freeze;

impl Op<f64> for Freeze {
    type SavedBackward = ();
    type SavedJvp = ();

    fn primal(&self, inputs: &[&f64]) -> AdResult<Vec<f64>> {
        Ok(vec![*inputs[0] + 1.0])
    }

    fn input_schema(&self, _inputs: &[&f64]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: false,
                auxiliary: false,
            }],
        })
    }

    fn save_for_backward(
        &self,
        _inputs: &[&f64],
        _outputs: &[f64],
    ) -> AdResult<Self::SavedBackward> {
        SAVE_COUNTER.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn save_for_jvp(&self, _inputs: &[&f64], _outputs: &[f64]) -> AdResult<Self::SavedJvp> {
        SAVE_COUNTER.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn backward(
        &self,
        _saved: &Self::SavedBackward,
        _grad_outputs: &[Option<f64>],
        _input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<f64>>> {
        Ok(vec![Some(1.0)])
    }

    fn jvp(
        &self,
        _saved: &Self::SavedJvp,
        _tangents: &[Option<f64>],
    ) -> AdResult<Vec<Option<f64>>> {
        Ok(vec![Some(1.0)])
    }
}

#[test]
fn op_apply_one_supports_single_output_reverse_mode() -> AdResult<()> {
    let x = Value::new(2.0).requires_grad_(true);
    let y = Square.apply_one(&[&x])?;
    y.backward()?;

    assert_eq!(x.grad()?.unwrap(), 4.0);
    Ok(())
}

#[test]
fn nondiff_output_skips_saved_state_and_returns_plain_value() -> AdResult<()> {
    SAVE_COUNTER.store(0, Ordering::SeqCst);
    let x = Value::new(2.0).requires_grad_(true);

    let y = Freeze.apply_one(&[&x])?;

    assert_eq!(*y.primal(), 3.0);
    assert!(!y.requires_grad());
    assert_eq!(SAVE_COUNTER.load(Ordering::SeqCst), 0);
    Ok(())
}

#[derive(Clone, Copy)]
struct Pair;

impl Op<f64> for Pair {
    type SavedBackward = ();
    type SavedJvp = ();

    fn primal(&self, inputs: &[&f64]) -> AdResult<Vec<f64>> {
        Ok(vec![*inputs[0], *inputs[0] + 1.0])
    }

    fn input_schema(&self, _inputs: &[&f64]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                },
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                },
            ],
        })
    }

    fn save_for_backward(
        &self,
        _inputs: &[&f64],
        _outputs: &[f64],
    ) -> AdResult<Self::SavedBackward> {
        Ok(())
    }

    fn save_for_jvp(&self, _inputs: &[&f64], _outputs: &[f64]) -> AdResult<Self::SavedJvp> {
        Ok(())
    }

    fn backward(
        &self,
        _saved: &Self::SavedBackward,
        grad_outputs: &[Option<f64>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<f64>>> {
        assert_eq!(input_grad_mask, &[true]);
        let g0 = grad_outputs[0].unwrap_or(0.0);
        let g1 = grad_outputs[1].unwrap_or(0.0);
        Ok(vec![Some(g0 + g1)])
    }

    fn jvp(&self, _saved: &Self::SavedJvp, tangents: &[Option<f64>]) -> AdResult<Vec<Option<f64>>> {
        let dx = tangents[0];
        Ok(vec![dx, dx])
    }
}

#[test]
fn multi_output_apply_tracks_each_differentiable_output_slot() -> AdResult<()> {
    let x = Value::new(3.0).requires_grad_(true);
    let ys = Pair.apply(&[&x])?;

    assert_eq!(ys.len(), 2);
    assert!(ys[0].requires_grad());
    assert!(ys[1].requires_grad());

    ys[0].backward()?;
    assert_eq!(x.grad()?.unwrap(), 1.0);

    x.zero_grad()?;
    ys[1].backward()?;
    assert_eq!(x.grad()?.unwrap(), 1.0);
    Ok(())
}

#[derive(Clone, Copy)]
struct Multiply;

impl Op<f64> for Multiply {
    type SavedBackward = (f64, f64);
    type SavedJvp = ();

    fn primal(&self, inputs: &[&f64]) -> AdResult<Vec<f64>> {
        Ok(vec![*inputs[0] * *inputs[1]])
    }

    fn input_schema(&self, _inputs: &[&f64]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                },
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                },
            ],
        })
    }

    fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn save_for_backward(
        &self,
        inputs: &[&f64],
        _outputs: &[f64],
    ) -> AdResult<Self::SavedBackward> {
        Ok((*inputs[0], *inputs[1]))
    }

    fn save_for_jvp(&self, _inputs: &[&f64], _outputs: &[f64]) -> AdResult<Self::SavedJvp> {
        Ok(())
    }

    fn backward(
        &self,
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<f64>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<f64>>> {
        assert_eq!(input_grad_mask, &[true, true]);
        let grad_out = grad_outputs[0].unwrap_or(0.0);
        Ok(vec![Some(saved.1 * grad_out), Some(saved.0 * grad_out)])
    }

    fn jvp(&self, _saved: &Self::SavedJvp, tangents: &[Option<f64>]) -> AdResult<Vec<Option<f64>>> {
        Ok(vec![Some(
            tangents[0].unwrap_or(0.0) + tangents[1].unwrap_or(0.0),
        )])
    }
}

#[test]
fn op_apply_can_join_previously_independent_reverse_subgraphs() -> AdResult<()> {
    let x = Value::new(2.0).requires_grad_(true);
    let x_sq = Square.apply_one(&[&x])?;

    let y = Value::new(3.0).requires_grad_(true);
    let y_sq = Square.apply_one(&[&y])?;

    let loss = Multiply.apply_one(&[&x_sq, &y_sq])?;
    loss.backward()?;

    assert_eq!(x.grad()?.unwrap(), 36.0);
    assert_eq!(y.grad()?.unwrap(), 24.0);
    Ok(())
}

#[derive(Clone, Copy)]
struct ValueAndIndex;

impl Op<f64> for ValueAndIndex {
    type SavedBackward = ();
    type SavedJvp = ();

    fn primal(&self, inputs: &[&f64]) -> AdResult<Vec<f64>> {
        Ok(vec![*inputs[0] + 10.0, 7.0])
    }

    fn input_schema(&self, _inputs: &[&f64]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                },
                SlotSchema {
                    differentiable: false,
                    auxiliary: true,
                },
            ],
        })
    }

    fn save_for_backward(
        &self,
        _inputs: &[&f64],
        _outputs: &[f64],
    ) -> AdResult<Self::SavedBackward> {
        Ok(())
    }

    fn save_for_jvp(&self, _inputs: &[&f64], _outputs: &[f64]) -> AdResult<Self::SavedJvp> {
        Ok(())
    }

    fn backward(
        &self,
        _saved: &Self::SavedBackward,
        grad_outputs: &[Option<f64>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<f64>>> {
        assert_eq!(input_grad_mask, &[true]);
        Ok(vec![Some(grad_outputs[0].unwrap_or(0.0))])
    }

    fn jvp(&self, _saved: &Self::SavedJvp, tangents: &[Option<f64>]) -> AdResult<Vec<Option<f64>>> {
        Ok(vec![tangents[0], None])
    }
}

#[test]
fn multi_output_apply_allows_auxiliary_nondiff_outputs() -> AdResult<()> {
    let x = Value::new(5.0).requires_grad_(true);
    let ys = ValueAndIndex.apply(&[&x])?;

    assert_eq!(ys.len(), 2);
    assert!(ys[0].requires_grad());
    assert!(!ys[1].requires_grad());

    ys[0].backward()?;
    assert_eq!(x.grad()?.unwrap(), 1.0);
    Ok(())
}
