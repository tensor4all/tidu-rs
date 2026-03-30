use std::sync::atomic::{AtomicUsize, Ordering};

use tidu::{AdResult, Differentiable, Op, Schema, SlotSchema, Value};

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

#[derive(Clone, Copy, Debug, PartialEq)]
struct Vec2([f64; 2]);

impl Differentiable for Vec2 {
    type Tangent = Self;

    fn zero_tangent(&self) -> Self::Tangent {
        Self([0.0, 0.0])
    }

    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        Self([a.0[0] + b.0[0], a.0[1] + b.0[1]])
    }

    fn num_elements(&self) -> usize {
        2
    }

    fn seed_cotangent(&self) -> Self::Tangent {
        Self([1.0, 1.0])
    }
}

#[derive(Clone, Copy)]
struct ScaleByTwo;

impl Op<Vec2> for ScaleByTwo {
    type SavedBackward = ();
    type SavedJvp = ();

    fn primal(&self, inputs: &[&Vec2]) -> AdResult<Vec<Vec2>> {
        let x = inputs[0];
        Ok(vec![Vec2([2.0 * x.0[0], 2.0 * x.0[1]])])
    }

    fn input_schema(&self, _inputs: &[&Vec2]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(&self, _inputs: &[&Vec2], _outputs: &[Vec2]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn save_for_backward(
        &self,
        _inputs: &[&Vec2],
        _outputs: &[Vec2],
    ) -> AdResult<Self::SavedBackward> {
        Ok(())
    }

    fn save_for_jvp(&self, _inputs: &[&Vec2], _outputs: &[Vec2]) -> AdResult<Self::SavedJvp> {
        Ok(())
    }

    fn backward(
        &self,
        _saved: &Self::SavedBackward,
        grad_outputs: &[Option<Vec2>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<Vec2>>> {
        assert_eq!(input_grad_mask, &[true]);
        let grad_out = grad_outputs[0].unwrap();
        Ok(vec![Some(Vec2([2.0 * grad_out.0[0], 2.0 * grad_out.0[1]]))])
    }

    fn jvp(
        &self,
        _saved: &Self::SavedJvp,
        tangents: &[Option<Vec2>],
    ) -> AdResult<Vec<Option<Vec2>>> {
        Ok(vec![
            tangents[0].map(|dx| Vec2([2.0 * dx.0[0], 2.0 * dx.0[1]]))
        ])
    }
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

static SAVE_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
struct SaveOnlyWhenGradRequired;

impl Op<f64> for SaveOnlyWhenGradRequired {
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
                differentiable: true,
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

    fn jvp(&self, _saved: &Self::SavedJvp, tangents: &[Option<f64>]) -> AdResult<Vec<Option<f64>>> {
        Ok(vec![tangents[0]])
    }
}

#[test]
fn custom_function_supports_scalar_value_types() -> AdResult<()> {
    let x = Value::new(2.0).requires_grad_(true);
    let y = Square.apply_one(&[&x])?;
    y.backward()?;

    assert_eq!(x.grad()?.unwrap(), 4.0);
    Ok(())
}

#[test]
fn custom_function_supports_user_defined_differentiable_types() -> AdResult<()> {
    let x = Value::new(Vec2([3.0, -1.0])).requires_grad_(true);
    let y = ScaleByTwo.apply_one(&[&x])?;
    y.backward_with_seed(Vec2([1.0, 1.0]))?;

    assert_eq!(x.grad()?.unwrap(), Vec2([2.0, 2.0]));
    Ok(())
}

#[test]
fn custom_function_connects_separately_created_reverse_leaves() -> AdResult<()> {
    let x = Value::new(2.0).requires_grad_(true);
    let y = Value::new(3.0).requires_grad_(true);
    let z = Multiply.apply_one(&[&x, &y])?;
    z.backward()?;

    assert_eq!(x.grad()?.unwrap(), 3.0);
    assert_eq!(y.grad()?.unwrap(), 2.0);
    Ok(())
}

#[test]
fn custom_function_skips_save_for_backward_when_inputs_do_not_require_grad() -> AdResult<()> {
    SAVE_COUNTER.store(0, Ordering::SeqCst);
    let x = Value::new(2.0);

    let y = SaveOnlyWhenGradRequired.apply_one(&[&x])?;

    assert_eq!(*y.primal(), 3.0);
    assert_eq!(SAVE_COUNTER.load(Ordering::SeqCst), 0);
    Ok(())
}
