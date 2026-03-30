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

#[test]
fn backward_accumulates_leaf_gradients_without_explicit_tape() -> AdResult<()> {
    let x = Value::new(2.0).requires_grad_(true);
    let y = Square.apply_one(&[&x])?;
    y.backward()?;

    assert_eq!(x.grad()?.unwrap(), 4.0);
    Ok(())
}

#[test]
fn zero_grad_clears_leaf_gradient() -> AdResult<()> {
    let x = Value::new(3.0).requires_grad_(true);
    let y = Square.apply_one(&[&x])?;
    y.backward()?;

    assert_eq!(x.grad()?.unwrap(), 6.0);
    x.zero_grad()?;
    assert!(x.grad()?.is_none());
    Ok(())
}

#[test]
fn grad_wrt_returns_functional_gradient_without_touching_leaf_cache() -> AdResult<()> {
    let x = Value::new(4.0).requires_grad_(true);
    let y = Square.apply_one(&[&x])?;

    let grads = y.grad_wrt_with_seed(1.0, &[&x])?;

    assert_eq!(grads, vec![Some(8.0)]);
    assert!(x.grad()?.is_none());
    Ok(())
}

#[test]
fn shares_reverse_graph_distinguishes_connected_and_disconnected_outputs() -> AdResult<()> {
    let x = Value::new(2.0).requires_grad_(true);
    let y = Square.apply_one(&[&x])?;
    let z = Square.apply_one(&[&x])?;
    assert!(y.shares_reverse_graph(&z));

    let other = Value::new(3.0).requires_grad_(true);
    let w = Square.apply_one(&[&other])?;
    assert!(!y.shares_reverse_graph(&w));
    Ok(())
}
