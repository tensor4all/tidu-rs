use tidu::{AdResult, LinearizableOp, LinearizedOp, Schema, SlotSchema, Value};

#[derive(Clone, Copy)]
struct Square;

struct SquareLinearized {
    x: f64,
}

impl LinearizedOp<f64> for SquareLinearized {
    fn jvp(&self, input_tangents: &[Option<f64>]) -> AdResult<Vec<Option<f64>>> {
        Ok(vec![input_tangents[0].map(|dx| 2.0 * self.x * dx)])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<f64>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<f64>>> {
        assert_eq!(input_grad_mask, &[true]);
        let grad_out = output_cotangents[0].unwrap_or(0.0);
        Ok(vec![Some(2.0 * self.x * grad_out)])
    }
}

impl LinearizableOp<f64> for Square {
    type Linearized = SquareLinearized;

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

    fn linearize(&self, inputs: &[&f64], _outputs: &[f64]) -> AdResult<Self::Linearized> {
        Ok(SquareLinearized { x: *inputs[0] })
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
