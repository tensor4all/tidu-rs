use tidu::{LinearizableOp, LinearizedOp, Schema, SlotSchema, Value};

#[derive(Clone, Copy)]
struct Square;

struct SquareLinearized {
    x: f64,
}

impl LinearizedOp<f64> for SquareLinearized {
    fn jvp(&self, input_tangents: &[Option<f64>]) -> tidu::AdResult<Vec<Option<f64>>> {
        Ok(vec![input_tangents[0].map(|dx| 2.0 * self.x * dx)])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<f64>],
        input_grad_mask: &[bool],
    ) -> tidu::AdResult<Vec<Option<f64>>> {
        assert_eq!(input_grad_mask, &[true]);
        let g = output_cotangents[0].unwrap_or(0.0);
        Ok(vec![Some(2.0 * self.x * g)])
    }
}

impl LinearizableOp<f64> for Square {
    type Linearized = SquareLinearized;

    fn primal(&self, inputs: &[&f64]) -> tidu::AdResult<Vec<f64>> {
        Ok(vec![inputs[0] * inputs[0]])
    }

    fn input_schema(&self, _inputs: &[&f64]) -> tidu::AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn linearize(
        &self,
        inputs: &[&f64],
        _outputs: &[f64],
    ) -> tidu::AdResult<Self::Linearized> {
        Ok(SquareLinearized { x: *inputs[0] })
    }
}

#[test]
fn linearized_op_drives_reverse_and_forward_contract() -> tidu::AdResult<()> {
    let x = Value::new(3.0_f64).requires_grad_(true);
    let y = Square.apply_one(&[&x])?;
    y.backward()?;
    assert_eq!(x.grad()?, Some(6.0));

    let lin = Square.linearize(&[x.primal()], &[*y.primal()])?;
    assert_eq!(lin.jvp(&[Some(1.0)])?, vec![Some(6.0)]);
    Ok(())
}
