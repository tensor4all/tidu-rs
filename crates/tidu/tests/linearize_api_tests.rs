use std::sync::atomic::{AtomicUsize, Ordering};

use tidu::{Differentiable, LinearizableOp, LinearizedOp, Schema, SlotSchema, Value};

static FREEZE_LINEARIZE_COUNT: AtomicUsize = AtomicUsize::new(0);

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

    fn linearize(&self, inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Self::Linearized> {
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

#[derive(Clone, Copy)]
struct Multiply;

struct MultiplyLinearized {
    lhs: f64,
    rhs: f64,
}

impl LinearizedOp<f64> for MultiplyLinearized {
    fn jvp(&self, input_tangents: &[Option<f64>]) -> tidu::AdResult<Vec<Option<f64>>> {
        let dl = input_tangents[0].unwrap_or(0.0);
        let dr = input_tangents[1].unwrap_or(0.0);
        Ok(vec![Some(self.rhs * dl + self.lhs * dr)])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<f64>],
        input_grad_mask: &[bool],
    ) -> tidu::AdResult<Vec<Option<f64>>> {
        assert_eq!(input_grad_mask, &[true, true]);
        let g = output_cotangents[0].unwrap_or(0.0);
        Ok(vec![Some(self.rhs * g), Some(self.lhs * g)])
    }
}

impl LinearizableOp<f64> for Multiply {
    type Linearized = MultiplyLinearized;

    fn primal(&self, inputs: &[&f64]) -> tidu::AdResult<Vec<f64>> {
        Ok(vec![*inputs[0] * *inputs[1]])
    }

    fn input_schema(&self, _inputs: &[&f64]) -> tidu::AdResult<Schema> {
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

    fn output_schema(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn linearize(&self, inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Self::Linearized> {
        Ok(MultiplyLinearized {
            lhs: *inputs[0],
            rhs: *inputs[1],
        })
    }
}

#[test]
fn linearized_vjp_handles_two_inputs_one_output() -> tidu::AdResult<()> {
    let lhs = Value::new(2.0_f64).requires_grad_(true);
    let rhs = Value::new(3.0_f64).requires_grad_(true);
    let out = Multiply.apply_one(&[&lhs, &rhs])?;

    out.backward()?;
    assert_eq!(lhs.grad()?, Some(3.0));
    assert_eq!(rhs.grad()?, Some(2.0));

    let lin = Multiply.linearize(&[lhs.primal(), rhs.primal()], &[*out.primal()])?;
    assert_eq!(lin.jvp(&[Some(1.0), Some(1.0)])?, vec![Some(5.0)]);
    Ok(())
}

#[derive(Clone, Copy)]
struct SquareWithAux;

struct SquareWithAuxLinearized {
    x: f64,
}

impl LinearizedOp<f64> for SquareWithAuxLinearized {
    fn jvp(&self, input_tangents: &[Option<f64>]) -> tidu::AdResult<Vec<Option<f64>>> {
        let dx = input_tangents[0];
        Ok(vec![dx.map(|value| 2.0 * self.x * value), None])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<f64>],
        input_grad_mask: &[bool],
    ) -> tidu::AdResult<Vec<Option<f64>>> {
        assert_eq!(input_grad_mask, &[true]);
        let g = output_cotangents[0].unwrap_or(0.0);
        assert!(output_cotangents[1].is_none());
        Ok(vec![Some(2.0 * self.x * g)])
    }
}

impl LinearizableOp<f64> for SquareWithAux {
    type Linearized = SquareWithAuxLinearized;

    fn primal(&self, inputs: &[&f64]) -> tidu::AdResult<Vec<f64>> {
        Ok(vec![*inputs[0] * *inputs[0], *inputs[0] + 1.0])
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

    fn linearize(&self, inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Self::Linearized> {
        Ok(SquareWithAuxLinearized { x: *inputs[0] })
    }
}

#[test]
fn auxiliary_outputs_stay_detached_while_diff_outputs_backpropagate() -> tidu::AdResult<()> {
    let x = Value::new(4.0_f64).requires_grad_(true);
    let outputs = SquareWithAux.apply(&[&x])?;

    assert_eq!(outputs.len(), 2);
    assert!(outputs[0].requires_grad());
    assert!(!outputs[1].requires_grad());
    assert_eq!(*outputs[1].primal(), 5.0);

    outputs[0].backward()?;
    assert_eq!(x.grad()?, Some(8.0));
    Ok(())
}

#[derive(Clone, Copy)]
struct Freeze;

struct FreezeLinearized;

impl LinearizedOp<f64> for FreezeLinearized {
    fn jvp(&self, _input_tangents: &[Option<f64>]) -> tidu::AdResult<Vec<Option<f64>>> {
        Ok(vec![None])
    }

    fn vjp(
        &self,
        _output_cotangents: &[Option<f64>],
        _input_grad_mask: &[bool],
    ) -> tidu::AdResult<Vec<Option<f64>>> {
        Ok(vec![None])
    }
}

impl LinearizableOp<f64> for Freeze {
    type Linearized = FreezeLinearized;

    fn primal(&self, inputs: &[&f64]) -> tidu::AdResult<Vec<f64>> {
        Ok(vec![*inputs[0] + 1.0])
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
                differentiable: false,
                auxiliary: false,
            }],
        })
    }

    fn linearize(&self, _inputs: &[&f64], _outputs: &[f64]) -> tidu::AdResult<Self::Linearized> {
        FREEZE_LINEARIZE_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(FreezeLinearized)
    }
}

#[test]
fn nondifferentiable_outputs_return_detached_values_without_linearizing() -> tidu::AdResult<()> {
    FREEZE_LINEARIZE_COUNT.store(0, Ordering::SeqCst);

    let x = Value::new(2.0_f64).requires_grad_(true);
    let y = Freeze.apply_one(&[&x])?;

    assert_eq!(*y.primal(), 3.0);
    assert!(!y.requires_grad());
    assert_eq!(x.grad()?, None);
    assert_eq!(FREEZE_LINEARIZE_COUNT.load(Ordering::SeqCst), 0);
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TaggedScalar {
    value: f64,
    tag: u8,
}

impl Differentiable for TaggedScalar {
    type Tangent = f64;

    fn zero_tangent(&self) -> Self::Tangent {
        0.0
    }

    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        a + *b
    }

    fn num_elements(&self) -> usize {
        1
    }

    fn seed_cotangent(&self) -> Self::Tangent {
        1.0
    }
}

#[derive(Clone, Copy)]
struct TaggedSquare;

struct TaggedSquareLinearized {
    value: f64,
}

impl LinearizedOp<TaggedScalar> for TaggedSquareLinearized {
    fn jvp(
        &self,
        input_tangents: &[Option<<TaggedScalar as Differentiable>::Tangent>],
    ) -> tidu::AdResult<Vec<Option<<TaggedScalar as Differentiable>::Tangent>>> {
        Ok(vec![input_tangents[0].map(|dx| 2.0 * self.value * dx)])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<<TaggedScalar as Differentiable>::Tangent>],
        input_grad_mask: &[bool],
    ) -> tidu::AdResult<Vec<Option<<TaggedScalar as Differentiable>::Tangent>>> {
        assert_eq!(input_grad_mask, &[true]);
        let grad_out = output_cotangents[0].unwrap_or(0.0);
        Ok(vec![Some(2.0 * self.value * grad_out)])
    }
}

impl LinearizableOp<TaggedScalar> for TaggedSquare {
    type Linearized = TaggedSquareLinearized;

    fn primal(&self, inputs: &[&TaggedScalar]) -> tidu::AdResult<Vec<TaggedScalar>> {
        Ok(vec![TaggedScalar {
            value: inputs[0].value * inputs[0].value,
            tag: inputs[0].tag,
        }])
    }

    fn input_schema(&self, _inputs: &[&TaggedScalar]) -> tidu::AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(
        &self,
        _inputs: &[&TaggedScalar],
        _outputs: &[TaggedScalar],
    ) -> tidu::AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn linearize(
        &self,
        inputs: &[&TaggedScalar],
        _outputs: &[TaggedScalar],
    ) -> tidu::AdResult<Self::Linearized> {
        Ok(TaggedSquareLinearized {
            value: inputs[0].value,
        })
    }
}

#[test]
fn linearized_ops_support_custom_differentiable_value_types() -> tidu::AdResult<()> {
    let x = Value::new(TaggedScalar { value: 3.0, tag: 7 }).requires_grad_(true);
    let y = TaggedSquare.apply_one(&[&x])?;

    assert_eq!(*y.primal(), TaggedScalar { value: 9.0, tag: 7 });

    y.backward()?;
    assert_eq!(x.grad()?, Some(6.0));

    let lin = TaggedSquare.linearize(&[x.primal()], &[*y.primal()])?;
    assert_eq!(lin.jvp(&[Some(1.5)])?, vec![Some(9.0)]);
    Ok(())
}
