use tidu::{Differentiable, LinearizableOp, LinearizedOp, Schema, SlotSchema, Value};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pair {
    x: f64,
    y: f64,
}

impl Differentiable for Pair {
    type Tangent = Self;

    fn zero_tangent(&self) -> Self::Tangent {
        Self { x: 0.0, y: 0.0 }
    }

    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        Self {
            x: a.x + b.x,
            y: a.y + b.y,
        }
    }

    fn num_elements(&self) -> usize {
        2
    }

    fn seed_cotangent(&self) -> Self::Tangent {
        Self { x: 1.0, y: 1.0 }
    }
}

#[derive(Clone, Copy)]
struct ScaleByTwo;

struct ScaleByTwoLinearized;

impl LinearizedOp<Pair> for ScaleByTwoLinearized {
    fn jvp(&self, input_tangents: &[Option<Pair>]) -> tidu::AdResult<Vec<Option<Pair>>> {
        Ok(vec![input_tangents[0].map(|dx| Pair {
            x: 2.0 * dx.x,
            y: 2.0 * dx.y,
        })])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<Pair>],
        input_grad_mask: &[bool],
    ) -> tidu::AdResult<Vec<Option<Pair>>> {
        assert_eq!(input_grad_mask, &[true]);
        let g = output_cotangents[0].unwrap_or(Pair { x: 0.0, y: 0.0 });
        Ok(vec![Some(Pair {
            x: 2.0 * g.x,
            y: 2.0 * g.y,
        })])
    }
}

impl LinearizableOp<Pair> for ScaleByTwo {
    type Linearized = ScaleByTwoLinearized;

    fn primal(&self, inputs: &[&Pair]) -> tidu::AdResult<Vec<Pair>> {
        Ok(vec![Pair {
            x: 2.0 * inputs[0].x,
            y: 2.0 * inputs[0].y,
        }])
    }

    fn input_schema(&self, _inputs: &[&Pair]) -> tidu::AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(&self, _inputs: &[&Pair], _outputs: &[Pair]) -> tidu::AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn linearize(&self, _inputs: &[&Pair], _outputs: &[Pair]) -> tidu::AdResult<Self::Linearized> {
        Ok(ScaleByTwoLinearized)
    }
}

#[test]
fn pullback_with_seed_accepts_tensor_like_outputs() {
    let x = Value::new(Pair { x: 1.0, y: -2.0 }).with_requires_grad(true);
    x.backward_with_seed(Pair { x: 3.0, y: 4.0 }).unwrap();
    assert_eq!(x.grad().unwrap(), Some(Pair { x: 3.0, y: 4.0 }));
}

#[test]
fn plain_pullback_still_requires_scalar_outputs() {
    let x = Value::new(Pair { x: 1.0, y: -2.0 }).with_requires_grad(true);
    let err = match x.backward() {
        Ok(_) => panic!("plain pullback should still reject non-scalar outputs"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        tidu::AutodiffError::NonScalarLoss { num_elements: 2 }
    ));
}

#[test]
fn seeded_reverse_mode_runs_through_linearized_vjp() {
    let x = Value::new(Pair { x: 1.0, y: -2.0 }).with_requires_grad(true);
    let y = ScaleByTwo.apply_one(&[&x]).unwrap();
    y.backward_with_seed(Pair { x: 3.0, y: 4.0 }).unwrap();
    assert_eq!(x.grad().unwrap(), Some(Pair { x: 6.0, y: 8.0 }));
}
