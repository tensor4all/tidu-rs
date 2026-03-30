//! Tests for tidu: Tape, TrackedValue, DualValue, Gradients,
//! PullbackPlan, and pullback with dummy operations.

use std::{collections::HashMap, f64::consts::PI};

use tidu::{
    expert::{Gradients, NodeId, PullbackPlan, ReverseRule, Tape, TrackedValue},
    AdResult, AutodiffError, Differentiable, DualValue,
};

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScalarBox(f64);

impl Differentiable for ScalarBox {
    type Tangent = Self;

    fn zero_tangent(&self) -> Self::Tangent {
        Self(0.0)
    }

    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        Self(a.0 + b.0)
    }

    fn num_elements(&self) -> usize {
        1
    }

    fn seed_cotangent(&self) -> Self::Tangent {
        Self(1.0)
    }
}

// ============================================================================
// Tape creation
// ============================================================================

#[test]
fn tape_new() {
    let tape = Tape::<f64>::new();
    // A fresh tape should be able to create leaves
    let x = tape.leaf(1.0);
    assert!(x.requires_grad());
    assert_eq!(*x.value(), 1.0);
}

#[test]
fn tape_default() {
    let tape = Tape::<f64>::default();
    // Default tape behaves the same as Tape::new
    let x = tape.leaf(2.0);
    assert!(x.requires_grad());
    assert_eq!(*x.value(), 2.0);
}

#[test]
fn tape_clone_shares_state() {
    let tape1 = Tape::<f64>::new();
    let tape2 = tape1.clone();
    // Both tapes share state: leaf on one is visible via pullback on the other
    let x = tape1.leaf(2.0);
    let grads = tape2.pullback(&x).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
}

#[test]
fn same_tape_id_and_node_count_reflect_shared_graph_state() {
    let tape = Tape::<f64>::new();
    let clone = tape.clone();
    assert!(tape.same_tape(&clone));
    assert_eq!(tape.id(), clone.id());
    assert_eq!(tape.node_count(), 0);

    let _x = tape.leaf(1.0);
    assert_eq!(tape.node_count(), 1);
    assert_eq!(clone.node_count(), 1);
}

// ============================================================================
// Tape::leaf
// ============================================================================

#[test]
fn leaf_requires_grad() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(PI);
    assert!(x.requires_grad());
}

#[test]
fn leaf_has_node_id() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(PI);
    assert!(x.node_id().is_some());
    assert_eq!(x.node_id().unwrap().index(), 0);
}

#[test]
fn leaf_sequential_node_ids() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(1.0);
    let y = tape.leaf(2.0);
    assert_eq!(x.node_id().unwrap().index(), 0);
    assert_eq!(y.node_id().unwrap().index(), 1);
}

#[test]
fn leaf_no_tangent() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(1.0);
    assert!(!x.has_tangent());
    assert!(x.tangent().is_none());
}

// ============================================================================
// Tape::leaf_with_tangent
// ============================================================================

#[test]
fn leaf_with_tangent_has_tangent() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf_with_tangent(PI, 1.0).unwrap();
    assert!(x.requires_grad());
    assert!(x.has_tangent());
    assert_eq!(*x.tangent().unwrap(), 1.0);
}

#[test]
fn leaf_with_tangent_has_node_id() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf_with_tangent(PI, 1.0).unwrap();
    assert!(x.node_id().is_some());
}

// ============================================================================
// TrackedValue::new
// ============================================================================

#[test]
fn tracked_new_no_grad() {
    let x = TrackedValue::new(42.0_f64);
    assert!(!x.requires_grad());
    assert!(x.node_id().is_none());
    assert!(!x.has_tangent());
}

#[test]
fn tracked_value() {
    let x = TrackedValue::new(42.0_f64);
    assert_eq!(*x.value(), 42.0);
}

#[test]
fn tracked_into_value() {
    let x = TrackedValue::new(42.0_f64);
    assert_eq!(x.into_value(), 42.0);
}

// ============================================================================
// TrackedValue::detach
// ============================================================================

#[test]
fn detach_removes_grad() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(PI);
    assert!(x.requires_grad());
    let d = x.detach();
    assert!(!d.requires_grad());
    assert!(d.node_id().is_none());
    assert!(!d.has_tangent());
    assert_eq!(*d.value(), PI);
}

#[test]
fn detach_removes_tangent() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf_with_tangent(PI, 1.0).unwrap();
    assert!(x.has_tangent());
    let d = x.detach();
    assert!(!d.has_tangent());
}

// ============================================================================
// DualValue
// ============================================================================

#[test]
fn dual_new_no_tangent() {
    let x = DualValue::new(PI);
    assert!(!x.has_tangent());
    assert_eq!(*x.primal(), PI);
}

#[test]
fn dual_with_tangent() {
    let x = DualValue::with_tangent(PI, 1.0).unwrap();
    assert!(x.has_tangent());
    assert_eq!(*x.tangent().unwrap(), 1.0);
    assert_eq!(*x.primal(), PI);
}

#[test]
fn dual_into_parts() {
    let x = DualValue::with_tangent(PI, 1.0).unwrap();
    let (p, t) = x.into_parts();
    assert_eq!(p, PI);
    assert_eq!(t, Some(1.0));
}

#[test]
fn dual_into_parts_no_tangent() {
    let x = DualValue::new(PI);
    let (p, t) = x.into_parts();
    assert_eq!(p, PI);
    assert_eq!(t, None);
}

#[test]
fn dual_detach_tangent() {
    let x = DualValue::with_tangent(PI, 1.0).unwrap();
    let c = x.detach_tangent();
    assert!(!c.has_tangent());
    assert_eq!(*c.primal(), PI);
}

// ============================================================================
// Gradients
// ============================================================================

#[test]
fn gradients_new_empty() {
    let grads = Gradients::<f64>::new();
    assert!(grads.entries().is_empty());
}

#[test]
fn gradients_default() {
    let grads = Gradients::<f64>::default();
    assert!(grads.entries().is_empty());
}

#[test]
fn gradients_get_missing() {
    let grads = Gradients::<f64>::new();
    assert!(grads.get(NodeId::new(0)).is_none());
}

#[test]
fn gradients_accumulate_insert() {
    let mut grads = Gradients::<f64>::new();
    grads.accumulate(NodeId::new(0), 3.0).unwrap();
    assert_eq!(*grads.get(NodeId::new(0)).unwrap(), 3.0);
}

#[test]
fn gradients_accumulate_adds() {
    let mut grads = Gradients::<f64>::new();
    grads.accumulate(NodeId::new(0), 2.0).unwrap();
    grads.accumulate(NodeId::new(0), 3.0).unwrap();
    assert_eq!(*grads.get(NodeId::new(0)).unwrap(), 5.0);
}

#[test]
fn gradients_accumulate_multiple_nodes() {
    let mut grads = Gradients::<f64>::new();
    grads.accumulate(NodeId::new(0), 1.0).unwrap();
    grads.accumulate(NodeId::new(1), 2.0).unwrap();
    assert_eq!(*grads.get(NodeId::new(0)).unwrap(), 1.0);
    assert_eq!(*grads.get(NodeId::new(1)).unwrap(), 2.0);
}

#[test]
fn gradients_entries() {
    let mut grads = Gradients::<f64>::new();
    grads.accumulate(NodeId::new(0), 1.0).unwrap();
    grads.accumulate(NodeId::new(1), 2.0).unwrap();
    assert_eq!(grads.entries().len(), 2);
}

// ============================================================================
// Differentiable: num_elements and seed_cotangent for f64
// ============================================================================

#[test]
fn f64_num_elements() {
    assert_eq!(42.0_f64.num_elements(), 1);
}

#[test]
fn f64_seed_cotangent() {
    assert_eq!(42.0_f64.seed_cotangent(), 1.0_f64);
}

#[test]
fn f32_num_elements() {
    assert_eq!(42.0_f32.num_elements(), 1);
}

#[test]
fn f32_seed_cotangent() {
    assert_eq!(42.0_f32.seed_cotangent(), 1.0_f32);
}

// ============================================================================
// Pullback: leaf only (d(x)/d(x) = 1)
// ============================================================================

#[test]
fn pullback_leaf_identity() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(2.0);
    let grads = tape.pullback(&x).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
}

#[test]
fn pullback_leaf_identity_custom_type() {
    let tape = Tape::<ScalarBox>::new();
    let x = tape.leaf(ScalarBox(2.0));
    let grads = tape.pullback(&x).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), ScalarBox(1.0));
}

#[test]
fn pullback_missing_node_error() {
    let tape = Tape::<f64>::new();
    let x = TrackedValue::new(2.0);
    let result = tape.pullback(&x);
    assert!(result.is_err());
    match result {
        Err(AutodiffError::MissingNode) => {}
        Err(other) => panic!("expected MissingNode, got {other:?}"),
        Ok(_) => panic!("expected error"),
    }
}

// ============================================================================
// Pullback with dummy operations
// ============================================================================

/// Rule: y = 2*x, so dy/dx = 2
struct MultiplyBy2Rule {
    input: NodeId,
}

impl ReverseRule<f64> for MultiplyBy2Rule {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        Ok(vec![(self.input, cotangent * 2.0)])
    }
    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input]
    }
}

#[test]
fn pullback_single_op() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    // y = 2*x = 6, dy/dx = 2
    let y = tape.record_op(
        6.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    let grads = tape.pullback(&y).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 2.0);
}

#[test]
fn placeholder_then_attach_rule_supports_pullback() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    let y = tape.placeholder(6.0, Some(2.0));
    assert!(y.requires_grad());
    assert!(y.has_tangent());

    tape.attach_rule(
        y.node_id().unwrap(),
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
    )
    .unwrap();

    let grads = tape.pullback(&y).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 2.0);
}

#[test]
fn attach_rule_missing_node_errors() {
    let tape = Tape::<f64>::new();
    let err = tape
        .attach_rule(
            NodeId::new(7),
            Box::new(MultiplyBy2Rule {
                input: NodeId::new(0),
            }),
        )
        .unwrap_err();
    assert!(matches!(err, AutodiffError::MissingNode));
}

#[test]
fn tracked_existing_rehydrates_known_node() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(2.0);
    let y = tape.record_op(
        4.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        Some(0.5),
    );

    let restored = tape
        .tracked_existing(y.node_id().unwrap(), 4.0, Some(0.5))
        .unwrap();
    assert!(restored.requires_grad());
    assert_eq!(restored.node_id(), y.node_id());
    assert_eq!(*restored.value(), 4.0);
    assert_eq!(*restored.tangent().unwrap(), 0.5);
}

#[test]
fn tracked_existing_rejects_unknown_node() {
    let tape = Tape::<f64>::new();
    match tape.tracked_existing(NodeId::new(3), 1.0, None) {
        Err(AutodiffError::InvalidArgument(msg)) => assert!(msg.contains("not present")),
        Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        Ok(_) => panic!("expected error"),
    }
}

#[test]
fn pullback_chain_of_ops() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    // y = 2*x
    let y = tape.record_op(
        6.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    // z = 2*y = 4*x
    let z = tape.record_op(
        12.0,
        Box::new(MultiplyBy2Rule {
            input: y.node_id().unwrap(),
        }),
        None,
    );
    let grads = tape.pullback(&z).unwrap();
    // dz/dx = dz/dy * dy/dx = 2 * 2 = 4
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 4.0);
}

/// Rule: z = x + y, so dz/dx = 1, dz/dy = 1
struct AddRule {
    inputs: Vec<NodeId>,
}

impl ReverseRule<f64> for AddRule {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        Ok(self.inputs.iter().map(|&id| (id, *cotangent)).collect())
    }
    fn inputs(&self) -> Vec<NodeId> {
        self.inputs.clone()
    }
}

#[test]
fn pullback_multi_input() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(2.0);
    let y = tape.leaf(3.0);
    // z = x + y = 5
    let z = tape.record_op(
        5.0,
        Box::new(AddRule {
            inputs: vec![x.node_id().unwrap(), y.node_id().unwrap()],
        }),
        None,
    );
    let grads = tape.pullback(&z).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
    assert_eq!(*grads.get(y.node_id().unwrap()).unwrap(), 1.0);
}

/// Rule: y = x * x (same input used twice), so dy/dx = 2*x
/// Pullback: returns two entries for the same input, each cotangent * x
struct SquareRule {
    input: NodeId,
    saved_x: f64,
}

impl ReverseRule<f64> for SquareRule {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        // d(x*x)/dx = 2x, but expressed as two contributions of x each
        Ok(vec![
            (self.input, cotangent * self.saved_x),
            (self.input, cotangent * self.saved_x),
        ])
    }
    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input]
    }
}

#[test]
fn pullback_repeated_input() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    // y = x^2 = 9, dy/dx = 2*x = 6
    let y = tape.record_op(
        9.0,
        Box::new(SquareRule {
            input: x.node_id().unwrap(),
            saved_x: 3.0,
        }),
        None,
    );
    let grads = tape.pullback(&y).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 6.0);
}

#[test]
fn pullback_diamond_graph() {
    // x -> y1 = 2*x, x -> y2 = 2*x, z = y1 + y2
    // dz/dx = dz/dy1 * dy1/dx + dz/dy2 * dy2/dx = 1*2 + 1*2 = 4
    let tape = Tape::<f64>::new();
    let x = tape.leaf(1.0);
    let y1 = tape.record_op(
        2.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    let y2 = tape.record_op(
        2.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    let z = tape.record_op(
        4.0,
        Box::new(AddRule {
            inputs: vec![y1.node_id().unwrap(), y2.node_id().unwrap()],
        }),
        None,
    );
    let grads = tape.pullback(&z).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 4.0);
}

#[test]
fn pullback_only_returns_leaf_grads() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    let y = tape.record_op(
        6.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    let grads = tape.pullback(&y).unwrap();
    // Only leaf node (x) should have a gradient, not intermediate (y)
    assert_eq!(grads.entries().len(), 1);
    assert_eq!(grads.entries()[0].0, x.node_id().unwrap());
}

// ============================================================================
// PullbackPlan
// ============================================================================

#[test]
fn pullback_plan_build() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(2.0);
    let plan = PullbackPlan::build(&x).unwrap();
    assert_eq!(plan.loss_node().index(), 0);
}

#[test]
fn pullback_plan_build_missing_node() {
    let x = TrackedValue::new(2.0_f64);
    let result = PullbackPlan::build(&x);
    match result {
        Err(AutodiffError::MissingNode) => {}
        Err(other) => panic!("expected MissingNode, got {other:?}"),
        Ok(_) => panic!("expected error"),
    }
}

#[test]
fn pullback_plan_execute() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(2.0);
    let plan = PullbackPlan::build(&x).unwrap();
    let grads = plan.execute(&x).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
}

#[test]
fn pullback_plan_execute_with_op() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    let y = tape.record_op(
        6.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    let plan = PullbackPlan::build(&y).unwrap();
    let grads = plan.execute(&y).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 2.0);
}

// ============================================================================
// HVP with dummy operation
// ============================================================================

/// Rule: y = x^2
/// forward_tangents: output tangent = 2*x*dx where dx comes from input_tangents
/// pullback: dy = 2*x * dL
/// pullback_with_tangents:
///   cotangent of input = 2*x * dL (same as pullback)
///   cotangent tangent of input = 2*dx * dL + 2*x * dL_tangent
///   where dx comes from the input_tangents closure
struct SquareRuleHvp {
    input: NodeId,
    saved_x: f64,
}

impl ReverseRule<f64> for SquareRuleHvp {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        Ok(vec![(self.input, 2.0 * self.saved_x * cotangent)])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input]
    }

    fn forward_tangents<'t>(
        &self,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t f64>,
    ) -> AdResult<Option<f64>>
    where
        f64: 't,
    {
        // d(x^2) = 2*x*dx
        let dx = input_tangents(self.input).copied().unwrap_or(0.0);
        Ok(Some(2.0 * self.saved_x * dx))
    }

    fn pullback_with_tangents<'t>(
        &self,
        cotangent: &f64,
        cotangent_tangent: &f64,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t f64>,
    ) -> AdResult<Vec<(NodeId, f64, f64)>>
    where
        f64: 't,
    {
        let dx = input_tangents(self.input).copied().unwrap_or(0.0);
        // grad = 2*x * cotangent
        let grad = 2.0 * self.saved_x * cotangent;
        // grad_tangent = 2*dx * cotangent + 2*x * cotangent_tangent
        let grad_tangent = 2.0 * dx * cotangent + 2.0 * self.saved_x * cotangent_tangent;
        Ok(vec![(self.input, grad, grad_tangent)])
    }
}

#[test]
fn hvp_square_function() {
    // f(x) = x^2, grad = 2x, H = 2, Hv = 2v
    // x = 3.0, v = 1.0 (tangent direction)
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    // y = x^2 = 9
    let y = tape.record_op(
        9.0,
        Box::new(SquareRuleHvp {
            input: x.node_id().unwrap(),
            saved_x: 3.0,
        }),
        None,
    );
    let mut leaf_tangents = HashMap::new();
    leaf_tangents.insert(x.node_id().unwrap(), 1.0);
    let result = tape.hvp(&y, &leaf_tangents).unwrap();
    // Gradient: d(x^2)/dx = 2*3 = 6
    assert_eq!(*result.gradients.get(x.node_id().unwrap()).unwrap(), 6.0);
    // HVP: H*v = 2*1 = 2
    assert_eq!(*result.hvp.get(x.node_id().unwrap()).unwrap(), 2.0);
}

/// Rule: z = a + b (HVP-aware addition)
/// forward_tangents: sum of input tangents
/// pullback: dz/da = cotangent, dz/db = cotangent
/// pullback_with_tangents: pass cotangent and cotangent_tangent through unchanged
struct AddRuleHvp {
    inputs: Vec<NodeId>,
}

impl ReverseRule<f64> for AddRuleHvp {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        Ok(self.inputs.iter().map(|&id| (id, *cotangent)).collect())
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.inputs.clone()
    }

    fn forward_tangents<'t>(
        &self,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t f64>,
    ) -> AdResult<Option<f64>>
    where
        f64: 't,
    {
        // d(a + b) = da + db
        let mut sum = 0.0;
        let mut any = false;
        for &id in &self.inputs {
            if let Some(&t) = input_tangents(id) {
                sum += t;
                any = true;
            }
        }
        Ok(if any { Some(sum) } else { None })
    }

    fn pullback_with_tangents<'t>(
        &self,
        cotangent: &f64,
        cotangent_tangent: &f64,
        _input_tangents: &dyn Fn(NodeId) -> Option<&'t f64>,
    ) -> AdResult<Vec<(NodeId, f64, f64)>>
    where
        f64: 't,
    {
        Ok(self
            .inputs
            .iter()
            .map(|&id| (id, *cotangent, *cotangent_tangent))
            .collect())
    }
}

#[test]
fn hvp_dag_merge_point() {
    // f(x) = x^2 + x^2 = 2x^2
    // grad = 4x, H = 4, Hv = 4v
    // At x = 3.0, v = 1.0: gradient = 12, HVP = 4
    //
    // DAG:  x --> y1 = x^2 --> z = y1 + y2
    //       \--> y2 = x^2 --/
    //
    // During reverse traversal, x receives cotangent contributions from
    // both y1 and y2, hitting the Some(existing) accumulation branches.
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);

    // y1 = x^2 = 9
    let y1 = tape.record_op(
        9.0,
        Box::new(SquareRuleHvp {
            input: x.node_id().unwrap(),
            saved_x: 3.0,
        }),
        None,
    );
    // y2 = x^2 = 9
    let y2 = tape.record_op(
        9.0,
        Box::new(SquareRuleHvp {
            input: x.node_id().unwrap(),
            saved_x: 3.0,
        }),
        None,
    );
    // z = y1 + y2 = 18
    let z = tape.record_op(
        18.0,
        Box::new(AddRuleHvp {
            inputs: vec![y1.node_id().unwrap(), y2.node_id().unwrap()],
        }),
        None,
    );

    let mut leaf_tangents = HashMap::new();
    leaf_tangents.insert(x.node_id().unwrap(), 1.0);
    let result = tape.hvp(&z, &leaf_tangents).unwrap();

    // Gradient: d(2x^2)/dx = 4*3 = 12
    assert_eq!(*result.gradients.get(x.node_id().unwrap()).unwrap(), 12.0);
    // HVP: H*v = 4*1 = 4
    assert_eq!(*result.hvp.get(x.node_id().unwrap()).unwrap(), 4.0);
}

#[test]
fn hvp_missing_node_error() {
    let tape = Tape::<f64>::new();
    let x = TrackedValue::new(2.0_f64);
    let leaf_tangents = HashMap::new();
    let result = tape.hvp(&x, &leaf_tangents);
    assert!(result.is_err());
    match result {
        Err(AutodiffError::MissingNode) => {}
        Err(other) => panic!("expected MissingNode, got {other:?}"),
        Ok(_) => panic!("expected error"),
    }
}

#[test]
fn hvp_empty_tangent_map_gives_zero_hvp() {
    // f(x) = x^2, but we pass an empty leaf_tangents map.
    // Gradient should still be correct (6.0), HVP should be 0 (no tangent direction).
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    let y = tape.record_op(
        9.0,
        Box::new(SquareRuleHvp {
            input: x.node_id().unwrap(),
            saved_x: 3.0,
        }),
        None,
    );
    let leaf_tangents = HashMap::new(); // empty — all tangents are zero
    let result = tape.hvp(&y, &leaf_tangents).unwrap();
    assert_eq!(*result.gradients.get(x.node_id().unwrap()).unwrap(), 6.0);
    assert_eq!(*result.hvp.get(x.node_id().unwrap()).unwrap(), 0.0);
}

#[test]
fn hvp_partial_tangent_map() {
    // f(a, b) = a^2 + b^2, tangent only on a (v_a=1, v_b=0).
    // Gradient: [2a, 2b] = [6, 10]
    // Hessian is diag(2, 2), HVP = [2*1, 2*0] = [2, 0]
    let tape = Tape::<f64>::new();
    let a = tape.leaf(3.0);
    let b = tape.leaf(5.0);
    let a2 = tape.record_op(
        9.0,
        Box::new(SquareRuleHvp {
            input: a.node_id().unwrap(),
            saved_x: 3.0,
        }),
        None,
    );
    let b2 = tape.record_op(
        25.0,
        Box::new(SquareRuleHvp {
            input: b.node_id().unwrap(),
            saved_x: 5.0,
        }),
        None,
    );
    let z = tape.record_op(
        34.0,
        Box::new(AddRuleHvp {
            inputs: vec![a2.node_id().unwrap(), b2.node_id().unwrap()],
        }),
        None,
    );
    // Only set tangent for a, not b
    let mut leaf_tangents = HashMap::new();
    leaf_tangents.insert(a.node_id().unwrap(), 1.0);
    let result = tape.hvp(&z, &leaf_tangents).unwrap();
    assert_eq!(*result.gradients.get(a.node_id().unwrap()).unwrap(), 6.0);
    assert_eq!(*result.gradients.get(b.node_id().unwrap()).unwrap(), 10.0);
    assert_eq!(*result.hvp.get(a.node_id().unwrap()).unwrap(), 2.0);
    assert_eq!(*result.hvp.get(b.node_id().unwrap()).unwrap(), 0.0);
}

#[test]
fn free_graph_blocks_existing_nodes_until_new_activity() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(2.0);
    tape.free_graph();

    match tape.pullback(&x) {
        Err(AutodiffError::GraphFreed) => {}
        Err(other) => panic!("expected GraphFreed, got {other:?}"),
        Ok(_) => panic!("expected error"),
    }

    let y = tape.leaf(5.0);
    let grads = tape.pullback(&y).unwrap();
    assert_eq!(*grads.get(y.node_id().unwrap()).unwrap(), 1.0);
}

// ============================================================================
// record_op basics
// ============================================================================

#[test]
fn record_op_creates_tracked() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(1.0);
    let y = tape.record_op(
        2.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    assert!(y.requires_grad());
    assert!(y.node_id().is_some());
    assert_eq!(y.node_id().unwrap().index(), 1);
    assert_eq!(*y.value(), 2.0);
}

#[test]
fn record_op_with_tangent() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(1.0);
    let y = tape.record_op(
        2.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        Some(2.0), // output tangent
    );
    assert!(y.has_tangent());
    assert_eq!(*y.tangent().unwrap(), 2.0);
}
