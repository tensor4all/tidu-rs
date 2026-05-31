use std::collections::HashMap;
use std::sync::Arc;

use crate::{ADKey, PrimitiveOp};
use computegraph::{GlobalOpKey, GlobalValKey, GraphOp, OpMode};

use super::trace::{Trace, TraceEdge, TraceNode};

/// Input descriptor for recording one eager primitive execution.
pub struct Input<Op: GraphOp> {
    /// User-visible eager value key used for cotangent accumulation.
    pub key: GlobalValKey<Op>,
    /// Trace node that produced this value, or `None` for leaves.
    pub trace: Option<Trace<Op>>,
    /// Whether this value participates in reverse-mode propagation.
    pub requires_grad: bool,
    /// Concrete primal data for saved forward replay.
    pub data: Arc<Op::Operand>,
}

/// Per-output trace metadata returned by [`Recorder::record`].
pub struct Output<Op: GraphOp> {
    /// User-visible eager output key.
    pub key: GlobalValKey<Op>,
    /// Shared trace node for all outputs when any input requires gradients.
    pub trace: Option<Trace<Op>>,
    /// Whether this output should be tracked by the downstream frontend.
    pub requires_grad: bool,
    /// Output slot within the recorded primitive.
    pub output_slot: usize,
}

/// Caller-provided source of stable eager value keys.
pub trait KeySource<Op: GraphOp> {
    /// Return a fresh input key that has not been used for another eager value.
    fn fresh_input_key(&mut self) -> Op::InputKey;
}

/// Stateful eager operation recorder.
pub struct Recorder<K> {
    key_source: K,
}

impl<K> Recorder<K> {
    /// Create a recorder from a downstream key source.
    pub fn new(key_source: K) -> Self {
        Self { key_source }
    }

    /// Borrow the underlying key source.
    pub fn key_source_mut(&mut self) -> &mut K {
        &mut self.key_source
    }

    /// Return the underlying key source.
    pub fn into_key_source(self) -> K {
        self.key_source
    }

    /// Record a concrete eager primitive execution for reverse-mode AD.
    pub fn record<Op>(
        &mut self,
        op: Op,
        inputs: &[Input<Op>],
        outputs: &[Arc<Op::Operand>],
    ) -> Vec<Output<Op>>
    where
        Op: PrimitiveOp,
        Op::InputKey: ADKey,
        K: KeySource<Op>,
    {
        assert_eq!(
            inputs.len(),
            op.n_inputs(),
            "Recorder::record for {:?} expected {} inputs, got {}",
            op,
            op.n_inputs(),
            inputs.len()
        );
        assert_eq!(
            outputs.len(),
            op.n_outputs(),
            "Recorder::record for {:?} expected {} outputs, got {}",
            op,
            op.n_outputs(),
            outputs.len()
        );
        assert!(
            outputs.len() <= u8::MAX as usize + 1,
            "Recorder::record for {:?} has too many outputs for GlobalValKey: {}",
            op,
            outputs.len()
        );

        let input_aliases = fresh_value_keys(&mut self.key_source, inputs.len());
        let output_keys = fresh_value_keys(&mut self.key_source, outputs.len());
        let requires_grad = inputs.iter().any(|input| input.requires_grad);

        let trace = requires_grad.then(|| {
            Trace::new(Arc::new(TraceNode::new(
                op.clone(),
                input_aliases.clone(),
                output_keys.clone(),
                saved_forward_values(&op, &input_aliases, inputs, outputs),
                inputs
                    .iter()
                    .map(|input| {
                        TraceEdge::new(
                            input.trace.as_ref().map(|trace| trace.node().clone()),
                            input.key.clone(),
                            input.requires_grad,
                        )
                    })
                    .collect(),
            )))
        });

        output_keys
            .into_iter()
            .enumerate()
            .map(|(output_slot, key)| Output {
                key,
                trace: trace.clone(),
                requires_grad,
                output_slot,
            })
            .collect()
    }
}

fn derived_output_key<Op: GraphOp>(
    op: &Op,
    input_aliases: &[GlobalValKey<Op>],
    output_slot: usize,
) -> GlobalValKey<Op> {
    assert!(
        output_slot <= u8::MAX as usize,
        "output slot {} is too large for GlobalValKey",
        output_slot
    );

    GlobalValKey::Derived {
        op: Arc::new(GlobalOpKey::new(
            op.clone(),
            input_aliases.to_vec(),
            OpMode::Primal,
        )),
        output_slot: output_slot as u8,
    }
}

fn saved_forward_values<Op: GraphOp>(
    op: &Op,
    input_aliases: &[GlobalValKey<Op>],
    inputs: &[Input<Op>],
    outputs: &[Arc<Op::Operand>],
) -> HashMap<GlobalValKey<Op>, Arc<Op::Operand>> {
    assert_eq!(
        input_aliases.len(),
        op.n_inputs(),
        "saved_forward_values for {:?} expected {} input aliases, got {}",
        op,
        op.n_inputs(),
        input_aliases.len()
    );
    assert_eq!(
        inputs.len(),
        op.n_inputs(),
        "saved_forward_values for {:?} expected {} inputs, got {}",
        op,
        op.n_inputs(),
        inputs.len()
    );
    assert_eq!(
        outputs.len(),
        op.n_outputs(),
        "saved_forward_values for {:?} expected {} outputs, got {}",
        op,
        op.n_outputs(),
        outputs.len()
    );
    assert!(
        input_aliases
            .iter()
            .all(|key| matches!(key, GlobalValKey::Input(_))),
        "saved_forward_values for {:?} requires GlobalValKey::Input aliases",
        op
    );

    let mut saved = HashMap::with_capacity(inputs.len() + outputs.len());
    for (key, input) in input_aliases.iter().zip(inputs.iter()) {
        saved.insert(key.clone(), input.data.clone());
    }
    for (slot, output) in outputs.iter().enumerate() {
        saved.insert(derived_output_key(op, input_aliases, slot), output.clone());
    }
    saved
}

fn fresh_value_keys<Op: GraphOp>(
    key_source: &mut impl KeySource<Op>,
    count: usize,
) -> Vec<GlobalValKey<Op>> {
    (0..count)
        .map(|_| GlobalValKey::Input(key_source.fresh_input_key()))
        .collect()
}
