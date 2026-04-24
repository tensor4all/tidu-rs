use std::collections::HashMap;
use std::sync::Arc;

use chainrules::{ADKey, PrimitiveOp};
use computegraph::{GlobalOpKey, GlobalValKey, GraphOp, OpMode};

use crate::{GradEdge, GradNode};

/// Eager frontend input descriptor for generic AD recording.
///
/// Downstream frontends execute the primal operation themselves, then pass one
/// `EagerValue` per concrete input to [`record_eager_op`].
///
/// # Examples
///
/// ```ignore
/// let input = tidu::EagerValue {
///     key: tensor.key.clone(),
///     node: tensor.grad_node.clone(),
///     requires_grad: tensor.requires_grad,
///     data: tensor.data.clone(),
/// };
/// ```
pub struct EagerValue<Op: GraphOp> {
    /// User-visible eager value key used for cotangent accumulation.
    pub key: GlobalValKey<Op>,
    /// Grad node that produced this value, or `None` for leaves.
    pub node: Option<Arc<GradNode<Op>>>,
    /// Whether this value participates in reverse-mode propagation.
    pub requires_grad: bool,
    /// Concrete primal data for saved forward replay.
    pub data: Arc<Op::Operand>,
}

/// Per-output trace metadata returned by [`record_eager_op`].
///
/// The downstream eager value type should embed these fields next to its own
/// concrete output data and gradient slot.
///
/// # Examples
///
/// ```ignore
/// let traces = tidu::record_eager_op(&mut keys, op, &inputs, &outputs);
/// let result = MyEagerValue::new(outputs[0].clone(), traces[0].key.clone(), traces[0].node.clone());
/// ```
pub struct EagerOutput<Op: GraphOp> {
    /// User-visible eager output key.
    pub key: GlobalValKey<Op>,
    /// Shared grad node for all outputs when any input requires gradients.
    pub node: Option<Arc<GradNode<Op>>>,
    /// Whether this output should be tracked by the downstream frontend.
    pub requires_grad: bool,
    /// Output slot within the recorded primitive.
    pub output_slot: usize,
}

/// Caller-provided source of stable eager value keys.
///
/// `tidu` wraps each fresh input key in `GlobalValKey::Input`. This guarantees
/// the aliases recorded in [`GradNode::primal_in_keys`] satisfy the current
/// single-op backward replay model.
///
/// # Examples
///
/// ```ignore
/// impl tidu::EagerKeySource<MyOp> for MyKeySource {
///     fn fresh_input_key(&mut self) -> MyInputKey {
///         self.next_key()
///     }
/// }
/// ```
pub trait EagerKeySource<Op: GraphOp> {
    /// Return a fresh input key that has not been used for another eager value.
    fn fresh_input_key(&mut self) -> Op::InputKey;
}

/// Record a concrete eager primitive execution for reverse-mode AD.
///
/// The downstream frontend is responsible for executing `op` and passing its
/// concrete `outputs`. `tidu` allocates stable input aliases and output keys,
/// builds saved forward data, constructs input edges, and returns per-output
/// metadata. Multi-output operations share one `GradNode`; each output receives
/// its own key, and `backward_dag` seeds the matching output slot by key.
///
/// # Examples
///
/// ```ignore
/// let output_data = Arc::new(execute_primal(&op, &input_data));
/// let traces = tidu::record_eager_op(
///     &mut key_source,
///     op,
///     &input_traces,
///     &[output_data.clone()],
/// );
/// ```
pub fn record_eager_op<Op: PrimitiveOp>(
    key_source: &mut impl EagerKeySource<Op>,
    op: Op,
    inputs: &[EagerValue<Op>],
    outputs: &[Arc<Op::Operand>],
) -> Vec<EagerOutput<Op>>
where
    Op::InputKey: ADKey,
{
    assert_eq!(
        inputs.len(),
        op.n_inputs(),
        "record_eager_op for {:?} expected {} inputs, got {}",
        op,
        op.n_inputs(),
        inputs.len()
    );
    assert_eq!(
        outputs.len(),
        op.n_outputs(),
        "record_eager_op for {:?} expected {} outputs, got {}",
        op,
        op.n_outputs(),
        outputs.len()
    );
    assert!(
        outputs.len() <= u8::MAX as usize + 1,
        "record_eager_op for {:?} has too many outputs for GlobalValKey: {}",
        op,
        outputs.len()
    );

    let input_aliases = fresh_value_keys(key_source, inputs.len());
    let output_keys = fresh_value_keys(key_source, outputs.len());
    let requires_grad = inputs.iter().any(|input| input.requires_grad);

    let node = requires_grad.then(|| {
        Arc::new(GradNode::new(
            op.clone(),
            input_aliases.clone(),
            output_keys.clone(),
            saved_forward_values(&op, &input_aliases, inputs, outputs),
            inputs
                .iter()
                .map(|input| {
                    GradEdge::new(input.node.clone(), input.key.clone(), input.requires_grad)
                })
                .collect(),
        ))
    });

    output_keys
        .into_iter()
        .enumerate()
        .map(|(output_slot, key)| EagerOutput {
            key,
            node: node.clone(),
            requires_grad,
            output_slot,
        })
        .collect()
}

/// Construct the derived key used to save a replayed primal output value.
///
/// # Examples
///
/// ```ignore
/// let key = tidu::derived_output_key(&op, &input_aliases, 0);
/// ```
pub fn derived_output_key<Op: GraphOp>(
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
        op: GlobalOpKey {
            primitive: op.clone(),
            inputs: input_aliases.to_vec(),
            mode: OpMode::Primal,
        },
        output_slot: output_slot as u8,
    }
}

/// Build saved forward data for one eager op.
///
/// Inputs are saved under stable input aliases. Outputs are saved under the
/// derived keys produced by replaying `op` with those aliases.
///
/// # Examples
///
/// ```ignore
/// let saved = tidu::saved_forward_values(&op, &input_aliases, &inputs, &outputs);
/// ```
pub fn saved_forward_values<Op: GraphOp>(
    op: &Op,
    input_aliases: &[GlobalValKey<Op>],
    inputs: &[EagerValue<Op>],
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
    assert!(
        outputs.len() <= u8::MAX as usize + 1,
        "saved_forward_values for {:?} has too many outputs for GlobalValKey: {}",
        op,
        outputs.len()
    );
    assert_eq!(
        input_aliases.len(),
        inputs.len(),
        "saved_forward_values for {:?} expected one alias per input, got {} aliases for {} inputs",
        op,
        input_aliases.len(),
        inputs.len()
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
    key_source: &mut impl EagerKeySource<Op>,
    count: usize,
) -> Vec<GlobalValKey<Op>> {
    (0..count)
        .map(|_| GlobalValKey::Input(key_source.fresh_input_key()))
        .collect()
}
