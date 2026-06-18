use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::{ADKey, ADRuleError, ADRuleKind, ADRuleResult, Primitive};
use computegraph::graph::{Graph, GraphBuilder};
use computegraph::resolve::resolve;
use computegraph::{GraphOperation, OperationRole, ValueKey, ValueRef};

use crate::LinearizedGraph;

use super::trace::{Trace, TraceEdge, TraceNode};

/// Error returned when eager graph recording receives inconsistent metadata.
///
/// These errors describe caller-visible recording contract violations. AD rule
/// failures during linearization are still reported as [`crate::ADRuleError`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EagerRecordError {
    /// A caller supplied a slice whose length must match graph metadata.
    CountMismatch {
        /// Name of the validated field.
        field: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// A caller supplied keys in an order that does not match the graph.
    KeyMismatch {
        /// Name of the validated field.
        field: &'static str,
        /// Mismatching slot.
        index: usize,
    },
    /// The graph has more outputs than eager tracing can address.
    TooManyOutputs {
        /// Supplied output count.
        actual: usize,
        /// Maximum accepted output count.
        max: usize,
    },
}

impl EagerRecordError {
    pub(crate) fn count_mismatch(field: &'static str, expected: usize, actual: usize) -> Self {
        Self::CountMismatch {
            field,
            expected,
            actual,
        }
    }

    pub(crate) fn key_mismatch(field: &'static str, index: usize) -> Self {
        Self::KeyMismatch { field, index }
    }

    fn too_many_outputs(actual: usize) -> Self {
        Self::TooManyOutputs {
            actual,
            max: u8::MAX as usize + 1,
        }
    }
}

impl fmt::Display for EagerRecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountMismatch {
                field,
                expected,
                actual,
            } => write!(f, "{field} expected {expected} entries, got {actual}"),
            Self::KeyMismatch { field, index } => {
                write!(f, "{field} does not match graph metadata at slot {index}")
            }
            Self::TooManyOutputs { actual, max } => write!(
                f,
                "eager recording supports at most {max} outputs, got {actual}"
            ),
        }
    }
}

impl Error for EagerRecordError {}

/// Result type used by eager graph recording APIs.
pub type EagerRecordResult<T> = Result<T, EagerRecordError>;

/// Graph invocation recorded as one eager reverse-mode trace node.
///
/// # Examples
///
/// ```
/// use tidu::eager::RecordedGraph;
/// use computegraph::GraphOperation;
///
/// #[derive(Clone, Debug, Hash, PartialEq, Eq)]
/// enum Op { Add }
///
/// impl GraphOperation for Op {
///     type Operand = f64;
///     type Context = ();
///     type InputKey = &'static str;
///
///     fn input_count(&self) -> usize { 2 }
///     fn output_count(&self) -> usize { 1 }
/// }
///
/// let recorded = RecordedGraph::from_primitive(Op::Add, vec!["x", "y"])?;
/// assert_eq!(recorded.input_keys(), &["x", "y"]);
/// assert_eq!(recorded.output_keys().len(), 1);
/// # Ok::<(), tidu::eager::EagerRecordError>(())
/// ```
pub struct RecordedGraph<Op: GraphOperation> {
    graph: Arc<Graph<Op>>,
    input_keys: Vec<Op::InputKey>,
    output_keys: Vec<ValueKey<Op>>,
}

impl<Op: GraphOperation> RecordedGraph<Op> {
    /// Create a recorded graph from an already-built graph and aligned keys.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use computegraph::graph::GraphBuilder;
    /// use computegraph::{GraphOperation, OperationRole, ValueRef};
    /// use tidu::eager::RecordedGraph;
    ///
    /// #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    /// enum Op { Id }
    ///
    /// impl GraphOperation for Op {
    ///     type Operand = f64;
    ///     type Context = ();
    ///     type InputKey = &'static str;
    ///
    ///     fn input_count(&self) -> usize { 1 }
    ///     fn output_count(&self) -> usize { 1 }
    /// }
    ///
    /// let mut builder = GraphBuilder::new();
    /// let x = builder.add_input("x");
    /// let y = builder.add_operation(Op::Id, vec![ValueRef::Local(x)], OperationRole::Primary);
    /// builder.set_outputs(y.clone());
    /// let graph = Arc::new(builder.build());
    /// let output_keys = y.iter().map(|id| graph.values()[*id].key.clone()).collect();
    /// let recorded = RecordedGraph::new(graph, vec!["x"], output_keys)?;
    ///
    /// assert_eq!(recorded.input_keys(), &["x"]);
    /// # Ok::<(), tidu::eager::EagerRecordError>(())
    /// ```
    pub fn new(
        graph: Arc<Graph<Op>>,
        input_keys: Vec<Op::InputKey>,
        output_keys: Vec<ValueKey<Op>>,
    ) -> EagerRecordResult<Self> {
        if graph.inputs().len() != input_keys.len() {
            return Err(EagerRecordError::count_mismatch(
                "RecordedGraph input keys",
                graph.inputs().len(),
                input_keys.len(),
            ));
        }
        if graph.outputs().len() != output_keys.len() {
            return Err(EagerRecordError::count_mismatch(
                "RecordedGraph output keys",
                graph.outputs().len(),
                output_keys.len(),
            ));
        }
        for (index, (&input_id, input_key)) in
            graph.inputs().iter().zip(input_keys.iter()).enumerate()
        {
            if graph.values()[input_id].key != ValueKey::Input(input_key.clone()) {
                return Err(EagerRecordError::key_mismatch(
                    "RecordedGraph input keys",
                    index,
                ));
            }
        }
        for (index, (&output_id, output_key)) in
            graph.outputs().iter().zip(output_keys.iter()).enumerate()
        {
            if &graph.values()[output_id].key != output_key {
                return Err(EagerRecordError::key_mismatch(
                    "RecordedGraph output keys",
                    index,
                ));
            }
        }

        Ok(Self {
            graph,
            input_keys,
            output_keys,
        })
    }

    /// Build a one-op recorded graph for an eager primitive invocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tidu::eager::RecordedGraph;
    /// use computegraph::GraphOperation;
    ///
    /// #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    /// enum Op { Add }
    ///
    /// impl GraphOperation for Op {
    ///     type Operand = f64;
    ///     type Context = ();
    ///     type InputKey = &'static str;
    ///
    ///     fn input_count(&self) -> usize { 2 }
    ///     fn output_count(&self) -> usize { 1 }
    /// }
    ///
    /// let recorded = RecordedGraph::from_primitive(Op::Add, vec!["x", "y"])?;
    /// assert_eq!(recorded.as_graph().operations().len(), 1);
    /// # Ok::<(), tidu::eager::EagerRecordError>(())
    /// ```
    pub fn from_primitive(op: Op, input_keys: Vec<Op::InputKey>) -> EagerRecordResult<Self> {
        let mut builder = GraphBuilder::new();
        let input_ids: Vec<_> = input_keys
            .iter()
            .cloned()
            .map(|key| builder.add_input(key))
            .collect();
        let output_ids = builder.add_operation(
            op,
            input_ids
                .iter()
                .map(|local_id| ValueRef::Local(*local_id))
                .collect(),
            OperationRole::Primary,
        );
        builder.set_outputs(output_ids.clone());
        let graph = Arc::new(builder.build());
        let output_keys = output_ids
            .iter()
            .map(|output_id| graph.values()[*output_id].key.clone())
            .collect();
        Self::new(graph, input_keys, output_keys)
    }

    /// Borrow the recorded graph.
    pub fn as_graph(&self) -> &Graph<Op> {
        &self.graph
    }

    /// Graph input keys aligned with eager input edges.
    pub fn input_keys(&self) -> &[Op::InputKey] {
        &self.input_keys
    }

    /// Graph output keys aligned with eager output slots.
    pub fn output_keys(&self) -> &[ValueKey<Op>] {
        &self.output_keys
    }
}

impl<Op: Primitive> RecordedGraph<Op>
where
    Op::InputKey: ADKey,
{
    pub(crate) fn linearize(
        &self,
        output_slots: &[usize],
        ctx: &mut Op::ADContext,
    ) -> ADRuleResult<LinearizedGraph<Op>> {
        let mut selected_outputs = Vec::with_capacity(output_slots.len());
        for &slot in output_slots {
            let Some(output_key) = self.output_keys.get(slot).cloned() else {
                return Err(ADRuleError::invalid_input(
                    "tidu::eager::RecordedGraph",
                    ADRuleKind::Jvp,
                    format!(
                        "requested output slot {slot}, but graph has {} outputs",
                        self.output_keys.len()
                    ),
                ));
            };
            selected_outputs.push(output_key);
        }
        let view = resolve(vec![Arc::clone(&self.graph)]);
        let aliases = HashMap::new();
        crate::linearize(&view, &selected_outputs, &self.input_keys, 0, ctx, &aliases)
    }
}

/// Input descriptor for recording one eager graph invocation.
pub struct EagerInput<Op: GraphOperation> {
    /// User-visible eager value key used for cotangent accumulation.
    pub key: ValueKey<Op>,
    /// Trace node that produced this value, or `None` for leaves.
    pub trace: Option<Trace<Op>>,
    /// Whether this value participates in reverse-mode propagation.
    pub requires_grad: bool,
    /// Concrete primal data for saved forward replay.
    pub data: Arc<Op::Operand>,
}

/// Per-output trace metadata returned by [`Recorder::record_graph`].
pub struct EagerOutput<Op: GraphOperation> {
    /// User-visible eager output key.
    pub key: ValueKey<Op>,
    /// Shared trace node for all outputs when any input requires gradients.
    pub trace: Option<Trace<Op>>,
    /// Whether this output should be tracked by the downstream frontend.
    pub requires_grad: bool,
    /// Output slot within the recorded graph invocation.
    pub output_slot: usize,
}

/// Caller-provided source of stable eager value keys.
pub trait KeySource<Op: GraphOperation> {
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

    /// Return fresh graph input keys for one eager graph invocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tidu::eager::{KeySource, Recorder};
    /// use computegraph::GraphOperation;
    ///
    /// #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    /// enum Op { Id }
    ///
    /// impl GraphOperation for Op {
    ///     type Operand = f64;
    ///     type Context = ();
    ///     type InputKey = usize;
    ///
    ///     fn input_count(&self) -> usize { 1 }
    ///     fn output_count(&self) -> usize { 1 }
    /// }
    ///
    /// struct Keys(usize);
    ///
    /// impl KeySource<Op> for Keys {
    ///     fn fresh_input_key(&mut self) -> usize {
    ///         let key = self.0;
    ///         self.0 += 1;
    ///         key
    ///     }
    /// }
    ///
    /// let mut recorder = Recorder::new(Keys(0));
    /// assert_eq!(recorder.fresh_input_keys::<Op>(2), vec![0, 1]);
    /// ```
    pub fn fresh_input_keys<Op>(&mut self, count: usize) -> Vec<Op::InputKey>
    where
        Op: GraphOperation,
        K: KeySource<Op>,
    {
        (0..count)
            .map(|_| self.key_source.fresh_input_key())
            .collect()
    }

    /// Record a concrete eager graph invocation for reverse-mode AD.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use std::sync::Arc;
    /// use computegraph::{GraphOperation, LocalValueId, OperationRole, ValueKey};
    /// use tidu::{
    ///     ADKey, DiffPassId, Primitive, PrimitiveBuilder, PrimitiveValue,
    /// };
    /// use tidu::eager::{EagerInput, KeySource, RecordedGraph, Recorder};
    ///
    /// #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    /// enum Key {
    ///     User(&'static str),
    ///     Generated(usize),
    ///     Tangent(Box<Key>, DiffPassId),
    /// }
    ///
    /// impl ADKey for Key {
    ///     fn tangent_of(&self, pass: DiffPassId) -> Self {
    ///         Self::Tangent(Box::new(self.clone()), pass)
    ///     }
    /// }
    ///
    /// #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    /// enum Op { Id }
    ///
    /// impl GraphOperation for Op {
    ///     type Operand = f64;
    ///     type Context = ();
    ///     type InputKey = Key;
    ///
    ///     fn input_count(&self) -> usize { 1 }
    ///     fn output_count(&self) -> usize { 1 }
    /// }
    ///
    /// impl Primitive for Op {
    ///     type ADContext = ();
    ///     fn add() -> Self { Self::Id }
    ///
    ///     fn jvp_rule(
    ///         &self,
    ///         _builder: &mut impl PrimitiveBuilder<Self>,
    ///         _primal_in: &[ValueKey<Self>],
    ///         _primal_out: &[ValueKey<Self>],
    ///         tangent_in: &[Option<LocalValueId>],
    ///         _ctx: &mut (),
    ///     ) -> tidu::ADRuleResult<Vec<Option<LocalValueId>>> {
    ///         Ok(vec![tangent_in[0]])
    ///     }
    ///
    ///     fn transpose_rule(
    ///         &self,
    ///         _builder: &mut impl PrimitiveBuilder<Self>,
    ///         cotangent_out: &[Option<LocalValueId>],
    ///         _inputs: &[PrimitiveValue<Self>],
    ///         _role: &OperationRole,
    ///         _ctx: &mut (),
    ///     ) -> tidu::ADRuleResult<Vec<Option<LocalValueId>>> {
    ///         Ok(vec![cotangent_out[0]])
    ///     }
    /// }
    ///
    /// struct Keys(usize);
    /// impl KeySource<Op> for Keys {
    ///     fn fresh_input_key(&mut self) -> Key {
    ///         let key = Key::Generated(self.0);
    ///         self.0 += 1;
    ///         key
    ///     }
    /// }
    ///
    /// let mut recorder = Recorder::new(Keys(0));
    /// let graph_inputs = recorder.fresh_input_keys::<Op>(1);
    /// let graph = RecordedGraph::from_primitive(Op::Id, graph_inputs)?;
    /// let input = EagerInput {
    ///     key: ValueKey::Input(Key::User("x")),
    ///     trace: None,
    ///     requires_grad: true,
    ///     data: Arc::new(2.0),
    /// };
    ///
    /// let outputs = recorder.record_graph(
    ///     graph,
    ///     &[input],
    ///     &[Arc::new(2.0)],
    ///     HashMap::new(),
    /// )?;
    /// assert!(outputs[0].trace.is_some());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn record_graph<Op>(
        &mut self,
        graph: RecordedGraph<Op>,
        inputs: &[EagerInput<Op>],
        outputs: &[Arc<Op::Operand>],
        retained_values: HashMap<ValueKey<Op>, Arc<Op::Operand>>,
    ) -> EagerRecordResult<Vec<EagerOutput<Op>>>
    where
        Op: Primitive,
        Op::InputKey: ADKey,
        K: KeySource<Op>,
    {
        if inputs.len() != graph.input_keys().len() {
            return Err(EagerRecordError::count_mismatch(
                "Recorder::record_graph inputs",
                graph.input_keys().len(),
                inputs.len(),
            ));
        }
        if outputs.len() != graph.output_keys().len() {
            return Err(EagerRecordError::count_mismatch(
                "Recorder::record_graph outputs",
                graph.output_keys().len(),
                outputs.len(),
            ));
        }
        if outputs.len() > u8::MAX as usize + 1 {
            return Err(EagerRecordError::too_many_outputs(outputs.len()));
        }

        let output_keys = fresh_value_keys(&mut self.key_source, outputs.len());
        let requires_grad = inputs.iter().any(|input| input.requires_grad);

        let trace = if requires_grad {
            let saved_data = saved_graph_values(&graph, inputs, &retained_values);
            Some(Trace::new(Arc::new(TraceNode::new(
                graph,
                output_keys.clone(),
                saved_data,
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
            )?)))
        } else {
            None
        };

        Ok(output_keys
            .into_iter()
            .enumerate()
            .map(|(output_slot, key)| EagerOutput {
                key,
                trace: trace.clone(),
                requires_grad,
                output_slot,
            })
            .collect())
    }
}

fn saved_graph_values<Op: GraphOperation>(
    graph: &RecordedGraph<Op>,
    inputs: &[EagerInput<Op>],
    retained_values: &HashMap<ValueKey<Op>, Arc<Op::Operand>>,
) -> HashMap<ValueKey<Op>, Arc<Op::Operand>> {
    let mut saved = HashMap::with_capacity(inputs.len() + retained_values.len());
    for (input_key, input) in graph.input_keys().iter().zip(inputs.iter()) {
        saved.insert(ValueKey::Input(input_key.clone()), input.data.clone());
    }
    for (key, value) in retained_values {
        saved.insert(key.clone(), value.clone());
    }
    saved
}

fn fresh_value_keys<Op: GraphOperation>(
    key_source: &mut impl KeySource<Op>,
    count: usize,
) -> Vec<ValueKey<Op>> {
    (0..count)
        .map(|_| ValueKey::Input(key_source.fresh_input_key()))
        .collect()
}
