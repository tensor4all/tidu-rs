use super::{ADKey, ADRuleResult, PrimitiveBuilder, PrimitiveValue};
use computegraph::{GraphOperation, LocalValueId, OperationRole, ValueKey};

/// Extends `GraphOperation` with primitive JVP and transpose rules for AD.
///
/// - `jvp_rule` is called by [`crate::linearize`]
/// - `transpose_rule` is called by [`crate::linear_transpose`]
///
/// Both methods add new primitive applications through a [`PrimitiveBuilder`]. The downstream
/// implementor is responsible for ensuring closure: every op emitted must also
/// implement `Primitive`.
///
/// # Examples
///
/// ```
/// use computegraph::{ValueKey, GraphOperation, LocalValueId, OperationRole};
/// use tidu::{ADKey, DiffPassId, Primitive, PrimitiveBuilder, PrimitiveValue};
///
/// #[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// enum Key { Base(String), Tan(Box<Key>, DiffPassId) }
///
/// impl ADKey for Key {
///     fn tangent_of(&self, p: DiffPassId) -> Self { Key::Tan(Box::new(self.clone()), p) }
/// }
///
/// #[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// struct AddOp;
///
/// impl GraphOperation for AddOp {
///     type Operand = f64;
///     type Context = ();
///     type InputKey = Key;
///     fn input_count(&self) -> usize { 2 }
///     fn output_count(&self) -> usize { 1 }
/// }
///
/// impl Primitive for AddOp {
///     type ADContext = ();
///
///     fn add() -> Self { AddOp }
///     fn jvp_rule(
///         &self, _b: &mut impl PrimitiveBuilder<Self>,
///         _pi: &[ValueKey<Self>], _po: &[ValueKey<Self>],
///         t: &[Option<LocalValueId>],
///         _ctx: &mut (),
///     ) -> tidu::ADRuleResult<Vec<Option<LocalValueId>>> {
///         Ok(vec![t[0].or(t[1])])
///     }
///     fn transpose_rule(
///         &self, _builder: &mut impl PrimitiveBuilder<Self>,
///         ct: &[Option<LocalValueId>], _i: &[PrimitiveValue<Self>], _m: &OperationRole,
///         _ctx: &mut (),
///     ) -> tidu::ADRuleResult<Vec<Option<LocalValueId>>> {
///         Ok(vec![ct[0], ct[0]])
///     }
/// }
/// ```
pub trait Primitive: GraphOperation
where
    Self::InputKey: ADKey,
{
    /// Runtime AD context threaded through linearization and transposition.
    ///
    /// This can carry information such as concrete shapes or guard decisions
    /// that influence how AD rules emit graph structure.
    type ADContext: Default;

    /// Returns the addition operation used for cotangent accumulation
    /// in [`crate::linear_transpose`]. When multiple cotangents flow to the same
    /// `ValueKey`, `linear_transpose` emits `Op::add()` nodes to sum them.
    fn add() -> Self
    where
        Self: Sized;

    /// Emit the JVP rule for this primitive.
    ///
    /// Must be linear in tangent inputs. May reference primal inputs/outputs
    /// through `External(ValueKey)`. Must emit ops in `OperationRole::Linearized`.
    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        primal_inputs: &[ValueKey<Self>],
        primal_outputs: &[ValueKey<Self>],
        tangent_inputs: &[Option<LocalValueId>],
        ctx: &mut Self::ADContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>
    where
        Self: Sized;

    /// Emit the transpose rule for this linear primitive.
    ///
    /// Receives cotangent outputs and produces cotangent inputs.
    /// Must only emit ops that themselves implement `Primitive`.
    fn transpose_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_outputs: &[Option<LocalValueId>],
        inputs: &[PrimitiveValue<Self>],
        role: &OperationRole,
        ctx: &mut Self::ADContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>
    where
        Self: Sized;
}
