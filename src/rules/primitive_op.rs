use computegraph::fragment::FragmentBuilder;
use computegraph::{GlobalValKey, GraphOp, LocalValId, OpEmitter, OpMode, ValRef};

use super::{ADKey, ADRuleResult};

/// Extends `GraphOp` with linearization and transpose rules for AD.
///
/// - `try_linearize` is called by [`crate::try_differentiate`]
/// - `try_transpose_rule` is called by [`crate::try_transpose`]
///
/// Both methods emit new ops into a `FragmentBuilder`. The downstream
/// implementor is responsible for ensuring closure: every op emitted must also
/// implement `PrimitiveOp`.
///
/// # Examples
///
/// ```
/// use computegraph::fragment::FragmentBuilder;
/// use computegraph::{GlobalValKey, GraphOp, LocalValId, OpEmitter, OpMode, ValRef};
/// use tidu::{ADKey, DiffPassId, PrimitiveOp};
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
/// impl GraphOp for AddOp {
///     type Operand = f64;
///     type Context = ();
///     type InputKey = Key;
///     fn n_inputs(&self) -> usize { 2 }
///     fn n_outputs(&self) -> usize { 1 }
/// }
///
/// impl PrimitiveOp for AddOp {
///     type ADContext = ();
///
///     fn add() -> Self { AddOp }
///     fn linearize(
///         &self, _b: &mut FragmentBuilder<Self>,
///         _pi: &[GlobalValKey<Self>], _po: &[GlobalValKey<Self>],
///         t: &[Option<LocalValId>],
///         _ctx: &mut (),
///     ) -> Vec<Option<LocalValId>> {
///         vec![t[0].or(t[1])]
///     }
///     fn transpose_rule(
///         &self, _emitter: &mut impl OpEmitter<Self>,
///         ct: &[Option<LocalValId>], _i: &[ValRef<Self>], _m: &OpMode,
///         _ctx: &mut (),
///     ) -> Vec<Option<LocalValId>> {
///         vec![ct[0], ct[0]]
///     }
/// }
/// ```
pub trait PrimitiveOp: GraphOp
where
    Self::InputKey: ADKey,
{
    /// Runtime AD context threaded through linearization and transpose.
    ///
    /// This can carry information such as concrete shapes or guard decisions
    /// that influence how AD rules emit graph structure.
    type ADContext: Default;

    /// Returns the addition operation used for cotangent accumulation
    /// in [`crate::transpose`]. When multiple cotangents flow to the same
    /// `GlobalValKey`, transpose emits `Op::add()` nodes to sum them.
    fn add() -> Self
    where
        Self: Sized;

    /// Emit the linear (JVP) rule for this primitive.
    ///
    /// Must be linear in tangent inputs. May reference primal inputs/outputs
    /// through `External(GlobalValKey)`. Must emit ops in `OpMode::Linear`.
    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValId>>
    where
        Self: Sized;

    /// Fallible variant of [`PrimitiveOp::linearize`].
    ///
    /// Implementors that can encounter missing extension rules should override
    /// this method and return [`super::ADRuleError`] instead of panicking. The
    /// default implementation preserves the infallible contract for existing
    /// primitive sets.
    fn try_linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        ctx: &mut Self::ADContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>>
    where
        Self: Sized,
    {
        Ok(self.linearize(builder, primal_in, primal_out, tangent_in, ctx))
    }

    /// Emit the transpose rule for this linear primitive.
    ///
    /// Receives cotangent outputs and produces cotangent inputs.
    /// Must only emit ops that themselves implement `PrimitiveOp`.
    ///
    /// Uses `OpEmitter` instead of `FragmentBuilder` to enable both
    /// graph-building and eager execution.
    fn transpose_rule(
        &self,
        emitter: &mut impl OpEmitter<Self>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<Self>],
        mode: &OpMode,
        ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValId>>
    where
        Self: Sized;

    /// Fallible variant of [`PrimitiveOp::transpose_rule`].
    ///
    /// Implementors that can encounter missing extension rules should override
    /// this method and return [`super::ADRuleError`] instead of panicking. The
    /// default implementation preserves the infallible contract for existing
    /// primitive sets.
    fn try_transpose_rule(
        &self,
        emitter: &mut impl OpEmitter<Self>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<Self>],
        mode: &OpMode,
        ctx: &mut Self::ADContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>>
    where
        Self: Sized,
    {
        Ok(self.transpose_rule(emitter, cotangent_out, inputs, mode, ctx))
    }
}
