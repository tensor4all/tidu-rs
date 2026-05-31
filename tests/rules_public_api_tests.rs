use computegraph::fragment::FragmentBuilder;
use computegraph::{GlobalValKey, GraphOp, LocalValId, OpEmitter, OpMode, ValRef};
use tidu::rules::{
    ADKey as ModuleADKey, ADRuleError as ModuleADRuleError, ADRuleKind as ModuleADRuleKind,
    ADRuleResult as ModuleADRuleResult, DiffPassId as ModuleDiffPassId,
    PrimitiveOp as ModulePrimitiveOp,
};
use tidu::{ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, PrimitiveOp};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Key {
    Base(&'static str),
    Tangent { of: Box<Key>, pass: DiffPassId },
}

impl ADKey for Key {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        Self::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AddOp;

impl GraphOp for AddOp {
    type Operand = f64;
    type Context = ();
    type InputKey = Key;

    fn n_inputs(&self) -> usize {
        2
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl PrimitiveOp for AddOp {
    type ADContext = ();

    fn add() -> Self {
        Self
    }

    fn linearize(
        &self,
        _builder: &mut FragmentBuilder<Self>,
        _primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValId>> {
        vec![tangent_in[0].or(tangent_in[1])]
    }

    fn transpose_rule(
        &self,
        _emitter: &mut impl OpEmitter<Self>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<Self>],
        _mode: &OpMode,
        _ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValId>> {
        vec![cotangent_out[0], cotangent_out[0]]
    }
}

#[test]
fn root_reexports_match_rules_module_contract() {
    fn assert_key<K: ADKey + ModuleADKey>() {}
    fn assert_op<Op: PrimitiveOp + ModulePrimitiveOp>()
    where
        Op::InputKey: ADKey,
    {
    }
    fn assert_result<T>(result: ADRuleResult<T>) -> ModuleADRuleResult<T> {
        result
    }
    fn assert_pass_id(pass: DiffPassId) -> ModuleDiffPassId {
        pass
    }

    assert_key::<Key>();
    assert_op::<AddOp>();
    let tangent = Key::Base("x").tangent_of(7);
    assert!(matches!(tangent, Key::Tangent { .. }));
    assert_eq!(assert_pass_id(7), 7);
    assert_eq!(
        ModuleADRuleKind::Linearize.as_str(),
        ADRuleKind::Linearize.as_str()
    );

    let err: ADRuleError = ModuleADRuleError::unsupported("test::op", ModuleADRuleKind::Transpose);
    assert_eq!(
        assert_result::<()>(Err(err)).unwrap_err().rule(),
        ADRuleKind::Transpose
    );
}
