use computegraph::{GraphOperation, LocalValueId, OperationRole, ValueKey, ValueRef};
use std::hint::black_box;
use tidu::rules::{
    ADKey as ModuleADKey, ADRuleError as ModuleADRuleError, ADRuleKind as ModuleADRuleKind,
    ADRuleResult as ModuleADRuleResult, DiffPassId as ModuleDiffPassId,
    Primitive as ModulePrimitive,
};
use tidu::{
    ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, Primitive, PrimitiveBuilder,
    PrimitiveValue,
};

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

impl GraphOperation for AddOp {
    type Operand = f64;
    type Context = ();
    type InputKey = Key;

    fn input_count(&self) -> usize {
        2
    }

    fn output_count(&self) -> usize {
        1
    }
}

impl Primitive for AddOp {
    type ADContext = ();

    fn add() -> Self {
        Self
    }

    fn jvp_rule(
        &self,
        _builder: &mut impl PrimitiveBuilder<Self>,
        _primal_inputs: &[ValueKey<Self>],
        _primal_outputs: &[ValueKey<Self>],
        tangent_inputs: &[Option<LocalValueId>],
        _ctx: &mut Self::ADContext,
    ) -> tidu::ADRuleResult<Vec<Option<LocalValueId>>> {
        Ok(vec![tangent_inputs[0].or(tangent_inputs[1])])
    }

    fn transpose_rule(
        &self,
        _builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_outputs: &[Option<LocalValueId>],
        _inputs: &[PrimitiveValue<Self>],
        _mode: &OperationRole,
        _ctx: &mut Self::ADContext,
    ) -> tidu::ADRuleResult<Vec<Option<LocalValueId>>> {
        Ok(vec![cotangent_outputs[0], cotangent_outputs[0]])
    }
}

#[test]
fn root_reexports_match_rules_module_contract() {
    fn assert_key<K: ADKey + ModuleADKey>() {}
    fn assert_primitive<Op: Primitive + ModulePrimitive>()
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
    assert_primitive::<AddOp>();
    let tangent = Key::Base("x").tangent_of(7);
    assert!(matches!(tangent, Key::Tangent { .. }));
    assert_eq!(assert_pass_id(7), 7);
    assert_eq!(ModuleADRuleKind::Jvp.as_str(), ADRuleKind::Jvp.as_str());
    assert_eq!(ModuleADRuleKind::Transpose.as_str(), "transpose");

    let err: ADRuleError = ModuleADRuleError::unsupported("test::op", ModuleADRuleKind::Jvp);
    assert_eq!(err.to_string(), "unsupported jvp AD rule for test::op");
    assert!(std::error::Error::source(&err).is_none());
    let rule_fn: fn(&ADRuleError) -> ADRuleKind = ADRuleError::rule;
    let runtime_err = black_box(assert_result::<()>(Err(err)).unwrap_err());
    assert_eq!(rule_fn(&runtime_err), ADRuleKind::Jvp);

    let transpose_err =
        ModuleADRuleError::unsupported("test::transpose", ModuleADRuleKind::Transpose);
    assert_eq!(
        transpose_err.to_string(),
        "unsupported transpose AD rule for test::transpose"
    );

    let invalid: ADRuleError = ModuleADRuleError::invalid_input(
        "test::solve",
        ModuleADRuleKind::Transpose,
        "expected rank >= 2",
    );
    assert_eq!(rule_fn(&invalid), ADRuleKind::Transpose);
    assert_eq!(
        invalid.to_string(),
        "invalid transpose AD input for test::solve: expected rank >= 2"
    );
}

#[test]
fn primitive_value_round_trips_computegraph_value_refs() {
    let local = PrimitiveValue::<AddOp>::Local(3);
    let local_ref: ValueRef<AddOp> = local.clone().into();
    assert_eq!(local_ref, ValueRef::Local(3));
    assert_eq!(PrimitiveValue::from(local_ref), local);

    let key = ValueKey::Input(Key::Base("x"));
    let external = PrimitiveValue::<AddOp>::External(key.clone());
    let external_ref: ValueRef<AddOp> = external.clone().into();
    assert_eq!(external_ref, ValueRef::External(key));
    assert_eq!(PrimitiveValue::from(external_ref), external);
}
