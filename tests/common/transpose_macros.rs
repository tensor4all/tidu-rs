#[allow(unused_imports)]
use computegraph::types::OperationRole;
#[allow(unused_imports)]
use tidu::PrimitiveValue;

#[macro_export]
macro_rules! transpose_add {
    ($ct:expr) => {
        Ok(vec![Some($ct), Some($ct)])
    };
}

#[macro_export]
macro_rules! transpose_mul_real {
    ($builder:expr, $OpMul:path, $inputs:expr, $ct:expr, $role:expr) => {{
        let active_mask = match $role {
            OperationRole::Linearized { active_mask } => active_mask,
            OperationRole::Primary => return Ok(vec![None, None]),
        };
        let mut result = vec![None, None];
        if active_mask[0] {
            let out = $builder.add_primitive(
                $OpMul,
                vec![$inputs[1].clone(), PrimitiveValue::Local($ct)],
                OperationRole::Linearized {
                    active_mask: vec![false, true],
                },
            );
            result[0] = Some(out[0]);
        }
        if active_mask[1] {
            let out = $builder.add_primitive(
                $OpMul,
                vec![$inputs[0].clone(), PrimitiveValue::Local($ct)],
                OperationRole::Linearized {
                    active_mask: vec![false, true],
                },
            );
            result[1] = Some(out[0]);
        }
        Ok(result)
    }};
}

#[macro_export]
macro_rules! transpose_mul_complex {
    ($builder:expr, $OpMul:path, $OpConj:path, $inputs:expr, $ct:expr, $role:expr) => {{
        let active_mask = match $role {
            OperationRole::Linearized { active_mask } => active_mask,
            OperationRole::Primary => return Ok(vec![None, None]),
        };
        let mut result = vec![None, None];
        if active_mask[0] {
            let conj_fixed = $builder.add_primitive(
                $OpConj,
                vec![$inputs[1].clone()],
                OperationRole::Linearized {
                    active_mask: vec![false],
                },
            );
            let out = $builder.add_primitive(
                $OpMul,
                vec![
                    PrimitiveValue::Local(conj_fixed[0]),
                    PrimitiveValue::Local($ct),
                ],
                OperationRole::Linearized {
                    active_mask: vec![false, true],
                },
            );
            result[0] = Some(out[0]);
        }
        if active_mask[1] {
            let conj_fixed = $builder.add_primitive(
                $OpConj,
                vec![$inputs[0].clone()],
                OperationRole::Linearized {
                    active_mask: vec![false],
                },
            );
            let out = $builder.add_primitive(
                $OpMul,
                vec![
                    PrimitiveValue::Local(conj_fixed[0]),
                    PrimitiveValue::Local($ct),
                ],
                OperationRole::Linearized {
                    active_mask: vec![false, true],
                },
            );
            result[1] = Some(out[0]);
        }
        Ok(result)
    }};
}

#[macro_export]
macro_rules! transpose_neg {
    ($builder:expr, $OpNeg:path, $ct:expr) => {{
        let out = $builder.add_primitive(
            $OpNeg,
            vec![PrimitiveValue::Local($ct)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        );
        Ok(vec![Some(out[0])])
    }};
}

#[macro_export]
macro_rules! transpose_conj {
    ($builder:expr, $OpConj:path, $ct:expr) => {{
        let out = $builder.add_primitive(
            $OpConj,
            vec![PrimitiveValue::Local($ct)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        );
        Ok(vec![Some(out[0])])
    }};
}
