#[allow(unused_imports)]
use computegraph::types::{LocalValueId, OperationRole, ValueKey};
#[allow(unused_imports)]
use tidu::PrimitiveValue;

#[macro_export]
macro_rules! linearize_add {
    ($builder:expr, $OpAdd:path, $t0:expr, $t1:expr) => {
        Ok(match ($t0, $t1) {
            (Some(dx), Some(dy)) => {
                let out = $builder.add_primitive(
                    $OpAdd,
                    vec![PrimitiveValue::Local(dx), PrimitiveValue::Local(dy)],
                    OperationRole::Linearized {
                        active_mask: vec![true, true],
                    },
                );
                vec![Some(out[0])]
            }
            (Some(dx), None) => vec![Some(dx)],
            (None, Some(dy)) => vec![Some(dy)],
            (None, None) => vec![None],
        })
    };
}

#[macro_export]
macro_rules! linearize_mul {
    ($builder:expr, $OpMul:path, $OpAdd:path, $primal_in:expr, $t0:expr, $t1:expr) => {{
        let mut terms = Vec::new();
        if let Some(dx) = $t0 {
            let term = $builder.add_primitive(
                $OpMul,
                vec![
                    PrimitiveValue::Local(dx),
                    PrimitiveValue::External($primal_in[1].clone()),
                ],
                OperationRole::Linearized {
                    active_mask: vec![true, false],
                },
            );
            terms.push(term[0]);
        }
        if let Some(dy) = $t1 {
            let term = $builder.add_primitive(
                $OpMul,
                vec![
                    PrimitiveValue::External($primal_in[0].clone()),
                    PrimitiveValue::Local(dy),
                ],
                OperationRole::Linearized {
                    active_mask: vec![false, true],
                },
            );
            terms.push(term[0]);
        }
        Ok(match terms.as_slice() {
            [] => vec![None],
            [only] => vec![Some(*only)],
            [lhs, rhs] => {
                let sum = $builder.add_primitive(
                    $OpAdd,
                    vec![PrimitiveValue::Local(*lhs), PrimitiveValue::Local(*rhs)],
                    OperationRole::Linearized {
                        active_mask: vec![true, true],
                    },
                );
                vec![Some(sum[0])]
            }
            _ => unreachable!("mul linearization creates at most two terms"),
        })
    }};
}

#[macro_export]
macro_rules! linearize_exp {
    ($builder:expr, $OpMul:path, $primal_out:expr, $t0:expr) => {
        Ok(match $t0 {
            Some(dx) => {
                let out = $builder.add_primitive(
                    $OpMul,
                    vec![
                        PrimitiveValue::External($primal_out.clone()),
                        PrimitiveValue::Local(dx),
                    ],
                    OperationRole::Linearized {
                        active_mask: vec![false, true],
                    },
                );
                vec![Some(out[0])]
            }
            None => vec![None],
        })
    };
}

#[macro_export]
macro_rules! linearize_neg {
    ($builder:expr, $OpNeg:path, $t0:expr) => {
        Ok(match $t0 {
            Some(dx) => {
                let out = $builder.add_primitive(
                    $OpNeg,
                    vec![PrimitiveValue::Local(dx)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                vec![Some(out[0])]
            }
            None => vec![None],
        })
    };
}

#[macro_export]
macro_rules! linearize_conj {
    ($builder:expr, $OpConj:path, $t0:expr) => {
        Ok(match $t0 {
            Some(dx) => {
                let out = $builder.add_primitive(
                    $OpConj,
                    vec![PrimitiveValue::Local(dx)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                vec![Some(out[0])]
            }
            None => vec![None],
        })
    };
}
