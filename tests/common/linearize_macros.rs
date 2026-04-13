#[allow(unused_imports)]
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};

#[macro_export]
macro_rules! linearize_add {
    ($builder:expr, $OpAdd:path, $t0:expr, $t1:expr) => {
        match ($t0, $t1) {
            (Some(dx), Some(dy)) => {
                let out = $builder.add_op(
                    $OpAdd,
                    vec![ValRef::Local(dx), ValRef::Local(dy)],
                    OpMode::Linear {
                        active_mask: vec![true, true],
                    },
                );
                vec![Some(out[0])]
            }
            (Some(dx), None) => vec![Some(dx)],
            (None, Some(dy)) => vec![Some(dy)],
            (None, None) => vec![None],
        }
    };
}

#[macro_export]
macro_rules! linearize_mul {
    ($builder:expr, $OpMul:path, $OpAdd:path, $primal_in:expr, $t0:expr, $t1:expr) => {{
        let mut terms = Vec::new();
        if let Some(dx) = $t0 {
            let term = $builder.add_op(
                $OpMul,
                vec![ValRef::Local(dx), ValRef::External($primal_in[1].clone())],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            terms.push(term[0]);
        }
        if let Some(dy) = $t1 {
            let term = $builder.add_op(
                $OpMul,
                vec![ValRef::External($primal_in[0].clone()), ValRef::Local(dy)],
                OpMode::Linear {
                    active_mask: vec![false, true],
                },
            );
            terms.push(term[0]);
        }
        match terms.as_slice() {
            [] => vec![None],
            [only] => vec![Some(*only)],
            [lhs, rhs] => {
                let sum = $builder.add_op(
                    $OpAdd,
                    vec![ValRef::Local(*lhs), ValRef::Local(*rhs)],
                    OpMode::Linear {
                        active_mask: vec![true, true],
                    },
                );
                vec![Some(sum[0])]
            }
            _ => unreachable!("mul linearization creates at most two terms"),
        }
    }};
}

#[macro_export]
macro_rules! linearize_exp {
    ($builder:expr, $OpMul:path, $primal_out:expr, $t0:expr) => {
        match $t0 {
            Some(dx) => {
                let out = $builder.add_op(
                    $OpMul,
                    vec![ValRef::External($primal_out.clone()), ValRef::Local(dx)],
                    OpMode::Linear {
                        active_mask: vec![false, true],
                    },
                );
                vec![Some(out[0])]
            }
            None => vec![None],
        }
    };
}

#[macro_export]
macro_rules! linearize_neg {
    ($builder:expr, $OpNeg:path, $t0:expr) => {
        match $t0 {
            Some(dx) => {
                let out = $builder.add_op(
                    $OpNeg,
                    vec![ValRef::Local(dx)],
                    OpMode::Linear {
                        active_mask: vec![true],
                    },
                );
                vec![Some(out[0])]
            }
            None => vec![None],
        }
    };
}

#[macro_export]
macro_rules! linearize_conj {
    ($builder:expr, $OpConj:path, $t0:expr) => {
        match $t0 {
            Some(dx) => {
                let out = $builder.add_op(
                    $OpConj,
                    vec![ValRef::Local(dx)],
                    OpMode::Linear {
                        active_mask: vec![true],
                    },
                );
                vec![Some(out[0])]
            }
            None => vec![None],
        }
    };
}
