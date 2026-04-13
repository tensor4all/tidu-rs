#[allow(unused_imports)]
use computegraph::types::{OpMode, ValRef};

#[macro_export]
macro_rules! transpose_add {
    ($ct:expr) => {
        vec![Some($ct), Some($ct)]
    };
}

#[macro_export]
macro_rules! transpose_mul_real {
    ($builder:expr, $OpMul:path, $inputs:expr, $ct:expr, $mode:expr) => {{
        let active_mask = match $mode {
            OpMode::Linear { active_mask } => active_mask,
            OpMode::Primal => return vec![None, None],
        };
        let mut result = vec![None, None];
        if active_mask[0] {
            let out = $builder.add_op(
                $OpMul,
                vec![$inputs[1].clone(), ValRef::Local($ct)],
                OpMode::Linear {
                    active_mask: vec![false, true],
                },
            );
            result[0] = Some(out[0]);
        }
        if active_mask[1] {
            let out = $builder.add_op(
                $OpMul,
                vec![$inputs[0].clone(), ValRef::Local($ct)],
                OpMode::Linear {
                    active_mask: vec![false, true],
                },
            );
            result[1] = Some(out[0]);
        }
        result
    }};
}

#[macro_export]
macro_rules! transpose_mul_complex {
    ($builder:expr, $OpMul:path, $OpConj:path, $inputs:expr, $ct:expr, $mode:expr) => {{
        let active_mask = match $mode {
            OpMode::Linear { active_mask } => active_mask,
            OpMode::Primal => return vec![None, None],
        };
        let mut result = vec![None, None];
        if active_mask[0] {
            let conj_fixed = $builder.add_op(
                $OpConj,
                vec![$inputs[1].clone()],
                OpMode::Linear {
                    active_mask: vec![false],
                },
            );
            let out = $builder.add_op(
                $OpMul,
                vec![ValRef::Local(conj_fixed[0]), ValRef::Local($ct)],
                OpMode::Linear {
                    active_mask: vec![false, true],
                },
            );
            result[0] = Some(out[0]);
        }
        if active_mask[1] {
            let conj_fixed = $builder.add_op(
                $OpConj,
                vec![$inputs[0].clone()],
                OpMode::Linear {
                    active_mask: vec![false],
                },
            );
            let out = $builder.add_op(
                $OpMul,
                vec![ValRef::Local(conj_fixed[0]), ValRef::Local($ct)],
                OpMode::Linear {
                    active_mask: vec![false, true],
                },
            );
            result[1] = Some(out[0]);
        }
        result
    }};
}

#[macro_export]
macro_rules! transpose_neg {
    ($builder:expr, $OpNeg:path, $ct:expr) => {{
        let out = $builder.add_op(
            $OpNeg,
            vec![ValRef::Local($ct)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        );
        vec![Some(out[0])]
    }};
}

#[macro_export]
macro_rules! transpose_conj {
    ($builder:expr, $OpConj:path, $ct:expr) => {{
        let out = $builder.add_op(
            $OpConj,
            vec![ValRef::Local($ct)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        );
        vec![Some(out[0])]
    }};
}
