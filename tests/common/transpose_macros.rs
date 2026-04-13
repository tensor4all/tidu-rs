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
    ($builder:expr, $Op:path, $inputs:expr, $ct:expr, $mode:expr) => {{
        let active_mask = match $mode {
            OpMode::Linear { active_mask } => active_mask,
            OpMode::Primal => return vec![None, None],
        };
        let mut result = vec![None, None];
        if active_mask[0] {
            let out = $builder.add_op(
                $Op::Mul,
                vec![$inputs[1].clone(), ValRef::Local($ct)],
                OpMode::Linear {
                    active_mask: vec![false, true],
                },
            );
            result[0] = Some(out[0]);
        }
        if active_mask[1] {
            let out = $builder.add_op(
                $Op::Mul,
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
    ($builder:expr, $Op:path, $Conj:path, $inputs:expr, $ct:expr, $mode:expr) => {{
        let active_mask = match $mode {
            OpMode::Linear { active_mask } => active_mask,
            OpMode::Primal => return vec![None, None],
        };
        let mut result = vec![None, None];
        if active_mask[0] {
            let conj_fixed = $builder.add_op($Conj, vec![$inputs[1].clone()], OpMode::Primal);
            let out = $builder.add_op(
                $Op::Mul,
                vec![ValRef::Local(conj_fixed[0]), ValRef::Local($ct)],
                OpMode::Linear {
                    active_mask: vec![false, true],
                },
            );
            result[0] = Some(out[0]);
        }
        if active_mask[1] {
            let conj_fixed = $builder.add_op($Conj, vec![$inputs[0].clone()], OpMode::Primal);
            let out = $builder.add_op(
                $Op::Mul,
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
    ($builder:expr, $Op:path, $ct:expr) => {{
        let out = $builder.add_op(
            $Op::Neg,
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
    ($builder:expr, $Op:path, $ct:expr) => {{
        let out = $builder.add_op(
            $Op::Conj,
            vec![ValRef::Local($ct)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        );
        vec![Some(out[0])]
    }};
}
