use ndarray::ArrayD;
use num_complex::Complex64;

#[allow(dead_code)]
pub fn assert_scalar_approx_eq(actual: f64, expected: f64, tol: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tol,
        "expected {expected}, got {actual}, |delta|={delta}"
    );
}

#[allow(dead_code)]
pub fn assert_complex_approx_eq(actual: Complex64, expected: Complex64, tol: f64) {
    let delta = actual - expected;
    assert!(
        delta.norm() <= tol,
        "expected {expected:?}, got {actual:?}, |delta|={}",
        delta.norm()
    );
}

#[allow(dead_code)]
pub fn assert_tensor_approx_eq(actual: &ArrayD<f64>, expected: &ArrayD<f64>, tol: f64) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "shape mismatch: expected {:?}, got {:?}",
        expected.shape(),
        actual.shape()
    );
    for (index, (av, ev)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (av - ev).abs();
        assert!(
            delta <= tol,
            "entry {index}: expected {ev}, got {av}, |delta|={delta}"
        );
    }
}

#[allow(dead_code)]
pub fn assert_ctensor_approx_eq(
    actual: &ArrayD<Complex64>,
    expected: &ArrayD<Complex64>,
    tol: f64,
) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "shape mismatch: expected {:?}, got {:?}",
        expected.shape(),
        actual.shape()
    );
    for (index, (av, ev)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (*av - *ev).norm();
        assert!(
            delta <= tol,
            "entry {index}: expected {ev:?}, got {av:?}, |delta|={delta}"
        );
    }
}
