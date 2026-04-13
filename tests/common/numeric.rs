#[allow(dead_code)]
pub fn five_point_derivative(sample: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (-sample(x + 2.0 * h) + 8.0 * sample(x + h) - 8.0 * sample(x - h) + sample(x - 2.0 * h))
        / (12.0 * h)
}
