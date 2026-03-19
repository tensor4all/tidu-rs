use tidu::{DualValue, Tape, TrackedValue};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn tidu_public_handles_are_send_sync() {
    assert_send_sync::<Tape<f64>>();
    assert_send_sync::<TrackedValue<f64>>();
    assert_send_sync::<DualValue<f64>>();
}
