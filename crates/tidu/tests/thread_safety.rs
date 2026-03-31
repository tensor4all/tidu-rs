use tidu::{AdExecutionPolicy, CheckpointMode, Value};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn tidu_public_handles_are_send_sync() {
    assert_send_sync::<Value<f64>>();
    assert_send_sync::<AdExecutionPolicy>();
    assert_send_sync::<CheckpointMode>();
}
