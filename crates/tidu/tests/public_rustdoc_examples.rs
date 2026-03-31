#[test]
fn public_rustdoc_examples_match_linearize_first_story() {
    let lib = include_str!("../src/lib.rs");

    assert!(
        !lib.contains("tenferro_"),
        "public rustdoc should not use tenferro-specific examples"
    );

    for required in [
        "Value-Centered Reverse Mode",
        "Local Directional Derivatives",
        "Checkpoint Policy",
        "LinearizableOp",
        "LinearizedOp",
    ] {
        assert!(
            lib.contains(required),
            "public rustdoc should mention `{required}`"
        );
    }

    for forbidden in ["Scalar Forward Mode", "Expert API", "DualValue", "HVP"] {
        assert!(
            !lib.contains(forbidden),
            "public rustdoc should not mention `{forbidden}`"
        );
    }
}
