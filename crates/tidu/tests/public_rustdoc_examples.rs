#[test]
fn public_rustdoc_examples_no_longer_reference_tenferro() {
    let tracked = include_str!("../src/engine/tracked.rs");
    let tape = include_str!("../src/engine/tape.rs");
    let results = include_str!("../src/engine/results.rs");
    let lib = include_str!("../src/lib.rs");

    for text in [tracked, tape, results, lib] {
        assert!(
            !text.contains("tenferro_"),
            "public rustdoc should not use tenferro-specific examples"
        );
    }

    assert!(lib.contains("Scalar Reverse Mode"));
    assert!(lib.contains("Scalar Forward Mode"));
    assert!(lib.contains("Scalar Hessian-Vector Product"));
    assert!(lib.contains("Custom Value Type"));
}
