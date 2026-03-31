use std::fs;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn src_file(path: &str) -> String {
    fs::read_to_string(crate_root().join("src").join(path)).expect("read source file")
}

fn src_subfile(path: &str) -> String {
    fs::read_to_string(crate_root().join("src").join(path)).expect("read source subfile")
}

fn repo_file(path: &str) -> String {
    fs::read_to_string(crate_root().join("..").join("..").join(path)).expect("read repo file")
}

#[test]
fn tidu_root_surface_exports_only_linearize_first_api() {
    let lib_rs = src_file("lib.rs");

    for required in [
        "pub use value::Value;",
        "pub use linearized::{LinearizableOp, LinearizedOp, Schema, SlotSchema};",
        "pub use checkpoint::{",
        "AdExecutionPolicy",
        "CheckpointMode",
        "with_ad_policy",
    ] {
        assert!(
            lib_rs.contains(required),
            "lib.rs should export `{required}`"
        );
    }

    for forbidden in [
        "pub mod expert",
        "DualValue",
        "Tape",
        "TrackedValue",
        "Gradients",
        "HvpResult",
    ] {
        assert!(
            !lib_rs.contains(forbidden),
            "lib.rs should not expose `{forbidden}` after the migration"
        );
    }
}

#[test]
fn readme_quick_start_stays_linearize_first() {
    let readme = repo_file("README.md");

    for required in [
        "LinearizableOp",
        "LinearizedOp",
        "CheckpointMode",
        "with_ad_policy",
    ] {
        assert!(
            readme.contains(required),
            "README.md should mention `{required}` in the public story"
        );
    }

    for forbidden in ["record_op", "Tape", "TrackedValue", "HVP"] {
        assert!(
            !readme.contains(forbidden),
            "README.md should not mention `{forbidden}`"
        );
    }
}

#[test]
fn crate_level_rustdoc_leads_with_linearize_first_examples() {
    let lib_rs = src_file("lib.rs");

    for required in [
        "Value-Centered Reverse Mode",
        "Local Directional Derivatives",
        "Checkpoint Policy",
        "LinearizableOp",
        "LinearizedOp",
    ] {
        assert!(
            lib_rs.contains(required),
            "crate-level rustdoc should mention `{required}`"
        );
    }

    for forbidden in ["Scalar Forward Mode", "Expert API", "DualValue", "HVP"] {
        assert!(
            !lib_rs.contains(forbidden),
            "crate-level rustdoc should not lead with `{forbidden}`"
        );
    }
}

#[test]
fn checkpoint_boundary_uses_public_hint_not_internal_class_name() {
    let checkpoint_rs = src_subfile("checkpoint.rs");
    let linearized_rs = src_subfile("linearized.rs");

    assert!(
        checkpoint_rs.contains("pub enum CheckpointHint"),
        "checkpoint.rs should expose CheckpointHint as the public trait-facing type"
    );
    assert!(
        !checkpoint_rs.contains("pub enum CheckpointClass"),
        "checkpoint.rs should not expose CheckpointClass publicly"
    );
    assert!(
        linearized_rs.contains("fn checkpoint_hint(&self) -> CheckpointHint"),
        "LinearizableOp should use checkpoint_hint instead of checkpoint_class"
    );
}
