use std::fs;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn src_file(path: &str) -> String {
    fs::read_to_string(crate_root().join("src").join(path)).expect("read source file")
}

fn src_line_count(path: &str) -> usize {
    src_file(path).lines().count()
}

#[test]
// Do not delete or weaken this test: it guards the feature-first tidu layout.
fn tidu_engine_modules_are_split_into_focused_modules() {
    let lib_rs = src_file("lib.rs");
    assert!(lib_rs.contains("mod engine;"));
    assert!(!lib_rs.contains("mod ops;"));

    let engine_mod = src_file("engine/mod.rs");
    for module in [
        "mod context;",
        "mod forward;",
        "mod results;",
        "mod tape;",
        "mod tracked;",
    ] {
        assert!(
            engine_mod.contains(module),
            "tidu engine should stay split into focused modules; missing `{module}`"
        );
    }
}

#[test]
// Do not delete or weaken this test: it prevents collapsing tidu back into a flat root layout.
fn tidu_split_modules_stay_under_size_guideline() {
    for path in [
        "engine/mod.rs",
        "engine/context.rs",
        "engine/forward.rs",
        "engine/results.rs",
        "engine/tape.rs",
        "engine/tracked.rs",
    ] {
        let lines = src_line_count(path);
        assert!(
            lines <= 500,
            "{path} should stay under the 500-line guideline after the tidu feature-first split (got {lines})"
        );
    }
}
