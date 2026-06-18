# JAX Terminology Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace tidu's current fragment/emitter/differentiate public vocabulary with a JAX-aligned primitive graph API and publish self-contained downstream-implementer documentation.

**Architecture:** This is a breaking tidu-only migration. The graph transform API becomes `linearize` and `linear_transpose`, rule contracts are expressed through `Primitive` and a tidu-owned `PrimitiveBuilder`, and raw computegraph fragments are wrapped by public graph types. Eager support remains as `tidu::eager`, but it is documented and typed as an integration layer rather than a separate AD mode.

**Tech Stack:** Rust 2021, `computegraph`, `cargo nextest`, rustdoc, Quarto docs site.

---

## Scope Check

This plan covers the tidu-rs API and documentation migration only. Downstream
crates such as tenferro-rs must be migrated in a separate follow-up plan after
the tidu PR merges, because they depend on the final tidu commit hash and API.

## File Structure

- Modify `src/lib.rs`: root docs and root re-exports.
- Rename `src/differentiate.rs` to `src/linearize.rs`: graph linearization transform.
- Rename `src/transpose.rs` to `src/linear_transpose.rs`: graph linear transpose transform and builder-backed transpose execution helper.
- Rename `src/linear_fragment.rs` to `src/linearized_graph.rs`: wrapper type that hides raw computegraph fragments.
- Modify `src/rules/primitive_op.rs`: rename `PrimitiveOp` to `Primitive`, rename local linearization rule to `jvp_rule`, and replace public `OpEmitter` usage with `PrimitiveBuilder`.
- Add `src/rules/primitive_builder.rs`: tidu-owned builder trait plus a crate-private adapter to computegraph graph builders.
- Add `src/primitive_graph.rs`: lightweight public wrapper for lower-level primitive graphs passed across eager executor boundaries.
- Modify `src/rules/ad_rule_error.rs` and `src/rules/mod.rs`: expose JAX-aligned rule kinds and names.
- Modify `src/eager/record.rs`, `src/eager/backward.rs`, `src/eager/mod.rs`, and `src/eager/trace.rs`: rename eager input/output types and remove raw fragment parameters from eager executor signatures.
- Delete or stop exporting `src/emit.rs`: replace `tidu::emit::linear_transpose_with_builder` with `linear_transpose_with_builder`.
- Modify `tests/common/*` and all Rust tests under `tests/`: migrate tests to new API names and add public API assertions.
- Modify `README.md` and `src/lib.rs` rustdoc: short front door and self-contained conceptual entry point.
- Create `docs/_quarto.yml`, `docs/index.md`, `docs/getting-started/*`, `docs/tutorials/*`, `docs/guides/*`, `docs/architecture/*`, `docs/api/index.md`, and `docs/internals/index.md`.
- Add tutorial code under `examples/` or `tests/tutorial_*` and run it in CI through existing workspace tests.
- Keep `scripts/build_docs_site.sh` unchanged; the planned `docs/_quarto.yml` writes Quarto output to `target/docs-site/design`, which the existing script already detects.

## Task 1: Red Tests For JAX-Aligned Public API

**Files:**
- Modify: `tests/rules_public_api_tests.rs`
- Modify: `tests/fallible_ad_tests.rs`
- Modify: `tests/eager_record_tests.rs`
- Modify: `tests/eager_backward_tests.rs`

- [ ] **Step 1: Update the public API contract test imports**

Replace the top of `tests/rules_public_api_tests.rs` with imports that use the new names:

```rust
use computegraph::{GlobalValKey, GraphOp, LocalValId, OpMode};
use std::hint::black_box;
use tidu::rules::{
    ADKey as ModuleADKey, ADRuleError as ModuleADRuleError,
    ADRuleKind as ModuleADRuleKind, ADRuleResult as ModuleADRuleResult,
    DiffPassId as ModuleDiffPassId, Primitive as ModulePrimitive,
};
use tidu::{
    ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, Primitive,
    PrimitiveBuilder, PrimitiveValue,
};
```

- [ ] **Step 2: Update the public API contract test primitive implementation**

Change the trait implementation in `tests/rules_public_api_tests.rs` to this shape:

```rust
impl Primitive for AddOp {
    type ADContext = ();

    fn add() -> Self {
        Self
    }

    fn jvp_rule(
        &self,
        _builder: &mut impl PrimitiveBuilder<Self>,
        _primal_inputs: &[GlobalValKey<Self>],
        _primal_outputs: &[GlobalValKey<Self>],
        tangent_inputs: &[Option<LocalValId>],
        _ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValId>> {
        vec![tangent_inputs[0].or(tangent_inputs[1])]
    }

    fn transpose_rule(
        &self,
        _builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_outputs: &[Option<LocalValId>],
        _inputs: &[PrimitiveValue<Self>],
        _mode: &OpMode,
        _ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValId>> {
        vec![cotangent_outputs[0], cotangent_outputs[0]]
    }
}
```

- [ ] **Step 3: Update the public API assertions**

In `root_reexports_match_rules_module_contract`, replace the primitive and rule-kind assertions with:

```rust
fn assert_primitive<Op: Primitive + ModulePrimitive>()
where
    Op::InputKey: ADKey,
{
}

assert_primitive::<AddOp>();
assert_eq!(ModuleADRuleKind::Jvp.as_str(), ADRuleKind::Jvp.as_str());
assert_eq!(ModuleADRuleKind::Transpose.as_str(), "transpose");

let err: ADRuleError = ModuleADRuleError::unsupported("test::op", ModuleADRuleKind::Jvp);
assert_eq!(err.to_string(), "unsupported jvp AD rule for test::op");
```

- [ ] **Step 4: Update fallible API test imports and names**

In `tests/fallible_ad_tests.rs`, change the import block to:

```rust
use computegraph::{GraphOp, LocalValId, OpMode};
use tidu::{
    linear_transpose, linear_transpose_with_builder, linearize,
    LinearizedGraph, PrimitiveBuilder, PrimitiveValue,
};
use tidu::{ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, Primitive};
```

Rename test functions:

```rust
fn linearize_propagates_jvp_rule_error()
fn linear_transpose_propagates_transpose_error()
fn linear_transpose_with_builder_propagates_transpose_error()
```

- [ ] **Step 5: Update eager tests to the intended names**

In eager tests, replace imports of `Input` and `Output` with `EagerInput` and `EagerOutput`, replace `tidu::LinearFragment` with `tidu::LinearizedGraph`, and replace `emit::linear_transpose_with_builder` with `tidu::linear_transpose_with_builder`.

Example target import:

```rust
use tidu::eager::{self, BackwardExecutor, EagerInput, KeySource, Recorder};
use tidu::{linearize, linear_transpose_with_builder, LinearizedGraph};
```

- [ ] **Step 6: Run red tests**

Run:

```bash
cargo nextest run --release --test rules_public_api_tests --test fallible_ad_tests --test eager_record_tests --test eager_backward_tests
```

Expected: FAIL with unresolved imports such as `tidu::Primitive`, `tidu::linearize`, `tidu::LinearizedGraph`, `tidu::PrimitiveBuilder`, and `tidu::eager::EagerInput`.

- [ ] **Step 7: Commit red tests**

```bash
git add tests/rules_public_api_tests.rs tests/fallible_ad_tests.rs tests/eager_record_tests.rs tests/eager_backward_tests.rs
git commit -m "test: specify JAX-aligned public API"
```

## Task 2: Introduce PrimitiveBuilder And PrimitiveValue

**Files:**
- Add: `src/rules/primitive_builder.rs`
- Modify: `src/rules/mod.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create `PrimitiveValue` and `PrimitiveBuilder`**

Create `src/rules/primitive_builder.rs`:

```rust
use computegraph::fragment::FragmentBuilder;
use computegraph::{GraphOp, LocalValId, OpMode, ValRef};

/// Reference to a value available to a primitive AD rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveValue<Op: GraphOp> {
    /// Value produced inside the graph being built.
    Local(LocalValId),
    /// Value from the source primitive computation graph.
    External(computegraph::GlobalValKey<Op>),
}

impl<Op: GraphOp> From<PrimitiveValue<Op>> for ValRef<Op> {
    fn from(value: PrimitiveValue<Op>) -> Self {
        match value {
            PrimitiveValue::Local(id) => ValRef::Local(id),
            PrimitiveValue::External(key) => ValRef::External(key),
        }
    }
}

impl<Op: GraphOp> From<ValRef<Op>> for PrimitiveValue<Op> {
    fn from(value: ValRef<Op>) -> Self {
        match value {
            ValRef::Local(id) => PrimitiveValue::Local(id),
            ValRef::External(key) => PrimitiveValue::External(key),
        }
    }
}

/// Builder used by primitive JVP and transpose rules to append primitive applications.
pub trait PrimitiveBuilder<Op: GraphOp> {
    /// Add one primitive application and return local ids for its outputs.
    fn add_primitive(
        &mut self,
        op: Op,
        inputs: Vec<PrimitiveValue<Op>>,
        mode: OpMode,
    ) -> Vec<LocalValId>;
}

pub(crate) struct FragmentPrimitiveBuilder<'a, Op: GraphOp> {
    inner: &'a mut FragmentBuilder<Op>,
}

impl<'a, Op: GraphOp> FragmentPrimitiveBuilder<'a, Op> {
    pub(crate) fn new(inner: &'a mut FragmentBuilder<Op>) -> Self {
        Self { inner }
    }
}

impl<Op: GraphOp> PrimitiveBuilder<Op> for FragmentPrimitiveBuilder<'_, Op> {
    fn add_primitive(
        &mut self,
        op: Op,
        inputs: Vec<PrimitiveValue<Op>>,
        mode: OpMode,
    ) -> Vec<LocalValId> {
        let inputs = inputs.into_iter().map(ValRef::from).collect();
        self.inner.add_op(op, inputs, mode)
    }
}
```

- [ ] **Step 2: Re-export the new builder contract**

In `src/rules/mod.rs`, add:

```rust
mod primitive_builder;
pub use primitive_builder::{PrimitiveBuilder, PrimitiveValue};
pub(crate) use primitive_builder::FragmentPrimitiveBuilder;
```

In `src/lib.rs`, add these names to the root re-export list:

```rust
pub use rules::{
    ADKey, ADRuleError, ADRuleKind, ADRuleResult, DiffPassId, Primitive,
    PrimitiveBuilder, PrimitiveValue,
};
```

- [ ] **Step 3: Run the public API contract test**

Run:

```bash
cargo nextest run --release --test rules_public_api_tests
```

Expected: still FAIL because `Primitive`, `ADRuleKind::Jvp`, and graph transform names are not yet implemented.

## Task 3: Rename PrimitiveOp To Primitive And JVP Rule Contract

**Files:**
- Modify: `src/rules/primitive_op.rs`
- Modify: `src/rules/mod.rs`
- Modify: `src/rules/ad_rule_error.rs`
- Modify: `src/rules/ad_key.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Rename the trait and rule method signatures**

In `src/rules/primitive_op.rs`, change the trait declaration and core method names to:

```rust
pub trait Primitive: GraphOp
where
    Self::InputKey: ADKey,
{
    type ADContext: Default;

    fn add() -> Self
    where
        Self: Sized;

    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        primal_inputs: &[GlobalValKey<Self>],
        primal_outputs: &[GlobalValKey<Self>],
        tangent_inputs: &[Option<LocalValId>],
        ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValId>>
    where
        Self: Sized;

    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        primal_inputs: &[GlobalValKey<Self>],
        primal_outputs: &[GlobalValKey<Self>],
        tangent_inputs: &[Option<LocalValId>],
        ctx: &mut Self::ADContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>>
    where
        Self: Sized,
    {
        Ok(self.jvp_rule(builder, primal_inputs, primal_outputs, tangent_inputs, ctx))
    }

    fn transpose_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_outputs: &[Option<LocalValId>],
        inputs: &[PrimitiveValue<Self>],
        mode: &OpMode,
        ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValId>>
    where
        Self: Sized;

    fn transpose_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_outputs: &[Option<LocalValId>],
        inputs: &[PrimitiveValue<Self>],
        mode: &OpMode,
        ctx: &mut Self::ADContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>>
    where
        Self: Sized,
    {
        Ok(self.transpose_rule(builder, cotangent_outputs, inputs, mode, ctx))
    }
}
```

Update imports in this file to include:

```rust
use super::{ADKey, ADRuleResult, PrimitiveBuilder, PrimitiveValue};
use computegraph::{GlobalValKey, GraphOp, LocalValId, OpMode};
```

- [ ] **Step 2: Rename rule kind from linearize to JVP**

In `src/rules/ad_rule_error.rs`, change the enum variant and string:

```rust
pub enum ADRuleKind {
    /// JVP rule for forward linearization.
    Jvp,
    /// Transpose / VJP rule for a linear primitive.
    Transpose,
}

impl ADRuleKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Jvp => "jvp",
            Self::Transpose => "transpose",
        }
    }
}
```

- [ ] **Step 3: Re-export `Primitive` instead of `PrimitiveOp`**

In `src/rules/mod.rs`, replace:

```rust
pub use primitive_op::PrimitiveOp;
```

with:

```rust
pub use primitive_op::Primitive;
```

In `src/lib.rs`, replace `PrimitiveOp` with `Primitive` in root re-exports.

- [ ] **Step 4: Update current source references**

Run:

```bash
rg -l "PrimitiveOp|linearize\\(|linearize\\(" src tests | xargs perl -0pi -e 's/PrimitiveOp/Primitive/g; s/linearize\\(/jvp_rule(/g'
```

Then run this check and fix only the printed lines:

```bash
rg -n "jvp_rule\\(&view|jvp_rule\\(view|fn jvp_rule<|pub fn jvp_rule<" src tests
```

Expected before fixes: any graph-transform function accidentally renamed from `linearize` to `jvp_rule` is printed. Expected after fixes: no graph-transform function definitions or calls are printed by this command.

- [ ] **Step 5: Run focused test**

Run:

```bash
cargo nextest run --release --test rules_public_api_tests
```

Expected: FAIL only on graph transform names/types that are implemented in later tasks, not on `Primitive` or `PrimitiveBuilder`.

- [ ] **Step 6: Commit primitive contract rename**

```bash
git add src/rules src/lib.rs tests/rules_public_api_tests.rs
git commit -m "refactor: rename primitive AD rule contract"
```

## Task 4: Rename LinearFragment To LinearizedGraph

**Files:**
- Rename: `src/linear_fragment.rs` -> `src/linearized_graph.rs`
- Modify: `src/lib.rs`
- Modify: `src/linearize.rs` after Task 5 rename
- Modify: `src/linear_transpose.rs` after Task 5 rename
- Modify: tests using `LinearFragment`

- [ ] **Step 1: Rename the file**

Run:

```bash
git mv src/linear_fragment.rs src/linearized_graph.rs
```

- [ ] **Step 2: Replace the struct with private graph storage and accessors**

Use this struct body in `src/linearized_graph.rs`:

```rust
use computegraph::fragment::Fragment;
use computegraph::{GraphOp, LocalValId};

/// Graph produced by linearizing a primitive computation graph.
pub struct LinearizedGraph<Op: GraphOp> {
    graph: Fragment<Op>,
    tangent_inputs: Vec<(Op::InputKey, LocalValId)>,
    tangent_outputs: Vec<Option<LocalValId>>,
}

impl<Op: GraphOp> LinearizedGraph<Op> {
    pub(crate) fn from_parts(
        graph: Fragment<Op>,
        tangent_inputs: Vec<(Op::InputKey, LocalValId)>,
        tangent_outputs: Vec<Option<LocalValId>>,
    ) -> Self {
        Self {
            graph,
            tangent_inputs,
            tangent_outputs,
        }
    }

    /// Borrow the lower-level graph representation.
    pub fn as_graph(&self) -> &Fragment<Op> {
        &self.graph
    }

    /// Consume this value and return the lower-level graph representation.
    pub fn into_graph(self) -> Fragment<Op> {
        self.graph
    }

    /// Tangent input keys and local value ids.
    pub fn tangent_inputs(&self) -> &[(Op::InputKey, LocalValId)] {
        &self.tangent_inputs
    }

    /// Tangent outputs aligned with requested primal outputs.
    pub fn tangent_outputs(&self) -> &[Option<LocalValId>] {
        &self.tangent_outputs
    }
}
```

- [ ] **Step 3: Update module exports**

In `src/lib.rs`, replace:

```rust
mod linear_fragment;
pub use linear_fragment::LinearFragment;
```

with:

```rust
mod linearized_graph;
pub use linearized_graph::LinearizedGraph;
```

- [ ] **Step 4: Update tests to access graph fields through methods**

Replace these patterns in tests:

```rust
linear.fragment
linear.tangent_inputs
linear.tangent_outputs
```

with:

```rust
linear.as_graph()
linear.tangent_inputs()
linear.tangent_outputs()
```

For owned graph use, use:

```rust
Arc::new(linear.into_graph())
```

- [ ] **Step 5: Run focused compile check**

Run:

```bash
cargo check --tests
```

Expected: FAIL until graph transform module names are updated in Task 5.

## Task 5: Rename Graph Transforms To linearize And linear_transpose

**Files:**
- Rename: `src/differentiate.rs` -> `src/linearize.rs`
- Rename: `src/transpose.rs` -> `src/linear_transpose.rs`
- Modify: `src/lib.rs`
- Modify: `src/eager/backward.rs`
- Modify: tests using graph transforms

- [ ] **Step 1: Rename files**

Run:

```bash
git mv src/differentiate.rs src/linearize.rs
git mv src/transpose.rs src/linear_transpose.rs
```

- [ ] **Step 2: Rename functions in `src/linearize.rs`**

Change function signatures to:

```rust
pub fn linearize<Op: Primitive>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
    wrt: &[Op::InputKey],
    pass: DiffPassId,
    ctx: &mut Op::ADContext,
    aliases: &HashMap<Op::InputKey, GlobalValKey<Op>>,
) -> LinearizedGraph<Op>
where
    Op::InputKey: ADKey,
{
    match linearize(view, outputs, wrt, pass, ctx, aliases) {
        Ok(linear) => linear,
        Err(err) => panic!("{err}"),
    }
}

pub fn linearize<Op: Primitive>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
    wrt: &[Op::InputKey],
    pass: DiffPassId,
    ctx: &mut Op::ADContext,
    aliases: &HashMap<Op::InputKey, GlobalValKey<Op>>,
) -> ADRuleResult<LinearizedGraph<Op>>
where
    Op::InputKey: ADKey,
```

Replace the call to primitive local rule with:

```rust
let mut primitive_builder = crate::rules::FragmentPrimitiveBuilder::new(&mut builder);
let tangent_out = op.jvp_rule(
    &mut primitive_builder,
    &input_keys,
    &output_keys,
    &tangent_in,
    ctx,
)?;
```

Construct the return value with:

```rust
Ok(LinearizedGraph::from_parts(
    builder.build(),
    tangent_inputs,
    tangent_outputs,
))
```

- [ ] **Step 3: Rename functions in `src/linear_transpose.rs`**

Change function signatures to:

```rust
pub fn linear_transpose<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
    ctx: &mut Op::ADContext,
) -> LinearizedGraph<Op>
where
    Op::InputKey: ADKey,

pub fn linear_transpose<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
    ctx: &mut Op::ADContext,
) -> ADRuleResult<LinearizedGraph<Op>>
where
    Op::InputKey: ADKey,
```

Inside the implementation, replace direct field reads with:

```rust
let graph = linear.as_graph();
for (index, maybe_tangent_output) in linear.tangent_outputs().iter().enumerate() {
    let Some(tangent_output_id) = maybe_tangent_output else {
        continue;
    };
    let source_key = graph.vals()[*tangent_output_id].key.clone();
    let seed_key = cotangent_seed_key(linear, index);
    let seed_id = builder.add_input(seed_key.clone());
    cotangent_env.insert(source_key, seed_id);
    cotangent_seed_inputs.push((seed_key, seed_id));
}
```

Use `graph` for every source graph access in the reverse traversal:

```rust
let mut primitive_builder = crate::rules::FragmentPrimitiveBuilder::new(&mut builder);
for op_node in graph.ops().iter().rev() {
    let cotangent_out: Vec<Option<LocalValId>> = op_node
        .outputs
        .iter()
        .map(|output_id| cotangent_env.get(&graph.vals()[*output_id].key).copied())
        .collect();
    if cotangent_out.iter().all(Option::is_none) {
        continue;
    }

    let rule_inputs: Vec<PrimitiveValue<Op>> = op_node
        .inputs
        .iter()
        .map(|input| match input {
            ValRef::Local(local_id) => {
                PrimitiveValue::External(graph.vals()[*local_id].key.clone())
            }
            ValRef::External(key) => PrimitiveValue::External(key.clone()),
        })
        .collect();

    let cotangent_in = op_node.op.transpose_rule(
        &mut primitive_builder,
        &cotangent_out,
        &rule_inputs,
        &op_node.mode,
        ctx,
    )?;
}
drop(primitive_builder);
```

After this edit, `src/linear_transpose.rs` must not read
`linear.fragment`, `linear.tangent_inputs`, or `linear.tangent_outputs`
directly.

Return:

```rust
Ok(LinearizedGraph::from_parts(
    builder.build(),
    cotangent_seed_inputs,
    tangent_outputs,
))
```

- [ ] **Step 4: Add builder-backed transpose helper**

In `src/linear_transpose.rs`, add:

```rust
pub fn linear_transpose_with_builder<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
    builder: &mut impl PrimitiveBuilder<Op>,
    cotangent_seeds: &[Option<LocalValId>],
    ctx: &mut Op::ADContext,
) -> ADRuleResult<Vec<Option<LocalValId>>>
where
    Op::InputKey: ADKey,
{
    let graph = linear.as_graph();
    let mut cotangent_env: HashMap<GlobalValKey<Op>, LocalValId> = HashMap::new();

    for (index, maybe_tangent_output) in linear.tangent_outputs().iter().enumerate() {
        if let (Some(output_id), Some(Some(seed_id))) =
            (maybe_tangent_output, cotangent_seeds.get(index))
        {
            let key = graph.vals()[*output_id].key.clone();
            cotangent_env.insert(key, *seed_id);
        }
    }

    for op_node in graph.ops().iter().rev() {
        let cotangent_out: Vec<Option<LocalValId>> = op_node
            .outputs
            .iter()
            .map(|output_id| cotangent_env.get(&graph.vals()[*output_id].key).copied())
            .collect();
        if cotangent_out.iter().all(Option::is_none) {
            continue;
        }

        let rule_inputs: Vec<PrimitiveValue<Op>> = op_node
            .inputs
            .iter()
            .map(|input| match input {
                ValRef::Local(local_id) => {
                    PrimitiveValue::External(graph.vals()[*local_id].key.clone())
                }
                ValRef::External(key) => PrimitiveValue::External(key.clone()),
            })
            .collect();

        let cotangent_in = op_node.op.transpose_rule(
            builder,
            &cotangent_out,
            &rule_inputs,
            &op_node.mode,
            ctx,
        )?;
        assert_eq!(
            cotangent_in.len(),
            rule_inputs.len(),
            "transpose_rule for {:?} returned {} cotangents for {} inputs",
            op_node.op,
            cotangent_in.len(),
            rule_inputs.len()
        );

        for (input, maybe_cotangent) in rule_inputs.iter().zip(cotangent_in) {
            let Some(cotangent_id) = maybe_cotangent else {
                continue;
            };
            let input_key = match input {
                PrimitiveValue::Local(_) => {
                    unreachable!("rule inputs are normalized to external refs")
                }
                PrimitiveValue::External(key) => key.clone(),
            };

            match cotangent_env.get(&input_key).copied() {
                Some(existing_id) => {
                    let sum = builder.add_primitive(
                        Op::add(),
                        vec![
                            PrimitiveValue::Local(existing_id),
                            PrimitiveValue::Local(cotangent_id),
                        ],
                        OpMode::Linear {
                            active_mask: vec![true, true],
                        },
                    );
                    cotangent_env.insert(input_key, sum[0]);
                }
                None => {
                    cotangent_env.insert(input_key, cotangent_id);
                }
            }
        }
    }

    Ok(linear
        .tangent_inputs()
        .iter()
        .map(|(_, tangent_input_id)| {
            let tangent_input_key = &graph.vals()[*tangent_input_id].key;
            cotangent_env.get(tangent_input_key).copied()
        })
        .collect())
}
```

Add these imports at the top of `src/linear_transpose.rs`:

```rust
use crate::{PrimitiveBuilder, PrimitiveValue};
use computegraph::{GlobalValKey, LocalValId, OpMode, ValRef};
```

- [ ] **Step 5: Update root modules**

In `src/lib.rs`, replace:

```rust
mod differentiate;
mod transpose;
pub use differentiate::{differentiate, linearize};
pub use transpose::{transpose, linear_transpose};
```

with:

```rust
mod linear_transpose;
mod linearize;
pub use linear_transpose::{
    linear_transpose, linear_transpose, linear_transpose_with_builder,
};
pub use linearize::{linearize, linearize};
```

- [ ] **Step 6: Remove `pub mod emit`**

Delete `src/emit.rs` after moving its helper into `src/linear_transpose.rs`, and remove `pub mod emit;` from `src/lib.rs`.

- [ ] **Step 7: Run focused graph transform tests**

Run:

```bash
cargo nextest run --release --test scalar_ad_tests --test vector_ad_tests --test fallible_ad_tests
```

Expected: PASS after tests are migrated to `linearize`, `linear_transpose`, and `LinearizedGraph`.

- [ ] **Step 8: Commit transform rename**

```bash
git add src tests
git commit -m "refactor: rename graph transforms to JAX terminology"
```

## Task 6: Migrate Eager Integration API

**Files:**
- Modify: `src/eager/record.rs`
- Modify: `src/eager/backward.rs`
- Modify: `src/eager/mod.rs`
- Modify: `src/eager/trace.rs`
- Modify: `tests/eager_record_tests.rs`
- Modify: `tests/eager_backward_tests.rs`

- [ ] **Step 1: Rename eager input/output types**

In `src/eager/record.rs`, rename:

`pub struct Input<Op: GraphOp>` becomes `pub struct EagerInput<Op: GraphOp>`.
`pub struct Output<Op: GraphOp>` becomes `pub struct EagerOutput<Op: GraphOp>`.
Keep all fields unchanged during this rename: `key`, `trace`, `requires_grad`,
`data`, and `output_slot`.

Update `Recorder::record` signature:

```rust
pub fn record<Op>(
    &mut self,
    op: Op,
    inputs: &[EagerInput<Op>],
    outputs: &[Arc<Op::Operand>],
) -> Vec<EagerOutput<Op>>
```

- [ ] **Step 2: Update eager re-exports**

In `src/eager/mod.rs`, replace:

```rust
pub use record::{Input, KeySource, Output, Recorder};
```

with:

```rust
pub use record::{EagerInput, EagerOutput, KeySource, Recorder};
```

- [ ] **Step 3: Add `PrimitiveGraph` wrapper**

Create `src/primitive_graph.rs`:

```rust
use computegraph::fragment::Fragment;
use computegraph::GraphOp;

/// Borrowed primitive computation graph passed to downstream executors.
pub struct PrimitiveGraph<'a, Op: GraphOp> {
    graph: &'a Fragment<Op>,
}

impl<'a, Op: GraphOp> PrimitiveGraph<'a, Op> {
    pub(crate) fn new(graph: &'a Fragment<Op>) -> Self {
        Self { graph }
    }

    /// Borrow the lower-level graph representation.
    pub fn as_graph(&self) -> &Fragment<Op> {
        self.graph
    }
}
```

In `src/lib.rs`, add:

```rust
mod primitive_graph;
pub use primitive_graph::PrimitiveGraph;
```

- [ ] **Step 4: Update `BackwardExecutor` signatures**

In `src/eager/backward.rs`, change the trait to:

```rust
pub trait BackwardExecutor<Op: Primitive>
where
    Op::InputKey: ADKey,
{
    fn execute_forward(
        &mut self,
        graph: PrimitiveGraph<'_, Op>,
        initial_data: &HashMap<GlobalValKey<Op>, Arc<Op::Operand>>,
    ) -> HashMap<GlobalValKey<Op>, Arc<Op::Operand>>;

    fn execute_transpose(
        &mut self,
        linear: &LinearizedGraph<Op>,
        cotangent_out: &[Option<Arc<Op::Operand>>],
        external_data: &HashMap<GlobalValKey<Op>, Arc<Op::Operand>>,
        ctx: &mut Op::ADContext,
    ) -> ADRuleResult<Vec<Option<Arc<Op::Operand>>>>;

    fn add_operands(&mut self, a: &Arc<Op::Operand>, b: &Arc<Op::Operand>) -> Arc<Op::Operand>;
}
```

- [ ] **Step 5: Pass `PrimitiveGraph` from eager backward**

In `backward`, replace the forward replay call with:

```rust
let replay_graph = PrimitiveGraph::new(linear.as_graph());
let all_values = executor.execute_forward(replay_graph, node.saved_data());
```

- [ ] **Step 6: Replace eager backward's internal linearization call**

In `src/eager/backward.rs`, replace:

```rust
crate::linearize(&view, &output_keys, &wrt_keys, 0, ctx, &aliases)
```

with:

```rust
crate::linearize(&view, &output_keys, &wrt_keys, 0, ctx, &aliases)
```

- [ ] **Step 7: Update eager tests**

Replace test helper constructors:

The helper named `scalar_input` must return `EagerInput<ScalarOp>` instead of
`Input<ScalarOp>`.

The helper named `scalar_input_from_output` must accept
`&tidu::eager::EagerOutput<ScalarOp>` and return `EagerInput<ScalarOp>`.

Replace all `emit::linear_transpose_with_builder(...)` calls with:

```rust
tidu::linear_transpose_with_builder(linear, &mut transpose_builder, &cotangent_seed_ids, ctx)
```

- [ ] **Step 8: Run eager tests**

Run:

```bash
cargo nextest run --release --test eager_record_tests --test eager_backward_tests
```

Expected: PASS.

- [ ] **Step 9: Commit eager migration**

```bash
git add src/eager tests/eager_record_tests.rs tests/eager_backward_tests.rs
git commit -m "refactor: align eager integration API"
```

## Task 7: Update All Remaining Tests And Rustdocs

**Files:**
- Modify: all `tests/*.rs`
- Modify: `tests/common/*.rs`
- Modify: `src/lib.rs`
- Modify: `src/rules/*.rs`
- Modify: `src/linearize.rs`
- Modify: `src/linear_transpose.rs`
- Modify: `src/linearized_graph.rs`

- [ ] **Step 1: Replace old graph transform names in tests**

Run:

```bash
rg -l "differentiate|linearize|transpose\\(|linear_transpose|LinearFragment|PrimitiveOp|OpEmitter|emit::" \
  tests src README.md docs/getting-started docs/tutorials docs/guides docs/architecture docs/api docs/internals \
  | xargs perl -0pi -e 's/linearize/linearize/g; s/differentiate/linearize/g; s/linear_transpose/linear_transpose/g; s/\\btranspose\\(/linear_transpose(/g; s/LinearFragment/LinearizedGraph/g; s/PrimitiveOp/Primitive/g; s/OpEmitter/PrimitiveBuilder/g'
```

Then inspect every line printed by:

```bash
rg -n "differentiate|linearize|transpose\\(|linear_transpose|LinearFragment|PrimitiveOp|OpEmitter|emit::" \
  src tests README.md docs/getting-started docs/tutorials docs/guides docs/architecture docs/api docs/internals \
  -g '*.rs' -g '*.md'
```

Expected: no matches in active source, tests, README, or active docs. Historical
plans under `docs/superpowers/` and design records under `docs/design/` are
intentionally excluded from this mechanical rewrite.

- [ ] **Step 2: Fix builder method calls in tests and macros**

Replace calls shaped like `builder.add_op(ScalarOp::Add, inputs, mode)` in AD
rule implementations and macros with:

```rust
builder.add_primitive(
    ScalarOp::Add,
    vec![
        PrimitiveValue::Local(lhs),
        PrimitiveValue::External(rhs_key.clone()),
    ],
    OpMode::Linear { active_mask: vec![true, false] },
)
```

For existing `ValRef` values in transpose helpers, use:

```rust
inputs[0].clone()
```

only after `inputs` has type `&[PrimitiveValue<Op>]`.

- [ ] **Step 3: Run all tests**

Run:

```bash
cargo nextest run --release --workspace --no-fail-fast
cargo test --doc --release --workspace
```

Expected: PASS.

- [ ] **Step 4: Commit full test/rustdoc migration**

```bash
git add src tests
git commit -m "test: migrate suite to JAX terminology"
```

## Task 8: Build Documentation Site And Tutorials

**Files:**
- Modify: `README.md`
- Modify: `src/lib.rs`
- Add: `docs/_quarto.yml`
- Add: `docs/index.md`
- Add: `docs/getting-started/index.md`
- Add: `docs/getting-started/terminology.md`
- Add: `docs/tutorials/index.md`
- Add: `docs/tutorials/primitive-linearization.md`
- Add: `docs/tutorials/eager-reverse-mode.md`
- Add: `docs/guides/implementing-primitives.md`
- Add: `docs/guides/linearize-and-transpose.md`
- Add: `docs/guides/eager-integration.md`
- Add: `docs/guides/complex-ad.md`
- Add: `docs/guides/higher-order-ad.md`
- Add: `docs/architecture/index.md`
- Add: `docs/architecture/public-boundaries.md`
- Add: `docs/architecture/computegraph-integration.md`
- Add: `docs/api/index.md`
- Add: `docs/internals/index.md`
- Add: `examples/primitive_linearization.rs`
- Add: `examples/eager_reverse_mode.rs`

- [ ] **Step 1: Replace README with short front door**

Use this opening in `README.md`:

```markdown
# tidu-rs

tidu builds automatic-differentiation transforms for primitive computation
graphs in Rust.

Downstream crates provide primitive operations, local AD rules, and concrete
runtimes. tidu builds new graphs for linearization, transposed linear maps, and
optional eager reverse-mode integration.

## Who This Is For

Read tidu docs if you are implementing a primitive operation set, AD rules, a
graph runtime, or an eager tensor frontend. If you only want tensor operations,
start with the downstream tensor/runtime crate that uses tidu.
```

- [ ] **Step 2: Add Quarto sidebar**

Create `docs/_quarto.yml`:

```yaml
project:
  type: website
  output-dir: ../target/docs-site/design
  render:
    - index.md
    - getting-started/**/*.md
    - tutorials/**/*.md
    - guides/**/*.md
    - architecture/**/*.md
    - api/**/*.md
    - internals/**/*.md

format:
  html:
    theme: flatly
    toc: true
    code-overflow: wrap
    html-math-method: katex

website:
  title: "tidu"
  sidebar:
    style: docked
    contents:
      - index.md
      - section: "Getting Started"
        contents:
          - getting-started/index.md
          - getting-started/terminology.md
      - section: "Tutorials"
        contents:
          - tutorials/index.md
          - tutorials/primitive-linearization.md
          - tutorials/eager-reverse-mode.md
      - section: "Guides"
        contents:
          - guides/implementing-primitives.md
          - guides/linearize-and-transpose.md
          - guides/eager-integration.md
          - guides/complex-ad.md
          - guides/higher-order-ad.md
      - section: "Architecture"
        contents:
          - architecture/index.md
          - architecture/public-boundaries.md
          - architecture/computegraph-integration.md
      - section: "API"
        contents:
          - api/index.md
      - section: "Internals"
        contents:
          - internals/index.md
```

- [ ] **Step 3: Add self-contained terminology page**

Create `docs/getting-started/terminology.md` with sections for primitive operation, primitive computation graph, linearization, linear transpose, JVP rule, transpose rule, and eager integration. Include this symbolic example:

```text
f(x) = x * x

linearize f at x with tangent dx:
  y  = x * x
  dy = x * dx + dx * x

linear_transpose of dy = J dx with seed ct_y:
  ct_x = x * ct_y + x * ct_y
```

- [ ] **Step 4: Add runnable examples**

Create `examples/primitive_linearization.rs` and `examples/eager_reverse_mode.rs`. Each example must:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build a small primitive graph,
    // call tidu APIs,
    // assert deterministic numeric results,
    Ok(())
}
```

Use the migrated tiny scalar primitive vocabulary from `tests/common/mod.rs`: `ScalarKey`, `ScalarOp::Add`, `ScalarOp::Mul`, `ScalarOp::Neg`, and `ScalarOp::Exp`.

- [ ] **Step 5: Link tutorial docs to runnable examples**

In `docs/tutorials/primitive-linearization.md`, include a normal fenced Rust
snippet copied from `examples/primitive_linearization.rs` and wrap it with:

```markdown
<!-- snippet-source: examples/primitive_linearization.rs -->
<!-- end-snippet-source -->
```

Use the same pattern in `docs/tutorials/eager-reverse-mode.md` for
`examples/eager_reverse_mode.rs`.

- [ ] **Step 6: Run docs checks**

Run:

```bash
cargo run --example primitive_linearization
cargo run --example eager_reverse_mode
bash scripts/build_docs_site.sh
```

Expected: both examples pass and docs-site builds into `target/docs-site`.

- [ ] **Step 7: Commit docs**

```bash
git add README.md src/lib.rs docs examples
git commit -m "docs: add downstream implementer guide"
```

## Task 9: Final Verification And PR Preparation

**Files:**
- All changed files.

- [ ] **Step 1: Format and lint**

Run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: PASS.

- [ ] **Step 2: Run full tests and docs**

Run:

```bash
cargo nextest run --release --workspace --no-fail-fast
cargo test --doc --release --workspace
cargo doc --workspace --no-deps
bash scripts/build_docs_site.sh
```

Expected: PASS.

- [ ] **Step 3: Check old public vocabulary is gone from active docs/source**

Run:

```bash
rg -n "differentiate|linearize|LinearFragment|PrimitiveOp|OpEmitter|tidu::emit|eager::Input|eager::Output" \
  README.md src tests docs/getting-started docs/tutorials docs/guides docs/architecture docs/api docs/internals
```

Expected: no matches.

- [ ] **Step 4: Review git diff**

Run:

```bash
git diff --check
git status --short
git log --oneline --max-count=8
```

Expected: no whitespace errors, clean staged/unstaged state after final commit, and commits split by public API, eager API, tests, and docs.

- [ ] **Step 5: Open PR**

Create the PR body:

```bash
cat >/tmp/tidu-jax-terminology-pr.md <<'MD'
## Summary

- rename graph transforms from `differentiate`/`transpose` to `linearize`/`linear_transpose`
- replace the intended primitive rule API with `Primitive`, `PrimitiveBuilder`, and `PrimitiveValue`
- move fragment/emitter terminology out of the primary public path
- rename eager integration types to `EagerInput`/`EagerOutput`
- add downstream implementer docs, runnable tutorials, and the Quarto docs site

## Follow-up

- migrate tenferro-rs to the merged tidu commit and updated API names

## Verification

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo nextest run --release --workspace --no-fail-fast`
- `cargo test --doc --release --workspace`
- `cargo doc --workspace --no-deps`
- `bash scripts/build_docs_site.sh`
MD
```

Push and open a PR:

```bash
git push -u origin codex/jax-terminology-docs
gh pr create -R tensor4all/tidu-rs --base main --head codex/jax-terminology-docs \
  --title "Refactor tidu public API around JAX terminology" \
  --body-file /tmp/tidu-jax-terminology-pr.md
```

## Follow-Up Plan Required After Merge

After this tidu PR merges, create a separate tenferro-rs migration plan and PR:

- update tidu git rev in tenferro manifests,
- migrate imports and API names,
- adjust eager `BackwardExecutor` implementations,
- run tenferro workspace clippy/tests/docs,
- open and merge the downstream PR.
