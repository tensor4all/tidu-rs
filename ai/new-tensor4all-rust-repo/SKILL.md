---
name: new-tensor4all-rust-repo
description: Use when creating or bootstrapping a new tensor4all numerical Rust repository from template-rs
---

# New Tensor4all Rust Repo

Create a new repository from `tensor4all/template-rs`, clone it locally, install the initial project-local agent assets, and verify the generated baseline.

## Required Inputs

- repository name
- short description

Optional inputs:

- organization
- visibility
- destination path
- `--rollback-on-failure`

## Workflow

1. Run the bootstrap script:

```bash
bash scripts/new-repo.sh --repo <name> --description "<short description>"
```

2. The script performs:
  - `gh` authentication and repository preflight checks
  - remote repo creation from `tensor4all/template-rs`
  - local clone
  - GitHub repo settings configuration for auto-merge and required status checks
  - initial agent-assets sync
  - baseline verification, including release-mode coverage verification and docs-site completeness

3. Report:
  - created GitHub URL
  - local clone path
  - verification result
  - any partial-success state if bootstrap failed mid-flight

## Safety Rules

- Do not overwrite a non-empty destination path.
- Do not delete a remotely created repo unless `--rollback-on-failure` was explicitly requested.
- Stop if GitHub repo settings cannot be configured correctly.
- Do not create a PR as part of bootstrap.
