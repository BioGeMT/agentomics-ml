# Contributing to Agentomics

Pull requests targeting `main` must declare **Release Impact** in the title:

```text
<type>[!]: <description>
```

Agentomics versions use `MAJOR.MINOR.PATCH`, for example `2.4.1`:

- `MAJOR` changes when compatibility is broken.
- `MINOR` changes for backward-compatible functionality.
- `PATCH` changes for backward-compatible fixes or performance improvements.

| Type | Use for | Default impact | Example |
| --- | --- | --- | --- |
| `feat` | New functionality | Minor | `feat: add CSV exports` |
| `fix` | Correcting a defect | Patch | `fix: preserve uploaded filenames` |
| `perf` | Improving performance | Patch | `perf: reduce parser memory use` |
| `docs` | Documentation only | None | `docs: explain workspace layout` |
| `test` | Tests only | None | `test: cover image selection` |
| `ci` | CI or workflow changes | None | `ci: cache container builds` |
| `chore` | Repository maintenance | None | `chore: update maintainers` |
| `refactor` | Internal changes without changed behavior | None | `refactor: share argument parsing` |

Add `!` to `feat`, `fix`, or `perf` for Major impact, for example `feat!: remove legacy configuration`. Authors propose the impact. Maintainers require or remove `!` after reviewing compatibility.

Pull requests are squash-merged using the reviewed PR title as the commit title.

## Testing standard

### What to test

- For each change in your PR, identify what could break and test each such potential regression.
- If a test already exists for that, update it. Delete tests for behavior that no longer exists.
- Before fixing a bug, ensure a test exists that fails because of that bug.

### How to test

- Use pytest. All tests must be deterministic.
- Name tests after the behavior being checked and assert that behavior. Do not rely on internal implementation details.
- Prefer testing through the **lowest stable interface**: the lowest-level function expected to stay the same when codebase changes. Avoid testing functions like private utilities and unnecessarily high-level functions. For example, test metric calculation through `get_metrics`, not a private _get_auroc or the entire `agentomics-run` workflow.
- Use production functions to create valid test state when possible. Do not duplicate production code or reimplement their ordering into fixtures or helpers (this can cause drift).
- Name fixtures after the state they create. E.g. a configuration fixture must not initialize a run.
- Keep scenario-specific setup local to its test module.

### Test placement

- **Unit** (`tests/unit/`): calls one lowest stable interface.
- **Integration** (`tests/integration/`): checks multiple lowest stable interfaces working together, directly or through a higher-level production function.
- **End-to-end** (`tests/end_to_end/`): runs outside the application container, invokes a user-facing CLI through Docker, and checks results. Scripted LLM behavior at the provider boundary is allowed.

### Agent evaluations

Assess real-LLM behavior through **agent evaluations**, not deterministic automated tests. Agent evaluations may be expensive, nondeterministic, and use LLM judges. (To be implemented)

### How to run tests

From the repository root, with Docker running:

```bash
pip install -e .
python scripts/run_tests.py --cpu-only
```

This will use the current repository code (including unstaged and uncommitted changes) to run the tests. Omit `--cpu-only` to require GPU access and include GPU tests.
