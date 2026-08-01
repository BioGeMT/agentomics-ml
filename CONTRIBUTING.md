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

Add `!` to `feat`, `fix`, or `perf` for Major impact, for example `feat!: remove legacy configuration`. Authors propose the impact; maintainers require or remove `!` after reviewing compatibility.

Pull requests are squash-merged using the reviewed PR title as the commit title.
