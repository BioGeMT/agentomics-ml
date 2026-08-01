# Releasing Agentomics

## Release a new version
On each release-worthy merge to main, an automatic Release Proposal PR is created. When merged to main, this releases a new version of agentomics to PyPI (pip).

To release a new version:

1. Open the current **Release Proposal PR**.
2. Review it.
3. Merge it. The merge is the approval to create the new Agentomics
   version. No additional manual release step is needed.

Do **not** manually:

- edit `project.version` in `pyproject.toml`
- create, move, delete, or reuse a Release tag
- create the corresponding GitHub Release.

To change the proposed version or contents, merge another pull
request to main and let the proposal automatically update.

## How the Release Proposal PR works

We use the Release Please (google tool) workflow to create the Release Proposal PR. The workflow continuously updates the proposal PR based on updates to `main` (Merging any pull request other than the Release Proposal only updates the
proposal). It calculates the next version number from their strongest Release Impact (see [Contributing](../../CONTRIBUTING.md)) automatically.
