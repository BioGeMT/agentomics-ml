# Releasing Agentomics

The pip package and the Docker image are released together at a single version,
the one declared in `src/agentomics/__init__.py` (`__version__`). The installed
package defaults to the matching image tag (`biogemt/agentomics:<version>`), so
users only ever bump one thing.

## Release steps

1. Bump `__version__` in `src/agentomics/__init__.py` (semantic versioning — bump
   the MAJOR when a persisted contract changes: config schema, workspace layout,
   or the host/container argument or environment contract). Merge it to `main`.
2. From an up-to-date `main`, tag the release commit and push the tag:
   ```bash
   git checkout main && git pull
   git tag v<version>
   git push origin v<version>
   ```
3. Ensure you are logged in for both registries:
   - `docker login` with push access to the agentomics image repo.
   - a PyPI token available to twine (`~/.pypirc`, or `TWINE_USERNAME`/`TWINE_PASSWORD`).
4. Run the release script — it builds and pushes the image (the tag the installed
   package pulls) from the git tag, then builds and uploads the pip package. It
   refuses to run if the tree is dirty, `HEAD` is not the tag, the tag is unpushed
   or not on `main`, or `<version>` is already on PyPI (a reminder to bump
   `__version__`):
   ```bash
   ./scripts/release.sh
   ```

## Recovering from a broken release

If the build fails or you spot a problem **before anything is published**, move
the tag freely: `git tag -d v<version> && git push origin :v<version>`, push the
fix to `main`, then re-tag. Once the image and wheel are **published**, the
version is immutable — never reuse it. Bump `__version__` to the next patch, and
release again.

## Testing a version locally before releasing

Build the image from your working tree (including uncommitted changes) and run
against it with `--image` — nothing is pushed to any registry:

```bash
docker build --build-arg REPOSITORY_SOURCE=. -t agentomics:dev .
agentomics-run --image agentomics:dev --dataset <name>
```

The `Dockerfile` copies `envs/` in before creating the conda environments and the
rest of the sources in after, so rebuilds skip the environments unless `envs/`
changed. When the code diff touches `envs/`, expect the environments to rebuild
(a build that skips them would test that code against your old dependencies).
