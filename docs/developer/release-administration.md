# Release administration

This runbook activates and verifies the production Release path. It contains no
credentials. Complete it with organization, repository, Docker Hub, and PyPI
administrators; do not mark the production Release path active until every
verification and the production record are complete.

The only production path is:

1. a reviewed pull request is squash-merged to `main`;
2. Release Please opens or updates the **Release Proposal**;
3. a maintainer merges the passing Release Proposal;
4. Release Please creates the `vX.Y.Z` tag and published GitHub Release; and
5. `.github/workflows/publish-release.yml` publishes Docker first and PyPI only
   after Docker succeeds.

Do not publish locally, manually create or move Release tags, or manually create
the corresponding GitHub Release.

## 1. Protect `main` and configure merges

In the repository's pull-request settings:

- enable **Allow squash merging** only;
- disable merge commits and rebase merging; and
- set the default squash commit message to **Pull request title**.

Create an active branch ruleset targeting `main` with no bypass for the Release
Please GitHub App. Require:

- a pull request before merging;
- at least one approval from a maintainer (use a required maintainer team or
  code-owner review where the repository's roles do not guarantee this);
- squash as the required merge type; and
- these required status checks:
  - `Validate PR title (Release Impact)`
  - `Automated tests (CPU-only)`

Apply the rule to administrators as well. Confirm a direct push is rejected; a
rule that administrators can silently bypass does not satisfy this policy.

Verify the settings with a representative normal pull request. Record evidence
that an invalid title fails, the corrected title passes, CPU CI passes, a
maintainer approves, other merge methods are unavailable, and the resulting
single commit title exactly matches the reviewed pull-request title.

## 2. Create the organization-owned GitHub App

This is a new private GitHub App created by a `BioGeMT` organization owner for
this Release workflow. It is not an existing Marketplace app. A suitable name
is **Agentomics Release Automation**. The workflow uses the App to obtain a
short-lived installation token so Release Please can create bot-authored
Release Proposals, tags, and GitHub Releases without a maintainer's personal
access token. App-authored proposals trigger the ordinary required checks;
proposals created with the built-in `GITHUB_TOKEN` might not.

An organization owner should create it as follows:

1. Open the `BioGeMT` organization on GitHub, then select **Settings > Developer
   settings > GitHub Apps > New GitHub App**.
2. Enter a globally unique app name and use the repository URL as the homepage
   URL. No callback URL is needed.
3. Clear **Active** under **Webhook**. This App does not receive webhook events.
4. Under **Repository permissions**, grant only the permissions in the table
   below. Leave every other permission at **No access**.
5. Under **Where can this GitHub App be installed?**, select **Only on this
   account**, then create the App.

| Repository permission | Access | Reason |
| --- | --- | --- |
| Contents | Read and write | Proposal branches, tags, and GitHub Releases |
| Pull requests | Read and write | Create and update the Release Proposal |
| Issues | Read and write | Manage Release Please lifecycle labels |
| Metadata | Read-only | Granted automatically by GitHub |

Do not grant Actions, Administration, Deployments, Environments, Members,
Packages, Secrets, or Workflows access. Do not add the App to a branch-rule
bypass list.

On the new App's settings page:

1. Copy **Client ID** (not **App ID**).
2. Under **Private keys**, select **Generate a private key**. GitHub downloads a
   `.pem` file; treat it as a credential.
3. Select **Install App**, choose `BioGeMT`, choose **Only select repositories**,
   select `Agentomics-ML`, and install it.
4. In `BioGeMT/Agentomics-ML`, open **Settings > Secrets and variables >
   Actions > Variables** and create a **repository variable** named
   `RELEASE_PLEASE_APP_CLIENT_ID` with the copied Client ID. Do not create an
   environment variable; the Release Proposal job does not use a GitHub
   environment. An organization variable is unnecessary for this single-repository
   App.
5. On the **Secrets** tab, create a **repository secret** named
   `RELEASE_PLEASE_APP_PRIVATE_KEY` and paste the
   complete contents of the downloaded PEM file, including its `BEGIN` and
   `END` lines.
6. Store or remove the downloaded PEM according to the organization's credential
   policy. Never commit it or paste it into an issue, log, or production record.

The existing `.github/workflows/release-proposal.yml` reads those two values and
requests a token limited to this repository and the permissions above. No App
code or separate server needs to be deployed.

After a release-worthy normal PR reaches `main`, confirm the App-authored
Release Proposal starts ordinary pull-request workflows and both required checks
finish. A proposal whose checks remain pending is not production-ready.

## 3. Configure Docker Hub

`DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` are credentials created in Docker
Hub, then stored as GitHub repository secrets. The username is not generated by
GitHub.

### Create a dedicated automation user and token

Create or use a dedicated Docker automation user with write access to
`biogemt/agentomics`. Do not use a maintainer's everyday account. Restrict this
user's organization and repository access to `biogemt/agentomics` only.

While signed in as that automation user:

1. Select the avatar in Docker Home, then **Account settings > Personal access
   tokens > Generate new token**.
2. Name it `Agentomics GitHub Actions`, set an expiry, and select **Read &
   Write** access. Do not grant Delete access.
3. Select **Generate** and immediately copy the token. Docker shows it only
   once.

The values are:

```text
DOCKERHUB_USERNAME = <the automation user's Docker ID>
DOCKERHUB_TOKEN    = <the generated personal access token>
```

The username is the automation user's Docker ID—not its email address, the
`biogemt` organization name, or the repository name. Personal access tokens
cannot themselves be restricted to one repository, so the dedicated user's
limited repository membership provides the access boundary. Record the user,
its repository access, and the token owner in the activation evidence.

### Store the credentials in GitHub

In `BioGeMT/Agentomics-ML`, open **Settings > Secrets and variables > Actions >
Secrets** and create these two **repository secrets**:

- `DOCKERHUB_USERNAME` with the username described above;
- `DOCKERHUB_TOKEN` with the copied token.

Do not put either value in source, an issue, or workflow logs. The publication
workflow passes them only to `docker/login-action` in the Docker job.

### Make Release tags immutable

In the `biogemt/agentomics` Docker Hub repository, open **Settings > General >
Tag mutability settings**. Select **Specific tags are immutable** and enter a
regular expression that matches complete stable semantic-version tags only:

```text
^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$
```

Save the setting. The workflow must publish only the exact `X.Y.Z` tag; do not
add `latest`, major-only, minor-only, or other aliases.

## 4. Configure PyPI Trusted Publishing

On the existing PyPI `agentomics` project, add one GitHub Actions Trusted
Publisher with these exact values:

| Field | Value |
| --- | --- |
| Repository owner | `BioGeMT` |
| Repository name | `Agentomics-ML` |
| Workflow filename | `publish-release.yml` |
| Environment name | `pypi` |

After adding the Trusted Publisher, create its matching GitHub environment:

1. Open `BioGeMT/Agentomics-ML` on GitHub.
2. Select **Settings > Environments > New environment**.
3. Enter the exact, case-sensitive name `pypi`, then select **Configure
   environment**.
4. Leave **Required reviewers** unconfigured.
5. Leave **Wait timer** disabled or set to `0` minutes.
6. Do not enable any custom deployment-protection rules.
7. Under **Deployment branches and tags**, leave the policy at **No
   restriction**. In particular, do not configure a branch-only policy: this
   workflow runs for a published `vX.Y.Z` tag.
8. Do not add any environment secrets or variables. No PyPI API token is
   required.

The environment's name is part of the OIDC identity that PyPI checks. The
`pypi` job in `.github/workflows/publish-release.yml` names this environment and
has `id-token: write`, so GitHub provides a short-lived OIDC identity after the
Docker job succeeds. PyPI exchanges that identity for temporary publication
credentials. No stored PyPI password or token is involved, and merging the
Release Proposal remains the only manual publication approval.

## 5. Test the setup

This test creates a real Release on Docker Hub and PyPI. Use a genuine
release-worthy change; do not merge the Release Proposal merely as a dry run.

1. Open a normal PR to `main`. Confirm an invalid title fails, then correct it to
   a valid title such as `fix: ...` and confirm title validation and CPU tests
   pass. Get maintainer approval and squash-merge it.
2. Confirm the GitHub App opens or updates one Release Proposal with the expected
   version and changelog, and that the proposal's normal required checks pass.
3. Merge the Release Proposal. This is the final approval; there should be no
   additional PyPI approval prompt.
4. Confirm the same `X.Y.Z` appears in the Git tag, GitHub Release, Docker Hub
   tag and AMD64/ARM64 manifest, PyPI version, installed
   `agentomics.__version__`, and default Docker image.
5. Record links to the normal PR, Release Proposal, checks, tag, GitHub Release,
   publication workflow, Docker artifact, and PyPI artifact in the activation
   issue. Never record credentials.

