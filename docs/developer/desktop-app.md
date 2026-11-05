# Building the Desktop App

The desktop app (`ui/`) is built into an installer for each OS:

| OS | Build it on | Installer (in `ui/desktop/release/`) |
|----|-------------|--------------------------------------|
| macOS (Apple silicon) | a Mac | `Agentomics-<version>-arm64.dmg` |
| Linux x64 | a Linux machine | `Agentomics-<version>-x86_64.AppImage` and `agentomics_<version>_amd64.deb` |
| Windows x64 | a Windows machine | `Agentomics-<version>-x64-setup.exe` |

!!! warning "You can't choose the OS: the build is for the machine it runs on"
    There is no option to pick an OS. `task dist` (or `npm run dist`) always builds the installer for
    the OS of the machine you run it on:

    - want the **Linux** installer → run it on a **Linux** machine
    - want the **Windows** installer → run it on a **Windows** machine
    - want the **macOS** installer → run it on a **Mac**

    A Mac can't build the Linux or Windows installers, and the other way round: the app carries its
    own `agentomics-run`, built with PyInstaller, which only builds for the OS it runs on. Without a
    machine of that OS, use [GitHub Actions](#all-three-with-github-actions) and download only the
    installer you want.

The installers are unsigned. The computer the app is installed on needs only Docker.

## What you need (every OS)

- [Node.js](https://nodejs.org/) 22 with npm
- [uv](https://docs.astral.sh/uv/getting-started/installation/), which builds the bundled `agentomics-run`
  (it fetches its own Python 3.12)
- [Task](https://taskfile.dev/installation/) (optional; the plain npm commands are given too)

## macOS

In Terminal, from the repository root:

```bash
cd ui
task dist
```

Without Task:

```bash
npm --prefix ui/renderer install
npm --prefix ui/desktop install
cd ui/desktop
npm run dist
```

Result: `ui/desktop/release/Agentomics-<version>-arm64.dmg`.

## Linux

In a terminal, from the repository root:

```bash
cd ui
task dist
```

Without Task:

```bash
npm --prefix ui/renderer install
npm --prefix ui/desktop install
cd ui/desktop
npm run dist
```

Result: `ui/desktop/release/Agentomics-<version>-x86_64.AppImage` and
`ui/desktop/release/agentomics_<version>_amd64.deb`.

## Windows

In PowerShell, from the repository root:

```powershell
cd ui
task dist
```

Without Task:

```powershell
npm --prefix ui/renderer install
npm --prefix ui/desktop install
cd ui/desktop
npm run dist
```

Result: `ui\desktop\release\Agentomics-<version>-x64-setup.exe`.

## All three with GitHub Actions

The **Desktop app** workflow (`.github/workflows/desktop.yml`) builds on macOS, Linux and Windows
runners at once. It doesn't run on its own; start it by hand:

- On GitHub: **Actions → Desktop app → Run workflow**, pick the branch, **Run workflow**.
- Or with the GitHub CLI:

    ```bash
    gh workflow run desktop.yml --ref <branch>
    gh run watch                      # follow it
    gh run download <run-id>          # fetch the installers
    ```

The workflow always builds all three; it has no option to pick one OS. The installers are attached
to the run as separate artifacts, so download only the one you want:

| You want | Download the artifact | With the GitHub CLI |
|----------|-----------------------|---------------------|
| macOS | `agentomics-desktop-macOS` | `gh run download <run-id> -n agentomics-desktop-macOS` |
| Linux | `agentomics-desktop-Linux` | `gh run download <run-id> -n agentomics-desktop-Linux` |
| Windows | `agentomics-desktop-Windows` | `gh run download <run-id> -n agentomics-desktop-Windows` |

!!! note
    GitHub only offers to run a workflow by hand once the workflow file is on the default branch
    (`main`). After that, you can run it on any branch.
