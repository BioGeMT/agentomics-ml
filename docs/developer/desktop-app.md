# Building the Desktop App

The desktop app (`ui/`) is built into an installer for each OS:

| OS | Build it on | Installer (in `ui/desktop/release/`) |
|----|-------------|--------------------------------------|
| macOS (Apple silicon) | a Mac | `Agentomics-<version>-arm64.dmg` |
| Linux x64 | a Linux machine | `Agentomics-<version>-x86_64.AppImage` and `agentomics_<version>_amd64.deb` |
| Windows x64 | a Windows machine | `Agentomics-<version>-x64-setup.exe` |

When a build finishes it prints where its installers are, and whether they are signed and notarized. They are
in `ui/desktop/release/` of the repository: open it with `open ui/desktop/release` (macOS),
`xdg-open ui/desktop/release` (Linux) or `explorer ui\desktop\release` (Windows).

`task dist` signs and notarizes the macOS app when a Developer ID certificate is available (see
[Signing](#signing)), which adds 10–15 minutes. To test on your own machine, `task dist-unsigned` builds the
same installers without signing or notarizing.

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

The macOS app is signed and notarized when a Developer ID certificate is available (see [Signing](#signing)); otherwise, and on Linux and Windows, the installers are unsigned. The computer the app is installed on needs only Docker.

## What you need (every OS)

- [Node.js](https://nodejs.org/) 22.12 or newer, with npm
- [uv](https://docs.astral.sh/uv/getting-started/installation/), which builds the bundled `agentomics-run`
  (it fetches its own Python 3.12)
- [Task](https://taskfile.dev/installation/) (optional; the plain npm commands are given too)

`task check` confirms these are in place (and on macOS, whether the build will be signed and notarized);
`task dist` runs it first.

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

Result: `ui/desktop/release/Agentomics-<version>-arm64.dmg`. To try it, see [Testing a build](#testing-a-build).

With Xcode 26 or newer, the app's icon is `ui/desktop/Agentomics.icon` (made in Icon Composer), which macOS
draws in Liquid Glass. Without it, the build uses the flat `ui/desktop/icon.png`.

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
`ui/desktop/release/agentomics_<version>_amd64.deb`. To try them, see [Testing a build](#testing-a-build).

## Windows

Build in PowerShell. Runs need [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)
with the WSL 2 backend, set to Linux containers (its default).

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

Result: `ui\desktop\release\Agentomics-<version>-x64-setup.exe`, the installer. It is unsigned, so Windows
SmartScreen warns before it runs (see [Signing](#signing)). To try it, see [Testing a build](#testing-a-build).

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

## Testing a build

The same on every OS, once the app is installed:

1. Build the Docker image runs use, with this checkout's code: from `ui`, run `task image`. It builds
   `agentomics:dev` (slow the first time).
2. Open the app. In **Settings**, select the image `agentomics:dev`, and connect **Codex** (ChatGPT) or
   **OpenRouter**.
3. **New Run → Try an Example**, set **Iterations** to 1, and **Start Run**.
4. Check:
    - the steps appear live, with their tool calls;
    - the run ends as **Finished**;
    - **Figures** shows the metrics, and the held-out test result once the run ends;
    - **Files** opens the run's files.

The example is `ui/examples/breast_cancer/` (152 KB), what `agentomics-download-dataset --dataset breast_cancer`
creates from scikit-learn's copy of the
[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
dataset (W. Wolberg, O. Mangasarian, N. Street, W. Street; UCI Machine Learning Repository, CC BY 4.0).

## Signing

Unsigned, the app still runs, but each OS warns first: macOS asks to allow it in **System Settings ›
Privacy & Security**, and Windows SmartScreen shows "Windows protected your PC" (**More info › Run
anyway**). Linux doesn't check signatures.

### macOS: Developer ID and notarization

This needs an Apple Developer Program membership. The app shows the account's name as its developer.

1. **Certificate.** In [Certificates, Identifiers & Profiles](https://developer.apple.com/account/resources/certificates/list),
   the team's Account Holder creates a **Developer ID Application** certificate (an *Apple Development*
   certificate can't sign apps for other people). Install it in your keychain, and export it with its
   private key as a `.p12` file, with a password, for CI.
2. **Notarization credentials.** An [app-specific password](https://support.apple.com/en-us/102654) for
   the Apple ID, and the team's Team ID (shown in the developer account under Membership).

On your Mac, with the certificate in your keychain, `task dist` signs the app and its DMG. To also notarize
both (and staple the tickets, so they open without warnings even offline), store the credentials once in the
keychain and name the profile when building:

```bash
xcrun notarytool store-credentials agentomics --apple-id <apple-id> --team-id <team-id>
APPLE_KEYCHAIN_PROFILE=agentomics task dist
```

If the keychain has more than one Developer ID certificate, pick one with
`CSC_NAME="<name as shown in Keychain Access>"`.

In the **Desktop app** workflow, add these repository secrets (**Settings › Secrets and variables ›
Actions**); the macOS build is then signed and notarized:

| Secret | Value |
|--------|-------|
| `MAC_CERTIFICATE_P12_BASE64` | the `.p12` file, base64-encoded: `base64 -i certificate.p12 \| pbcopy` |
| `MAC_CERTIFICATE_PASSWORD` | the `.p12` file's password |
| `APPLE_ID` | the Apple ID used for notarization |
| `APPLE_APP_SPECIFIC_PASSWORD` | its app-specific password |
| `APPLE_TEAM_ID` | the team's Team ID |

Check a signed build with `codesign --verify --deep --strict --verbose=2 Agentomics.app` and
`spctl --assess --type execute --verbose Agentomics.app` (which should say "Notarized Developer ID").

The app runs with the hardened runtime, with the entitlements in `ui/desktop/signing/entitlements.mac.plist`:
Electron needs JIT, and the bundled `agentomics-run` loads its Python extension modules at run time.

### Windows

Signing needs a code-signing certificate from a certificate authority; its key has to live in hardware or a
cloud signing service, for example [Azure Trusted Signing](https://learn.microsoft.com/azure/trusted-signing/)
or [SignPath](https://about.signpath.io/) (free for open-source projects). Not set up yet.
