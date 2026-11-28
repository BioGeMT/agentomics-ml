// Where things are. In development the app runs from ui/desktop (app.getAppPath()) inside the repo and
// uses its .venv and outputs/; the packaged app carries its own agentomics-run (a standalone build of the
// same CLI, scripts/build-cli.mjs) and keeps runs in the user's Documents.
import { app } from "electron";
import os from "node:os";
import path from "node:path";

// Electron's app.isPackaged keys off the executable's name, which the dev bundle renames to
// Agentomics; process.defaultApp is only set when launched with `electron .`.
export const PACKAGED = !process.defaultApp;
// The packaged app (or AGENTOMICS_UI=export) loads the exported UI through agentomics://; otherwise
// the Vite dev server is used, for instant reloads while developing.
export const USE_PROTOCOL = PACKAGED || process.env.AGENTOMICS_UI === "export";

const APP_DIR = app.getAppPath();
export const REPO_ROOT = path.resolve(APP_DIR, "..", "..");
export const RENDERER_DIR = path.join(REPO_ROOT, "ui", "renderer");
export const DEV_SERVER_URL = "http://127.0.0.1:3000";
const EXE = process.platform === "win32" ? ".exe" : "";
export const AGENTOMICS_RUN = PACKAGED
  ? path.join(process.resourcesPath, "agentomics-run", `agentomics-run${EXE}`)
  : path.join(REPO_ROOT, ".venv", process.platform === "win32" ? "Scripts" : "bin", `agentomics-run${EXE}`);
// AGENTOMICS_RUNS_DIR points a test copy somewhere else.
export const RUNS_DIR =
  process.env.AGENTOMICS_RUNS_DIR || (PACKAGED ? path.join(os.homedir(), "Documents", "Agentomics", "Runs") : path.join(REPO_ROOT, "outputs", "app"));
/** Where agentomics-run starts: it reads an optional .env there. The repo in development; the runs folder once packaged. */
export const RUN_CWD = PACKAGED ? RUNS_DIR : REPO_ROOT;
export const CONTAINER_PYTHON = "/opt/conda/envs/agentomics-env/bin/python";
/** The window and Dock icon: the Mac grid one on macOS, the full-square one (icon-square.svg) on Windows and Linux. Both rendered by npm run icon. */
export const ICON = path.join(APP_DIR, process.platform === "darwin" ? "icon.png" : "icon-square.png");
/** The built UI: bundled into the app's resources, or ui/renderer/out when trying the build in development. */
/** The example datasets the app carries: bundled into its resources, or ui/examples in development. */
export const EXAMPLES_DIR = PACKAGED ? path.join(process.resourcesPath, "examples") : path.join(REPO_ROOT, "ui", "examples");
export const RENDERER_ROOT = PACKAGED ? path.join(process.resourcesPath, "renderer") : path.join(RENDERER_DIR, "out");
