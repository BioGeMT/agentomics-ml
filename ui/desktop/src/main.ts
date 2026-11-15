// Agentomics desktop app.
//
// The window shows the UI in ui/renderer. This main process does what a browser can't:
// picks folders, runs Docker, starts `agentomics-run`, and keeps the run history.
import { app, BrowserWindow, dialog, nativeImage, shell } from "electron";
import fs from "node:fs";
import path from "node:path";
import * as accounts from "./accounts";
import { applyAppearance, getAppearance, setAppearance } from "./appearance";
import { checkDataset } from "./datasets";
import { exampleDataset } from "./examples";
import { addDockerToPath, dockerStatus, openDocker } from "./docker";
import { handleDownloads } from "./downloads";
import { datasetFolderOf } from "./history";
import { requireDirectory, requireImage } from "./exec";
import * as images from "./images";
import { handle, send } from "./ipc";
import { ICON, PACKAGED, REPO_ROOT, RUNS_DIR, USE_PROTOCOL, RENDERER_ROOT } from "./paths";
import { registerScheme, serveApp } from "./protocol";
import { deleteRun, isRunning, listRuns, runEnding, runMenu, runnerOutcome, startRun, stopRun } from "./runs";
import { createWindow, setMenu, setViewState, stopDevServer } from "./window";

app.setName("Agentomics");
// The run history and the saved OpenRouter key. The development app keeps the folder from before the
// rename; the packaged app has its own, since its runs live elsewhere. AGENTOMICS_DATA_DIR points a test
// copy at its own folder.
app.setPath("userData", process.env.AGENTOMICS_DATA_DIR || path.join(app.getPath("appData"), PACKAGED ? "Agentomics" : "agentomics-desktop"));
registerScheme();
addDockerToPath();

// ---------- IPC: what the UI can ask for ----------

handle("example-dataset", () => exampleDataset());
handle("pick-folder", async () => {
  const result = await dialog.showOpenDialog({ properties: ["openDirectory"], title: "Choose a dataset folder" });
  return result.canceled ? null : result.filePaths[0];
});

handle("list-images", () => images.listImages());
handle("set-image", (_event, name) => {
  requireImage(name);
  return images.setImage(name);
});
handle("check-live-view", (_event, name) => {
  requireImage(name);
  return images.checkLiveView(name);
});
handle("pull-image", (event, name) => {
  requireImage(name);
  return images.pullImage(name, (progress) => send(event.sender, "pull-progress", progress));
});

handle("get-accounts", () => accounts.status(REPO_ROOT));
handle("codex-sign-in", () => accounts.codexSignIn());
handle("save-openrouter-key", (_event, key) => accounts.saveOpenRouterKey(key));
handle("remove-openrouter-key", () => accounts.removeOpenRouterKey());
handle("list-models", (_event, provider = "Codex") => (provider === "OpenRouter" ? accounts.openRouterModels(REPO_ROOT) : accounts.codexModels()));

handle("check-dataset", (_event, folder, image) => checkDataset(folder, image));
handle("start-run", (_event, options) => startRun(options));
handle("list-runs", () => listRuns());
handle("stop-run", (_event, id) => stopRun(id));
handle("runner-outcome", (_event, workspace) => runnerOutcome(String(workspace)));
handle("show-file", (_event, file) => {
  // Only a run's runner log, which the app wrote itself.
  const resolved = path.resolve(String(file));
  if (path.dirname(resolved) === path.resolve(RUNS_DIR) && resolved.endsWith(".log") && fs.existsSync(resolved)) shell.showItemInFolder(resolved);
});
handle("delete-run", (event, id) => deleteRun(event, id));
handle("run-menu", (event, id) => runMenu(event, id));
handle("set-view-state", (_event, state) => {
  const valid = state === null || (typeof state?.contents === "boolean" && typeof state?.figures === "boolean");
  setViewState(valid ? (state && { contents: state.contents, figures: state.figures }) : null);
});
handle("get-appearance", () => getAppearance());
handle("docker-status", () => dockerStatus());
handle("open-docker", () => openDocker());
handle("set-appearance", (_event, appearance) => setAppearance(appearance));
handle("show-folder", async (_event, folder) => {
  requireDirectory(folder);
  await shell.openPath(folder);
});

// ---------- lifecycle ----------

// One copy of the app: launching it again brings the open window forward instead of quitting silently.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    const window = BrowserWindow.getAllWindows()[0];
    if (!window) return;
    if (window.isMinimized()) window.restore();
    window.focus();
  });
}

app.whenReady().then(() => {
  if (USE_PROTOCOL) serveApp({ webRoot: RENDERER_ROOT, runRoots: [RUNS_DIR], datasetFor: datasetFolderOf, isRunning, runEnding });
  handleDownloads();
  applyAppearance(); // the Mac's appearance unless Settings pins light or dark
  setMenu();
  if (process.platform === "darwin") app.dock?.setIcon(nativeImage.createFromPath(ICON));
  return createWindow();
});

app.on("window-all-closed", () => {
  stopDevServer();
  app.quit();
});
