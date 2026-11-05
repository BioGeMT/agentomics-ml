// The only bridge between the web UI and the desktop app: a fixed set of calls, no Node access.
// Bundled on its own, since a sandboxed preload can require nothing but electron.
import type { DesktopApi, EventChannels, InvokeChannels } from "@agentomics/contract";
import { contextBridge, ipcRenderer, webUtils } from "electron";

// On macOS the page draws Apple window chrome (see ui/renderer/styles/mac-shell.css).
if (process.platform === "darwin") {
  const mark = () => document.documentElement?.setAttribute("data-shell", "mac");
  mark();
  window.addEventListener("DOMContentLoaded", mark);
}

function invoke<C extends keyof InvokeChannels>(channel: C, ...args: InvokeChannels[C]["args"]): Promise<InvokeChannels[C]["result"]> {
  return ipcRenderer.invoke(channel, ...args);
}

/** Listens for an event from the app; returns a function that stops listening. */
function listen<C extends keyof EventChannels>(channel: C, callback: (payload: EventChannels[C]) => void): () => void {
  const listener = (_event: Electron.IpcRendererEvent, payload: EventChannels[C]) => callback(payload);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

const api: DesktopApi = {
  pickFolder: () => invoke("pick-folder"),
  listImages: () => invoke("list-images"),
  setImage: (name) => invoke("set-image", name),
  checkLiveView: (name) => invoke("check-live-view", name),
  pullImage: (name) => invoke("pull-image", name),
  onPullProgress: (callback) => listen("pull-progress", callback),
  listModels: (provider) => invoke("list-models", provider),
  getAccounts: () => invoke("get-accounts"),
  codexSignIn: () => invoke("codex-sign-in"),
  saveOpenRouterKey: (key) => invoke("save-openrouter-key", key),
  removeOpenRouterKey: () => invoke("remove-openrouter-key"),
  pathForFile: (file) => webUtils.getPathForFile(file),
  checkDataset: (folder, image) => invoke("check-dataset", folder, image),
  startRun: (options) => invoke("start-run", options),
  listRuns: () => invoke("list-runs"),
  stopRun: (id) => invoke("stop-run", id),
  onMenu: (callback) => listen("menu", callback),
  deleteRun: (id) => invoke("delete-run", id),
  runMenu: (id) => invoke("run-menu", id),
  showFolder: (folder) => invoke("show-folder", folder),
  setViewState: (state) => invoke("set-view-state", state),
  getAppearance: () => invoke("get-appearance"),
  dockerStatus: () => invoke("docker-status"),
  openDocker: () => invoke("open-docker"),
  setAppearance: (appearance) => invoke("set-appearance", appearance),
};

contextBridge.exposeInMainWorld("agentomics", api);
