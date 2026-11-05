// Starting, listing, stopping and deleting runs. A run is `agentomics-run` in its own process group,
// with AGENTOMICS_UI_EVENTS=1 so it logs its events for the run page; its workspace lives in the app's runs folder.
import type { DeleteRunResult, RunEntry, RunMenuAction, StartRunOptions } from "@agentomics/contract";
import { BrowserWindow, dialog, Menu, shell, type IpcMainInvokeEvent, type MenuItemConstructorOptions } from "electron";
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import * as accounts from "./accounts";
import { imagePlatform, requireDirectory, requireImage } from "./exec";
import { endingOf, eventsFileOf, isAlive, loadHistory, runSummary, saveHistory, updateRun } from "./history";
import { selectedImage } from "./images";
import { AGENTOMICS_RUN, PACKAGED, REPO_ROOT, RUN_CWD, RUNS_DIR } from "./paths";
import { stopRunProcess } from "./processes";
import { useGpu } from "./gpu";

const processes = new Map<string, ChildProcess>(); // run id -> child process
// What the system calls its file manager and its trash, for the menu and the confirmation.
const SHOW_IN_FILES = process.platform === "darwin" ? "Show in Finder" : process.platform === "win32" ? "Show in File Explorer" : "Show in Files";
const TRASH = process.platform === "win32" ? "Recycle Bin" : "Trash";

/** Whether a run is still going: started in this session, or before the app was last quit and still alive. */
function going(entry: RunEntry | undefined): boolean {
  return Boolean(entry && (processes.has(entry.id) || (entry.status === "running" && isAlive(entry.pid))));
}

export async function startRun(options: StartRunOptions): Promise<RunEntry> {
  const { folder, image = await selectedImage(), provider = "Codex", model, iterations, timeoutMinutes, valMetric, userPrompt } = options ?? ({} as StartRunOptions);
  if (provider !== "Codex" && provider !== "OpenRouter") throw new Error("Choose ChatGPT (Codex) or OpenRouter.");
  const openRouterKey = provider === "OpenRouter" ? accounts.openRouterKey(REPO_ROOT) : null;
  if (provider === "OpenRouter" && !openRouterKey) throw new Error("Add an OpenRouter key first.");
  requireDirectory(folder);
  requireImage(image);
  if (typeof model !== "string" || !/^[\w.\-/:]+$/.test(model)) throw new Error("Choose a model.");
  const iterationCount = Number(iterations);
  const timeout = Number(timeoutMinutes);
  if (!Number.isInteger(iterationCount) || iterationCount < 1 || iterationCount > 20) throw new Error("Iterations must be 1-20.");
  if (!Number.isInteger(timeout) || timeout < 5 || timeout > 24 * 60) throw new Error("Time limit must be 5-1440 minutes.");
  if (!fs.existsSync(AGENTOMICS_RUN)) {
    throw new Error(PACKAGED ? "This copy of Agentomics is missing its run engine. Download the app again." : `agentomics-run not found at ${AGENTOMICS_RUN}. Install the repo into .venv.`);
  }

  const platform = await imagePlatform(image);
  if (!platform) throw new Error(`Docker image ${image} was not found.`);
  const dataset = path.basename(folder);
  const id = `${dataset}-${new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "").replace("T", "-")}`;
  const workspace = path.join(RUNS_DIR, id);
  fs.mkdirSync(workspace, { recursive: true });

  const args = [
    "--image", image, "--provider", provider, "--model", model,
    "--datasets-dir", path.dirname(folder), "--dataset", dataset,
    "--iterations", String(iterationCount), "--timeout", String(timeout * 60),
    "--workspace-dir", workspace, "--verbosity", "summary",
  ];
  if (!(await useGpu(image))) args.push("--cpu-only");
  if (valMetric) args.push("--val-metric", String(valMetric));
  if (userPrompt) args.push("--user-prompt", String(userPrompt));

  const log = fs.openSync(path.join(RUNS_DIR, `${id}.log`), "w");
  const child = spawn(AGENTOMICS_RUN, args, {
    cwd: RUN_CWD, // agentomics-run reads an optional .env from here
    env: { ...process.env, AGENTOMICS_UI_EVENTS: "1", DOCKER_DEFAULT_PLATFORM: platform, ...(openRouterKey ? { OPENROUTER_API_KEY: openRouterKey } : {}) },
    stdio: ["ignore", log, log],
    detached: true, // own process group, so Stop also stops docker
    windowsHide: true, // no console window on Windows
  });
  processes.set(id, child);
  child.on("exit", (code) => {
    processes.delete(id);
    updateRun(id, { status: code === 0 ? "finished" : "failed", exitCode: code, endedAt: Date.now() });
  });

  const entry: RunEntry = { id, dataset, folder, image, provider, model, iterations: iterationCount, pid: child.pid, workspace, status: "running", startedAt: Date.now() };
  saveHistory([entry, ...loadHistory()]);
  return entry;
}

export function listRuns(): RunEntry[] {
  return loadHistory().map((entry) => {
    entry.eventsFile = eventsFileOf(entry);
    if (entry.status === "running" && !going(entry)) {
      // The app was restarted while this run was going, and it has since ended. Its event log says how;
      // without an ending it was stopped. Recorded, so its pid is not mistaken for a later process.
      entry.status = endingOf(entry.eventsFile) ?? "interrupted";
      updateRun(entry.id, { status: entry.status });
    }
    entry.summary = runSummary(entry);
    return entry;
  });
}

export function stopRun(id: string): boolean {
  const entry = loadHistory().find((r) => r.id === id);
  const pid = processes.get(id)?.pid ?? entry?.pid;
  if (!pid || !going(entry)) return false;
  stopRunProcess(pid, path.basename(entry?.workspace ?? id));
  return true;
}

/** Whether the run in runFolder is still going (started by this app, and its process is alive). */
export async function isRunning(runFolder: string): Promise<boolean> {
  return going(loadHistory().find((r) => fs.existsSync(r.workspace) && fs.realpathSync(r.workspace) === runFolder));
}

/** Moves a finished run's workspace and log to the Trash (restorable) and drops it from the history. */
export async function deleteRun(event: IpcMainInvokeEvent, id: string): Promise<DeleteRunResult> {
  const entry = loadHistory().find((r) => r.id === id);
  if (!entry) return { ok: false, error: "This run is no longer in the list." };
  if (going(entry)) return { ok: false, error: `Stop the run before moving it to the ${TRASH}.` };
  const workspace = path.resolve(entry.workspace);
  if (!workspace.startsWith(RUNS_DIR + path.sep)) return { ok: false, error: "This run's folder is outside the app's runs folder." };
  const options = {
    type: "warning" as const,
    message: `Move “${entry.dataset}” to the ${TRASH}?`,
    detail: `The run from ${new Date(entry.startedAt).toLocaleString()} and its workspace move to the ${TRASH}. You can put them back from there.`,
    buttons: [`Move to ${TRASH}`, "Cancel"],
    defaultId: 0,
    cancelId: 1,
  };
  const window = BrowserWindow.fromWebContents(event.sender);
  const { response } = await (window ? dialog.showMessageBox(window, options) : dialog.showMessageBox(options));
  if (response !== 0) return { ok: false, cancelled: true };
  for (const item of [workspace, `${workspace}.log`]) {
    if (fs.existsSync(item)) await shell.trashItem(item);
  }
  saveHistory(loadHistory().filter((r) => r.id !== id));
  return { ok: true };
}

/** The native right-click menu for a run in the lobby. Resolves with the chosen action. */
export function runMenu(event: IpcMainInvokeEvent, id: string): Promise<RunMenuAction | null> {
  const running = going(loadHistory().find((r) => r.id === id));
  return new Promise((resolve) => {
    let chosen: RunMenuAction | null = null;
    const template: MenuItemConstructorOptions[] = [
      { label: "Open", click: () => (chosen = "open") },
      { label: SHOW_IN_FILES, click: () => (chosen = "show") },
      { type: "separator" },
      ...(running ? [{ label: "Stop Run", click: () => (chosen = "stop") }] : []),
      { label: `Move to ${TRASH}`, enabled: !running, click: () => (chosen = "trash") },
    ];
    Menu.buildFromTemplate(template).popup({
      window: BrowserWindow.fromWebContents(event.sender) ?? undefined,
      callback: () => setTimeout(() => resolve(chosen)),
    });
  });
}
