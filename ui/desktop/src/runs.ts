// Starting, listing, stopping and deleting runs. A run is `agentomics-run` in its own process group,
// whose agent traces the run page follows (logs/traces/); its workspace lives in the app's runs folder.
import type { DeleteRunResult, RunEntry, RunMenuAction, RunnerLog, RunnerOutcome, StartRunOptions } from "@agentomics/contract";
import { BrowserWindow, dialog, Menu, shell, type IpcMainInvokeEvent, type MenuItemConstructorOptions } from "electron";
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import * as accounts from "./accounts";
import { imagePlatform, requireDirectory, requireImage } from "./exec";
import { endingOf, isAlive, loadHistory, runSummary, saveHistory, updateRun } from "./history";
import { selectedImage } from "./images";
import { AGENTOMICS_RUN, PACKAGED, REPO_ROOT, RUN_CWD, RUNS_DIR } from "./paths";
import { stopRunProcess } from "./processes";
import { useGpu } from "./gpu";
import { RUN_CONFIG } from "@agentomics/contract/snapshot";

const processes = new Map<string, ChildProcess>(); // run id -> child process
// What the system calls its file manager and its trash, for the menu and the confirmation.
const SHOW_IN_FILES = process.platform === "darwin" ? "Show in Finder" : process.platform === "win32" ? "Show in File Explorer" : "Show in Files";
const TRASH = process.platform === "win32" ? "Recycle Bin" : "Trash";

/** Whether a run is still going: started in this session, or before the app was last quit and still alive. */
function going(entry: RunEntry | undefined): boolean {
  return Boolean(entry && (processes.has(entry.id) || (entry.status === "running" && isAlive(entry.pid))));
}

export async function startRun(options: StartRunOptions): Promise<RunEntry> {
  const { forkFrom } = options ?? ({} as StartRunOptions);
  const source = forkFrom ? forkable(forkFrom) : undefined;
  // A fork uses its source's dataset, which agentomics-run mounts again.
  const folder = source ? source.entry.folder : options?.folder;
  const { image = await selectedImage(), provider = "Codex", model, iterations, timeoutMinutes, valMetric, userPrompt } = options ?? ({} as StartRunOptions);
  if (provider !== "Codex" && provider !== "OpenRouter") throw new Error("Choose ChatGPT (Codex) or OpenRouter.");
  const openRouterKey = provider === "OpenRouter" ? accounts.openRouterKey(REPO_ROOT) : null;
  if (provider === "OpenRouter" && !openRouterKey) throw new Error("Add an OpenRouter key first.");
  requireDirectory(folder);
  requireImage(image);
  if (typeof model !== "string" || !/^[\w.\-/:~]+$/.test(model)) throw new Error("Choose a model.");
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
  // A fork keeps its source's goal (agentomics-run inherits it) unless told what to change, which is added to it.
  const prompt = source?.change ? `${goalOf(source.entry)}\n\nFrom iteration ${source.kept + 1} on: ${source.change}`.trim() : userPrompt;
  if (prompt) args.push("--user-prompt", String(prompt));
  // Forking from an iteration's end checkpoint: the new run keeps iterations 1..kept and adds `iterations` more.
  if (source) args.push("--fork-from-run", source.entry.workspace, "--fork-from-iteration", String(source.kept - 1));

  const logFile = runnerLogOf(id);
  const log = fs.openSync(logFile, "w");
  const child = spawn(AGENTOMICS_RUN, args, {
    cwd: RUN_CWD, // agentomics-run reads an optional .env from here
    // Unbuffered, so the runner's output reaches its log even if the process is killed (Windows: exit -1).
    env: { ...process.env, PYTHONUNBUFFERED: "1", DOCKER_DEFAULT_PLATFORM: platform, ...(openRouterKey ? { OPENROUTER_API_KEY: openRouterKey } : {}) },
    stdio: ["ignore", log, log],
    detached: true, // own process group, so Stop also stops docker
    windowsHide: true, // no console window on Windows
  });
  processes.set(id, child);
  // Either event can end the run, and "exit" may never follow an "error" (the OS refused to start it).
  const finish = (changes: Partial<RunEntry>) => {
    if (!processes.delete(id)) return;
    fs.closeSync(log);
    updateRun(id, { ...changes, endedAt: Date.now() });
  };
  child.on("error", (error) => {
    fs.writeSync(log, `\nCouldn't start agentomics-run: ${error.message}\n`);
    finish({ status: "failed", exitCode: null, startError: error.message });
  });
  child.on("exit", (code) => finish({ status: code === 0 ? "finished" : "failed", exitCode: signedExitCode(code) }));

  const entry: RunEntry = {
    id, dataset, folder, image, provider, model, iterations: (source?.kept ?? 0) + iterationCount, timeoutMinutes: timeout, pid: child.pid, workspace, status: "running", startedAt: Date.now(),
    ...(source ? { forkedFrom: { id: source.entry.id, iterations: source.kept } } : {}),
  };
  saveHistory([entry, ...loadHistory()]);
  return entry;
}

/**
 * The run to fork, how many of its finished iterations to keep, and what the new ones should change. A run still
 * going can be forked too: the fork rolls its copy back to the chosen iteration's checkpoint.
 */
function forkable({ workspace, iterations, change }: NonNullable<StartRunOptions["forkFrom"]>): { entry: RunEntry; kept: number; change?: string } {
  const entry = typeof workspace === "string" ? entryIn(workspace) : undefined;
  if (!entry) throw new Error("That run isn't in the app's history.");
  const done = runSummary(entry).done;
  const kept = Number(iterations);
  if (!Number.isInteger(kept) || kept < 1 || kept > done) throw new Error(done ? `Fork from iteration 1-${done}.` : "That run has no finished iteration to fork from.");
  return { entry, kept, change: typeof change === "string" && change.trim() ? change.trim() : undefined };
}

/** The goal a run was given (its config's user prompt), or "" when it can't be read. */
function goalOf(entry: RunEntry): string {
  try {
    const goal = JSON.parse(fs.readFileSync(path.join(entry.workspace, RUN_CONFIG), "utf8")).user_prompt;
    return typeof goal === "string" ? goal : "";
  } catch {
    return "";
  }
}

function runnerLogOf(id: string): string {
  return path.join(RUNS_DIR, `${id}.log`);
}

/** Windows reports exit codes as unsigned 32-bit numbers: 4294967295 is -1. */
function signedExitCode(code: number | null): number | null {
  return code !== null && code > 0x7fffffff ? code - 0x100000000 : code;
}

/** The run whose workspace is this folder, by real path. */
function entryIn(workspace: string): RunEntry | undefined {
  const real = (folder: string) => {
    try {
      return fs.realpathSync(folder);
    } catch {
      return null;
    }
  };
  const wanted = real(workspace);
  return wanted ? loadHistory().find((r) => real(r.workspace) === wanted) : undefined;
}

// The CLI colours its output for a terminal; in the app the codes would show as text.
// eslint-disable-next-line no-control-regex
const TERMINAL_COLOURS = /\x1b\[[0-9;]*[A-Za-z]/g;

/** The end of a run's runner output: its last `chars` characters, from the start of a line. */
function tailOf(logFile: string, chars: number): string {
  try {
    const text = fs.readFileSync(logFile, "utf8").replace(TERMINAL_COLOURS, "");
    if (text.length <= chars) return text.trimEnd();
    const tail = text.slice(-chars);
    return tail.slice(tail.indexOf("\n") + 1).trimEnd();
  } catch {
    return "";
  }
}

/** How the run in workspace ended, and the end of its runner's output. */
export function runnerOutcome(workspace: string): RunnerOutcome | null {
  const entry = entryIn(workspace);
  if (!entry || going(entry)) return null;
  const logFile = runnerLogOf(entry.id);
  return { exitCode: entry.exitCode ?? null, stopped: entry.stopped, startError: entry.startError, log: tailOf(logFile, 8000), logFile };
}

/** The run in workspace's runner output so far, while it runs or after. */
export function runnerLog(workspace: string): RunnerLog | null {
  const entry = entryIn(workspace);
  if (!entry) return null;
  const logFile = runnerLogOf(entry.id);
  return { log: tailOf(logFile, 64_000), logFile };
}

export function listRuns(): RunEntry[] {
  return loadHistory().map((entry) => {
    if (entry.status === "running" && !going(entry)) {
      // The app was restarted while this run was going, and it has since ended. Its event log says how;
      // without an ending it was stopped. Recorded, so its pid is not mistaken for a later process.
      entry.status = endingOf(entry) ?? "interrupted";
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
  updateRun(id, { stopped: true });
  stopRunProcess(pid, path.basename(entry?.workspace ?? id));
  return true;
}

/** The run page's Stop: asks first, since the run can't be resumed afterwards (only forked). */
export async function stopRunIn(event: IpcMainInvokeEvent, workspace: string): Promise<boolean> {
  const entry = entryIn(workspace);
  if (!entry || !going(entry)) return false;
  const options = {
    type: "warning" as const,
    message: `Stop the run on “${entry.dataset}”?`,
    detail: "agentomics-run and its container stop. The steps it has recorded stay.",
    buttons: ["Stop Run", "Cancel"],
    defaultId: 1,
    cancelId: 1,
  };
  const window = BrowserWindow.fromWebContents(event.sender);
  const { response } = await (window ? dialog.showMessageBox(window, options) : dialog.showMessageBox(options));
  return response === 0 && stopRun(entry.id);
}

/** How the run in runFolder (a real path) ended, for the end of its event stream; null if the app didn't start it. */
export async function runEnding(runFolder: string): Promise<{ exitCode: number | null; error?: string; endedAt?: number; stopped?: boolean } | null> {
  const entry = loadHistory().find((r) => fs.existsSync(r.workspace) && fs.realpathSync(r.workspace) === runFolder);
  if (!entry || going(entry)) return null;
  const { endedAt } = entry;
  if (entry.startError) return { exitCode: null, error: `agentomics-run couldn't be started: ${entry.startError}`, endedAt };
  if (entry.stopped) return { exitCode: null, error: "The run was stopped from the app.", endedAt, stopped: true };
  if (typeof entry.exitCode === "number") return { exitCode: entry.exitCode, endedAt };
  const status = entry.status === "running" ? endingOf(entry) ?? "interrupted" : entry.status;
  if (status === "finished") return { exitCode: 0, endedAt };
  return { exitCode: null, error: status === "interrupted" ? "The run stopped while the app was closed." : "The run failed.", endedAt };
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
  const entry = loadHistory().find((r) => r.id === id);
  const running = going(entry);
  const forkable = entry ? runSummary(entry).done > 0 : false;
  return new Promise((resolve) => {
    let chosen: RunMenuAction | null = null;
    const template: MenuItemConstructorOptions[] = [
      { label: "Open", click: () => (chosen = "open") },
      { label: "Fork…", enabled: forkable, click: () => (chosen = "fork") },
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
