// What the desktop app and the UI agree on: the calls the page can make (window.agentomics), the
// events the app sends it, and the data both pass. Types only, so nothing here ends up in a bundle;
// import it with `import type`. The runtime parts are small pure functions: ./protocol.ts, the
// agentomics:// protocol (how the page reads runs); ./events.ts, the schema of Agentomics' own event;
// and ./snapshot.ts, how a finished iteration's files read.

export type Provider = "Codex" | "OpenRouter";
export type Model = { id: string; label: string };

export type Accounts = {
  codex: { signedIn: boolean };
  openrouter: { configured: boolean; source: "app" | "environment" | null; masked: string | null };
};

export type ImageOption = { name: string; tag?: string; updated?: string; size?: number | string; created?: string; downloaded?: boolean; live?: boolean };
export type ImageList = { selected: string; dockerError: string | null; hubError: string | null; official: ImageOption[]; local: ImageOption[] };
export type PullProgress = { name: string; done: number; total: number; status: string };

export type DatasetDetails = { title?: string; classes?: Record<string, number>; features?: number };
export type CheckResult =
  | { ok: true; summary: Record<string, any>; details?: DatasetDetails }
  | { ok: false; error: string; suggestion?: string };

export type StartRunOptions = {
  folder: string;
  image?: string;
  provider?: Provider;
  model: string;
  iterations: number;
  timeoutMinutes: number;
  valMetric?: string;
  userPrompt?: string;
};

export type RunStatus = "running" | "finished" | "failed" | "interrupted";

/** The lobby's figures for a run: its main metric at its best validation score, the test score, iterations done. */
export type RunSummary = { metric: string | null; best: number | null; test: number | null; done: number; total: number | null; history: (number | null)[] };

export type RunEntry = {
  id: string;
  dataset: string;
  folder: string;
  image: string;
  provider?: Provider;
  model: string;
  iterations: number;
  /** The agentomics-run process, which leads its own process group. */
  pid?: number;
  workspace: string;
  status: RunStatus;
  startedAt: number;
  endedAt?: number;
  exitCode?: number | null;
  /** Set by listRuns. */
  eventsFile?: string;
  summary?: RunSummary;
};

export type Result = { ok: true } | { ok: false; error: string };
export type DeleteRunResult = { ok: true } | { ok: false; cancelled?: boolean; error?: string };
export type SaveKeyResult = { ok: true; masked: string | null } | { ok: false; error: string };
/** Whether Docker is installed, and whether its engine answers. */
export type DockerStatus = { installed: boolean; running: boolean; version?: string };

/** Light or dark: following the Mac (the default), or pinned to one. */
export type Appearance = "system" | "light" | "dark";

export type MenuAction = "new-run" | "settings" | "toggle-contents" | "toggle-figures";
/** Which of the run page's sidebars are showing, for the View menu's Show/Hide items; null where there are none. */
export type ViewState = { contents: boolean; figures: boolean } | null;
export type RunMenuAction = "open" | "show" | "stop" | "trash";

/** ipcRenderer.invoke channels: the arguments each takes and what it resolves with. */
export type InvokeChannels = {
  "pick-folder": { args: []; result: string | null };
  "list-images": { args: []; result: ImageList };
  "set-image": { args: [name: string]; result: string };
  "check-live-view": { args: [name: string]; result: boolean | null };
  "pull-image": { args: [name: string]; result: Result };
  "get-accounts": { args: []; result: Accounts };
  "codex-sign-in": { args: []; result: Result };
  "save-openrouter-key": { args: [key: string]; result: SaveKeyResult };
  "remove-openrouter-key": { args: []; result: boolean };
  "list-models": { args: [provider: Provider]; result: Model[] };
  "check-dataset": { args: [folder: string, image?: string]; result: CheckResult };
  "start-run": { args: [options: StartRunOptions]; result: RunEntry };
  "list-runs": { args: []; result: RunEntry[] };
  "stop-run": { args: [id: string]; result: boolean };
  "delete-run": { args: [id: string]; result: DeleteRunResult };
  "run-menu": { args: [id: string]; result: RunMenuAction | null };
  "show-folder": { args: [folder: string]; result: void };
  "set-view-state": { args: [state: ViewState]; result: void };
  "get-appearance": { args: []; result: Appearance };
  "docker-status": { args: []; result: DockerStatus };
  "open-docker": { args: []; result: boolean };
  "set-appearance": { args: [appearance: Appearance]; result: Appearance };
};

/** webContents.send events from the app to the page. */
export type EventChannels = {
  menu: MenuAction;
  "pull-progress": PullProgress;
};

/** window.agentomics, exposed by the desktop app's preload. Absent in a plain browser. */
export type DesktopApi = {
  pickFolder(): Promise<string | null>;
  listImages(): Promise<ImageList>;
  setImage(name: string): Promise<string>;
  checkLiveView(name: string): Promise<boolean | null>;
  pullImage(name: string): Promise<Result>;
  /** Returns a function that stops listening. */
  onPullProgress(callback: (progress: PullProgress) => void): () => void;
  listModels(provider: Provider): Promise<Model[]>;
  getAccounts(): Promise<Accounts>;
  codexSignIn(): Promise<Result>;
  saveOpenRouterKey(key: string): Promise<SaveKeyResult>;
  removeOpenRouterKey(): Promise<boolean>;
  /** The real path of a file or folder dropped on the window. */
  pathForFile(file: File): string;
  checkDataset(folder: string, image?: string): Promise<CheckResult>;
  startRun(options: StartRunOptions): Promise<RunEntry>;
  listRuns(): Promise<RunEntry[]>;
  stopRun(id: string): Promise<boolean>;
  /** Menu commands (New Run, Settings…, the View menu's sidebars). Returns a function that stops listening. */
  onMenu(callback: (action: MenuAction) => void): () => void;
  deleteRun(id: string): Promise<DeleteRunResult>;
  runMenu(id: string): Promise<RunMenuAction | null>;
  showFolder(folder: string): Promise<void>;
  /** Tells the View menu which sidebars are showing (null: this page has none). */
  setViewState(state: ViewState): Promise<void>;
  getAppearance(): Promise<Appearance>;
  dockerStatus(): Promise<DockerStatus>;
  /** Starts Docker Desktop. */
  openDocker(): Promise<boolean>;
  setAppearance(appearance: Appearance): Promise<Appearance>;
};

declare global {
  interface Window {
    agentomics?: DesktopApi;
  }
}
