// The window, the menu bar, and (in development) the Vite dev server the window loads.
import type { MenuAction, ViewState } from "@agentomics/contract";
import { app, BrowserWindow, Menu, session, shell, type MenuItemConstructorOptions } from "electron";
import { spawn, type ChildProcess } from "node:child_process";
import path from "node:path";
import { send } from "./ipc";
import { endTree, WINDOWS } from "./processes";
import { ICON, USE_PROTOCOL, RENDERER_DIR, DEV_SERVER_URL } from "./paths";
import { ORIGIN } from "./protocol";

const UI_ORIGIN = USE_PROTOCOL ? ORIGIN : DEV_SERVER_URL;
let devServer: ChildProcess | null = null;
// The run page's sidebars, as the page last reported them; null on pages without them.
let viewState: ViewState = null;

async function devServerIsUp(): Promise<boolean> {
  try {
    return (await fetch(DEV_SERVER_URL)).ok;
  } catch {
    return false;
  }
}

async function ensureDevServer(): Promise<void> {
  if (await devServerIsUp()) return;
  // npm is npm.cmd on Windows, which only starts through a shell.
  devServer = spawn("npm", ["run", "dev"], { cwd: RENDERER_DIR, stdio: "inherit", detached: !WINDOWS, shell: WINDOWS });
  for (let i = 0; i < 120 && !(await devServerIsUp()); i++) await new Promise((r) => setTimeout(r, 500));
}

/** Stops the dev server this app started, if any. */
export function stopDevServer(): void {
  if (devServer?.pid) endTree(devServer.pid);
}

export async function createWindow(): Promise<void> {
  if (!USE_PROTOCOL) {
    await ensureDevServer();
    // The dev server's pages and redirects change from build to build (the old Next.js server
    // permanently redirected /app/ to /app), so nothing from an earlier one is kept.
    await session.defaultSession.clearCache();
  }
  const window = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 820,
    minHeight: 560,
    title: "Agentomics",
    icon: ICON,
    // macOS: no title bar, window buttons inside the frosted sidebar, as in Finder or Notes.
    ...(process.platform === "darwin" && {
      titleBarStyle: "hiddenInset",
      trafficLightPosition: { x: 20, y: 21 },
      vibrancy: "sidebar",
      visualEffectState: "active", // stay light and vibrant even when the window is in the background
      backgroundColor: "#00000000",
    }),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      plugins: true, // Chromium's built-in PDF viewer, for reports in the file viewer
    },
  });
  // Only the local UI runs inside the app; anything else opens in the user's browser.
  window.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
  window.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith(UI_ORIGIN)) {
      event.preventDefault();
      shell.openExternal(url);
    }
  });
  // A new page starts without sidebars until it reports its own.
  window.webContents.on("did-navigate", () => setViewState(null));
  await window.loadURL(`${UI_ORIGIN}/app/`);
}

/** Menu commands go to the page in the focused window, which shows the right screen. */
function sendMenu(action: MenuAction): void {
  const window = BrowserWindow.getFocusedWindow() ?? BrowserWindow.getAllWindows()[0];
  if (window) send(window.webContents, "menu", action);
}

/** Called by the page whenever its sidebars show or hide; the View menu follows. */
export function setViewState(state: ViewState): void {
  viewState = state;
  setMenu();
}

/** The View menu: the run page's sidebars (titles follow their state, dimmed where there are none), then the standard items. */
function viewMenu(): MenuItemConstructorOptions {
  const toggle = (label: string, shown: boolean | undefined, accelerator: string, action: MenuAction): MenuItemConstructorOptions => ({
    label: `${shown ? "Hide" : "Show"} ${label}`,
    accelerator,
    enabled: viewState !== null,
    click: () => sendMenu(action),
  });
  return {
    label: "View",
    submenu: [
      toggle("Contents", viewState?.contents ?? true, "CmdOrCtrl+[", "toggle-contents"),
      toggle("Details", viewState?.figures ?? true, "CmdOrCtrl+]", "toggle-figures"),
      { type: "separator" },
      { role: "reload" },
      { role: "forceReload" },
      { role: "toggleDevTools" },
      { type: "separator" },
      { role: "resetZoom" },
      { role: "zoomIn" },
      { role: "zoomOut" },
      { type: "separator" },
      { role: "togglefullscreen" },
    ],
  };
}

/** Standard menus, named for the app rather than Electron; on Windows and Linux Settings sits in File. */
export function setMenu(): void {
  const mac = process.platform === "darwin";
  const settings: MenuItemConstructorOptions = { label: "Settings…", accelerator: "CmdOrCtrl+,", click: () => sendMenu("settings") };
  const template: MenuItemConstructorOptions[] = [
    ...(mac
      ? [{
          role: "appMenu" as const,
          submenu: [
            { role: "about" as const },
            { type: "separator" as const },
            settings,
            { type: "separator" as const },
            { role: "services" as const },
            { type: "separator" as const },
            { role: "hide" as const },
            { role: "hideOthers" as const },
            { role: "unhide" as const },
            { type: "separator" as const },
            { role: "quit" as const },
          ],
        }]
      : []),
    {
      label: "File",
      submenu: [
        { label: "New Run", accelerator: "CmdOrCtrl+N", click: () => sendMenu("new-run") },
        { type: "separator" },
        ...(mac ? [] : [settings, { type: "separator" as const }]),
        { role: "close" },
      ],
    },
    { role: "editMenu" },
    viewMenu(),
    { role: "windowMenu" },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
  app.setAboutPanelOptions({ applicationName: "Agentomics", iconPath: ICON });
}
