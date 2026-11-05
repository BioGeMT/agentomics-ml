// The app's settings file (userData/settings.json): the Docker image, image checks and the appearance.
import type { Appearance } from "@agentomics/contract";
import { app } from "electron";
import fs from "node:fs";
import path from "node:path";

export type Settings = { image?: string; eventLog?: Record<string, boolean>; appearance?: Appearance };

function settingsPath(): string {
  return path.join(app.getPath("userData"), "settings.json");
}

export function loadSettings(): Settings {
  try {
    return JSON.parse(fs.readFileSync(settingsPath(), "utf8"));
  } catch {
    return {};
  }
}

export function saveSettings(changes: Settings): Settings {
  const next = { ...loadSettings(), ...changes };
  fs.mkdirSync(path.dirname(settingsPath()), { recursive: true });
  fs.writeFileSync(settingsPath(), JSON.stringify(next, null, 2));
  return next;
}
