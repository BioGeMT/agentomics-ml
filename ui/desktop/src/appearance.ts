// Light or dark. The app follows the Mac by default (dark-mode.md); Settings can pin it to one, as Xcode
// does. Electron's themeSource sets the window's appearance and the page's prefers-color-scheme together.
import type { Appearance } from "@agentomics/contract";
import { nativeTheme } from "electron";
import { loadSettings, saveSettings } from "./settings";

const CHOICES: Appearance[] = ["system", "light", "dark"];

export function getAppearance(): Appearance {
  const saved = loadSettings().appearance;
  return saved && CHOICES.includes(saved) ? saved : "system";
}

export function applyAppearance(): void {
  nativeTheme.themeSource = getAppearance();
}

export function setAppearance(choice: unknown): Appearance {
  if (!CHOICES.includes(choice as Appearance)) throw new Error(`Unknown appearance: ${choice}`);
  saveSettings({ appearance: choice as Appearance });
  applyAppearance();
  return getAppearance();
}
