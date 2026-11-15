// The words and keys of the system the app runs on. The page can't see process.platform, but the
// user agent names the OS; without one the page reads as macOS.
const agent = typeof navigator === "undefined" ? "" : navigator.userAgent;
const windows = /Windows/.test(agent);
const linux = !windows && /Linux|X11/.test(agent) && !/Android/.test(agent);

/** The modifier key as a shortcut shows it: ⌘N on macOS, Ctrl+N elsewhere. */
export const MOD = windows || linux ? "Ctrl+" : "⌘";
export const SHOW_IN_FILES = windows ? "Show in File Explorer" : linux ? "Show in Files" : "Show in Finder";
export const TRASH = windows ? "Recycle Bin" : "Trash";
export const DOCKER = linux ? "Docker" : "Docker Desktop";

/** A path's segments, whichever separator the OS uses. */
export const segments = (folder: string) => folder.split(/[\\/]/).filter(Boolean);
