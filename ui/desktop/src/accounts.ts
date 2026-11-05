// Model provider accounts for the desktop app: ChatGPT (Codex) sign-in and an OpenRouter API key.
//
// The OpenRouter key is encrypted with the OS keychain (safeStorage) and kept in the app's data
// folder. It never goes back to the web page; it is only handed to runs this app starts.
import type { Accounts, Model, Result, SaveKeyResult } from "@agentomics/contract";
import { app, safeStorage } from "electron";
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const CODEX_AUTH = path.join(os.homedir(), ".codex", "auth.json");
const CODEX_MODELS = path.join(os.homedir(), ".codex", "models_cache.json");
const OPENROUTER = "https://openrouter.ai/api/v1";
// Shown first in the OpenRouter model list; the rest follow alphabetically.
const OPENROUTER_PICKS = ["openai/gpt-6-sol", "openai/gpt-6.1-sol", "anthropic/claude-sonnet-5-5", "anthropic/claude-opus-5-5", "openai/gpt-5.6-luna"];

function keyFile(): string {
  return path.join(app.getPath("userData"), "openrouter.key");
}

function codexSignedIn(): boolean {
  try {
    const tokens = JSON.parse(fs.readFileSync(CODEX_AUTH, "utf8")).tokens ?? {};
    return Boolean(tokens.access_token && tokens.refresh_token);
  } catch {
    return false;
  }
}

function storedOpenRouterKey(): string | null {
  try {
    return safeStorage.decryptString(fs.readFileSync(keyFile()));
  } catch {
    return null;
  }
}

function envOpenRouterKey(repoRoot: string): string | null {
  if (process.env.OPENROUTER_API_KEY) return process.env.OPENROUTER_API_KEY;
  try {
    const line = fs.readFileSync(path.join(repoRoot, ".env"), "utf8").split("\n").find((l) => l.startsWith("OPENROUTER_API_KEY="));
    return line ? line.slice("OPENROUTER_API_KEY=".length).trim().replace(/^["']|["']$/g, "") || null : null;
  } catch {
    return null;
  }
}

/** The OpenRouter key runs should use: the one saved in the app, else one from the environment or .env. */
export function openRouterKey(repoRoot: string): string | null {
  return storedOpenRouterKey() ?? envOpenRouterKey(repoRoot);
}

function mask(key: string | null): string | null {
  return key ? `${key.slice(0, 9)}••••${key.slice(-4)}` : null;
}

export function status(repoRoot: string): Accounts {
  const saved = storedOpenRouterKey();
  const fromEnv = saved ? null : envOpenRouterKey(repoRoot);
  return {
    codex: { signedIn: codexSignedIn() },
    openrouter: {
      configured: Boolean(saved || fromEnv),
      source: saved ? "app" : fromEnv ? "environment" : null,
      masked: mask(saved ?? fromEnv),
    },
  };
}

/** Runs `codex login`, which opens the browser for ChatGPT sign-in and writes ~/.codex/auth.json. */
export function codexSignIn(): Promise<Result> {
  const missing: Result = { ok: false, error: "The Codex CLI isn't installed. Install it with: npm install -g @openai/codex" };
  return new Promise((resolve) => {
    let child: ChildProcess;
    try {
      // codex is codex.cmd on Windows (an npm shim), which only starts through a shell.
      child = spawn("codex", ["login"], { stdio: "ignore", shell: process.platform === "win32", windowsHide: true });
    } catch {
      resolve(missing);
      return;
    }
    child.on("error", () => resolve(missing));
    child.on("exit", () =>
      resolve(codexSignedIn() ? { ok: true } : { ok: false, error: "Sign-in didn't finish. Try again and complete it in your browser." }),
    );
  });
}

export async function saveOpenRouterKey(key: unknown): Promise<SaveKeyResult> {
  if (typeof key !== "string" || !/^sk-or-[A-Za-z0-9_-]{20,}$/.test(key.trim())) {
    return { ok: false, error: "That doesn't look like an OpenRouter key. Keys start with sk-or-." };
  }
  key = key.trim();
  if (!safeStorage.isEncryptionAvailable()) {
    return { ok: false, error: "The system keychain isn't available, so the key can't be stored safely." };
  }
  let response: Response;
  try {
    response = await fetch(`${OPENROUTER}/key`, { headers: { Authorization: `Bearer ${key}` } });
  } catch {
    return { ok: false, error: "Couldn't reach OpenRouter to check the key. Check your connection and try again." };
  }
  if (response.status === 401) return { ok: false, error: "OpenRouter rejected this key. Copy it again from openrouter.ai/keys." };
  if (!response.ok) return { ok: false, error: `OpenRouter couldn't check the key right now (HTTP ${response.status}). Try again in a moment.` };
  fs.mkdirSync(path.dirname(keyFile()), { recursive: true });
  fs.writeFileSync(keyFile(), safeStorage.encryptString(key as string), { mode: 0o600 });
  return { ok: true, masked: mask(key as string) };
}

export function removeOpenRouterKey(): boolean {
  fs.rmSync(keyFile(), { force: true });
  return true;
}

export function codexModels(): Model[] {
  try {
    return (JSON.parse(fs.readFileSync(CODEX_MODELS, "utf8")).models ?? [])
      .filter((m: any) => m.supported_in_api !== false && m.visibility !== "hide")
      .sort((a: any, b: any) => (a.priority ?? 999) - (b.priority ?? 999))
      .map((m: any) => ({ id: m.slug ?? m.id, label: m.display_name ?? m.slug }));
  } catch {
    return [];
  }
}

export async function openRouterModels(repoRoot: string): Promise<Model[]> {
  const key = openRouterKey(repoRoot);
  if (!key) return [];
  try {
    const response = await fetch(`${OPENROUTER}/models`, { headers: { Authorization: `Bearer ${key}` } });
    const models: { id: string; supported_parameters?: string[] }[] = ((await response.json()).data ?? []).filter((m: any) =>
      (m.supported_parameters ?? []).includes("tools"),
    );
    const rank = (id: string) => (OPENROUTER_PICKS.includes(id) ? OPENROUTER_PICKS.indexOf(id) : OPENROUTER_PICKS.length);
    return models
      .sort((a, b) => rank(a.id) - rank(b.id) || a.id.localeCompare(b.id))
      .map((m) => ({ id: m.id, label: m.id }));
  } catch {
    return [];
  }
}
