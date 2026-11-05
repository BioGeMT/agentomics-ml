// Docker, which every run needs. An app opened from Finder, the Start menu or a launcher gets a minimal
// PATH, so the places Docker Desktop (and Homebrew) install the docker command are added first; then the
// app can tell whether Docker is installed and whether its engine is running, and start Docker Desktop.
import type { DockerStatus } from "@agentomics/contract";
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { run } from "./exec";

const PROGRAM_FILES = process.env.ProgramFiles ?? "C:\\Program Files";
const WINDOWS_DOCKER_DESKTOP = path.join(PROGRAM_FILES, "Docker", "Docker", "Docker Desktop.exe");

const DOCKER_PATHS: Record<string, string[]> = {
  darwin: ["/usr/local/bin", "/opt/homebrew/bin", path.join(os.homedir(), ".docker", "bin"), "/Applications/Docker.app/Contents/Resources/bin"],
  linux: ["/usr/local/bin", "/usr/bin", path.join(os.homedir(), ".docker", "bin")],
  win32: [path.join(PROGRAM_FILES, "Docker", "Docker", "resources", "bin")],
};

/** Must run before the first docker call. */
export function addDockerToPath(): void {
  const current = (process.env.PATH ?? "").split(path.delimiter).filter(Boolean);
  const extra = (DOCKER_PATHS[process.platform] ?? []).filter((p) => !current.includes(p));
  process.env.PATH = [...current, ...extra].join(path.delimiter);
}

export async function dockerStatus(): Promise<DockerStatus> {
  const client = await run("docker", ["--version"], { timeout: 5000 });
  if (client.code !== 0) return { installed: false, running: false };
  const engine = await run("docker", ["info", "--format", "{{.ServerVersion}}"], { timeout: 8000 });
  return { installed: true, running: engine.code === 0 && Boolean(engine.stdout.trim()), version: engine.stdout.trim() || undefined };
}

/** Starts Docker Desktop; the page polls dockerStatus until the engine answers. False where the app can't. */
export async function openDocker(): Promise<boolean> {
  if (process.platform === "darwin") return (await run("open", ["-a", "Docker"])).code === 0;
  if (process.platform === "win32" && fs.existsSync(WINDOWS_DOCKER_DESKTOP)) {
    spawn(WINDOWS_DOCKER_DESKTOP, [], { detached: true, stdio: "ignore" }).unref();
    return true;
  }
  // Linux: Docker Desktop runs as a user service; Docker Engine is a system service the user starts.
  if (process.platform === "linux") return (await run("systemctl", ["--user", "start", "docker-desktop"])).code === 0;
  return false;
}
