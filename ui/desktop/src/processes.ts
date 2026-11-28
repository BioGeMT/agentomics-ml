// Process handling that differs by OS. On macOS and Linux a run is its own process group, ended with SIGTERM;
// Windows has no signals, so the process tree is ended with taskkill. Either way the run's containers are then
// stopped by the label the CLI gives them: docker passes SIGTERM on to a container, but the agent inside
// carries on, and its docker client with it.
import { execFileSync, spawn, spawnSync } from "node:child_process";

export const WINDOWS = process.platform === "win32";
// Must match the label _workspace_label_arguments sets in src/agentomics/cli/run.py.
const WORKSPACE_LABEL = "agentomics.workspace";

/** Ends the process tree started at pid (a detached process group on macOS and Linux). */
export function endTree(pid: number): void {
  if (WINDOWS) {
    spawnSync("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true });
  } else {
    process.kill(-pid, "SIGTERM");
  }
}

/** Ends a run: its process tree, then its containers (in the background: docker stop can take seconds). */
export function stopRunProcess(pid: number, workspaceName: string): void {
  try {
    endTree(pid);
  } catch {} // already gone; its containers may not be
  const ids = spawnSync("docker", ["ps", "-q", "--filter", `label=${WORKSPACE_LABEL}=${workspaceName}`], { encoding: "utf8", windowsHide: true })
    .stdout?.split(/\s+/)
    .filter(Boolean);
  if (ids?.length) spawn("docker", ["stop", ...ids], { stdio: "ignore", windowsHide: true }).on("error", () => {});
}

/** The command line of a live process, or "" when there is none. */
export function commandOf(pid: number): string {
  if (!WINDOWS) return spawnSync("ps", ["-o", "command=", "-p", String(pid)], { encoding: "utf8" }).stdout ?? "";
  try {
    // tasklist gives the image name (agentomics-run.exe), which is all the check needs.
    return execFileSync("tasklist", ["/FI", `PID eq ${pid}`, "/FO", "CSV", "/NH"], { encoding: "utf8", windowsHide: true });
  } catch {
    return "";
  }
}
