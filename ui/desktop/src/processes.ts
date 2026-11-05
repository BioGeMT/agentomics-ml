// Process handling that differs by OS. On macOS and Linux a run is its own process group, and SIGTERM
// to the group reaches docker, which passes it on to the container. Windows has no signals: the process
// tree is ended with taskkill, and the run's containers are stopped by the label the CLI gives them.
import { execFileSync, spawnSync } from "node:child_process";

export const WINDOWS = process.platform === "win32";
// Must match WORKSPACE_LABEL in src/agentomics/cli/run.py.
const WORKSPACE_LABEL = "org.biogemt.agentomics.workspace";

/** Ends the process tree started at pid (a detached process group on macOS and Linux). */
export function endTree(pid: number): void {
  if (WINDOWS) {
    spawnSync("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true });
  } else {
    process.kill(-pid, "SIGTERM");
  }
}

/** Ends a run: its process tree, then (on Windows, where docker can't forward a signal) its containers. */
export function stopRunProcess(pid: number, workspaceName: string): void {
  endTree(pid);
  if (!WINDOWS) return;
  const ids = spawnSync("docker", ["ps", "-q", "--filter", `label=${WORKSPACE_LABEL}=${workspaceName}`], { encoding: "utf8", windowsHide: true })
    .stdout?.split(/\s+/)
    .filter(Boolean);
  if (ids?.length) spawnSync("docker", ["stop", ...ids], { windowsHide: true });
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
