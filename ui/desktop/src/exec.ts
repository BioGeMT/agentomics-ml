// Running commands (mostly docker) and checking what the page sends before acting on it.
import { execFile, type ExecFileOptions } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

export type Output = { code: number; stdout: string; stderr: string };

export function run(command: string, args: string[], options: ExecFileOptions = {}): Promise<Output> {
  return new Promise((resolve) => {
    execFile(command, args, { maxBuffer: 20 * 1024 * 1024, windowsHide: true, ...options, encoding: "utf8" }, (error, stdout, stderr) => {
      const code = error ? (typeof error.code === "number" ? error.code : 1) : 0;
      resolve({ code, stdout: String(stdout), stderr: String(stderr) });
    });
  });
}

/** os/arch of a local image, or null when Docker doesn't have it. */
export async function imagePlatform(image: string): Promise<string | null> {
  const { code, stdout } = await run("docker", ["image", "inspect", image, "--format", "{{.Os}}/{{.Architecture}}"]);
  return code === 0 ? stdout.trim() : null;
}

// The page is trusted no more than any other input: these run on every value it sends.

export function requireDirectory(folder: unknown): asserts folder is string {
  if (typeof folder !== "string" || !path.isAbsolute(folder) || !fs.statSync(folder, { throwIfNoEntry: false })?.isDirectory()) {
    throw new Error("Choose an existing folder.");
  }
}

export function requireImage(image: unknown): asserts image is string {
  if (typeof image !== "string" || !/^[a-z0-9][a-z0-9._\-/:@]*$/i.test(image)) {
    throw new Error(`Invalid image name: ${image}`);
  }
}
