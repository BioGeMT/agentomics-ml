// Builds agentomics-run, the same CLI as the repo's, as a standalone folder the packaged app carries
// (Python and its libraries included), so a computer with only Docker can run Agentomics. Needs uv to make
// the build environment; nothing of uv ships. PyInstaller builds for the OS it runs on, so each OS's app is
// built on that OS. Output: build/agentomics-run/.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repo = path.resolve(here, "..", "..");
const build = path.join(here, "build");
const windows = process.platform === "win32";
const venv = path.join(build, "cli-venv");
const bin = (name) => path.join(venv, windows ? "Scripts" : "bin", windows ? `${name}.exe` : name);
const sh = (command, args) => execFileSync(command, args, { stdio: "inherit" });

fs.mkdirSync(build, { recursive: true });
sh("uv", ["venv", "--quiet", "--allow-existing", "--python", "3.12", venv]);
sh("uv", ["pip", "install", "--quiet", "--python", bin("python"), "--reinstall-package", "agentomics", repo, "pyinstaller==6.22.3"]);
const entry = path.join(build, "agentomics_run.py");
fs.writeFileSync(entry, 'import sys\nfrom agentomics.cli.run import main\n\nif __name__ == "__main__":\n    sys.exit(main())\n');
sh(bin("pyinstaller"), [
  "--noconfirm", "--log-level", "WARN", "--name", "agentomics-run", "--onedir",
  "--copy-metadata", "agentomics", "--collect-data", "agentomics",
  "--distpath", build, "--workpath", path.join(build, "cli-work"), "--specpath", build, entry,
]);
const exe = path.join(build, "agentomics-run", windows ? "agentomics-run.exe" : "agentomics-run");
execFileSync(exe, ["--help"], { stdio: "ignore" });
console.log(`Built ${path.dirname(exe)}`);
