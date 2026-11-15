// Checks what building the installers needs before a build starts, so a missing tool is reported up front
// rather than as an error halfway through. Run by `task check` (and so by `task dist`). Docker isn't needed to
// build the app, only to run it.
import { spawnSync } from "node:child_process";

const problems = [];
const notes = [];
const windows = process.platform === "win32";
const version = (command) => {
  const result = spawnSync(command, ["--version"], { encoding: "utf8", shell: windows });
  return result.status === 0 ? result.stdout.trim().split("\n")[0] : null;
};

const [major, minor] = process.versions.node.split(".").map(Number);
if (major < 22 || (major === 22 && minor < 12)) {
  problems.push(`Node.js 22.12 or newer is needed; this is ${process.version}. With nvm: nvm install 22 && nvm use 22`);
}
if (!version("npm")) problems.push("npm is missing (it comes with Node.js).");
if (!version("uv")) problems.push("uv is missing; it builds the bundled agentomics-run: https://docs.astral.sh/uv/getting-started/installation/");

if (process.platform === "darwin") {
  const identities = spawnSync("security", ["find-identity", "-v", "-p", "codesigning"], { encoding: "utf8" }).stdout ?? "";
  if (process.env.AGENTOMICS_UNSIGNED === "1") notes.push("Unsigned build (task dist-unsigned): not signed or notarized.");
  else if (!identities.includes("Developer ID Application") && !process.env.CSC_LINK) {
    notes.push("No Developer ID Application certificate: the app will be signed ad hoc, not notarized.");
  } else if (!process.env.APPLE_KEYCHAIN_PROFILE && !process.env.APPLE_ID) {
    notes.push("Signed with your Developer ID, but not notarized: set APPLE_KEYCHAIN_PROFILE (docs/developer/desktop-app.md, Signing).");
  } else notes.push("Will be signed with your Developer ID and notarized.");
}

for (const note of notes) console.log(note);
if (problems.length) {
  console.error(`\nCan't build the installers yet:\n${problems.map((p) => `  - ${p}`).join("\n")}`);
  process.exit(1);
}
console.log(`Build tools OK: Node ${process.version}, ${version("uv")}.`);
