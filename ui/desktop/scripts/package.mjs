// Packages the app into this OS's installers with electron-builder. On macOS it signs only with a
// "Developer ID Application" certificate (from CSC_LINK, or the keychain): without one, electron-builder
// would fall back to an "Apple Development" certificate, which other Macs don't trust, so the app stays
// signed ad hoc instead (scripts/after-pack.js). With a Developer ID, the DMG is signed too, and when Apple
// credentials are set it is notarized and stapled like the app inside it, so a downloaded DMG opens without
// warnings. See docs/developer/desktop-app.md, "Signing".
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const env = { ...process.env };
const args = ["--publish", "never", ...process.argv.slice(2)];
const run = (command, commandArgs) => {
  const result = spawnSync(command, commandArgs, { cwd: here, env, stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
};

let developerId = false;
if (process.platform === "darwin") {
  const identities = spawnSync("security", ["find-identity", "-v", "-p", "codesigning"], { encoding: "utf8" }).stdout ?? "";
  // AGENTOMICS_UNSIGNED=1 (task dist-unsigned) skips signing and notarization for a quick local build.
  developerId = env.AGENTOMICS_UNSIGNED !== "1" && (Boolean(env.CSC_LINK) || identities.includes("Developer ID Application"));
  if (developerId) args.push("-c.dmg.sign=true");
  else {
    console.log(env.AGENTOMICS_UNSIGNED === "1" ? "Unsigned build: the app is signed ad hoc, not notarized." : "No Developer ID Application certificate: the app is signed ad hoc, not notarized.");
    env.CSC_IDENTITY_AUTO_DISCOVERY = "false";
  }
}

run(process.execPath, [path.join(here, "node_modules", "electron-builder", "cli.js"), ...args]);

/** notarytool's credentials: a keychain profile (local), or the same Apple ID env vars electron-builder reads (CI). */
function notaryCredentials() {
  if (env.APPLE_KEYCHAIN_PROFILE) return ["--keychain-profile", env.APPLE_KEYCHAIN_PROFILE];
  if (env.APPLE_ID && env.APPLE_APP_SPECIFIC_PASSWORD && env.APPLE_TEAM_ID) {
    return ["--apple-id", env.APPLE_ID, "--password", env.APPLE_APP_SPECIFIC_PASSWORD, "--team-id", env.APPLE_TEAM_ID];
  }
  return null;
}

const { version } = JSON.parse(fs.readFileSync(path.join(here, "package.json"), "utf8"));
const release = path.join(here, "release");
const credentials = notaryCredentials();
const notarized = developerId && credentials;
if (notarized) {
  const dmg = path.join(release, `Agentomics-${version}-arm64.dmg`);
  console.log(`Notarizing ${path.basename(dmg)}…`);
  run("xcrun", ["notarytool", "submit", dmg, ...credentials, "--wait"]);
  run("xcrun", ["stapler", "staple", dmg]);
}

// Where the installers are: this version's, for this OS.
const installers = fs.readdirSync(release).filter((name) => name.includes(version) && /\.(dmg|AppImage|deb|exe)$/.test(name));
const signing = process.platform !== "darwin" ? "unsigned" : notarized ? "signed and notarized" : developerId ? "signed, not notarized" : "unsigned";
console.log(`\nInstallers, ${signing}, in ${release}:`);
for (const name of installers) console.log(`  ${path.join(release, name)}`);
