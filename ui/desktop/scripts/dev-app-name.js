// In development macOS names the app (menu bar, Dock, app switcher, Activity Monitor) after the bundle
// that runs it: node_modules/electron/dist/Electron.app. Make an Agentomics.app copy with our name,
// executable and icon, and point the electron launcher at it. Packaged builds do this themselves.
// Runs after npm install; safe to run again.
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

if (process.platform !== "darwin") process.exit(0);

const electron = path.join(__dirname, "..", "node_modules", "electron");
const source = path.join(electron, "dist", "Electron.app");
const target = path.join(electron, "dist", "Agentomics.app");
const plist = path.join(target, "Contents", "Info.plist");
const set = (key, value) => execFileSync("plutil", ["-replace", key, "-string", value, plist]);

fs.rmSync(target, { recursive: true, force: true });
execFileSync("cp", ["-R", source, target]); // cp keeps the framework symlinks intact
fs.renameSync(path.join(target, "Contents", "MacOS", "Electron"), path.join(target, "Contents", "MacOS", "Agentomics"));
set("CFBundleName", "Agentomics");
set("CFBundleDisplayName", "Agentomics");
set("CFBundleExecutable", "Agentomics");
set("CFBundleIdentifier", "org.biogemt.agentomics.dev");

// The Dock icon before the app starts: build an .icns from icon.png.
const iconset = fs.mkdtempSync(path.join(os.tmpdir(), "agentomics-")) + "/icon.iconset";
fs.mkdirSync(iconset);
const png = path.join(__dirname, "..", "icon.png");
for (const size of [16, 32, 128, 256, 512]) {
  execFileSync("sips", ["-z", `${size}`, `${size}`, png, "--out", `${iconset}/icon_${size}x${size}.png`], { stdio: "ignore" });
  execFileSync("sips", ["-z", `${size * 2}`, `${size * 2}`, png, "--out", `${iconset}/icon_${size}x${size}@2x.png`], { stdio: "ignore" });
}
execFileSync("iconutil", ["-c", "icns", iconset, "-o", path.join(target, "Contents", "Resources", "agentomics.icns")]);
set("CFBundleIconFile", "agentomics.icns");

// The bundle changed, so re-sign it ad hoc or macOS refuses to launch it.
execFileSync("codesign", ["--force", "--deep", "--sign", "-", target], { stdio: "ignore" });
fs.writeFileSync(path.join(electron, "path.txt"), "Agentomics.app/Contents/MacOS/Agentomics");
console.log("Development app bundle: Agentomics.app");
