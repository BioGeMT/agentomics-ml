// electron-builder afterPack hook: signs the app ad hoc, since packaging rewrites Info.plist and adds the
// agentomics-run folder. Without a valid signature macOS reports the app as damaged; with an ad hoc one,
// people can open it from System Settings › Privacy & Security. When a "Developer ID Application"
// certificate is available (CSC_LINK, or the keychain), electron-builder then signs the app properly
// with the hardened runtime, replacing this signature, and notarizes it if Apple credentials are set
// (see docs/developer/desktop-app.md, "Signing").
const { execFileSync } = require("node:child_process");
const path = require("node:path");

exports.default = async function afterPack(context) {
  if (context.electronPlatformName !== "darwin") return;
  const app = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`);
  execFileSync("codesign", ["--force", "--deep", "--sign", "-", app], { stdio: "inherit" });
  execFileSync("codesign", ["--verify", "--deep", "--strict", app], { stdio: "inherit" });
};
