// electron-builder afterPack hook. Until there is a Developer ID certificate the app is signed ad hoc:
// packaging rewrites Info.plist and adds the agentomics-run folder, so the whole bundle is re-signed.
// Without a valid signature macOS reports the app as damaged; with one, people can open it from
// System Settings › Privacy & Security.
const { execFileSync } = require("node:child_process");
const path = require("node:path");

exports.default = async function afterPack(context) {
  if (context.electronPlatformName !== "darwin") return;
  const app = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`);
  execFileSync("codesign", ["--force", "--deep", "--sign", "-", app], { stdio: "inherit" });
  execFileSync("codesign", ["--verify", "--deep", "--strict", app], { stdio: "inherit" });
};
