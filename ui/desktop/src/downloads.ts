// Saving workspace files. The page links to GET /api/files?…&download=1 (ui/contract/protocol.ts);
// Chromium streams the attachment to disk while a native Save dialog asks where it goes.
import { app, session } from "electron";
import path from "node:path";

export function handleDownloads(): void {
  session.defaultSession.on("will-download", (_event, item) => {
    item.setSaveDialogOptions({ title: "Save File", defaultPath: path.join(app.getPath("downloads"), item.getFilename()) });
    // As Safari does: the Downloads stack in the Dock bounces when a file arrives.
    item.once("done", (_done, state) => {
      if (state === "completed" && process.platform === "darwin") app.dock?.downloadFinished(item.getSavePath());
    });
  });
}
