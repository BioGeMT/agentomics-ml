// The example dataset the app carries (ui/examples), so a first run needs no dataset of one's own.
// Docker can't read from inside the installed app (macOS doesn't share /Applications with containers, and
// an AppImage's files are a private mount), so the packaged app copies it once next to the runs folder.
import fs from "node:fs";
import path from "node:path";
import { EXAMPLES_DIR, PACKAGED, RUNS_DIR } from "./paths";

const EXAMPLE = "breast_cancer";

export function exampleDataset(): string {
  const source = path.join(EXAMPLES_DIR, EXAMPLE);
  if (!PACKAGED) return source;
  const target = path.join(path.dirname(RUNS_DIR), "Examples", EXAMPLE);
  if (!fs.existsSync(path.join(target, "metadata.json"))) {
    // Copied under another name first, so an interrupted copy is never taken for the example.
    const partial = `${target}.partial`;
    fs.rmSync(partial, { recursive: true, force: true });
    fs.cpSync(source, partial, { recursive: true });
    fs.rmSync(target, { recursive: true, force: true });
    fs.renameSync(partial, target);
  }
  return target;
}
