// Paths in a run's folder, following the links the agent makes inside its container. Those point at
// /datasets/<dataset>/…, where the CLI mounts the items of the run's dataset folder; on the Mac the same
// files are in that dataset folder. The dataset folder comes from the app's own run history, never from
// the request, and a link is only followed into it.
import { lstat, readlink, realpath } from "node:fs/promises";
import path from "node:path";

const CONTAINER_DATASETS = "/datasets/";

/** The dataset folder a run was started on, from the app's run history; null when unknown. */
export type DatasetFor = (runFolder: string) => Promise<string | null>;

export type Located = { ok: true; real: string } | { ok: false; reason: "missing" | "container-link" | "outside" };

const inside = (child: string, parent: string) => child === parent || child.startsWith(parent + path.sep);

/** Where a container link points on the Mac, with the dataset folder it lands in; null if it doesn't. */
export async function followContainerLink(link: string, runFolder: string, datasetFor?: DatasetFor): Promise<{ real: string; within: string } | null> {
  const target = await readlink(link).catch(() => null);
  if (!target?.startsWith(CONTAINER_DATASETS) || !datasetFor) return null;
  const dataset = await datasetFor(runFolder);
  const within = dataset ? await realpath(dataset).catch(() => null) : null;
  if (!dataset || !within) return null;
  const [name, ...rest] = target.slice(CONTAINER_DATASETS.length).split("/");
  if (name !== path.basename(dataset)) return null;
  const real = await realpath(path.resolve(within, ...rest)).catch(() => null);
  return real && inside(real, within) ? { real, within } : null;
}

/**
 * The real path of relative within the run folder realRoot, one segment at a time, so a file inside a
 * linked folder resolves too. Stays inside the run folder, or inside its dataset folder through a link.
 */
export async function locate(realRoot: string, relative: string, datasetFor?: DatasetFor): Promise<Located> {
  const requested = path.resolve(realRoot, relative);
  if (!inside(requested, realRoot)) return { ok: false, reason: "outside" };
  let current = realRoot;
  let allowed = realRoot;
  for (const segment of path.relative(realRoot, requested).split(path.sep).filter(Boolean)) {
    const next = path.join(current, segment);
    const info = await lstat(next).catch(() => null);
    if (!info) return { ok: false, reason: "missing" };
    let real = await realpath(next).catch(() => null);
    if (!real && info.isSymbolicLink()) {
      const followed = await followContainerLink(next, realRoot, datasetFor);
      if (!followed) return { ok: false, reason: "container-link" };
      real = followed.real;
      allowed = followed.within; // past a container link, the walk stays inside the dataset folder
    }
    if (!real || !inside(real, allowed)) return { ok: false, reason: "outside" };
    current = real;
  }
  return { ok: true, real: current };
}
