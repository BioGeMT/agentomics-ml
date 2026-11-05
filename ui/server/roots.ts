// The one rule for every path a request names: it must resolve, symlinks included, inside a runs folder.
import { realpath } from "node:fs/promises";
import path from "node:path";

export type Roots = {
  /** The real path of target if it lies inside a runs folder, else null. */
  within(target: string): Promise<string | null>;
};

export function createRoots(roots: string[]): Roots {
  let real: Promise<string[]> | null = null;
  const allowed = () =>
    (real ??= Promise.all(roots.map((r) => realpath(r).catch(() => null))).then((all) => all.filter((r): r is string => r !== null)));

  return {
    async within(target) {
      try {
        const resolved = await realpath(path.resolve(target));
        return (await allowed()).some((root) => resolved.startsWith(root + path.sep)) ? resolved : null;
      } catch {
        return null;
      }
    },
  };
}
