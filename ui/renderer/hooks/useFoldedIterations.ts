import { useState } from "react";

/** Which iterations are folded, shared by the contents and the notebook. Option-click folds or unfolds them all. */
export function useFoldedIterations(all: number[]) {
  const [folded, setFolded] = useState<Set<number>>(new Set());
  function toggle(iteration: number, event?: React.MouseEvent) {
    setFolded((current) => {
      const willFold = !current.has(iteration);
      if (event?.altKey) return new Set(willFold ? all : []);
      const next = new Set(current);
      if (willFold) next.add(iteration);
      else next.delete(iteration);
      return next;
    });
  }
  function unfold(iteration: number) {
    setFolded((current) => {
      if (!current.has(iteration)) return current;
      const next = new Set(current);
      next.delete(iteration);
      return next;
    });
  }
  return { folded, toggle, unfold };
}
