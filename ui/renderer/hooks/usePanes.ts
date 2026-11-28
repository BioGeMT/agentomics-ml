import { useEffect, useRef, useState } from "react";
import { useLinger } from "./useLinger";

export type Side = "contents" | "figures";

/**
 * The run page's two sidebars. Wide windows show them side by side, each open or folded to a rail
 * (remembered per viewer); medium windows keep both as rails and open one at a time over the page,
 * as a drawer. ⌘[ and ⌘] toggle them (from the View menu in the desktop app); Escape closes a drawer.
 */
const MEDIUM = "(min-width: 761px) and (max-width: 1180px)";

/** The sidebars this viewer last left open or folded. */
function savedPanes(): Record<Side, boolean> {
  try {
    const saved = JSON.parse(localStorage.getItem("agentomics.panes") ?? "null");
    if (saved && typeof saved.contents === "boolean" && typeof saved.figures === "boolean") return saved;
  } catch {}
  return { contents: true, figures: true };
}

export function usePanes() {
  const [panes, setPanes] = useState(savedPanes);
  // Known before the first paint, so a medium window doesn't open laid out as a wide one.
  const [compact, setCompact] = useState(() => window.matchMedia(MEDIUM).matches);
  const compactRef = useRef(compact);
  const [drawer, setDrawer] = useState<null | Side>(null);
  const shown = compact ? { contents: drawer === "contents", figures: drawer === "figures" } : panes;
  // Keep a closing sidebar on screen until its slide-out has finished.
  const onScreen = { contents: useLinger(shown.contents) || shown.contents, figures: useLinger(shown.figures) || shown.figures };

  useEffect(() => {
    const query = window.matchMedia(MEDIUM);
    const update = () => {
      compactRef.current = query.matches;
      setCompact(query.matches);
      setDrawer(null);
    };
    update();
    query.addEventListener("change", update);
    const onEscape = (event: KeyboardEvent) => { if (event.key === "Escape") setDrawer(null); };
    window.addEventListener("keydown", onEscape);
    return () => {
      query.removeEventListener("change", update);
      window.removeEventListener("keydown", onEscape);
    };
  }, []);

  function toggle(side: Side) {
    if (compactRef.current) {
      setDrawer((current) => (current === side ? null : side));
      return;
    }
    setPanes((current) => {
      const next = { ...current, [side]: !current[side] };
      try {
        localStorage.setItem("agentomics.panes", JSON.stringify(next));
      } catch {}
      return next;
    });
  }

  /** Makes sure a sidebar is showing: opens its drawer in compact windows, unfolds it otherwise. */
  function show(side: Side) {
    if (compactRef.current) setDrawer(side);
    else if (!panes[side]) toggle(side);
  }

  // An open drawer takes focus; closing it hands focus back to the rail that opened it.
  const lastDrawer = useRef<null | Side>(null);
  useEffect(() => {
    const previous = lastDrawer.current;
    lastDrawer.current = drawer;
    if (drawer) document.querySelector<HTMLButtonElement>(`.nb > .${drawer} button`)?.focus();
    else if (previous) document.querySelector<HTMLButtonElement>(`.toggle-${previous}`)?.focus();
  }, [drawer]);

  // The desktop app's View menu shows Show or Hide for each sidebar, so it hears about every change.
  useEffect(() => {
    window.agentomics?.setViewState({ contents: shown.contents, figures: shown.figures });
  }, [shown.contents, shown.figures]);
  useEffect(() => () => { window.agentomics?.setViewState(null); }, []);

  // In the desktop app the View menu's ⌘[ and ⌘] toggle the sidebars; in a browser there is no menu,
  // so the page listens for the keys itself.
  useEffect(() => {
    const desktop = window.agentomics;
    if (desktop) {
      return desktop.onMenu((action) => {
        if (action === "toggle-contents") toggle("contents");
        if (action === "toggle-figures") toggle("figures");
      });
    }
    const onKey = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey)) return;
      if (event.key === "[") { event.preventDefault(); toggle("contents"); }
      if (event.key === "]") { event.preventDefault(); toggle("figures"); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { compact, drawer, shown, onScreen, toggle, show, closeDrawer: () => setDrawer(null), behindDrawer: compact && drawer !== null };
}
