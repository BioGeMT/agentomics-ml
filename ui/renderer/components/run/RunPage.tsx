import { useEffect, useMemo, useRef, useState } from "react";
import { useEvaluation } from "@/hooks/useEvaluation";
import { useFoldedIterations } from "@/hooks/useFoldedIterations";
import { useFollow } from "@/hooks/useFollow";
import { useMenuToLobby } from "@/hooks/useMenuToLobby";
import { useNow } from "@/hooks/useNow";
import { usePanes } from "@/hooks/usePanes";
import { useReadingEntry } from "@/hooks/useReadingEntry";
import { useRunStream } from "@/hooks/useRunStream";
import { useSnapshots } from "@/hooks/useSnapshots";
import { workspacePath } from "@/model/container-paths";
import { nextStep, withUpcoming, type Entry } from "@/model/run";
import { parseRunLink, type RunMeta } from "@/model/run-link";
import { ContentsSidebar } from "../contents/ContentsSidebar";
import { DetailsSidebar } from "../details/DetailsSidebar";
import type { Details } from "../details/DetailsTabs";
import { FileViewer } from "../files/FileViewer";
import { Notebook } from "../notebook/Notebook";
import { NarrowSummary } from "./NarrowSummary";
import { NarrowTabs, type Tab } from "./NarrowTabs";
import { RunToolbar } from "./RunToolbar";

/**
 * A run as a lab notebook: contents on the left, the entries in the middle, figures and files on the right.
 * Opened from the lobby with a run link (model/run-link.ts); follows a live run or replays a saved one.
 */
export function RunPage() {
  const { run, source, unreachable, connect, setSnapshots } = useRunStream();
  const [meta, setMeta] = useState<RunMeta>({});
  // The run's workspace on disk, for the file explorer: two levels above its event log.
  const [workspace, setWorkspace] = useState<string | null>(null);
  const [live, setLive] = useState(false);
  const [details, setDetails] = useState<Details>("figures");
  const [openFile, setOpenFile] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("notebook"); // narrow windows only
  const panes = usePanes();
  const pageRef = useRef<HTMLElement>(null);
  const now = useNow(run.status === "running");
  const follow = useFollow(pageRef, run, live);
  useSnapshots(workspace, run, setSnapshots);
  useMenuToLobby();

  useEffect(() => {
    const link = parseRunLink(window.location.search);
    if (!link) return;
    setMeta(link.meta);
    setWorkspace(link.workspace);
    setLive(link.live);
    connect(link.source);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const totalIterations = meta.iterations ?? run.snapshots[run.snapshots.length - 1]?.iterations_total;
  const entries = useMemo(() => withUpcoming(run, totalIterations), [run, totalIterations]);
  const chapters = useMemo(() => {
    const byIteration = new Map<number, Entry[]>();
    entries.filter((e) => e.position > 0).forEach((e) => byIteration.set(e.iteration, [...(byIteration.get(e.iteration) ?? []), e]));
    return [...byIteration.entries()];
  }, [entries]);
  const evaluation = useEvaluation(workspace, run.snapshots, run.status === "finished");
  const { folded, toggle: toggleIteration, unfold } = useFoldedIterations(chapters.map(([iteration]) => iteration));
  // Re-observed when iterations fold or unfold, since that mounts and unmounts their entries.
  const active = useReadingEntry(pageRef, [entries.length, run.entries.length, folded]);
  const liveEntry = run.entries.find((e) => e.status === "running" && e.position > 0);
  // Between steps (or before the first) a live run is still working: the step it is about to start.
  const starting = nextStep(run, entries);
  const counts = useMemo(() => {
    const records = run.entries.flatMap((e) => e.records);
    return { tools: records.filter((r) => r.kind === "tool").length, done: run.entries.filter((e) => e.status === "done" && e.position > 0).length };
  }, [run.entries]);

  /** Scrolls the notebook to an entry, unfolding its iteration first if needed; an open file gives way to it. */
  function goTo(key: string) {
    setOpenFile(null);
    follow.stopFollowing();
    setTab("notebook");
    panes.closeDrawer();
    const iteration = Number(key.split(".")[0]);
    const scroll = () => document.getElementById(`entry-${key}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    if (folded.has(iteration)) {
      unfold(iteration);
      requestAnimationFrame(() => requestAnimationFrame(scroll)); // after the iteration's entries render
    } else scroll();
  }

  function open(rel: string) {
    setOpenFile(rel);
    panes.closeDrawer();
  }

  /** Opens a /workspace/… path an entry mentions in the file viewer. */
  async function openContainerPath(path: string, entry: Entry) {
    if (workspace) open(await workspacePath(workspace, path, entry));
  }

  const elapsed = run.startedAt ? (run.endedAt ?? now) - run.startedAt : undefined;
  const title = meta.dataset ?? source?.log.split("/").slice(-3, -2)[0] ?? "Run";
  const { compact, drawer, shown, onScreen } = panes;

  return (
    <div className="nb" data-tab={tab} data-contents={shown.contents ? "open" : "folded"} data-figures={shown.figures ? "open" : "folded"} data-drawer={compact && drawer ? drawer : undefined}>
      <RunToolbar
        title={title}
        model={meta.model}
        run={run}
        elapsed={elapsed}
        offline={Boolean(unreachable) && run.entries.length === 0}
        counts={counts}
        shown={shown}
        onToggle={panes.toggle}
      />

      <NarrowSummary run={run} live={liveEntry} now={now} />
      <NarrowTabs tab={tab} onTab={setTab} />

      {compact && drawer && <button className="drawer-scrim" aria-label="Close sidebar" tabIndex={-1} onClick={panes.closeDrawer} />}
      <ContentsSidebar
        run={run}
        chapters={chapters}
        folded={folded}
        active={active}
        current={liveEntry ?? starting}
        now={now}
        hidden={!(compact ? shown.contents : onScreen.contents)}
        inert={!shown.contents}
        onToggleIteration={toggleIteration}
        onGoTo={goTo}
      />

      <Notebook
        pageRef={pageRef}
        run={run}
        source={source}
        unreachable={unreachable}
        folded={folded}
        now={now}
        elapsed={elapsed}
        done={counts.done}
        inert={panes.behindDrawer}
        showJump={live && run.status === "running" && !follow.following}
        liveEntry={liveEntry}
        starting={starting}
        onScroll={follow.onScroll}
        onJump={() => { setOpenFile(null); follow.jumpToLive(); }}
        onConnect={connect}
        onToggleIteration={toggleIteration}
        onPath={workspace ? openContainerPath : undefined}
      />

      <DetailsSidebar
        run={run}
        workspace={workspace}
        details={details}
        onDetails={setDetails}
        evaluation={evaluation}
        provider={meta.provider}
        openFile={openFile}
        now={now}
        hidden={!(compact ? shown.figures : onScreen.figures)}
        inert={!shown.figures}
        onOpenFile={open}
        onGoTo={goTo}
      />
      {openFile && workspace && <FileViewer root={workspace} rel={openFile} onClose={() => setOpenFile(null)} />}
    </div>
  );
}
