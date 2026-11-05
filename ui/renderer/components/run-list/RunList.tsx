import type { DesktopApi, RunEntry } from "@agentomics/contract";
import { Fragment, useState } from "react";
import { filterRuns, groupByDay, isFiltering, NO_FILTER, type RunFilter } from "@/model/run-list";
import { TRASH } from "@/model/platform";
import { runLink } from "@/model/run-link";
import { RunFilters } from "./RunFilters";
import { RunListItem } from "./RunListItem";

/** The sidebar's runs: search and filters, then the runs grouped by day. */
export function RunList({ api, runs: allRuns, onChange }: { api: DesktopApi; runs: RunEntry[]; onChange: () => void }) {
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<RunFilter>(NO_FILTER);
  if (allRuns.length === 0) return <p className="sidebar-empty">Runs you start appear here.</p>;
  const runs = filterRuns(allRuns, filter);

  async function trash(run: RunEntry) {
    const result = await api.deleteRun(run.id);
    setError(result.ok || result.cancelled ? null : result.error ?? `The run couldn't be moved to the ${TRASH}.`);
    onChange();
  }

  async function stop(run: RunEntry) {
    await api.stopRun(run.id);
    onChange();
  }

  /** The native right-click menu. */
  async function menu(event: React.MouseEvent, run: RunEntry) {
    event.preventDefault();
    const action = await api.runMenu(run.id);
    if (action === "open") window.location.href = runLink(run);
    if (action === "show") api.showFolder(run.workspace);
    if (action === "stop") stop(run);
    if (action === "trash") trash(run);
  }

  return (
    <>
      {error && <p className="sidebar-error" role="alert">{error}</p>}
      <RunFilters runs={allRuns} filter={filter} onChange={setFilter} />
      {isFiltering(filter) && runs.length === 0 && (
        <p className="sidebar-empty">No runs match. <button className="link-button" onClick={() => setFilter(NO_FILTER)}>Clear Filters</button></p>
      )}
      <ul className="run-list">
        {groupByDay(runs).map(({ day, runs }) => (
          <Fragment key={runs[0].id}>
            <li className="run-day" role="presentation">{day}</li>
            {runs.map((run) => (
              <RunListItem
                key={run.id}
                run={run}
                onMenu={(event) => menu(event, run)}
                onStop={() => stop(run)}
                onTrash={() => trash(run)}
                onShow={() => api.showFolder(run.workspace)}
              />
            ))}
          </Fragment>
        ))}
      </ul>
    </>
  );
}
