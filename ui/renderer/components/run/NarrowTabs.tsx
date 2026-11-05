export type Tab = "notebook" | "contents" | "figures";
const ORDER: Tab[] = ["notebook", "contents", "figures"];

/** Narrow windows show one section at a time, chosen here; arrow keys move between tabs. */
export function NarrowTabs({ tab, onTab }: { tab: Tab; onTab: (tab: Tab) => void }) {
  return (
    <div className="narrow-tabs" role="tablist" aria-label="Sections">
      {ORDER.map((name) => (
        <button
          key={name}
          id={`tab-${name}`}
          role="tab"
          aria-selected={tab === name}
          aria-controls={`panel-${name}`}
          tabIndex={tab === name ? 0 : -1}
          onClick={() => onTab(name)}
          onKeyDown={(event) => {
            const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
            if (!step) return;
            const next = ORDER[(ORDER.indexOf(name) + step + ORDER.length) % ORDER.length];
            onTab(next);
            document.getElementById(`tab-${next}`)?.focus();
          }}
        >
          {name.charAt(0).toUpperCase() + name.slice(1)}
        </button>
      ))}
    </div>
  );
}
