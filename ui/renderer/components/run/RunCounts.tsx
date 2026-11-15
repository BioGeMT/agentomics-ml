/** How much the notebook holds: finished entries and tool calls. */
export function RunCounts({ done, tools }: { done: number; tools: number }) {
  return (
    <div className="cover-stats">
      <span><b>{done}</b> entries</span>
      <span><b>{tools}</b> tool calls</span>
    </div>
  );
}
