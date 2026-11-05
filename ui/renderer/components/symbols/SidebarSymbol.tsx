// SF Symbols “sidebar.left” / “sidebar.right”, drawn to match at toolbar size.
export function SidebarSymbol({ side = "left", size = 18 }: { side?: "left" | "right"; size?: number }) {
  const pane = side === "left" ? 7.5 : 12.5; // the divider between the sidebar and the content
  const lines = side === "left" ? 3.6 : 14.4; // where the sidebar's list lines start
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" aria-hidden className="sidebar-icon">
      <rect x="2" y="3.5" width="16" height="13" rx="3" stroke="currentColor" strokeWidth="1.4" />
      <path d={`M${pane} 3.8v12.4`} stroke="currentColor" strokeWidth="1.4" />
      <path d={`M${lines} 6.8h2M${lines} 9h2M${lines} 11.2h2`} stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}
