/** Placeholder lines while the contents load. */
export function ContentsSkeleton() {
  return (
    <div className="skeleton" aria-hidden>
      {Array.from({ length: 9 }).map((_, i) => <span key={i} style={{ width: `${55 + ((i * 17) % 35)}%` }} />)}
    </div>
  );
}
