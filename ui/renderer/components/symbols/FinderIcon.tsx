// Finder's icons, drawn small: a blue folder and a document sheet with a folded corner.
export function FinderIcon({ folder }: { folder: boolean }) {
  return folder ? (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden className="finder-icon">
      <path d="M1.5 4.2c0-.8.6-1.4 1.4-1.4h3.3l1.4 1.4h5.5c.8 0 1.4.6 1.4 1.4v6.9c0 .8-.6 1.4-1.4 1.4H2.9c-.8 0-1.4-.6-1.4-1.4z" fill="#5aa8f0" />
      <path d="M1.5 6c0-.8.6-1.4 1.4-1.4h10.2c.8 0 1.4.6 1.4 1.4v6.5c0 .8-.6 1.4-1.4 1.4H2.9c-.8 0-1.4-.6-1.4-1.4z" fill="#8cc6fb" />
    </svg>
  ) : (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden className="finder-icon">
      <path d="M3.5 1.5h6l3 3v10h-9z" fill="#fff" stroke="#b7bdc6" />
      <path d="M9.5 1.5v3h3" fill="#eef0f3" stroke="#b7bdc6" strokeLinejoin="round" />
    </svg>
  );
}
