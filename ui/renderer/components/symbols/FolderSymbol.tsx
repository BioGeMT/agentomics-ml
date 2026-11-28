// SF Symbol “folder”, drawn to match Apple's glyph at small sizes (the symbol font
// isn't available to a web page).

type Props = { size?: number };

export function FolderSymbol({ size = 15 }: Props) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" aria-hidden>
      <path d="M2.6 6.1c0-1.1.9-2 2-2h3.1c.5 0 1 .2 1.4.6l1 1c.4.4.9.6 1.4.6h3.9c1.1 0 2 .9 2 2v6.2c0 1.1-.9 2-2 2H4.6c-1.1 0-2-.9-2-2z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M2.8 8.6h14.4" stroke="currentColor" strokeWidth="1.3" />
    </svg>
  );
}
