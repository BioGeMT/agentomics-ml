// SF Symbol “trash”, drawn to match Apple's glyph at small sizes (the symbol font
// isn't available to a web page).

type Props = { size?: number };

export function TrashSymbol({ size = 15 }: Props) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" aria-hidden>
      <path d="M3.5 5.2h13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M7.6 5V3.9c0-.7.5-1.2 1.2-1.2h2.4c.7 0 1.2.5 1.2 1.2V5" stroke="currentColor" strokeWidth="1.5" />
      <path d="M5.1 5.4l.8 10.3c.1.9.8 1.6 1.7 1.6h4.8c.9 0 1.6-.7 1.7-1.6l.8-10.3" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M8.4 8.2v6M11.6 8.2v6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}
