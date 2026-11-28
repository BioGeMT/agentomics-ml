// SF Symbol “stop.fill”, drawn to match Apple's glyph at small sizes (the symbol font
// isn't available to a web page).

type Props = { size?: number };

export function StopSymbol({ size = 15 }: Props) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" aria-hidden>
      <rect x="5" y="5" width="10" height="10" rx="2.4" fill="currentColor" />
    </svg>
  );
}
