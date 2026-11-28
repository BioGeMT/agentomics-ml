// SF Symbol “plus”, drawn to match Apple's glyph at small sizes (the symbol font
// isn't available to a web page).

type Props = { size?: number };

export function PlusSymbol({ size = 17 }: Props) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" aria-hidden>
      <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}
