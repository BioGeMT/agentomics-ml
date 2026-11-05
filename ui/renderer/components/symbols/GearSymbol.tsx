// SF Symbol “gearshape”, drawn to match Apple's glyph at small sizes (the symbol font
// isn't available to a web page).

type Props = { size?: number };

export function GearSymbol({ size = 16 }: Props) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" aria-hidden>
      <path
        d="M15.39 8.15L17.29 8.75L17.29 11.25L15.39 11.85L15.12 12.51L16.04 14.27L14.27 16.04L12.51 15.12L11.85 15.39L11.25 17.29L8.75 17.29L8.15 15.39L7.49 15.12L5.73 16.04L3.96 14.27L4.88 12.51L4.61 11.85L2.71 11.25L2.71 8.75L4.61 8.15L4.88 7.49L3.96 5.73L5.73 3.96L7.49 4.88L8.15 4.61L8.75 2.71L11.25 2.71L11.85 4.61L12.51 4.88L14.27 3.96L16.04 5.73L15.12 7.49Z"
        stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round"
      />
      <circle cx="10" cy="10" r="2.4" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}
