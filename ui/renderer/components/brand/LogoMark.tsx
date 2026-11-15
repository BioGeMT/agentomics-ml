import { useId } from "react";

// The Agentomics mark, as in the app icon (ui/desktop/icon.svg): a helix of two strands on brand blue,
// the back strand translucent so the overlap reads as depth. At toolbar sizes a rounded square with
// Apple's corner ratio stands in for the icon's continuous-corner shape.
export function LogoMark({ size = 28 }: { size?: number }) {
  const gradient = useId();
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" role="img" aria-label="Agentomics" className="logo-mark">
      <defs>
        <linearGradient id={gradient} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#5a86f5" />
          <stop offset="1" stopColor="#2442b0" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="14.4" fill={`url(#${gradient})`} />
      <g fill="none" stroke="#ffffff" strokeLinecap="round" transform="translate(32 32) scale(0.75) translate(-32 -32)">
        <path d="M42 10C42 23 22 19 22 32C22 45 42 41 42 54" strokeWidth="7" strokeOpacity="0.42" />
        <path d="M28 32H36" strokeWidth="3.85" strokeOpacity="0.42" />
        <path d="M22 10C22 23 42 19 42 32C42 45 22 41 22 54" strokeWidth="7" />
      </g>
    </svg>
  );
}
