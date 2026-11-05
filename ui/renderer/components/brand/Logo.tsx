// The Agentomics mark with its wordmark.
import { LogoMark } from "./LogoMark";

export function Logo({ size = 28 }: { size?: number }) {
  return (
    <span className="logo">
      <LogoMark size={size} />
      <span className="wordmark">
        Agent<span className="wordmark-omics">omics</span>
      </span>
    </span>
  );
}
