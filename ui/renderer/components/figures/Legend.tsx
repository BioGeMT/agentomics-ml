import { TRAIN, VALIDATION } from "./series";

/** Validation as filled dots, train as hollow ones. */
export function Legend() {
  return (
    <div className="legend" aria-hidden>
      <span><svg width="14" height="10"><circle cx="7" cy="5" r="4" fill={VALIDATION} /></svg> Validation</span>
      <span><svg width="14" height="10"><circle cx="7" cy="5" r="3.3" fill="var(--page)" stroke={TRAIN} strokeWidth="2" /></svg> Train</span>
    </div>
  );
}
