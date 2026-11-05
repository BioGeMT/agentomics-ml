import type { Appearance, DesktopApi } from "@agentomics/contract";
import { useEffect, useState } from "react";

const CHOICES: { value: Appearance; label: string }[] = [
  { value: "system", label: "Match System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

/** Light or dark: follows the Mac unless pinned to one. */
export function AppearanceCard({ api }: { api: Pick<DesktopApi, "getAppearance" | "setAppearance"> }) {
  const [appearance, setAppearance] = useState<Appearance | null>(null);
  useEffect(() => { api.getAppearance().then(setAppearance); }, [api]);

  async function choose(value: Appearance) {
    setAppearance(value);
    setAppearance(await api.setAppearance(value));
  }

  return (
    <section className="home-card appearance-card">
      <div className="appearance-row">
        <div>
          <h2>Appearance</h2>
          <p className="hint">Match System follows the Mac’s light or dark setting.</p>
        </div>
        <div className="segmented" role="radiogroup" aria-label="Appearance">
          {CHOICES.map((choice) => (
            <button
              key={choice.value}
              role="radio"
              aria-checked={appearance === choice.value}
              aria-selected={appearance === choice.value}
              onClick={() => choose(choice.value)}
              disabled={appearance === null}
            >
              {choice.label}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
