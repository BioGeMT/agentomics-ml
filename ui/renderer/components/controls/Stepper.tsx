import { Minus, Plus } from "lucide-react";

/** A number field with − and + buttons, as in macOS forms. */
export function Stepper({ labelledBy, value, min, max, step = 1, unit, onChange }: {
  labelledBy: string; value: number; min: number; max: number; step?: number; unit?: string; onChange: (v: number) => void;
}) {
  const clamp = (v: number) => Math.min(max, Math.max(min, v));
  return (
    <div className="stepper">
      <input
        type="number" aria-labelledby={labelledBy} value={value} min={min} max={max}
        onChange={(e) => onChange(clamp(Number(e.target.value) || min))}
      />
      {unit && <span className="unit">{unit}</span>}
      <div className="stepper-buttons">
        <button aria-label="Decrease" onClick={() => onChange(clamp(value - step))} disabled={value <= min}><Minus size={12} aria-hidden /></button>
        <button aria-label="Increase" onClick={() => onChange(clamp(value + step))} disabled={value >= max}><Plus size={12} aria-hidden /></button>
      </div>
    </div>
  );
}
