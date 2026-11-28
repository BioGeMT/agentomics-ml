/** An optional instruction for the agent: a new run's goal, or what a fork should do differently. */
export function GoalField({ value, onChange, label = "Goal", placeholder = "Develop a model that generalizes well to unseen data." }: {
  value: string; onChange: (value: string) => void; label?: string; placeholder?: string;
}) {
  return (
    <section className="form-group goal">
      <label htmlFor="goal">{label} <em>optional</em></label>
      <textarea id="goal" rows={3} placeholder={placeholder} value={value} onChange={(e) => onChange(e.target.value)} />
    </section>
  );
}
