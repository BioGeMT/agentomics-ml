/** An optional goal for the agent, passed to the run as its user prompt. */
export function GoalField({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  return (
    <section className="form-group goal">
      <label htmlFor="goal">Goal <em>optional</em></label>
      <textarea id="goal" rows={3} placeholder="Develop a model that generalizes well to unseen data." value={value} onChange={(e) => onChange(e.target.value)} />
    </section>
  );
}
