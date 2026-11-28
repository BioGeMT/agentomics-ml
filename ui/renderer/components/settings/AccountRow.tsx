/** A provider's name and state on the left, its actions on the right. */
export function AccountRow({ name, meta, children }: { name: string; meta: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="account">
      <div>
        <strong>{name}</strong>
        <span className="account-meta">{meta}</span>
      </div>
      {children}
    </div>
  );
}
