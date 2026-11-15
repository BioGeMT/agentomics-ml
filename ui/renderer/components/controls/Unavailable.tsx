/** Apple's "content unavailable" pattern: an icon, a short title, one line of explanation and an action. */
export function Unavailable({ title, message, detail, action, compact, tone }: {
  title: string; message: string; detail?: string; action?: () => void; compact?: boolean; tone?: "failed";
}) {
  return (
    <div className={`unavailable ${compact ? "compact" : ""} ${tone ?? ""}`} role="alert">
      <svg width={compact ? 22 : 40} height={compact ? 22 : 40} viewBox="0 0 24 24" fill="none" aria-hidden className="unavailable-icon">
        <path d="M12 3.5 2.8 19.5h18.4z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
        <path d="M12 9.5v4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        <circle cx="12" cy="16.8" r="1" fill="currentColor" />
      </svg>
      <div className="unavailable-text">
        <h2>{title}</h2>
        <p>{message}</p>
        {detail && <p className="unavailable-detail">{detail}</p>}
      </div>
      {action && <button className="unavailable-action" onClick={action}>Try Again</button>}
    </div>
  );
}
