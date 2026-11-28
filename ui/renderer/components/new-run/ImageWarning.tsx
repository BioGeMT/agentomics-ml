/** The selected Docker image is too old to stream the live notebook. */
export function ImageWarning({ image, onOpenSettings }: { image: string | null; onOpenSettings: () => void }) {
  return (
    <p className="form-note warn">
      The Docker image <code>{image}</code> predates the live notebook: the run works, but you won&apos;t see its progress here.{" "}
      <button className="link-button" onClick={onOpenSettings}>Choose Another in Settings…</button>
    </p>
  );
}
