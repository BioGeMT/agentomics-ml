import type { DesktopApi, DockerStatus } from "@agentomics/contract";
import { LoaderCircle } from "lucide-react";
import { useState } from "react";
import { DOCKER } from "@/model/platform";

const DOWNLOAD = "https://www.docker.com/products/docker-desktop/";

/** Runs need Docker: says what's missing and how to fix it. Disappears by itself once Docker answers. */
export function DockerSetup({ status, api }: { status: DockerStatus; api: Pick<DesktopApi, "openDocker"> }) {
  const [opening, setOpening] = useState(false);
  async function open() {
    setOpening(true);
    await api.openDocker();
  }
  return (
    <section className="home-card docker-setup" role="status">
      <h2>{status.installed ? `${DOCKER} isn’t running` : `Install ${DOCKER}`}</h2>
      <p className="hint">
        {status.installed
          ? `Every run executes inside a Docker container. Start ${DOCKER} and this page continues when it’s ready.`
          : `Every run executes inside a Docker container, so Agentomics needs ${DOCKER}. Install it, start it once, and this page continues when it’s ready.`}
      </p>
      <div className="docker-actions">
        {status.installed ? (
          <button className="mac-primary" onClick={open} disabled={opening}>
            {opening && <LoaderCircle size={14} className="spin" aria-hidden />}
            {opening ? "Starting Docker…" : `Start ${DOCKER}`}
          </button>
        ) : (
          <a className="mac-primary" href={DOWNLOAD} target="_blank" rel="noreferrer">Download {DOCKER}</a>
        )}
      </div>
    </section>
  );
}
