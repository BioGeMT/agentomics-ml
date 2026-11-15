import type { DesktopApi, ImageList, ImageOption, PullProgress } from "@agentomics/contract";
import { LoaderCircle } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { gigabytes } from "@/model/format";
import { ImageRow, type PullState } from "./ImageRow";

type ImagesApi = Pick<DesktopApi, "listImages" | "setImage" | "checkLiveView" | "pullImage" | "onPullProgress">;

/** Settings: which Docker image new runs use, with official releases to download and local builds. */
export function ImagesCard({ api }: { api: ImagesApi }) {
  const [list, setList] = useState<ImageList | null>(null);
  const [live, setLive] = useState<Record<string, boolean | null | "checking">>({});
  const [pulling, setPulling] = useState<Record<string, PullState>>({});

  const refresh = useCallback(async () => setList(await api.listImages()), [api]);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => api.onPullProgress((p: PullProgress) => setPulling((current) => ({ ...current, [p.name]: p }))), [api]);

  // Check, one image at a time, whether each available image can stream the live notebook.
  useEffect(() => {
    if (!list) return;
    const unknown = [...list.official.filter((i) => i.downloaded), ...list.local].filter((i) => i.live === undefined && live[i.name] === undefined);
    if (!unknown.length) return;
    const [next] = unknown;
    setLive((current) => ({ ...current, [next.name]: "checking" }));
    api.checkLiveView(next.name).then((result) => setLive((current) => ({ ...current, [next.name]: result })));
  }, [api, list, live]);

  async function choose(name: string) {
    await api.setImage(name);
    refresh();
  }

  async function download(name: string) {
    setPulling((current) => ({ ...current, [name]: { name, done: 0, total: 0, status: "Starting…" } }));
    const result = await api.pullImage(name);
    setPulling((current) => {
      const next = { ...current };
      if (result.ok) delete next[name];
      else next[name] = { error: result.error ?? "The download failed." };
      return next;
    });
    refresh();
  }

  const row = (image: ImageOption, meta: string) => (
    <ImageRow
      key={image.name}
      image={image}
      meta={meta}
      selected={list?.selected === image.name}
      live={image.live ?? live[image.name]}
      pull={pulling[image.name]}
      onChoose={() => choose(image.name)}
      onDownload={() => download(image.name)}
    />
  );

  return (
    <section className="home-card images-card">
      <h2>Docker image</h2>
      <p className="hint">Every run executes inside this image. Official releases come from Docker Hub; images you build locally are found automatically.</p>
      {!list && <p className="image-loading"><LoaderCircle size={14} className="spin" aria-hidden /> Looking for images…</p>}
      {list?.dockerError && <p className="field-error">{list.dockerError}</p>}
      {list && (
        <>
          <h3 className="image-group">Official releases <span>biogemt/agentomics</span></h3>
          {list.hubError && <p className="image-note">{list.hubError}</p>}
          <ul className="image-list">
            {list.official.map((image) => row(image, [image.updated && new Date(image.updated).toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" }), gigabytes(image.size)].filter(Boolean).join(" · ")))}
          </ul>
          <h3 className="image-group">Local builds</h3>
          {list.local.length === 0 && <p className="image-note">None found. Build one from the repository with <code>docker build -t agentomics:dev .</code></p>}
          <ul className="image-list">
            {list.local.map((image) => row(image, [image.created, gigabytes(image.size)].filter(Boolean).join(" · ")))}
          </ul>
        </>
      )}
    </section>
  );
}
