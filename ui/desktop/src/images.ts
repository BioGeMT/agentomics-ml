// Docker images for runs: official releases from Docker Hub (biogemt/agentomics) and local builds.
//
// The image new runs use is a setting (settings.ts). Whether an image can stream the live
// notebook depends on its code, so the app checks once per image ID that it can log its events, and remembers.
import type { ImageList, PullProgress, Result } from "@agentomics/contract";
import { spawn } from "node:child_process";
import { run } from "./exec";
import { app } from "electron";
import { CONTAINER_PYTHON, PACKAGED } from "./paths";
import { loadSettings, saveSettings } from "./settings";

export const OFFICIAL = "biogemt/agentomics";
const HUB_TAGS = `https://hub.docker.com/v2/repositories/${OFFICIAL}/tags?page_size=25&ordering=last_updated`;
const HOST_PLATFORM = `linux/${process.arch === "arm64" ? "arm64" : "amd64"}`;

type LocalImage = { name: string; repository: string; tag: string; id: string; created: string; size: string };
type HubTag = { tag: string; updated: string; size: number };

let hubCache: { at: number; tags: HubTag[] } | null = null;

async function localImages(): Promise<LocalImage[] | null> {
  const { code, stdout } = await run("docker", ["images", "--format", "{{json .}}"]);
  if (code !== 0) return null;
  return stdout
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .filter((image) => image.Repository.includes("agentomics") && image.Tag !== "<none>")
    .map((image) => ({ name: `${image.Repository}:${image.Tag}`, repository: image.Repository, tag: image.Tag, id: image.ID, created: image.CreatedSince, size: image.Size }));
}

/** Release tags of the official image, newest first. Cached for ten minutes. */
async function officialTags(): Promise<HubTag[]> {
  if (hubCache && Date.now() - hubCache.at < 10 * 60 * 1000) return hubCache.tags;
  const response = await fetch(HUB_TAGS);
  if (!response.ok) throw new Error(`Docker Hub answered ${response.status}`);
  const tags: HubTag[] = ((await response.json()).results ?? [])
    .filter((t: any) => /^\d+\.\d+\.\d+$/.test(t.name))
    .filter((t: any) => (t.images ?? []).some((i: any) => `${i.os}/${i.architecture}` === HOST_PLATFORM))
    .slice(0, 5)
    .map((t: any) => ({ tag: t.name, updated: t.last_updated, size: t.full_size }));
  hubCache = { at: Date.now(), tags };
  return tags;
}

/**
 * The image new runs use: the saved choice, else a local build that streams the notebook, else (packaged) the
 * release matching this app's bundled CLI, else the latest release.
 */
export async function selectedImage(): Promise<string> {
  const saved = loadSettings().image;
  if (saved) return saved;
  const local = (await localImages()) ?? [];
  const live = loadSettings().eventLog ?? {};
  const build = local.find((i) => i.repository !== OFFICIAL && live[i.id]) ?? local.find((i) => i.repository !== OFFICIAL);
  if (build) return build.name;
  if (PACKAGED) return `${OFFICIAL}:${app.getVersion()}`;
  try {
    const [latest] = await officialTags();
    if (latest) return `${OFFICIAL}:${latest.tag}`;
  } catch {}
  return `${OFFICIAL}:latest`;
}

export async function listImages(): Promise<ImageList> {
  const local = await localImages();
  const live = loadSettings().eventLog ?? {};
  let tags: HubTag[] = [];
  let hubError: string | null = null;
  try {
    tags = await officialTags();
  } catch {
    hubError = "Couldn't reach Docker Hub to list releases. Check your connection.";
  }
  const byName = new Map((local ?? []).map((i) => [i.name, i]));
  return {
    selected: await selectedImage(),
    dockerError: local === null ? "Docker isn't running. Start Docker Desktop to use or download images." : null,
    hubError,
    official: tags.map((t) => {
      const name = `${OFFICIAL}:${t.tag}`;
      const downloaded = byName.get(name);
      return { name, tag: t.tag, updated: t.updated, size: t.size, downloaded: Boolean(downloaded), live: downloaded ? live[downloaded.id] : undefined };
    }),
    local: (local ?? [])
      .filter((i) => i.repository !== OFFICIAL)
      .map((i) => ({ name: i.name, created: i.created, size: i.size, live: live[i.id] })),
  };
}

export function setImage(name: string): string {
  saveSettings({ image: name });
  return name;
}

/** Whether the image can log its run's events (AGENTOMICS_UI_EVENTS), which the notebook needs. Cached per image ID. */
export async function checkLiveView(name: string): Promise<boolean | null> {
  const inspect = await run("docker", ["image", "inspect", name, "--format", "{{.Id}}"]);
  if (inspect.code !== 0) return null;
  const id = inspect.stdout.trim().replace(/^sha256:/, "").slice(0, 12);
  const cached = loadSettings().eventLog?.[id];
  if (cached !== undefined) return cached;
  const probe = await run("docker", [
    "run", "--rm", "-e", "PYTHONPATH=/repository/src", "--entrypoint", CONTAINER_PYTHON, name,
    "-c", "from agentomics.runtime.ui_events import UI_EVENTS_ENV",
  ]);
  const live = probe.code === 0;
  saveSettings({ eventLog: { ...(loadSettings().eventLog ?? {}), [id]: live } });
  return live;
}

/** Pulls an official release, reporting layer progress. Resolves when the pull ends. */
export function pullImage(name: string, onProgress: (progress: PullProgress) => void): Promise<Result> {
  return new Promise((resolve) => {
    if (!name.startsWith(`${OFFICIAL}:`)) {
      resolve({ ok: false, error: "Only official releases can be downloaded." });
      return;
    }
    const layers = new Map<string, string>();
    const child = spawn("docker", ["pull", "--platform", HOST_PLATFORM, name]);
    let tail = "";
    const read = (chunk: Buffer) => {
      for (const line of chunk.toString().split(/\r?\n/)) {
        const match = line.match(/^([0-9a-f]{12}): (.+)$/);
        if (match) layers.set(match[1], match[2]);
        if (line.trim()) tail = line.trim();
      }
      const done = [...layers.values()].filter((s) => /complete|Already exists/i.test(s)).length;
      onProgress({ name, done, total: layers.size, status: tail });
    };
    child.stdout.on("data", read);
    child.stderr.on("data", read);
    child.on("error", () => resolve({ ok: false, error: "Docker isn't available." }));
    child.on("exit", (code) => resolve(code === 0 ? { ok: true } : { ok: false, error: tail || "The download failed." }));
  });
}
