// Checking a dataset folder before a run: inside the run's Docker image, with Agentomics' own checks,
// plus a few cheap facts read straight from the folder for the dataset card.
import type { CheckResult, DatasetDetails } from "@agentomics/contract";
import fs from "node:fs";
import path from "node:path";
import { imagePlatform, requireDirectory, requireImage, run } from "./exec";
import { selectedImage } from "./images";
import { CONTAINER_PYTHON, REPO_ROOT } from "./paths";

/** A prepared copy (what Agentomics makes from a dataset) rather than a dataset to start from, with its original if found. */
function preparedCopy(folder: string): { original: string | null } | null {
  const labels = path.join(folder, "train", "labels.csv");
  const header = fs.existsSync(labels) ? fs.readFileSync(labels, "utf8").split("\n", 1)[0].trim() : "";
  const prepared = folder.split(path.sep).includes("prepared_datasets") || header === "id,numeric_label";
  if (!prepared) return null;
  const original = path.join(REPO_ROOT, "datasets", path.basename(folder));
  return { original: fs.existsSync(original) ? original : null };
}

function firstLine(file: string, bytes = 65536): string | null {
  try {
    const fd = fs.openSync(file, "r");
    const buffer = Buffer.alloc(bytes);
    const read = fs.readSync(fd, buffer, 0, bytes, 0);
    fs.closeSync(fd);
    return buffer.subarray(0, read).toString("utf8").split(/\r?\n/, 1)[0];
  } catch {
    return null;
  }
}

/** Cheap facts read straight from the dataset folder, to show alongside the check's summary. */
function datasetDetails(folder: string): DatasetDetails {
  const details: DatasetDetails = {};
  try {
    const description = fs.readFileSync(path.join(folder, "dataset_description.md"), "utf8");
    details.title = description.split(/\r?\n/).find((line) => line.startsWith("# "))?.slice(2).trim();
  } catch {}
  try {
    const [header, ...rows] = fs.readFileSync(path.join(folder, "train", "labels.csv"), "utf8").trim().split(/\r?\n/);
    const column = header.split(",").indexOf("label");
    if (column >= 0 && rows.length <= 2_000_000) {
      const counts: Record<string, number> = {};
      for (const row of rows) {
        const value = row.split(",")[column];
        counts[value] = (counts[value] ?? 0) + 1;
      }
      if (Object.keys(counts).length <= 50) details.classes = counts;
    }
  } catch {}
  try {
    const inputs = fs.readdirSync(path.join(folder, "train", "input")).filter((name) => /\.(csv|tsv)$/i.test(name));
    if (inputs.length === 1) {
      const header = firstLine(path.join(folder, "train", "input", inputs[0]));
      if (header) details.features = header.split(/\.tsv$/i.test(inputs[0]) ? "\t" : ",").filter((c) => c.trim() && c.trim() !== "id").length;
    }
  } catch {}
  return details;
}

/** The check's error in the user's terms: paths relative to their folder, no Python exception names. */
function friendlyError(message: string): string {
  return String(message)
    .replace(/^[A-Za-z]*(Error|Exception): /, "")
    .replaceAll("/data/", "");
}

export async function checkDataset(folder: unknown, chosen?: unknown): Promise<CheckResult> {
  const image = chosen ?? (await selectedImage());
  requireDirectory(folder);
  requireImage(image);
  const copy = preparedCopy(folder);
  if (copy) {
    return {
      ok: false,
      error: copy.original
        ? `This is the prepared copy Agentomics makes from a dataset. Choose the original instead: ${copy.original}`
        : "This is a prepared copy Agentomics makes from a dataset. Choose the original dataset folder, whose train/labels.csv has the columns id,label.",
      suggestion: copy.original ?? undefined,
    };
  }
  const platform = await imagePlatform(image);
  if (!platform) return { ok: false, error: `Docker image ${image} was not found. Build or pull it first.` };
  const script = [
    "import json, warnings",
    "warnings.filterwarnings('ignore')",
    "from pathlib import Path",
    "from agentomics.datasets.dataset_preparation import check_dataset",
    "try:",
    "    print(json.dumps({'ok': True, 'summary': check_dataset(source_dir=Path('/data'), interactive=False)}, default=str))",
    "except Exception as error:",
    "    print(json.dumps({'ok': False, 'error': f'{type(error).__name__}: {error}'}))",
  ].join("\n");
  const { stdout, stderr } = await run("docker", [
    "run", "--rm", "--platform", platform, "-e", "PYTHONPATH=/repository/src",
    "--mount", `type=bind,src=${folder},dst=/data,readonly`,
    "--entrypoint", CONTAINER_PYTHON, image, "-c", script,
  ]);
  const lastLine = stdout.trim().split("\n").pop() ?? "";
  try {
    const result = JSON.parse(lastLine);
    return result.ok ? { ...result, details: datasetDetails(folder) } : { ...result, error: friendlyError(result.error) };
  } catch {
    return { ok: false, error: friendlyError(stderr.trim().split("\n").pop() || "The dataset check failed.") };
  }
}
