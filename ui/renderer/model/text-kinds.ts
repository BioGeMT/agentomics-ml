// Recognising what a block of text is, so it can be shown as what it is: code by file name, CSV by
// shape, JSON by parsing, and the parts of a shell command and its output.

export const LANGUAGE: Record<string, string> = { py: "python", json: "json", md: "markdown", sh: "bash", yaml: "yaml", yml: "yaml" };

export const IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "svg"];

export function extension(name: string): string {
  const base = name.split("/").pop() ?? name;
  return base.includes(".") ? base.split(".").pop()!.toLowerCase() : "";
}

export function languageFor(name: string): string | undefined {
  return LANGUAGE[extension(name)];
}

export function splitRow(line: string, separator: string): string[] {
  if (separator === "\t" || !line.includes('"')) return line.split(separator);
  const cells: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"' && line[i + 1] === '"' && quoted) { cell += '"'; i++; }
    else if (c === '"') quoted = !quoted;
    else if (c === separator && !quoted) { cells.push(cell); cell = ""; }
    else cell += c;
  }
  cells.push(cell);
  return cells;
}

/** A block of lines is CSV when every line has the same number of commas (at least one). */
export function looksLikeCsv(text: string): boolean {
  const lines = text.replace(/\n$/, "").split("\n").slice(0, 20);
  if (lines.length < 2) return false;
  const count = (line: string) => splitRow(line, ",").length;
  const first = count(lines[0]);
  return first > 1 && lines.every((line) => count(line) === first);
}

export function prettyJson(text: string): string | null {
  const trimmed = text.trim();
  if (!/^[[{]/.test(trimmed)) return null;
  try {
    return JSON.stringify(JSON.parse(trimmed), null, 2);
  } catch {
    return null;
  }
}

/** The file a shell command prints, for commands like `cat x.csv` or `head -n 5 x.csv`; else undefined. */
export function printedFile(command: string): string | undefined {
  const match = /^\s*(?:cat|head|tail|less|more)\b(?:\s+-{1,2}[\w=-]+(?:\s+\d+)?)*\s+([^\s|;&>]+)\s*$/.exec(command.trim());
  return match?.[1];
}

/** Splits a command chain at top-level &&, || , ; and | (outside quotes), keeping each connector. */
export function splitCommand(command: string): { connector: string; text: string }[] {
  const parts: { connector: string; text: string }[] = [];
  let connector = "";
  let current = "";
  let quote: string | null = null;
  for (let i = 0; i < command.length; i++) {
    const c = command[i];
    if (quote) {
      current += c;
      if (c === quote && command[i - 1] !== "\\") quote = null;
      continue;
    }
    if (c === "'" || c === '"') { quote = c; current += c; continue; }
    const two = command.slice(i, i + 2);
    const op = two === "&&" || two === "||" ? two : c === ";" || c === "|" || c === "\n" ? c : null;
    if (op) {
      if (current.trim()) parts.push({ connector, text: current.trim() });
      connector = op === "\n" ? "" : op;
      current = "";
      i += op.length - 1;
      continue;
    }
    current += c;
  }
  if (current.trim()) parts.push({ connector, text: current.trim() });
  return parts;
}

export type Segment = { kind: "text"; lines: string[] } | { kind: "csv"; lines: string[] } | { kind: "warning"; lines: string[] };

/** Splits command output into plain text, CSV blocks (2+ lines with the same number of columns) and warnings. */
export function segmentOutput(text: string): Segment[] {
  const lines = text.replace(/\n?\[Tool call took [\d.]+ seconds\]\s*$/, "").replace(/\n$/, "").split("\n");
  const segments: Segment[] = [];
  const push = (kind: Segment["kind"], line: string) => {
    const last = segments[segments.length - 1];
    if (last && last.kind === kind) last.lines.push(line);
    else segments.push({ kind, lines: [line] } as Segment);
  };
  const columns = (line: string) => (line.includes(",") ? splitRow(line, ",").length : 0);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // Python warnings: "...: SomethingWarning: message" plus the indented source line that follows.
    if (/\b\w*Warning: /.test(line)) {
      push("warning", line);
      while (i + 1 < lines.length && /^\s+\S/.test(lines[i + 1]) && !lines[i + 1].includes(",")) push("warning", lines[++i]);
      continue;
    }
    const n = columns(line);
    if (n > 1 && (columns(lines[i + 1] ?? "") === n || columns(lines[i - 1] ?? "") === n)) push("csv", line);
    else push("text", line);
  }
  return segments.filter((s) => s.lines.some((l) => l.trim()));
}
