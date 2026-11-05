---
name: Agentomics
description: The agent's lab notebook, bound in graphite, for following autonomous model development live.
colors:
  cover: "#1c1f24"
  cover-deep: "#15171b"
  cover-edge: "#2e333a"
  cover-ink: "#f4f5f7"
  cover-muted: "#a3a9b3"
  wordmark-accent: "#9db8ff"
  done-on-cover: "#8fd3a8"
  logo-tile: "#1c1f24"
  logo-strand-front: "#f4f5f7"
  logo-strand-accent: "#9db8ff"
  logo-strand-back: "#7a828f"
  logo-strand-deep: "#4d5b7c"
  logo-rung: "#5c6b8a"
  page: "#ffffff"
  page-tint: "#f7f8f9"
  rail: "#f4f5f7"
  grid: "rgba(28, 31, 36, 0.04)"
  rule: "#e3e6ea"
  rule-strong: "#cfd4da"
  margin-rule: "#d64a40"
  ink: "#16181d"
  ink-2: "#474d56"
  ink-3: "#636a74"
  ballpoint: "#2f5bd3"
  ballpoint-tint: "#eef2fd"
  live: "#d9861a"
  live-text: "#8a520c"
  live-tint: "#fdf3e3"
  done: "#2e7d50"
  done-tint: "#eaf5ee"
  failed: "#b8322a"
  failed-tint: "#fcecea"
  series-validation: "#2f5bd3"
  series-train: "#b0358a"
  hover-row: "#f3f5f8"
typography:
  chapter:
    fontFamily: "Source Sans 3, Segoe UI, system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 700
    letterSpacing: "-0.02em"
  entry-title:
    fontFamily: "Source Sans 3, Segoe UI, system-ui, sans-serif"
    fontSize: "19px"
    fontWeight: 650
    letterSpacing: "-0.01em"
  card-title:
    fontFamily: "Source Sans 3, Segoe UI, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 650
    letterSpacing: "-0.01em"
  wordmark:
    fontFamily: "Source Sans 3, Segoe UI, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.015em"
  note:
    fontFamily: "Source Sans 3, Segoe UI, system-ui, sans-serif"
    fontSize: "14.5px"
    fontWeight: 400
    lineHeight: 1.6
  ui:
    fontFamily: "Source Sans 3, Segoe UI, system-ui, sans-serif"
    fontSize: "13.5px"
    fontWeight: 600
  label:
    fontFamily: "Source Sans 3, Segoe UI, system-ui, sans-serif"
    fontSize: "12.5px"
    fontWeight: 500
  code:
    fontFamily: "Source Code Pro, ui-monospace, Menlo, Consolas, monospace"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.6
  data:
    fontFamily: "Source Code Pro, ui-monospace, Menlo, Consolas, monospace"
    fontSize: "11.5px"
    fontWeight: 500
    fontFeature: "tnum"
  margin-time:
    fontFamily: "Source Code Pro, ui-monospace, Menlo, Consolas, monospace"
    fontSize: "11px"
    fontWeight: 500
    fontFeature: "tnum"
rounded:
  tag: "6px"
  control: "8px"
  card: "10px"
  panel: "12px"
  overlay: "14px"
  pill: "999px"
spacing:
  bar: "56px"
  margin-column: "92px"
  margin-column-narrow: "64px"
  record-gutter: "20px"
  paper-grid: "24px"
  entry-gap: "30px"
  contents-width: "clamp(248px, 19vw, 340px)"
  figures-width: "clamp(320px, 27vw, 520px)"
  folded-rail: "64px"
  sheet-max: "1240px"
components:
  button-primary:
    backgroundColor: "{colors.cover}"
    textColor: "{colors.cover-ink}"
    typography: "{typography.ui}"
    rounded: "{rounded.control}"
    height: "38px"
    padding: "0 18px"
  button-primary-hover:
    backgroundColor: "{colors.cover-deep}"
  button-secondary:
    backgroundColor: "{colors.page}"
    textColor: "{colors.ink}"
    typography: "{typography.ui}"
    rounded: "{rounded.control}"
    height: "34px"
    padding: "0 14px"
  button-secondary-hover:
    backgroundColor: "{colors.page-tint}"
  button-open:
    backgroundColor: "{colors.ballpoint-tint}"
    textColor: "{colors.ballpoint}"
    rounded: "{rounded.control}"
    height: "34px"
    padding: "0 14px"
  field-input:
    backgroundColor: "{colors.page}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    height: "36px"
    padding: "0 10px"
  entry-code:
    backgroundColor: "{colors.ballpoint-tint}"
    textColor: "{colors.ballpoint}"
    rounded: "{rounded.tag}"
    padding: "2px 7px"
  best-tag:
    backgroundColor: "{colors.ballpoint-tint}"
    textColor: "{colors.ballpoint}"
    padding: "1px 6px"
  run-pill-running:
    backgroundColor: "{colors.live-tint}"
    textColor: "{colors.live-text}"
    rounded: "{rounded.pill}"
    padding: "2px 9px"
  run-pill-finished:
    backgroundColor: "{colors.done-tint}"
    textColor: "{colors.done}"
    rounded: "{rounded.pill}"
    padding: "2px 9px"
  run-pill-failed:
    backgroundColor: "{colors.failed-tint}"
    textColor: "{colors.failed}"
    rounded: "{rounded.pill}"
    padding: "2px 9px"
  tool-card:
    backgroundColor: "{colors.page}"
    textColor: "{colors.ink}"
    typography: "{typography.code}"
    rounded: "{rounded.card}"
  tool-card-head:
    backgroundColor: "{colors.page-tint}"
    textColor: "{colors.ink-2}"
    typography: "{typography.label}"
    height: "34px"
    padding: "0 12px"
  toc-entry:
    textColor: "{colors.ink-2}"
    rounded: "{rounded.control}"
    height: "32px"
    padding: "0 8px"
  toc-entry-reading:
    backgroundColor: "{colors.page}"
    textColor: "{colors.ink}"
  home-card:
    backgroundColor: "{colors.page}"
    textColor: "{colors.ink}"
    rounded: "{rounded.panel}"
    padding: "20px 22px"
---

# Design System: Agentomics

## Overview

**Creative North Star: "The Lab Notebook"**

A run is the agent's lab notebook. Every step is an entry the agent writes, timestamped in a ruled margin and closed with a signed result, so a researcher reads it the way they read a colleague's notebook: skim the entries by their results, open the commands and code behind any claim, turn to the figures to judge the model. The system refuses the dark devtools agent console and the generic white chat window; it is a bound book with a dark cover and white pages.

The cover is graphite bookcloth: the 56px top bar and the app home's header. Pages are white with a faint engineering-paper grid and one red margin rule down the whole sheet. Ink is near-black, the agent's own marks (entry codes, figure labels, signatures, links) are ballpoint blue, and exactly one warm colour, amber, is reserved for work happening right now. Density is that of a working document: a single workhorse sans for UI and prose, a monospace only for code, commands and numbers-as-data, hairline rules, flat fills.

Where the direction contract's hex values differ from the build (its ballpoint, done and failed), the build wins and the values recorded here are the shipped ones; the cover is graphite in both, and the helix mark sits on that same graphite.

**Key Characteristics:**
- Graphite cover, white ruled pages, one red margin rule.
- Entry codes (1.1 to 3.9) shared by contents, page, figures and tables; iteration = chapter, step = entry.
- Margin timestamps (HH:MM:SS) in tabular mono, repeated only when the displayed time changes.
- Signed results close an entry; chapter breaks open an iteration with its validation result.
- Amber means now; blue means the agent; done green and failed red mark outcomes.
- Flat surfaces with hairline lift; depth only where something floats.

## Colors

A graphite-and-white notebook with ballpoint-blue ink and three state colours, one of which (amber) is reserved.

### Primary
- **Ballpoint Blue** (`ballpoint`): the agent's hand. Entry codes, figure and table labels ("Fig. 1", "Table 1"), the signer's name, the `$` prompt, margin labels ("Result"), fold toggles, metric-select links, the "best" tag, every focus ring on the page. Its tint (`ballpoint-tint`) backs entry-code chips, best tags, entry references and the Open button.
- **Validation Series** (`series-validation`): the same blue as ballpoint, deliberately. Validation is the number the agent is judged by, so it is drawn in the agent's ink: filled markers on a solid line.

### Secondary
- **Train Plum** (`series-train`): the train series only. Always paired with a second, non-colour cue: hollow markers on a dashed (5 4) line, hollow swatches in legends and tooltips. Its colour-blind separation from validation is moderate, so the shape difference is load-bearing.

### Tertiary (state)
- **Live Amber** (`live`): spinners, the breathing status dot, the running TOC mark, the jump-to-live focus ring. As text it always drops to **Live Amber Text** (`live-text`, 6.4:1 on white); its tint (`live-tint`) backs the running pill and the jump-to-live button.
- **Done Green** (`done`, `done-tint`): done marks, the run-finished closing line, the dataset-valid summary, "Ready" accounts. On the cover it becomes **Done on Cover** (`done-on-cover`, about 8:1).
- **Failed Red** (`failed`, `failed-tint`): failed marks, the page error, field errors, failed pills.

### Neutral
- **Graphite Cover** (`cover`, `cover-deep`, `cover-edge`): top bar, primary buttons, the logo tile; `cover-deep` for the bar's bottom edge and inputs set into it, `cover-edge` for hover fills and the divider before the run title.
- **Cover Ink / Cover Muted** (`cover-ink`, `cover-muted`): text on the cover; muted for secondary stats and controls.
- **Paper** (`page`, `page-tint`, `rail`): the notebook page and figures column; tint for tool-card heads, the home background and provider cards; rail for the contents sidebar and narrow tabs.
- **Paper Grid** (`grid`): the 24px engineering-paper lines, so faint they read as texture.
- **Rules** (`rule`, `rule-strong`): hairline dividers; strong under entry headings and around form fields.
- **Margin Red** (`margin-rule`): the single vertical margin rule, at 75% opacity. Decorative, not a state.
- **Ink ramp** (`ink`, `ink-2`, `ink-3`): headings and code; secondary text and outputs; timestamps, captions, hints.
- **Hover Row** (`hover-row`): row highlight inside charts.
- **Logo** (`logo-tile`, `logo-strand-front`, `logo-strand-accent`, `logo-strand-back`, `logo-strand-deep`, `logo-rung`): the helix mark on a graphite tile; the two front strands are cover white and pale blue, the back strands recede in grey and deep slate-blue, joined by a slate rung. `wordmark-accent` colours "omics" to match the blue front strand.

### Named Rules
**The One Warm Colour Rule.** Amber marks work happening right now (a live run, a running entry, a pending command, a dataset check in progress) and nothing else. Nothing decorative is ever amber, and amber text always uses `live-text`, never `live`.

**The Agent's Ink Rule.** Ballpoint blue belongs to what the agent wrote or what you can follow: codes, labels, signatures, references, focus. It is not a general accent for chrome.

**The Two Cues Rule.** Validation and train are never told apart by colour alone: validation is filled and solid, train is hollow and dashed.

## Typography

**Display Font:** none; headings use the body family at heavier weights.
**Body Font:** Source Sans 3 (with Segoe UI, system-ui), self-hosted via next/font as `--font-sans`.
**Label/Mono Font:** Source Code Pro (with ui-monospace, Menlo, Consolas), self-hosted as `--font-mono`.

**Character:** A plain, legible humanist sans for everything a person reads, and a quiet mono for everything a machine produced or a person compares column to column. Both are self-hosted so every desktop platform renders the same faces.

### Hierarchy
- **Chapter** (700, 24px, -0.02em): iteration chapter breaks on the page.
- **Entry title** (650, 19px, -0.01em): each step's heading beside its entry code.
- **Card title** (650, 17px): home card headings; the wordmark shares the size at 700.
- **Note** (400, 14.5px, 1.6, max 72ch): the agent's messages on the page. Signed-result values sit at 13.5px / 1.55, max 78ch.
- **UI** (600, 13 to 13.5px): buttons, TOC entries, figure captions, table text.
- **Label** (500, 12 to 12.5px): tool heads, hints, legends, status lines, sidebar headings (12px, 600, `ink-3`).
- **Code** (mono 400, 12.5px, 1.6): commands, outputs, written code, diffs.
- **Data** (mono 500, 10.5 to 12.5px, tabular): entry codes, durations, metrics, chapter results, axis labels, paths, run meta.
- **Margin time** (mono 500, 11px, tabular): margin timestamps; 10px in narrow windows.

### Named Rules
**The Mono Means Data Rule.** Mono is for code, commands, paths and identifiers (entry codes, file names, metric names) and for numbers that stand alone as data (margin times, contents durations, table, axis and chapter results); prose, headings and UI labels are sans, and numbers inside a sentence or status line stay sans with tabular figures.

**The Tabular Rule.** Every number that can change or be compared (durations, metrics, timestamps, counts) uses tabular figures so columns never jitter while a run is live.

## Layout

A three-column notebook shell under a 56px cover bar: Contents, the page, Figures. The sidebars scale with the window (Contents 248 to 340px at 19vw, Figures 320 to 520px at 27vw), so wide screens get bigger figures rather than empty margins; charts measure their own width and redraw at it, so axis and label text stays its true size. Each panel keeps a fixed grid column, so folding a sidebar never shifts the page into its place. Either sidebar folds to a 64px rail (Cmd+[ and Cmd+]) that keeps informing: a fold button, a single icon, and a vertical label reading bottom to top ("Contents · 6/18 steps", "Figures · best AUROC 0.995"). Rails carry no dots or progress pips; the label is the content.

The page is a sheet (max 1240px, right padding 20 to 56px at 3vw) on a 24px paper grid. A 92px margin column on the left holds right-aligned timestamps and labels, divided from the writing by the continuous red rule; records sit 20px past the rule. Entries are 30px apart; a chapter break (2px ink rule above, hairline below) opens each iteration. Page-level messages, errors and the closing line indent 112px to align with the writing.

Responsive behaviour: from 761px to 1180px both sidebars stay as 64px rails and open one at a time as drawers over the page (Contents up to 340px, Figures up to 460px) above a light graphite scrim; clicking the scrim or pressing Escape closes the drawer, and cover stats hide. At 760px and below the layout stacks under a narrow summary strip (best metric, a sparkline, what the agent is doing now) and Notebook / Contents / Figures tabs; the margin narrows to 64px. The app home is a single 920px column of cards on `page-tint`, two-column field grids collapsing below 700px.

**The Shared Code Rule.** An entry's code (iteration.step) is the same in the contents, on the page, in figure references and in the metrics table, and every reference navigates to the entry.

## Elevation & Depth

Flat with hairline lift. Surfaces are separated by 1px rules and tonal steps (cover, rail, tint, page), not by shadow. Resting containers that need edges (tool cards, home cards, the reading TOC row) carry a near-invisible 1 to 2px ambient shadow with a hairline. Real lift is reserved for things that float over content: the jump-to-live button, chart tooltips, sidebar drawers in medium windows, the folder drop overlay. Exact values live in the sidecar.

**The Float Only Rule.** A shadow you can see means the element floats above the page; nothing in the document flow gets more than a hairline.

## Shapes

Gently rounded, small radii scaled by element size: tags and entry codes 5 to 6px, controls and TOC rows 7 to 8px, tool cards, results and notices 10px, home cards 12px, the drop overlay 14px, status pills fully round. The logo tile is a rounded square (radius one quarter of its side). Strokes are 1px hairlines by default; the chapter break uses a 2px ink rule to mark the chapter. Dashed strokes carry specific meaning: the train series, the signature divider, empty figure placeholders, and the drop target.

## Components

### Buttons
- **Shape:** gently rounded (8px; 7px on the cover).
- **Primary:** graphite fill with cover ink, 38px tall, 0 18px; the one primary per card ("Start run"). Hover deepens to `cover-deep`.
- **Secondary:** white with a strong hairline and ink text, 34px; hover tints the fill and darkens the border.
- **Open (tonal):** ballpoint tint with ballpoint text, for opening a run from the history table.
- **Focus:** 2px ballpoint outline, 1px offset; on the cover the ring uses `wordmark-accent`.
- **Disabled:** 45% opacity, not-allowed cursor.

### Chips and Pills
- **Entry code:** mono 600 12px, ballpoint on ballpoint tint, 6px radius. Upcoming and skipped codes fade to grey.
- **Best tag:** sans 650 10.5px, ballpoint on tint, 5px radius. The only label for "new best".
- **Run pill:** fully round, 12px 650, capitalised state word; running uses live tint and live text, finished done tint, failed failed tint.
- **Run status (cover):** 28px round pill on `cover-deep` with an 8px dot in currentColor; live breathes amber, finished turns `done-on-cover`.

### Cards / Containers
- **Tool card:** 10px radius, hairline border, white body, hairline ambient shadow. A 34px tinted head carries a 14px line icon, a verb ("Shell", "Ran", "Wrote", "Edited") with the file name in mono, then duration and a right-truncating path in data type. Bodies are mono: commands with a ballpoint `$`, outputs on a faint grey folded after 12 lines with a fading mask and a "Show all N lines" toggle, written code collapsed by default with line numbers, diffs as red and green tinted lines.
- **Home card:** 12px radius, white, hairline border and ambient shadow, 20px 22px padding, 17px title with a muted hint beneath.
- **Notices:** 10px radius, state tint with a matching mid-tone border: failed for errors (with a Reconnect action), done for the run-finished closing line and the dataset summary.

### Inputs / Fields
- **Style:** 36px, white, strong hairline, 8px radius, 14px sans (13px mono for API keys and the cover port field). Labels above in 12.5px 600 `ink-2`.
- **Focus:** 2px ballpoint outline, 1px offset.
- **Checking:** amber spinner with live-text "Checking…" line beneath.
- **Error:** failed-tint box with failed text below the field.

### Navigation
- **Contents:** chapters titled "Iteration N" with the validation metric and a best tag; entries as 32px rows of status mark, mono code, title and right-aligned tabular duration. Hover tints the row; the entry being read lifts to white with a hairline ring; the running entry goes 600 with its duration in live text; upcoming entries are greyed and disabled; skipped entries are struck through.
- **Status marks:** line icons at 13px: check (done), spinning loader (running, amber), cross (failed), minus (skipped), small hollow circle (upcoming).
- **Narrow tabs:** equal-width 42px tabs on the rail colour; the selected tab turns white with a 2px graphite underline.

### Signed Result (signature component)
The agent's `final_result` closes the entry: the margin reads "Result" in ballpoint, the card is a pale-blue 10px box with a definition list of the result's fields (capitalised labels, 13.5px values), and a dashed divider leads to the signature line, "Signed **Agentomics** 14:32:07", signer in ballpoint 650.

### Chapter Break and Margin
Each iteration opens with a chapter break: a 2px ink top rule, "Iteration N" at 24px 700, the validation result in mono, and either a "best so far" tag or a muted "did not beat the best". The margin column shows a timestamp (HH:MM:SS) only when it differs from the record above.

### Figures
Captioned "Fig. 1", "Fig. 2", "Table 1" with the label in ballpoint 700, a legend, and an entry-reference chip linking back to the evaluation entry. Fig. 1 plots the run's metric by iteration with the best point ringed and direct-labelled; points sit inset from the axis; hovering a column draws a dashed crosshair and a dark ink tooltip, and clicking opens the entry. Fig. 2 is a dumbbell per bounded metric (train hollow, validation filled, grey connector), primary metric bold. Table 1 lists every metric in tabular mono; bounded metric names are underlined ballpoint buttons that re-plot Fig. 1.

### Logo
The helix mark: an upright double helix seen side on, on a rounded graphite tile. The front strands (pale blue above, cover white below) pass over grey and deep slate back strands, with a tile-coloured outline opening the gap where they cross and one slate rung at the waist; beside it the wordmark "Agent" in cover ink and "omics" in `wordmark-accent`. Used at 26 to 28px in the cover bar, and as the favicon and Dock icon.

### Drop Overlay
Dragging a folder onto the window dims everything with a translucent graphite scrim and shows a white 14px card with a 2px dashed ballpoint border, a ballpoint folder icon and a two-line instruction; the only element with a large shadow.

## Do's and Don'ts

### Do:
- **Do** keep amber for work happening right now, and set amber text in `live-text`.
- **Do** give every step an entry code and use that same code wherever the step is referenced.
- **Do** put timestamps in the margin column, in tabular mono, repeating only when the displayed time changes.
- **Do** close a step that produced a result with the signed-result card and signature line.
- **Do** distinguish train from validation by shape as well as colour (hollow and dashed versus filled and solid).
- **Do** show the agent's real commands, outputs, code and diffs in mono inside tool cards; fold long output rather than truncating it away.
- **Do** keep folded sidebars informative with a vertical label carrying one live fact.
- **Do** let sidebars and charts grow with the window instead of widening empty margins; draw charts at their measured width.
- **Do** use line icons (lucide, 13 to 16px) beside text; icon-only buttons are limited to fold and back controls, and carry an aria-label and title.

### Don't:
- **Don't** make the notebook a dark devtools console or a generic white chat thread.
- **Don't** use margin red for anything but the margin rule; it is never a state colour (failure has its own red).
- **Don't** use decorative gradient fills; gradients appear only as techniques (the paper grid, the output fade mask, the loading shimmer).
- **Don't** add visible shadows to elements that sit in the document flow.
- **Don't** add dots or progress pips to the folded rails.
- **Don't** set prose, headings or UI labels in mono, or code, commands, identifiers and stand-alone data values in the sans.
- **Don't** introduce a second warm accent or a decorative use of amber.
