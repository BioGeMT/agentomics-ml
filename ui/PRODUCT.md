# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

(The desktop app is Electron wrapping the Next.js UI in `ui/web`; its design language is the web UI's, not native macOS/Linux.)

## Users
Lab researchers in bioinformatics and biomedicine (e.g. BioGeMT and collaborators). They hand Agentomics a dataset, then check in on long autonomous runs over hours: what the agent is doing, whether it is on track, and how good the resulting model is. They are ML-literate but not necessarily software engineers.

## Product Purpose
Agentomics is an autonomous agentic system for supervised machine learning model development on biomedical data. The input is a folder-based dataset (plus an optional description); the agent experiments with models and strategies over iterations; the output is a trained model ready for inference and a report of the development process and metrics. The desktop app lets a researcher pick a dataset folder, start a run, and follow the agent's actual work live (steps, commands, code, results, metrics), then revisit finished runs.

## Positioning
An agent that does the model development itself and has outperformed human experts, creating new state-of-the-art models in Protein Engineering, Drug Discovery and Regulatory Genomics, rather than a tool that assists a human doing the work. The app's job is to make that autonomous work visible and trustworthy.

## Operating Context
- Runs take roughly 5-15 minutes per iteration and often several iterations; researchers glance in rather than watch continuously.
- Each run executes inside a Docker image; the app streams the run's events over AG-UI from the container.
- Datasets live in local folders (train/, optional validation/ and test/, metadata.json, dataset_description.md).
- Runs use the researcher's own LLM access (Codex subscription, OpenAI/Anthropic/OpenRouter keys) and optional Weights & Biases logging.
- A run is a sequence of iterations; each iteration runs fixed steps: iteration plan, data exploration, splitting, representation, architecture, training, inference, prediction exploration, validation evaluation. Each iteration reports validation/train metrics (e.g. AUROC, ACC, F1, MCC, AUPRC, LOG_LOSS) and whether it is the new best.

## Capabilities and Constraints
- Live view of a run: steps, the agent's messages, every tool call (bash, write_python, run_python, replace, step outputs) with arguments and results, and per-iteration metrics.
- Tool output appears when a command finishes, not line by line.
- Codex reasoning arrives encrypted: the app can show that the agent is thinking, not what it thinks.
- Undecided: settings screen (logins, API keys, Docker images), dataset preparation inputs (positive class, label column), human-in-the-loop approvals.

## Brand Commitments
- Name: Agentomics.
- No binding colours or existing logo: the docs' DNA-helix glyph (docs/assets/logo.svg) and teal/green palette are not commitments; the app gets a new mark.

## Evidence on Hand
- Published in the Bioinformatics journal; to be presented at ISMB 2026 (README).
- Real run data for UI work: recorded AG-UI event logs under outputs/ (e.g. breast_cancer runs, test AUROC ~0.995).
- No testimonials, user counts or benchmarks beyond the README claims; do not fabricate any.

## Product Principles
- Make autonomous work legible: a researcher should understand what the agent did and why the model is good without reading raw logs.
- Show progress and state at a glance; detail on demand.
- Trust through fidelity: show the agent's real commands, code and outputs, never a paraphrase.
- Results first: metrics and the best model are what the researcher ultimately came for.
