# Virtual Embryo Challenge 2026, Task 1: running Agentomics

Agent Team track. Task 1: forecast the single-cell transcriptome of a later mouse embryo stage
(E10.5 leaderboard, E12.5 final test) from E8.5 and E9.5. Task type: `temporal_generation`.

## 1. Data

Download `T1/E8.5_RNA.h5ad` and `T1/E9.5_RNA.h5ad` with the challenge API into `<t1_dir>`.
Request the E10.5 validation answers (P3 allows it; it ends P2 eligibility).

## 2. Build the dataset

```bash
PYTHONPATH=src python scripts/build_ve_t1_dataset.py \
    --source <t1_dir> --e105 <E10.5 answers .h5ad> --output <datasets_dir>/ve_t1
```

The dataset has a fixed split, mirroring the leaderboard:

- `train/`: E8.5 observed, E9.5 as target.
- `validation/`: E8.5 and E9.5 observed, E10.5 as target.

The agent keeps this split and only builds `mini_train`.

Before the E10.5 answers arrive, `--mock-e105` adds a synthetic E10.5 so the pipeline can be tested.
Its scores mean nothing biologically.

## 3. Run Agentomics

```bash
agentomics-run --dev \
    --provider codex \
    --model gpt-6.1-sol \
    --datasets-dir <datasets_dir> \
    --dataset ve_t1 \
    --iterations <n> \
    --disable-training-reporting
```

Validation metric: `VE_SCORE`. This is a local reconstruction of the Task 1 score from the published metric
definitions: 50 = copy_last, 100 = split-half replicate. The official scorer is not public, so the value
approximates the leaderboard. The component metrics are `VE_DE_SCORE`, `VE_DE_DIRECTION`, `VE_MMD` and
`VE_VARIOGRAM`.

The Agent Team rules require an autonomous, configuration-locked run. Keep the run's trajectory, prompts
and harness code (this branch) as evidence.

## 4. Make the submission

E12.5, final test (E10.5 given to the model as an observed stage):

```bash
PYTHONPATH=src python scripts/ve_t1_submission.py \
    --source <t1_dir> --e105 <E10.5 answers .h5ad> --target-stage E12.5 \
    --agent-dir outputs/<agent_id> --work-dir <work_dir> --dev \
    [--score-script <starter kit>/score_h5ad.py]
```

Use `--target-stage E10.5` without `--e105` for a leaderboard-style check. Upload `<work_dir>/pred.h5ad`
from the challenge account page. The deadline is 2 December 2026.
