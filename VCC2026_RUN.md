# VCC 2026: running Agentomics on the full dataset

## 1. Install

```bash
git clone -b vcc_task git@github.com:BioGeMT/agentomics-ml.git
cd agentomics-ml
pip install -e .
```

## 2. Cap the validation cells per perturbation

From the repository root:

```bash
PYTHONPATH=src python scripts/cap_validation_cells.py \
    --source <path/to/vcc_agentomics> \
    --output <datasets_dir>/vcc_agentomics
```

`--output` must not exist yet and needs about 1.1 TB of free space.

## 3. Run Agentomics

From the repository root:

```bash
agentomics-run --dev \
    --provider codex \
    --model gpt-6.1-sol \
    --datasets-dir <datasets_dir> \
    --dataset vcc_agentomics \
    --iterations 10000 \
    --disable-training-reporting
```

## 4. Make the submission

Install the VCC CLI and log in with an API token from the Credentials page of the VCC web app:

```bash
pip install vcc-cli
vcc login --token-stdin   # paste the token, then press Enter
vcc whoami
```
Download the final-phase controls and unzip them into `<vcc_dir>` (`vcc datasets list` shows the dataset id):

```bash
vcc datasets download <dataset_id> -d <download_dir>
unzip <download_dir>/<file>.zip -d <vcc_dir>
```

From the repository root, generate and package the prediction:

```bash
python scripts/vcc2026_submission.py \
    --vcc-dir <vcc_dir> \
    --controls-dir <path/to/prediction_inputs> \
    --dataset-dir <datasets_dir>/vcc_agentomics \
    --agent-dir outputs/<agent_id> \
    --work-dir <work_dir> \
    --vcc $(which vcc) \
    --dev
```

`--controls-dir` is the folder with `VCC2026_inputs__{D,E,F}.h5ad`.

Submit:

```bash
vcc submit <work_dir>/pred.vcc -m "<message>" --wait
```