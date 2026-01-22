#!/usr/bin/env bash

source "./bash_helpers.sh"

ENV_FILE="competitors/environment_datasets.yaml"
ENV_NAME="agentomics-datasets"

need_cmd conda
[[ -f "$ENV_FILE" ]] || die "Missing ${ENV_FILE}."

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
[[ -n "$CONDA_BASE" ]] || die "Unable to locate conda base. Is conda initialized?"

CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
[[ -f "$CONDA_SH" ]] || die "Missing conda activation script at ${CONDA_SH}."

# shellcheck source=/dev/null
source "$CONDA_SH"

if conda env list | awk '!/^#/{print $1}' | grep -qx "$ENV_NAME"; then
    info "Conda env ${ENV_NAME} exists; updating from ${ENV_FILE}."
    conda env update -n "$ENV_NAME" -f "$ENV_FILE"
else
    info "Creating conda env ${ENV_NAME} from ${ENV_FILE}."
    conda env create -f "$ENV_FILE"
fi

info "Activating ${ENV_NAME}."
conda activate "$ENV_NAME"

info "Downloading example datasets."
python src/utils/create_datasets.py

conda deactivate
info "Done. Example datasets are in ./datasets."
