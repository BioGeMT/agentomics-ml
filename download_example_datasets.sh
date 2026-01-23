#!/usr/bin/env bash
conda env remove -n agentomics-datasets -y 2>/dev/null || true
conda env create -f competitors/environment_datasets.yaml
conda run -n agentomics-datasets python src/utils/create_datasets.py