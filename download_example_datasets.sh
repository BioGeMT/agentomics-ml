#!/usr/bin/env bash
conda env create -f competitors/environment_datasets.yaml
conda activate agentomics-datasets
python src/utils/create_datasets.py
conda deactivate
