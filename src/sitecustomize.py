"""Process-wide defaults loaded by Python before application imports."""

import os

os.environ.setdefault("WANDB_ERROR_REPORTING", "false")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WEAVE_LOG_LEVEL", "ERROR")
