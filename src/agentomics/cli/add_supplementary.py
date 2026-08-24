from __future__ import annotations

import argparse
import shutil
import sys
from importlib.resources import files
from pathlib import Path

from rich.console import Console

from agentomics.datasets.data_contract import SUPPLEMENTARY_DIR_NAME

console = Console()

CATALOG_DIR_NAME = "supplementary_catalog"
FOUNDATION_MODELS_DIR_NAME = "foundation_models"
CATALOG_README_NAME = "README.md"

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy foundation models' documentation into a dataset's supplementary/ folder"
        ),
        allow_abbrev=False,
    )
    fm_selection = parser.add_mutually_exclusive_group(required=True)
    fm_selection.add_argument(
        "--all",
        action="store_true",
        help=f"Attach all foundation models from this list: {', '.join(_available_model_names())}",
    )
    fm_selection.add_argument(
        "--model",
        help=f"Foundation model to attach. Available: {', '.join(_available_model_names())}",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory to attach the model documentation to",
    )
    return parser

def _catalog() -> Path:
    return Path(str(files("agentomics"))) / CATALOG_DIR_NAME

def _foundation_models_catalog() -> Path:
    return _catalog() / FOUNDATION_MODELS_DIR_NAME

def _available_model_names() -> list[str]:
    return sorted(doc.stem for doc in _foundation_models_catalog().glob("*.md"))

def _resolve_model_name(requested_model: str) -> str:
    available_names = _available_model_names()
    for name in available_names:
        if name.lower() == requested_model.lower():
            return name
    raise ValueError(
        f"Unknown foundation model: {requested_model}. Available: {', '.join(available_names)}"
    )

def add_supplementary_fm(model: str, dataset_dir: Path) -> Path:
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset_dir}")

    model_name = _resolve_model_name(model)
    destination_dir = dataset_dir / SUPPLEMENTARY_DIR_NAME / FOUNDATION_MODELS_DIR_NAME
    destination_dir.mkdir(parents=True, exist_ok=True)

    model_doc = _foundation_models_catalog() / f"{model_name}.md"
    for source in (model_doc, _catalog() / CATALOG_README_NAME):
        shutil.copy(source, destination_dir)
    return destination_dir / model_doc.name

def main() -> int:
    arguments = build_parser().parse_args()
    try:
        models = _available_model_names() if arguments.all else [arguments.model]
        for model in models:
            model_doc = add_supplementary_fm(model, arguments.dataset_dir)
            console.print(f"Added {model_doc}", style="green")
    except (ValueError, OSError) as error:
        console.print(f"{error}", style="red")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
