import json
from pathlib import Path

import pandas as pd

from agentomics.datasets.dataset_preparation import prepare_dataset


def test_continued_run_can_reprepare_csv_from_saved_metadata(tmp_path: Path):
    source_dir = tmp_path / "datasets" / "toy"
    source_dir.mkdir(parents=True)
    (source_dir / "metadata.json").write_text(
        json.dumps({"id_column": "sample_id"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "sample_id": ["row-1", "row-2"],
            "feature": [1, 2],
            "outcome": ["no", "yes"],
        }
    ).to_csv(source_dir / "train.csv", index=False)

    first_metadata = prepare_dataset(
        source_dir=source_dir,
        destination_dir=tmp_path / "first_prepared",
        task_type="classification",
        label_to_scalar={"no": 0, "yes": 1},
        label_column="outcome",
        interactive=False,
    )
    second_prepared_dir = tmp_path / "second_prepared"
    # Continuing a run passes its saved dataset metadata back into preparation.
    second_metadata = prepare_dataset(
        source_dir=source_dir,
        destination_dir=second_prepared_dir,
        task_type=first_metadata["task_type"],
        label_to_scalar=first_metadata["label_to_scalar"],
        label_column=first_metadata["label_column"],
        interactive=False,
    )

    assert second_metadata["label_column"] == "outcome"
    assert second_metadata["id_column"] == "sample_id"
    assert second_metadata["label_to_scalar"] == {"no": 0, "yes": 1}
    replayed_labels = pd.read_csv(
        second_prepared_dir / "train" / "labels.csv",
        dtype={"id": str},
    )
    assert replayed_labels["id"].tolist() == ["row-1", "row-2"]
