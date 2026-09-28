from pathlib import Path

TEST_SEQUENCE_PREFIX = "TEST_SEQUENCE_ONLY_"


def write_raw_classification_split(
    dataset_dir: Path,
    split_name: str,
    *,
    input_files: dict[str, str],
    labels: tuple[str, ...],
) -> Path:
    """Create one raw folder-based classification split."""
    split_dir = dataset_dir / split_name
    input_dir = split_dir / "input"
    input_dir.mkdir(parents=True)
    for relative_path, content in input_files.items():
        input_path = input_dir / relative_path
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(content, encoding="utf-8")
    label_rows = "".join(
        f"{split_name}-{index},{label}\n" for index, label in enumerate(labels)
    )
    (split_dir / "labels.csv").write_text(
        f"id,label\n{label_rows}",
        encoding="utf-8",
    )
    return split_dir


def create_classification_dataset(
    test_root: Path,
    *,
    include_validation_split: bool = False,
    include_test_split: bool = False,
) -> Path:
    """Create a representative folder-based classification dataset."""
    dataset_dir = test_root / "datasets" / "promoter_sequences"
    split_names = ["train"]
    if include_validation_split:
        split_names.append("validation")
    if include_test_split:
        split_names.append("test")
    for split_name in split_names:
        sequences = (
            (f"A_{TEST_SEQUENCE_PREFIX}0", f"T_{TEST_SEQUENCE_PREFIX}1")
            if split_name == "test"
            else ("ACGTACGT", "TGCATGCA")
        )
        write_raw_classification_split(
            dataset_dir,
            split_name,
            input_files={
                "data.csv": (
                    f"id,sequence\n{split_name}-0,{sequences[0]}\n"
                    f"{split_name}-1,{sequences[1]}\n"
                )
            },
            labels=("positive", "negative"),
        )
    (dataset_dir / "supplementary").mkdir()
    (dataset_dir / "supplementary" / "paper.txt").write_text(
        "Supporting material.\n",
        encoding="utf-8",
    )
    (dataset_dir / "metadata.json").write_text(
        '{"task_type": "classification"}',
        encoding="utf-8",
    )
    (dataset_dir / "dataset_description.md").write_text(
        "Synthetic promoter-sequence classification dataset.\n",
        encoding="utf-8",
    )
    return dataset_dir
