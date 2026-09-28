import os
from pathlib import Path

import pytest
from pydantic_ai import ModelRetry

from agentomics.agents.steps.data_split import DataSplitStep
from agentomics.runtime.filesystem import (
    rewrite_symlinks_to_absolute,
    validate_symlinks_targets_in,
)


def test_relative_symlink_is_rewritten_to_absolute_target(tmp_path: Path):
    target_file = tmp_path / "data" / "file.txt"
    target_file.parent.mkdir()
    target_file.write_text("content", encoding="utf-8")
    link_dir = tmp_path / "links"
    link_dir.mkdir()
    link_path = link_dir / "link.txt"
    os.symlink("../data/file.txt", link_path)
    assert not link_path.readlink().is_absolute()

    rewrite_symlinks_to_absolute(link_dir)

    assert link_path.readlink().is_absolute()
    assert link_path.resolve() == target_file.resolve()


def test_symlink_validation_rejects_target_outside_allowed_directory(tmp_path: Path):
    allowed = tmp_path / "dataset"
    allowed.mkdir()
    (allowed / "ok.txt").write_text("ok", encoding="utf-8")
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "bad.txt").write_text("bad", encoding="utf-8")
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    os.symlink(allowed / "ok.txt", split_dir / "good_link")
    os.symlink(outside / "bad.txt", split_dir / "bad_link")

    with pytest.raises(ValueError):
        validate_symlinks_targets_in(split_dir, allowed)


def test_symlink_validation_accepts_targets_in_allowed_directory(tmp_path: Path):
    allowed = tmp_path / "dataset"
    allowed.mkdir()
    (allowed / "a.txt").write_text("a", encoding="utf-8")
    (allowed / "b.txt").write_text("b", encoding="utf-8")
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    os.symlink(allowed / "a.txt", split_dir / "link_a")
    os.symlink(allowed / "b.txt", split_dir / "link_b")

    validate_symlinks_targets_in(split_dir, allowed)


def test_symlink_validation_accepts_target_through_prepared_dataset_symlink(
    tmp_path: Path,
):
    raw_data = tmp_path / "raw" / "input"
    raw_data.mkdir(parents=True)
    (raw_data / "file.txt").write_text("content", encoding="utf-8")
    prepared = tmp_path / "prepared" / "dataset"
    prepared.mkdir(parents=True)
    os.symlink(raw_data, prepared / "input")
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    link_path = split_dir / "link.txt"
    os.symlink(prepared / "input" / "file.txt", link_path)
    assert not link_path.resolve().is_relative_to(prepared.resolve())

    validate_symlinks_targets_in(split_dir, prepared)


def test_generated_split_rejects_unchanged_copy_of_source_file(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    source_file = dataset_dir / "train" / "input" / "file.txt"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("hello", encoding="utf-8")
    split = tmp_path / "split"
    copied_file = split / "input" / "file.txt"
    copied_file.parent.mkdir(parents=True)
    copied_file.write_text("hello", encoding="utf-8")

    with pytest.raises(ModelRetry):
        DataSplitStep.validate_generated_split(split, dataset_dir)


def test_generated_split_accepts_modified_source_file(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    source_file = dataset_dir / "train" / "input" / "file.txt"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("hello", encoding="utf-8")
    split = tmp_path / "split"
    modified_file = split / "input" / "file.txt"
    modified_file.parent.mkdir(parents=True)
    modified_file.write_text("hello transformed", encoding="utf-8")

    DataSplitStep.validate_generated_split(split, dataset_dir)


def test_generated_split_accepts_symlink_to_source_file(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    source_file = dataset_dir / "train" / "input" / "file.txt"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("hello", encoding="utf-8")
    split = tmp_path / "split"
    linked_file = split / "input" / "file.txt"
    linked_file.parent.mkdir(parents=True)
    os.symlink(source_file, linked_file)

    DataSplitStep.validate_generated_split(split, dataset_dir)
