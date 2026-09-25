"""Agent-side executable that checks dataset visibility and permissions."""

import argparse
import json
import os
from pathlib import Path


def walk_tree(root):
    """Walk Agent-visible paths while safely following dataset symlinks."""
    visited_directories = set()
    for directory, directory_names, filenames in os.walk(root, followlinks=True):
        directory_path = Path(directory)
        try:
            resolved_directory = directory_path.resolve()
        except OSError:
            directory_names[:] = []
            continue
        if resolved_directory in visited_directories:
            directory_names[:] = []
            continue
        visited_directories.add(resolved_directory)
        yield directory_path, directory_names, filenames


def read_public_files(dataset_source):
    """Read every file exposed through the public source dataset mount."""
    return {
        str(path.relative_to(dataset_source)): path.read_text(encoding="utf-8")
        for directory, _, filenames in walk_tree(dataset_source)
        for path in (directory / filename for filename in filenames)
    }


def find_held_out_data(prefix):
    """Find held-out paths or identifiable sequences visible during development."""
    test_paths = set()
    sequence_matches = set()
    for root in (Path("/datasets"), Path("/workspace")):
        for directory, directory_names, filenames in walk_tree(root):
            test_paths.update(
                str(directory / name)
                for name in directory_names
                if name.startswith("test")
            )
            for path in (directory / filename for filename in filenames):
                try:
                    content = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    continue
                if prefix in content:
                    sequence_matches.add(str(path))
    return sorted(test_paths), sorted(sequence_matches)


def run_dataset_access_check(dataset_source, evidence_path, test_sequence_prefix):
    """Exercise dataset access from the Agent Execution Environment."""
    public_files = read_public_files(dataset_source)
    write_target = dataset_source / "supplementary" / "paper.txt"
    try:
        write_target.write_text(
            "Agent-authored write reached source data.\n", encoding="utf-8"
        )
    except OSError:
        public_source_write_succeeded = False
    else:
        public_source_write_succeeded = True

    test_paths, test_sequence_matches = find_held_out_data(test_sequence_prefix)
    evidence = {
        "public_files": public_files,
        "public_source_write_succeeded": public_source_write_succeeded,
        "write_target_content_after_attempt": write_target.read_text(encoding="utf-8"),
        "test_paths": test_paths,
        "test_sequence_matches": test_sequence_matches,
    }
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")

    if public_source_write_succeeded:
        raise RuntimeError("Public source dataset was writable")
    if test_paths or test_sequence_matches:
        raise RuntimeError("Held-out test data was available")
    print("Dataset access check completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--test-sequence-prefix", required=True)
    args = parser.parse_args()
    run_dataset_access_check(
        args.dataset_source,
        args.evidence,
        args.test_sequence_prefix,
    )


if __name__ == "__main__":
    main()
