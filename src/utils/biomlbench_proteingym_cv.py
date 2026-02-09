import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd


def _run_command(command):
    result = subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"Command: {command}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def _assert_zero_based_contiguous(unique_folds, fold_col):
    normalized = sorted(int(v) for v in unique_folds)
    expected = list(range(len(normalized)))
    assert normalized == expected, (
        f"Fold values in column '{fold_col}' must be contiguous and zero-based. "
        f"Found: {normalized}, Expected: {expected}"
    )
    return normalized


def generate_proteingym_submission_with_cv(
    env_path,
    train_script_path,
    inference_script_path,
    data_csv_path,
    submission_path,
    submission_extended_path,
    source_label_col="fitness_score",
    script_label_col="numeric_label",
):
    """
    Generate ProteinGym submission by retraining and inferring across folds.

    Expects train/inference scripts with Agentomics-compatible interfaces:
    - train.py --train-data --validation-data --artifacts-dir
    - inference.py --input --output --artifacts-dir
    """
    env_path = Path(env_path)
    train_script_path = Path(train_script_path)
    inference_script_path = Path(inference_script_path)
    data_csv_path = Path(data_csv_path)
    submission_path = Path(submission_path)
    submission_extended_path = Path(submission_extended_path)

    assert env_path.exists(), f"Conda environment not found: {env_path}"
    assert train_script_path.is_file(), f"Train script not found: {train_script_path}"
    assert inference_script_path.is_file(), f"Inference script not found: {inference_script_path}"
    assert data_csv_path.is_file(), f"ProteinGym data file not found: {data_csv_path}"

    og_data = pd.read_csv(data_csv_path)
    if "fold_random_5" in og_data.columns:
        og_data = og_data[og_data["fold_random_5"] != -1].copy()
    assert len(og_data) > 0, "No rows available for ProteinGym CV generation"

    required_cols = ["id", "sequence", source_label_col]
    for col in required_cols:
        assert col in og_data.columns, f"Missing required column in ProteinGym data: {col}"

    fold_cols = [col for col in og_data.columns if col.startswith("fold_")]
    assert len(fold_cols) > 0, "No fold_* columns found in ProteinGym data"

    start_time = time.time()
    with tempfile.TemporaryDirectory(prefix="proteingym_cv_", dir=str(submission_path.parent)) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        fold_to_predictions = []

        for fold_col in fold_cols:
            unique_folds = _assert_zero_based_contiguous(og_data[fold_col].dropna().unique().tolist(), fold_col)
            fold_predictions = []
            num_folds = len(unique_folds)

            for test_fold in unique_folds:
                validation_fold = (test_fold + 1) % num_folds
                train_df = og_data[
                    (og_data[fold_col] != test_fold) & (og_data[fold_col] != validation_fold)
                ][required_cols].copy()
                valid_df = og_data[og_data[fold_col] == validation_fold][required_cols].copy()
                test_df = og_data[og_data[fold_col] == test_fold][required_cols].copy()

                assert len(train_df) > 0, f"Empty train split for {fold_col}={test_fold}"
                assert len(valid_df) > 0, f"Empty validation split for {fold_col}={test_fold}"
                assert len(test_df) > 0, f"Empty test split for {fold_col}={test_fold}"

                if script_label_col != source_label_col:
                    train_df = train_df.rename(columns={source_label_col: script_label_col}, errors="raise")
                    valid_df = valid_df.rename(columns={source_label_col: script_label_col}, errors="raise")
                    test_df = test_df.rename(columns={source_label_col: script_label_col}, errors="raise")

                split_name = f"{fold_col}_{test_fold}"
                train_csv = tmp_dir / f"{split_name}_train.csv"
                valid_csv = tmp_dir / f"{split_name}_valid.csv"
                test_csv = tmp_dir / f"{split_name}_test.csv"
                predictions_csv = tmp_dir / f"{split_name}_predictions.csv"
                artifacts_dir = tmp_dir / f"{split_name}_artifacts"
                artifacts_dir.mkdir(parents=True, exist_ok=True)

                train_df.to_csv(train_csv, index=False)
                valid_df.to_csv(valid_csv, index=False)
                test_df.to_csv(test_csv, index=False)

                train_cmd = (
                    f'cd "{train_script_path.parent}" && '
                    f'conda run -p "{env_path}" --no-capture-output '
                    f'python "{train_script_path}" '
                    f'--train-data "{train_csv}" '
                    f'--validation-data "{valid_csv}" '
                    f'--artifacts-dir "{artifacts_dir}"'
                )
                _run_command(train_cmd)

                inference_cmd = (
                    f'cd "{inference_script_path.parent}" && '
                    f'conda run -p "{env_path}" --no-capture-output '
                    f'python "{inference_script_path}" '
                    f'--input "{test_csv}" '
                    f'--output "{predictions_csv}" '
                    f'--artifacts-dir "{artifacts_dir}"'
                )
                _run_command(inference_cmd)

                preds_df = pd.read_csv(predictions_csv)
                assert len(preds_df) == len(test_df), (
                    f"Prediction row mismatch for {split_name}: "
                    f"{len(preds_df)} predictions vs {len(test_df)} test rows"
                )
                assert "id" in preds_df.columns, f"Predictions for {split_name} must contain 'id' column"
                fold_predictions.append(preds_df)

            assert len(fold_predictions) > 0, f"No predictions produced for fold column: {fold_col}"
            fold_pred_df = pd.concat(fold_predictions, ignore_index=True)
            fold_pred_col = "fitness_score" if len(fold_cols) == 1 else f"fitness_score_{fold_col}"

            if fold_pred_col not in fold_pred_df.columns and "numeric_label" in fold_pred_df.columns:
                fold_pred_df = fold_pred_df.rename(columns={"numeric_label": fold_pred_col})
            if fold_pred_col not in fold_pred_df.columns and "prediction" in fold_pred_df.columns:
                fold_pred_df = fold_pred_df.rename(columns={"prediction": fold_pred_col})

            assert fold_pred_col in fold_pred_df.columns, (
                f"Predictions must include '{fold_pred_col}' column after processing. "
                f"Available columns: {fold_pred_df.columns.tolist()}"
            )
            fold_to_predictions.append(fold_pred_df[["id", fold_pred_col]].copy())

        final_df = fold_to_predictions[0]
        for df in fold_to_predictions[1:]:
            final_df = final_df.merge(df, on="id", how="inner")

        assert len(final_df) == len(og_data), (
            f"Final merged predictions count mismatch: {len(final_df)} vs {len(og_data)}"
        )

        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission_extended_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(submission_path, index=False)
        final_df.to_csv(submission_extended_path, index=False)

    return {
        "rows": len(final_df),
        "columns": final_df.columns.tolist(),
        "seconds": time.time() - start_time,
    }
