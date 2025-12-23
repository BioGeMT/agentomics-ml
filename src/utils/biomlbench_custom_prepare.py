# This script has some duplicate code with run_agent_biomlbench.py and is used for stealth-test eval only
import shutil
import pandas as pd
from pathlib import Path
from utils.biomlbench_target_utils import get_target_col_from_description
from utils.dataset_utils import prepare_dataset
from run_agent_biomlbench import extract_val_metric_from_description, extract_task_type_from_val_metric

def prepare_biomlbench_dataset(
        agentomics_dir,
        dataset_name,
    ):
    raw_datasets_dir = f'{agentomics_dir}/datasets'
    prepared_datasets_dir = f'{agentomics_dir}/prepared_datasets'
    prepared_test_sets_dir = f'{agentomics_dir}/prepared_test_sets'

    biomlbench_dset_path = (Path("~/.cache/bioml-bench/data/") / dataset_name).expanduser()
    is_proteingym = 'proteingym' in dataset_name.lower()
    description_path = biomlbench_dset_path / "prepared/public/description.md"
    val_metric = extract_val_metric_from_description(description_path, is_proteingym=is_proteingym)
    task_type = extract_task_type_from_val_metric(val_metric)

    Path(f"{raw_datasets_dir}/{dataset_name}").mkdir(parents=True, exist_ok=True)
    
    if is_proteingym:
        train_data_path = biomlbench_dset_path / "prepared/public/data.csv"
    else:
        train_data_path = biomlbench_dset_path / "prepared/public/train.csv"

    if(is_proteingym):
        # Only copy the main heading and ## Description section
        with open(description_path, 'r') as f:
            lines = f.readlines()

        output_lines = []
        in_description = False

        for i, line in enumerate(lines):
            # Always include the main heading (first line starting with #)
            if i == 0 or (not output_lines and line.startswith('#') and not line.startswith('##')):
                output_lines.append(line)
            # Start capturing when we find ## Description
            elif line.strip() == '## Description':
                in_description = True
                output_lines.append('\n')
                output_lines.append(line)
            # Stop capturing when we encounter the next ## section
            elif in_description and line.startswith('##'):
                break
            # Capture lines within the Description section
            elif in_description:
                output_lines.append(line)

        # Write the filtered content
        with open(f'{raw_datasets_dir}/{dataset_name}/dataset_description.md', 'w') as f:
            f.writelines(output_lines)
    else:
        shutil.copy(description_path, f'{raw_datasets_dir}/{dataset_name}/dataset_description.md')

    if is_proteingym:
        # Copy the csv, but leave out the following columns: fold_random_5,fold_modulo_5,fold_contiguous_5
        df = pd.read_csv(train_data_path)
        df = df[df['fold_random_5'] != -1]
        columns_to_drop = ['fold_random_5', 'fold_modulo_5', 'fold_contiguous_5']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        df.to_csv(f'{raw_datasets_dir}/{dataset_name}/train.csv', index=False)
    else:
        shutil.copy(train_data_path, f'{raw_datasets_dir}/{dataset_name}/train.csv')

    target_col = get_target_col_from_description(description_path)

    test_labels_path = biomlbench_dset_path / "prepared/private/answers.csv"
    test_features_path = biomlbench_dset_path / "prepared/public/test_features.csv"

    if test_labels_path.exists() and test_features_path.exists():
        # Join the two csvs on id and output as test csv
        test_features = pd.read_csv(test_features_path)
        test_labels = pd.read_csv(test_labels_path)
        test_data = test_features.merge(test_labels, on='id', how='inner')
        assert len(test_data) == len(test_labels)
        test_data.to_csv(f'{raw_datasets_dir}/{dataset_name}/test.csv', index=False)
    else:
        print(f'WARNING: Biomlbench dataset {dataset_name} has no private test set answers.csv')

    prepare_dataset(
        dataset_dir=f'{raw_datasets_dir}/{dataset_name}',
        target_col=target_col,
        positive_class=None,
        negative_class=None,
        task_type=task_type,
        output_dir=f'{prepared_datasets_dir}',
        test_sets_output_dir=f'{prepared_test_sets_dir}',
    )

    # Reorganize output to preserve group/dset_name structure
    dset_name_only = dataset_name.split('/')[-1]
    prepared_src = f'{prepared_datasets_dir}/{dset_name_only}'
    prepared_dst = f'{prepared_datasets_dir}/{dataset_name}'
    test_src = f'{prepared_test_sets_dir}/{dset_name_only}'
    test_dst = f'{prepared_test_sets_dir}/{dataset_name}'

    if Path(prepared_src).exists():
        Path(prepared_dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(prepared_src, prepared_dst)

    if Path(test_src).exists():
        Path(test_dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(test_src, test_dst)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--agentomics-dir', required=True)
    parser.add_argument('--dataset-name', required=True)
    args = parser.parse_args()

    prepare_biomlbench_dataset(
        agentomics_dir=args.agentomics_dir,
        dataset_name=args.dataset_name,
    )