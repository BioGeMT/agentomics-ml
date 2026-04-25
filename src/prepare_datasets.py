import argparse
import sys
from pathlib import Path
from rich.console import Console

from datasets.dataset_utils import prepare_dataset, check_dataset_prepared
from datasets.datasets_interactive import prepare_all_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preparation")
    parser.add_argument('--dataset-dir', type=Path, help='Single dataset directory to prepare')
    parser.add_argument('--prepare-all', action='store_true', help='Prepare all datasets in datasets-dir')
    parser.add_argument('--target-col', type=str, default=None, help='Target column name (auto-detected if not provided)')
    parser.add_argument('--task-type', choices=['classification', 'regression'], default=None, help='Task type (prompted if not provided)')
    parser.add_argument('--positive-class', help='Value used in the label column for a positive class (affects some binary classification metrics). If not provided, numeric labels are assigned based on the label appearance order in the train csv file.', default=None)
    parser.add_argument('--negative-class', help='Value used in the label column for a negative class (affects some binary classification metrics). If not provided, numeric labels are assigned based on the label appearance order in the train csv file.', default=None)
    parser.add_argument('--datasets-dir', default='./datasets', help='Directory containing raw datasets')
    parser.add_argument('--prepared-datasets-dir', default='./prepared_datasets', help='Output directory for prepared datasets')
    parser.add_argument('--prepared-test-sets-dir', default='./prepared_test_sets', help='Output directory for prepared test sets')
    return parser.parse_args()

def main():
    args = parse_args()
    console = Console()

    datasets_dir = args.datasets_dir
    dataset_dir = args.dataset_dir
    prepared_datasets_dir = args.prepared_datasets_dir
    prepared_test_sets_dir = args.prepared_test_sets_dir
    
    Path(prepared_datasets_dir).mkdir(parents=True, exist_ok=True)
    Path(prepared_test_sets_dir).mkdir(parents=True, exist_ok=True)

    if args.prepare_all or not dataset_dir:
        prepare_all_datasets(datasets_dir, prepared_datasets_dir, prepared_test_sets_dir)
    elif check_dataset_prepared(str(dataset_dir), str(prepared_datasets_dir)):
        console.print(f'[blue]Dataset "{dataset_dir.name}" already prepared, skipping preparation[/blue]')
    else:
        console.print(f'[blue]Preparing dataset "{dataset_dir.name}"[/blue]')# for {task_type} task with target column "{target_col}"[/blue]')
        try:
            prepare_dataset(
                dataset_dir=dataset_dir,
                target_col=args.target_col,
                positive_class=args.positive_class,
                negative_class=args.negative_class,
                task_type=args.task_type,
                output_dir=prepared_datasets_dir,
                test_sets_output_dir=prepared_test_sets_dir
            )
            console.print(f"[green]Dataset '{dataset_dir.name}' prepared successfully![/green]")
        except Exception as e:
            console.print(f"[red]Dataset '{dataset_dir.name}' preparation failed! {e}[/red]")
            sys.exit(1)
        
if __name__ == "__main__":
    main()