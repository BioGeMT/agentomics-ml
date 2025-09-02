import argparse
import os
from pathlib import Path
from rich.console import Console
from dataset_utils import prepare_dataset
from datasets_interactive_utils import prepare_all_datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preparation with auto-detection")
    parser.add_argument('--dataset-dir', type=Path, help='Single dataset directory to prepare')
    parser.add_argument('--prepare-all', action='store_true', help='Prepare all datasets in datasets-dir and auto-detect their targets and tasks')
    parser.add_argument('--target-col', type=str, default=None, help='Target column name (auto-detected if not provided)')
    parser.add_argument('--task-type', choices=['classification', 'regression'], default=None, help='Task type (auto-detected if not provided)')
    parser.add_argument('--positive-class', help='Value used in the label column for a positive class (affects some binary classification metrics). If not provided, numeric labels are assigned based on the label appearance order in the train csv file.', default=None)
    parser.add_argument('--negative-class', help='Value used in the label column for a negative class (affects some binary classification metrics). If not provided, numeric labels are assigned based on the label appearance order in the train csv file.', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    console = Console()
    
    # If environment variables are set, override defaults
    datasets_dir = os.environ.get("DATASETS_DIR", default="./datasets")
    prepared_datasets_dir = os.environ.get("PREPARED_DATASETS_DIR", default="./prepared_datasets")
    
    Path(prepared_datasets_dir).mkdir(parents=True, exist_ok=True)

    if args.prepare_all or not args.dataset_dir:
        prepare_all_datasets(datasets_dir, prepared_datasets_dir)
    else:
        console.print(f'[blue]Preparing dataset "{args.dataset_dir.name}"')# for {task_type} task with target column "{target_col}"[/blue]')
        try:
            prepare_dataset(
                dataset_dir=args.dataset_dir,
                target_col=args.target_col,
                positive_class=args.positive_class, #is auto-detected inside - do the same for target/task ?
                negative_class=args.negative_class,
                task_type=args.task_type,
                output_dir=prepared_datasets_dir,
            )
            console.print(f"[green]Dataset '{args.dataset_dir.name}' prepared successfully![/green]")
        except Exception as e:
            console.print(f"[red]Dataset '{args.dataset_dir.name}' preparation failed! {e}[/red]")
        
if __name__ == "__main__":
    main()