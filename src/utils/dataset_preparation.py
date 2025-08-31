import argparse
import os
import sys
import csv
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from dataset_utils import prepare_dataset

console = Console()

def count_csv_rows(csv_file: str) -> int:
    """
    Count rows in a CSV file (excluding header).
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        Number of data rows (excluding header)
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            row_count = sum(1 for _ in reader)
            return max(0, row_count - 1)  # Subtract 1 for header
    except (FileNotFoundError, IOError, UnicodeDecodeError):
        return 0

def check_dataset_prepared(dataset_dir: str, prepared_datasets_dir: str) -> bool:
    """Check if a dataset is already prepared."""
    dataset_name = Path(dataset_dir).name
    prepared_path = Path(prepared_datasets_dir) / dataset_name
    metadata_file = prepared_path / "metadata.json"
    train_file = prepared_path / "train.csv"
    return metadata_file.exists() and train_file.exists()

def get_single_dataset_info(dataset_dir: str, prepared_datasets_dir: str) -> Dict:
    if not dataset_dir.is_dir():
        return None
        
    dataset_name = dataset_dir.name
    train_file = dataset_dir / "train.csv"
    test_file = dataset_dir / "test.csv"
    
    # Count rows in raw files
    train_rows = count_csv_rows(str(train_file)) if train_file.exists() else 0
    test_rows = count_csv_rows(str(test_file)) if test_file.exists() else 0
    
    # Check if already prepared
    is_prepared = check_dataset_prepared(str(dataset_dir), prepared_datasets_dir)
    
    # Check if can be prepared
    can_prepare = train_file.exists() and train_rows > 0
    
    if not train_file.exists():
        status = "Missing train.csv"
    elif train_rows == 0:
        status = "Empty train.csv"
    elif is_prepared:
        status = "Already prepared"
    elif can_prepare:
        status = "Ready to prepare"
    else:
        status = "Cannot prepare"
        
    return {
        "name": dataset_name,
        "path": dataset_dir,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "status": status,
        "can_prepare": can_prepare,
        "should_prepare": can_prepare and not is_prepared,
        "is_prepared": is_prepared
    }

def get_all_datasets_info(datasets_dir: str, prepared_datasets_dir: str) -> List[Dict]:
    """
    Collect information about all datasets for preparation.
    
    Args:
        datasets_dir: Path to raw datasets directory
        prepared_datasets_dir: Path to prepared datasets directory
        
    Returns:
        List of dataset information dictionaries
    """
    datasets_path = Path(datasets_dir)
    
    if not datasets_path.exists():
        return []
    
    datasets_info = []
    
    for dataset_dir in datasets_path.iterdir():
        dataset_info = get_single_dataset_info(dataset_dir, prepared_datasets_dir)
        if(dataset_info):
            datasets_info.append(dataset_info)
        
    # Sort by name for consistent ordering
    datasets_info.sort(key=lambda x: x["name"])
    return datasets_info

def display_preparation_table(datasets: List[Dict], title: str = "Dataset Preparation Status") -> None:
    """
    Display datasets in a Rich table for preparation.
    
    Args:
        datasets: List of dataset dictionaries
        title: Table title
    """
    if not datasets:
        console.print("[red]No datasets found[/red]")
        return
    
    # Create table with preparation-focused styling
    table = Table(title=title, show_header=True, header_style="bold blue", expand=False)
    
    table.add_column("#", style="dim", width=4)
    table.add_column("Dataset Name", style="cyan", no_wrap=True, width=25)
    table.add_column("Train Rows", justify="right", style="green", width=12)
    table.add_column("Test Rows", justify="right", style="green", width=12)
    table.add_column("Status", style="magenta", width=25)
    
    for i, dataset in enumerate(datasets, 1):
        # Format row counts with colors
        if dataset['train_rows'] > 0:
            train_display = f"[green]{dataset['train_rows']:,}[/green]"
        else:
            train_display = "[dim]N/A[/dim]"
            
        if dataset['test_rows'] > 0:
            test_display = f"[blue]{dataset['test_rows']:,}[/blue]"
        else:
            test_display = "[dim]N/A[/dim]"
        
        # Color-code status with better styling
        status = dataset["status"]
        if status == "Already prepared":
            status_display = f"[bold green]✓ {status}[/bold green]"
            style_override = {"style": "dim"}
        elif status == "Prepared":
            status_display = f"[bold green]✓ {status}[/bold green]"
            style_override = {"style": "bold"}
        elif status == "Ready to prepare":
            status_display = f"[bold yellow]⏳ {status}[/bold yellow]"
            style_override = {"style": "bold"}
        elif status == "Failed":
            status_display = f"[bold red]✗ {status}[/bold red]"
            style_override = {}
        elif "Missing" in status or "Empty" in status:
            status_display = f"[bold red]⚠ {status}[/bold red]"
            style_override = {}
        else:
            status_display = f"[red]{status}[/red]"
            style_override = {}
        
        row = [
            f"[dim]{i}[/dim]",
            f"[bold cyan]{dataset['name']}[/bold cyan]",
            train_display,
            test_display,
            status_display
        ]
        
        table.add_row(*row, **style_override)
    
    console.print(table)

def prepare_all_datasets(datasets_dir: str, prepared_datasets_dir: str) -> None:
    """
    Prepare multiple datasets with Rich progress display.
    
    Args:
        datasets_dir: Path to raw datasets directory
        prepared_datasets_dir: Path to prepared datasets directory
    """
    console.print("[bold blue]Agentomics-ML Dataset Preparation[/bold blue]")
    console.print("=" * 50)
    console.print("")
    
    # Get initial dataset information
    console.print(f"[dim]Scanning datasets in {datasets_dir}...[/dim]")
    datasets_info = get_all_datasets_info(datasets_dir, prepared_datasets_dir)
    
    if not datasets_info:
        console.print(f"[red]No datasets found in {datasets_dir}[/red]")
        console.print("")
        console.print("[blue]To add datasets:[/blue]")
        console.print(f"   1. Create directories in {datasets_dir}/")
        console.print("   2. Add train.csv (and optionally test.csv) to each directory")
        console.print("   3. Run preparation again")
        sys.exit(1)
    
    console.print("")
    
    # Show initial status
    #TODO make sure table shows classification/regression that was detected
    display_preparation_table(datasets_info, "Datasets Found - Preparation Status")
    console.print("")
    
    # Count datasets needing preparation
    need_preparation = [d for d in datasets_info if d["should_prepare"]]
    already_prepared = [d for d in datasets_info if d["is_prepared"]]
    cannot_prepare = [d for d in datasets_info if not d["can_prepare"] and not d["is_prepared"]]
    
    if not need_preparation:
        if already_prepared:
            console.print(f"[green]All {len(already_prepared)} dataset(s) are already prepared![/green]")
        else:
            console.print("[yellow]No datasets can be prepared. Check for missing train.csv files.[/yellow]")
        
        console.print("")
        console.print("[blue]Summary:[/blue]")
        console.print(f"  Total datasets: {len(datasets_info)}")
        console.print(f"  Already prepared: [green]{len(already_prepared)}[/green]")
        console.print(f"  Cannot prepare: [red]{len(cannot_prepare)}[/red]")
        
        if already_prepared:
            console.print("")
            console.print("[green]Ready to use! Run: ./run.sh[/green]")
        
        sys.exit(0)
    
    # Prepare datasets with progress display
    console.print(f"[yellow]Preparing {len(need_preparation)} dataset(s)...[/yellow]")
    console.print("")
    
    prepared_now = 0
    failed_now = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for dataset_info in need_preparation:
            task = progress.add_task(f"Preparing {dataset_info['name']}...", total=None)
            try:
                console.print(f"[yellow]Preparing dataset '{dataset_info['name']}'[/yellow]")
                prepare_dataset(
                    dataset_dir=dataset_info['path'],
                    target_col=None, #auto-detected inside
                    positive_class=None, #auto-detected inside
                    negative_class=None, #auto-detected inside
                    task_type=None, #auto-detected inside
                    output_dir=prepared_datasets_dir,
                )
                success = True
            except Exception as e:
                console.print(f"[red]Error preparing dataset '{dataset_info['name']}': {e}[/red]")
                success = False
            
            if success:
                progress.update(task, description=f"{dataset_info['name']} prepared successfully")
                prepared_now += 1
                dataset_info["status"] = "Prepared"
                dataset_info["is_prepared"] = True
                dataset_info["can_prepare"] = False
                dataset_info["should_prepare"] = False
            else:
                progress.update(task, description=f"{dataset_info['name']} preparation failed")
                failed_now += 1
                dataset_info["status"] = "Failed"
    
    console.print("")
    
    # Show final status
    if prepared_now > 0:
        display_preparation_table(datasets_info, "Final Preparation Results")
        console.print("")
    
    # Summary
    console.print("[blue]Preparation Summary:[/blue]")
    console.print(f"  Total datasets: {len(datasets_info)}")
    console.print(f"  Already prepared: [green]{len(already_prepared)}[/green]")
    console.print(f"  Prepared now: [green]{prepared_now}[/green]")
    console.print(f"  Failed: [red]{failed_now}[/red]")
    console.print(f"  Ready to use: [green]{len(already_prepared) + prepared_now}[/green]")
    
    console.print("")
    
    if prepared_now > 0:
        console.print(f"[green]Successfully prepared {prepared_now} dataset(s)![/green]")
    
    if len(already_prepared) + prepared_now > 0:
        console.print("")
        console.print("[green]Next steps:[/green]")
        console.print("   1. Run the agent: [cyan]./run.sh[/cyan]")
        console.print("   2. Select your prepared dataset")
        console.print("   3. Choose your AI model")
        console.print("   4. Let the agent work its magic!")
    
    if failed_now > 0:
        console.print("")
        console.print(f"[yellow]{failed_now} dataset(s) failed to prepare. Check your data files.[/yellow]")

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