import argparse
import os
import sys
import csv
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from .dataset_utils import prepare_dataset

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

def auto_detect_task_type_and_target(dataset_dir: Path, quiet: bool = False) -> tuple:
    """Auto-detect target column and task type from the dataset."""
    train_path = dataset_dir / 'train.csv'
    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found in {dataset_dir}")
    
    df = pd.read_csv(train_path)
    
    # Auto-detect target column
    possible_target_cols = ['class', 'target', 'label', 'y']
    target_col = None
    for col in possible_target_cols:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Use the last column as target
        target_col = df.columns[-1]
        if not quiet:
            console.print(f'[dim]INFO: Using last column as target: {target_col}[/dim]')
    else:
        if not quiet:
            console.print(f'[dim]INFO: Auto-detected target column: {target_col}[/dim]')
    
    # Auto-detect task type based on target column
    target_values = df[target_col].dropna()
    unique_values = target_values.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_values)
    
    if is_numeric and unique_values > 10:
        task_type = 'regression'
        if not quiet:
            console.print(f'[dim]INFO: Auto-detected regression task (numeric target with {unique_values} unique values)[/dim]')
    else:
        task_type = 'classification' 
        if not quiet:
            console.print(f'[dim]INFO: Auto-detected classification task ({unique_values} unique values)[/dim]')
    
    return target_col, task_type



def check_dataset_prepared(dataset_dir: str, prepared_datasets_dir: str) -> bool:
    """Check if a dataset is already prepared."""
    dataset_name = Path(dataset_dir).name
    prepared_path = Path(prepared_datasets_dir) / dataset_name
    metadata_file = prepared_path / "metadata.json"
    train_file = prepared_path / "train.csv"
    return metadata_file.exists() and train_file.exists()

def get_dataset_info(datasets_dir: str, prepared_datasets_dir: str) -> List[Dict]:
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
    
    dataset_info = []
    
    for dataset_dir in datasets_path.iterdir():
        if not dataset_dir.is_dir():
            continue
            
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
            
        dataset_info.append({
            "name": dataset_name,
            "path": dataset_dir,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "status": status,
            "can_prepare": can_prepare and not is_prepared,
            "is_prepared": is_prepared
        })
    
    # Sort by name for consistent ordering
    dataset_info.sort(key=lambda x: x["name"])
    return dataset_info

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

def prepare_single_dataset(dataset_info: Dict, prepared_datasets_dir: str, 
                          target_col: Optional[str] = None, task_type: Optional[str] = None,
                          positive_class: Optional[str] = None, negative_class: Optional[str] = None) -> bool:
    """
    Prepare a single dataset using the dataset_utils.prepare_dataset function directly.
    
    Args:
        dataset_info: Dataset information dictionary
        prepared_datasets_dir: Path to prepared datasets directory
        target_col: Optional target column override
        task_type: Optional task type override
        positive_class: Optional positive class override
        negative_class: Optional negative class override
        
    Returns:
        True if preparation was successful
    """
    try:
        dataset_path = dataset_info["path"]
        
        # Auto-detect target column and task type (fully automatic in batch mode)
        if target_col is None or task_type is None:
            detected_target_col, detected_task_type = auto_detect_task_type_and_target(dataset_path, quiet=True)
            
            # In batch mode, always use auto-detected values (no interactive selection)
            final_target_col = target_col or detected_target_col
            final_task_type = task_type or detected_task_type
            
            console.print(f"[dim]Auto-detected for {dataset_path.name}: target='{final_target_col}', task='{final_task_type}'[/dim]")
        else:
            final_target_col = target_col
            final_task_type = task_type
        
        # Prepare file paths
        train_file = dataset_path / 'train.csv'
        test_file = dataset_path / 'test.csv' if (dataset_path / 'test.csv').exists() else None
        description_file = dataset_path / 'dataset_description.md' if (dataset_path / 'dataset_description.md').exists() else None
        dataset_name = dataset_path.name
        
        # Call the actual preparation function
        prepare_dataset(
            train=train_file,
            test=test_file,
            target_col=final_target_col,
            description=description_file,
            name=dataset_name,
            positive_class=positive_class,
            negative_class=negative_class,
            task_type=final_task_type,
            output_dir=Path(prepared_datasets_dir)
        )
        return True
        
    except Exception as e:
        console.print(f"[red]Error preparing {dataset_info['name']}: {e}[/red]")
        return False

def batch_prepare_datasets(datasets_dir: str, prepared_datasets_dir: str, 
                          task_type: Optional[str] = None) -> None:
    """
    Batch prepare multiple datasets with Rich progress display.
    
    Args:
        datasets_dir: Path to raw datasets directory
        prepared_datasets_dir: Path to prepared datasets directory
        task_type: Optional task type override
    """
    console.print("[bold blue]Agentomics-ML Dataset Preparation[/bold blue]")
    console.print("=" * 50)
    console.print("")
    
    # Ensure prepared datasets directory exists
    Path(prepared_datasets_dir).mkdir(parents=True, exist_ok=True)
    
    # Get initial dataset information
    console.print(f"[dim]Scanning datasets in {datasets_dir}...[/dim]")
    datasets = get_dataset_info(datasets_dir, prepared_datasets_dir)
    
    if not datasets:
        console.print(f"[red]No datasets found in {datasets_dir}[/red]")
        console.print("")
        console.print("[blue]To add datasets:[/blue]")
        console.print(f"   1. Create directories in {datasets_dir}/")
        console.print("   2. Add train.csv (and optionally test.csv) to each directory")
        console.print("   3. Run preparation again")
        sys.exit(1)
    
    console.print("")
    
    # Show initial status
    display_preparation_table(datasets, "Datasets Found - Preparation Status")
    console.print("")
    
    # Count datasets needing preparation
    need_preparation = [d for d in datasets if d["can_prepare"]]
    already_prepared = [d for d in datasets if d["is_prepared"]]
    cannot_prepare = [d for d in datasets if not d["can_prepare"] and not d["is_prepared"]]
    
    if not need_preparation:
        if already_prepared:
            console.print(f"[green]All {len(already_prepared)} dataset(s) are already prepared![/green]")
        else:
            console.print("[yellow]No datasets can be prepared. Check for missing train.csv files.[/yellow]")
        
        console.print("")
        console.print("[blue]Summary:[/blue]")
        console.print(f"  Total datasets: {len(datasets)}")
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
        
        for dataset in need_preparation:
            task = progress.add_task(f"Preparing {dataset['name']}...", total=None)
            
            success = prepare_single_dataset(dataset, prepared_datasets_dir, task_type=task_type)
            
            if success:
                progress.update(task, description=f"{dataset['name']} prepared successfully")
                prepared_now += 1
                dataset["status"] = "Prepared"
                dataset["is_prepared"] = True
                dataset["can_prepare"] = False
            else:
                progress.update(task, description=f"{dataset['name']} preparation failed")
                failed_now += 1
                dataset["status"] = "Failed"
    
    console.print("")
    
    # Show final status
    if prepared_now > 0:
        display_preparation_table(datasets, "Final Preparation Results")
        console.print("")
    
    # Summary
    console.print("[blue]Preparation Summary:[/blue]")
    console.print(f"  Total datasets: {len(datasets)}")
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
        console.print("   1. Run the main agent: [cyan]./run.sh[/cyan]")
        console.print("   2. Select your prepared dataset")
        console.print("   3. Choose your AI model")
        console.print("   4. Let the agent work its magic!")
    
    if failed_now > 0:
        console.print("")
        console.print(f"[yellow]{failed_now} dataset(s) failed to prepare. Check your data files.[/yellow]")

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preparation with auto-detection")
    parser.add_argument('--dataset-dir', type=Path, help='Single dataset directory to prepare')
    parser.add_argument('--batch', action='store_true', help='Batch mode: prepare all datasets in datasets-dir')
    parser.add_argument('--target-col', type=str, default=None, help='Target column name (auto-detected if not provided)')
    parser.add_argument('--task-type', choices=['classification', 'regression'], default=None, help='Task type (auto-detected if not provided)')
    return parser.parse_args()

def main():
    global console
    args = parse_args()
    
    # Auto-detect environment (Docker vs Local)
    if os.path.exists("/repository"):
        # Docker environment
        default_datasets_dir = "/repository/datasets"
        default_prepared_dir = "/repository/prepared_datasets"
    else:
        # Local environment
        default_datasets_dir = "./datasets"
        default_prepared_dir = "./prepared_datasets"
    
    if args.batch or not args.dataset_dir:
        # Batch mode - prepare all datasets
        datasets_dir = os.environ.get("DATASETS_DIR", default_datasets_dir)
        prepared_datasets_dir = os.environ.get("PREPARED_DATASETS_DIR", default_prepared_dir)
        task_type = args.task_type or os.environ.get("TASK_TYPE")
        
        batch_prepare_datasets(datasets_dir, prepared_datasets_dir, task_type)
        
    else:
        # Single dataset mode
        dataset_dir = args.dataset_dir
        prepared_datasets_dir = Path(os.environ.get("PREPARED_DATASETS_DIR", default_prepared_dir))
        
        # Auto-detect target column and task type
        if args.target_col is None or args.task_type is None:
            target_col, task_type = auto_detect_task_type_and_target(dataset_dir)
            console.print(f'[dim]Auto-detected: target="{target_col}", task="{task_type}"[/dim]')
        else:
            target_col = args.target_col
            task_type = args.task_type
        
        # Prepare single dataset
        console.print(f'[blue]Preparing dataset "{dataset_dir.name}" for {task_type} task with target column "{target_col}"[/blue]')
        
        train_file = dataset_dir / 'train.csv'
        test_file = dataset_dir / 'test.csv' if (dataset_dir / 'test.csv').exists() else None
        description_file = dataset_dir / 'dataset_description.md' if (dataset_dir / 'dataset_description.md').exists() else None
        
        prepare_dataset(
            train=train_file,
            test=test_file,
            target_col=target_col,
            description=description_file,
            name=dataset_dir.name,
            task_type=task_type,
            output_dir=prepared_datasets_dir
        )
        
        console.print(f"[green]Dataset '{dataset_dir.name}' prepared successfully![/green]")

if __name__ == "__main__":
    main()
