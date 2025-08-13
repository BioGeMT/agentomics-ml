import os
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.table import Table

console = Console()

def detect_environment() -> Tuple[str, str]:
    """
    Return dataset directories for Docker-only environment.
    
    Returns:
        Tuple of (datasets_dir, prepared_datasets_dir)
    """
    # Docker-only environment - no detection needed
    return "/repository/datasets", "/repository/prepared_datasets"

def detect_task_type(dataset_dir: str) -> str:
    """
    Auto-detect task type using direct function call.
    
    Args:
        dataset_dir: Path to the dataset directory
        
    Returns:
        Task type (classification, regression, etc.) or "Unknown"
    """
    train_file = Path(dataset_dir) / "train.csv"
    
    if not train_file.exists():
        return "Unknown"
    
    try:
        # Import the detection function directly from the same utils directory
        from .dataset_preparation import auto_detect_task_type_and_target
        
        # Call the function directly
        _, task_type = auto_detect_task_type_and_target(Path(dataset_dir), quiet=True)
        return task_type
    except Exception:
        return "Unknown"

# Import count_csv_rows from dataset_preparation to avoid duplication
from .dataset_preparation import count_csv_rows

def get_dataset_info(datasets_dir: str, prepared_datasets_dir: str) -> List[Dict]:
    """
    Collect information about all available datasets.
    
    Args:
        datasets_dir: Path to raw datasets directory
        prepared_datasets_dir: Path to prepared datasets directory
        
    Returns:
        List of dataset information dictionaries
    """
    datasets_path = Path(datasets_dir)
    prepared_path = Path(prepared_datasets_dir)
    
    if not datasets_path.exists():
        return []
    
    dataset_info = []
    
    for dataset_dir in datasets_path.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        
        # Get dataset statistics
        raw_train_file = dataset_dir / "train.csv"
        raw_test_file = dataset_dir / "test.csv"
        prep_train_file = prepared_path / dataset_name / "train.csv"
        prep_test_file = prepared_path / dataset_name / "test.csv"
        
        # Count rows (prefer prepared files, fallback to raw)
        if prep_train_file.exists():
            train_rows = count_csv_rows(str(prep_train_file))
        elif raw_train_file.exists():
            train_rows = count_csv_rows(str(raw_train_file))
        else:
            train_rows = 0
            
        if prep_test_file.exists():
            test_rows = count_csv_rows(str(prep_test_file))
        elif raw_test_file.exists():
            test_rows = count_csv_rows(str(raw_test_file))
        else:
            test_rows = 0
        
        # Auto-detect task type from prepared dataset directory
        prepared_dataset_dir = prepared_path / dataset_name
        if prepared_dataset_dir.exists():
            task_type = detect_task_type(str(prepared_dataset_dir))
        else:
            task_type = "Unknown"
        
        # Determine status
        metadata_file = prepared_path / dataset_name / "metadata.json"
        if metadata_file.exists() and (prepared_path / dataset_name).exists():
            status = "Prepared"
        elif raw_train_file.exists():
            status = "Not prepared"
        else:
            status = "Missing files"
        
        dataset_info.append({
            "name": dataset_name,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "task_type": task_type,
            "status": status
        })
    
    # Sort by name for consistent ordering
    dataset_info.sort(key=lambda x: x["name"])
    return dataset_info

def display_datasets_table(datasets: List[Dict]) -> None:
    """
    Display datasets in a Rich table matching the model listing style.
    
    Args:
        datasets: List of dataset dictionaries
    """
    if not datasets:
        console.print("[red]No datasets found[/red]")
        return
    
    # Create table with same styling approach as models
    table = Table(title="Available Datasets (Agentomics-ML)", show_header=True, header_style="bold blue")
    
    table.add_column("#", style="dim", width=3)
    table.add_column("Dataset Name", style="cyan", no_wrap=True)
    table.add_column("Train Rows", justify="right", style="green")
    table.add_column("Test Rows", justify="right", style="green")
    table.add_column("Task Type", style="yellow")
    table.add_column("Status", style="magenta")
    
    prepared_count = 0
    total_count = len(datasets)
    
    for i, dataset in enumerate(datasets, 1):
        # Highlight prepared datasets
        style_override = {}
        if dataset["status"] == "Prepared":
            style_override = {"style": "bold"}
            prepared_count += 1
        
        # Format row counts
        train_display = f"{dataset['train_rows']:,}" if dataset['train_rows'] > 0 else "N/A"
        test_display = f"{dataset['test_rows']:,}" if dataset['test_rows'] > 0 else "N/A"
        
        # Color-code status
        status_display = dataset["status"]
        if dataset["status"] == "Prepared":
            status_display = f"[green]{dataset['status']}[/green]"
        elif dataset["status"] == "Not prepared":
            status_display = f"[yellow]{dataset['status']}[/yellow]"
        else:
            status_display = f"[red]{dataset['status']}[/red]"
        
        row = [
            str(i),
            dataset["name"],
            train_display,
            test_display,
            dataset["task_type"],
            status_display
        ]
        
        table.add_row(*row, **style_override)
    
    console.print(table)
    
    # Summary section (matching model listing approach)
    console.print(f"[dim]Bold entries are prepared and ready to use[/dim]")
    console.print("")
    
    # Status summary
    console.print(f"[blue]Summary:[/blue]")
    console.print(f"  Total datasets: {total_count}")
    console.print(f"  Prepared: [green]{prepared_count}[/green]")
    console.print(f"  Not prepared: [yellow]{total_count - prepared_count}[/yellow]")
    console.print("")
    
    # Action suggestions
    if prepared_count == total_count:
        console.print("[green]All datasets are prepared and ready to use![/green]")
    elif prepared_count == 0:
        console.print("[yellow]No datasets are prepared yet.[/yellow]")
        console.print("[dim]To prepare datasets, run: ./run.sh --prepare-only[/dim]")
    else:
        console.print("[yellow]Some datasets need preparation.[/yellow]")
        console.print("[dim]To prepare remaining datasets, run: ./run.sh --prepare-only[/dim]")

def output_selection_format(datasets: List[Dict]) -> None:
    """
    Output datasets in parseable format for bash scripts.
    Format: number|dataset_name|train_rows|task_type|status
    
    Args:
        datasets: List of dataset dictionaries
    """
    if not datasets:
        print("ERROR: No datasets available", file=sys.stderr)
        return
    
    for i, dataset in enumerate(datasets, 1):
        print(f'{i}|{dataset["name"]}|{dataset["train_rows"]}|{dataset["task_type"]}|{dataset["status"]}')

def interactive_dataset_selection(datasets: List[Dict]) -> Optional[str]:
    """
    Interactive dataset selection with status display.
    
    Args:
        datasets: List of dataset dictionaries
        
    Returns:
        Selected dataset name or None if cancelled
    """
    if not datasets:
        console.print("[red]No datasets available[/red]")
        return None
    
    # Display datasets table
    display_datasets_table(datasets)
    
    # Filter only prepared datasets for selection
    prepared_datasets = [d for d in datasets if d["status"] == "Prepared"]
    
    if not prepared_datasets:
        console.print("[red]No prepared datasets available for selection[/red]")
        console.print("[dim]Run ./run.sh --prepare-only to prepare datasets first[/dim]")
        return None
    
    console.print(f"\n[green]Select a prepared dataset (1-{len(prepared_datasets)}) or press Enter for first:[/green]")
    
    try:
        choice_input = input("Enter choice: ").strip()
        if not choice_input:
            return prepared_datasets[0]["name"]  # Return first prepared dataset as default
        
        choice = int(choice_input) - 1
        if 0 <= choice < len(prepared_datasets):
            selected_dataset = prepared_datasets[choice]["name"]
            console.print(f"[green]Selected: {selected_dataset}[/green]")
            return selected_dataset
        else:
            console.print("[red]Invalid choice[/red]")
            return None
    except ValueError:
        console.print("[red]Invalid input[/red]")
        return None

if __name__ == "__main__":
    import argparse
    
    try:
        import dotenv
        from .env_utils import ensure_dotenv_loaded
        ensure_dotenv_loaded()
    except ImportError:
        # dotenv is optional
        pass
    
    parser = argparse.ArgumentParser(description='Dataset listing utilities')
    parser.add_argument('--format', choices=['display', 'selection'], default='display',
                       help='Output format: display (rich table) or selection (bash-parseable)')
    parser.add_argument('--interactive-select', action='store_true',
                       help='Interactive dataset selection mode')
    args = parser.parse_args()
    
    # Get dataset directories
    datasets_dir, prepared_datasets_dir = detect_environment()
    
    # Collect dataset information
    datasets = get_dataset_info(datasets_dir, prepared_datasets_dir)
    
    if not datasets:
        if args.format == 'selection':
            print('ERROR: No datasets found', file=sys.stderr)
            sys.exit(1)
        else:
            console.print(f"[red]No datasets found in {datasets_dir} directory[/red]")
            console.print("")
            console.print("[blue]To add datasets:[/blue]")
            console.print(f"  1. Create a directory in {datasets_dir}/")
            console.print("  2. Add your dataset files (train.csv, test.csv, etc.)")
            console.print("  3. Run this script again")
            sys.exit(1)
    
    # Handle different modes
    if args.interactive_select:
        selected = interactive_dataset_selection(datasets)
        if selected:
            print(selected)  # Output selected dataset for shell capture
        else:
            sys.exit(1)
    elif args.format == 'selection':
        output_selection_format(datasets)
    else:
        display_datasets_table(datasets)
