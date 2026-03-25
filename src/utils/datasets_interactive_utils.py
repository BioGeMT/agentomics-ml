import sys
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from utils.user_input import get_user_input_for_int
from utils.dataset_utils import prepare_dataset, get_all_datasets_info

console = Console()

def print_datasets_table(datasets: List[Dict], skip_status_column: bool = False) -> None:
    """
    Display datasets in a Rich table for preparation.

    Args:
        datasets: List of dataset dictionaries
    """
    if not datasets:
        console.print("[red]No datasets found[/red]")
        return

    table = Table(border_style="cyan")
    table.add_column("#", style="dim")
    table.add_column("Dataset Name", style="white")
    table.add_column("Train", justify="right")
    table.add_column("Validation", justify="right")
    table.add_column("Test", justify="right")
    if not skip_status_column:
        table.add_column("Status")

    for i, dataset in enumerate(datasets, 1):
        train = f"{dataset['train_rows']:,}" if dataset['train_rows'] > 0 else "[dim]N/A[/dim]"
        val = f"{dataset['validation_rows']:,}" if dataset['validation_rows'] > 0 else "[dim]N/A[/dim]"
        test = f"{dataset['test_rows']:,}" if dataset['test_rows'] > 0 else "[dim]N/A[/dim]"

        row = [str(i), dataset['name'], train, val, test]

        if not skip_status_column:
            status = dataset["status"]
            if status in ("Already prepared", "Prepared"):
                row.append(f"[green]✓ {status}[/green]")
            elif status == "Ready to prepare":
                row.append(f"[yellow]⏳ {status}[/yellow]")
            elif status == "Failed":
                row.append(f"[red]✗ {status}[/red]")
            elif "Missing" in status or "Empty" in status:
                row.append(f"[red]⚠ {status}[/red]")
            else:
                row.append(f"[red]{status}[/red]")

        table.add_row(*row)

    console.print(table)

def prepare_all_datasets(datasets_dir: str, prepared_datasets_dir: str, prepared_test_sets_dir: str) -> None:
    """
    Prepare multiple datasets with Rich progress display.
    
    Args:
        datasets_dir: Path to raw datasets directory
        prepared_datasets_dir: Path to prepared datasets directory
    """
    console.print("[bold blue]■ Dataset Preparation[/bold blue]")
    console.print("")
    
    # Get initial dataset information
    datasets_info = get_all_datasets_info(datasets_dir, prepared_datasets_dir)
    
    if not datasets_info:
        console.print(f"[red]No datasets found in {datasets_dir}[/red]")
        console.print("")
        console.print("[blue]To add datasets:[/blue]")
        console.print(f"   1. Create directories in {datasets_dir}/ (e.g. {datasets_dir}/my_dataset/)")
        console.print("   2. Add train.csv (and optionally test.csv) to each dataset directory")
        console.print("   3. Run preparation again")
        sys.exit(1)
    
    console.print("")
    
    # Show initial status
    #TODO make sure table shows classification/regression that was detected
    print_datasets_table(datasets_info)
    console.print("")
    
    # Count datasets needing preparation
    need_preparation = [d for d in datasets_info if d["should_prepare"]]
    already_prepared = [d for d in datasets_info if d["is_prepared"]]
    
    if not need_preparation:
        if not already_prepared:
            console.print("[yellow]No datasets can be prepared. Check for missing train.csv files.[/yellow]")
        sys.exit(0)
    
    # Prepare datasets with progress display
    console.print(f"[yellow]Preparing {len(need_preparation)} dataset(s)...[/yellow]")
    console.print("")
    
    prepared_now = 0
    failed_now = 0

    for dataset_info in need_preparation:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
        
            task = progress.add_task(f"Preparing {dataset_info['name']}", total=None)
            try:
                prepare_dataset(
                    dataset_dir=dataset_info['path'],
                    target_col=None, #auto-detected inside
                    positive_class=None, #auto-detected inside
                    negative_class=None, #auto-detected inside
                    task_type=None, #auto-detected inside
                    output_dir=prepared_datasets_dir,
                    interactive=True,
                    test_sets_output_dir=prepared_test_sets_dir
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
        print_datasets_table(datasets_info)
        console.print("")
    
    if failed_now > 0:
        console.print("")
        console.print(f"[yellow]{failed_now} dataset(s) failed to prepare. Check your data files.[/yellow]")



def interactive_dataset_selection(datasets: List[Dict]) -> Optional[str]:
    """
    Interactive dataset selection with info display.
    
    Args:
        datasets: List of dataset dictionaries
        
    Returns:
        Selected dataset name or None if cancelled
    """
    if not datasets:
        console.print("[red]No datasets available[/red]")
        return None
    
    print_datasets_table(datasets, skip_status_column=True)
    prepared_datasets_table_indicies = [i+1 for i,d in enumerate(datasets)]
    
    choice = get_user_input_for_int(
        f"Select a prepared dataset", 
        valid_options=prepared_datasets_table_indicies,
        default=prepared_datasets_table_indicies[0],
    )
    selected_dataset = datasets[choice-1]["name"]
    console.print(f"[cyan]Selected: {selected_dataset}[/cyan]")
    return selected_dataset
