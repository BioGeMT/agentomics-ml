import sys
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.console import Group
from rich.progress import Progress, SpinnerColumn, TextColumn
from utils.user_input import get_user_input_for_int
from utils.dataset_utils import prepare_dataset, get_all_datasets_info

console = Console()

def print_datasets_table(datasets: List[Dict], title: str = "Dataset Preparation Status") -> None:
    """
    Display datasets in a Rich table for preparation.

    Args:
        datasets: List of dataset dictionaries
        title: Table title
    """
    if not datasets:
        console.print("[red]No datasets found[/red]")
        return

    console.print(f"[bold blue]{title}[/bold blue] ({len(datasets)} datasets)\n")

    name_lines = []
    train_lines = []
    val_lines = []
    test_lines = []
    status_lines = []

    max_num_width = len(str(len(datasets)))

    for i, dataset in enumerate(datasets, 1):
        num_str = str(i).rjust(max_num_width)
        name_lines.append(f"[white]{num_str}[/white] [bold cyan]{dataset['name']}[/bold cyan]")

        if dataset['train_rows'] > 0:
            train_lines.append(f"[green]{dataset['train_rows']:,}[/green]")
        else:
            train_lines.append("[dim]N/A[/dim]")

        if dataset['validation_rows'] > 0:
            val_lines.append(f"[yellow]{dataset['validation_rows']:,}[/yellow]")
        else:
            val_lines.append("[dim]N/A (Will be created by agent)[/dim]")

        if dataset['test_rows'] > 0:
            test_lines.append(f"[blue]{dataset['test_rows']:,}[/blue]")
        else:
            test_lines.append("[dim]N/A[/dim]")

        status = dataset["status"]
        if status == "Already prepared":
            status_lines.append(f"[bold green]✓ {status}[/bold green]")
        elif status == "Prepared":
            status_lines.append(f"[bold green]✓ {status}[/bold green]")
        elif status == "Ready to prepare":
            status_lines.append(f"[bold yellow]⏳ {status}[/bold yellow]")
        elif status == "Failed":
            status_lines.append(f"[bold red]✗ {status}[/bold red]")
        elif "Missing" in status or "Empty" in status:
            status_lines.append(f"[bold red]⚠ {status}[/bold red]")
        else:
            status_lines.append(f"[red]{status}[/red]")

    boxes = [
        Panel("\n".join(name_lines), title="[bold]Dataset Name[/bold]", border_style="cyan"),
        Panel("\n".join(train_lines), title="[bold]Train Rows[/bold]", border_style="green"),
        Panel("\n".join(val_lines), title="[bold]Validation Rows[/bold]", border_style="yellow"),
        Panel("\n".join(test_lines), title="[bold]Test Rows[/bold]", border_style="blue"),
        Panel("\n".join(status_lines), title="[bold]Status[/bold]", border_style="magenta")
    ]

    console.print(Columns(boxes, padding=(0, 1), expand=False))

def prepare_all_datasets(datasets_dir: str, prepared_datasets_dir: str, prepared_test_sets_dir: str) -> None:
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
    print_datasets_table(datasets_info, "Datasets Found - Preparation Status")
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
            console.print("[green]Ready to use![/green]")
        
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
                progress.stop()
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
                progress.start()
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
        print_datasets_table(datasets_info, "Final Preparation Results")
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
        console.print("   1. Run the agent:")
        console.print("   2. Select your prepared dataset")
        console.print("   3. Choose your AI model")
        console.print("   4. Let the agent work its magic!")
    
    if failed_now > 0:
        console.print("")
        console.print(f"[yellow]{failed_now} dataset(s) failed to prepare. Check your data files.[/yellow]")



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
    
    console.print("Selecting dataset interactively...", style="cyan")
    print_datasets_table(datasets)
    prepared_datasets_table_indicies = [i+1 for i,d in enumerate(datasets)]

    if not datasets:
        console.print("[red]No prepared datasets available for selection[/red]")
        return None
    
    choice = get_user_input_for_int(
        f"Select a prepared dataset", 
        valid_options=prepared_datasets_table_indicies,
        default=prepared_datasets_table_indicies[0],
    )
    selected_dataset = datasets[choice-1]["name"]
    console.print(f"[green]Selected: {selected_dataset}[/green]")
    return selected_dataset
