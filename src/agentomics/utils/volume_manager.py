import argparse
import subprocess
import sys
from typing import List, Optional

try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich import box
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    print("Error: rich library not found. Please install it with: pip install rich", file=sys.stderr)
    sys.exit(1)


class VolumeManager:
    def __init__(self):
        self.console = Console(stderr=True)  # Use stderr for UI, stdout for result
        self.default_volume = "agentomics-workspace"
    
    def get_existing_volumes(self) -> List[str]:
        """Get list of existing Docker volumes."""
        try:
            result = subprocess.run(
                ["docker", "volume", "ls", "--format", "{{.Name}}"],
                capture_output=True,
                text=True,
                check=True
            )
            volumes = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return volumes
        except subprocess.CalledProcessError:
            return []
    
    def create_volume(self, name: str) -> bool:
        """Create a new Docker volume."""
        try:
            subprocess.run(["docker", "volume", "create", name], 
                         capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def display_volumes_table(self, volumes: List[str]) -> None:
        """Display volumes in a rich table."""
        if not volumes:
            self.console.print("No existing Docker volumes found", style="yellow")
            return
            
        table = Table(title="Existing Docker Volumes", box=box.ROUNDED)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Volume Name", style="green")
        table.add_column("Status", style="blue")
        
        for i, volume in enumerate(volumes, 1):
            status = "Available"
            if volume == self.default_volume:
                status = "Default"
            table.add_row(str(i), volume, status)
        
        self.console.print(table)
    
    def select_volume_interactive(self) -> Optional[str]:
        """Interactive volume selection with rich UI."""
        # Show header
        header = Panel(
            Text("Docker Workspace Volume Selection", style="bold blue"),
            subtitle="Choose or create a workspace volume for your experiments",
            border_style="blue"
        )
        self.console.print(header)
        self.console.print()
        
        # Get existing volumes
        volumes = self.get_existing_volumes()
        
        # Display volumes table
        self.display_volumes_table(volumes)
        self.console.print()
        
        # Show options
        options_text = Text()
        options_text.append("Options:\n", style="bold cyan")
        options_text.append(f"  • Press Enter for default: {self.default_volume}\n", style="white")
        options_text.append("  • Enter volume number (1-{})\n".format(len(volumes)) if volumes else "", style="white")
        options_text.append("  • Type 'new' to create a new volume\n", style="white")
        options_text.append("  • Type 'quit' to exit", style="white")
        
        self.console.print(Panel(options_text, border_style="cyan"))
        self.console.print()
        
        while True:
            choice = Prompt.ask(
                "Select volume",
                default=self.default_volume,
                console=self.console
            ).strip()
            
            # Handle default (empty input)
            if not choice or choice == self.default_volume:
                self.console.print(f"Using default volume: [green]{self.default_volume}[/green]")
                return self.default_volume
            
            # Handle quit
            if choice.lower() in ['quit', 'exit', 'q']:
                self.console.print("Volume selection cancelled", style="red")
                return None
            
            # Handle new volume creation
            if choice.lower() == 'new':
                return self._create_new_volume()
            
            # Handle numeric selection
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(volumes):
                    selected = volumes[index]
                    self.console.print(f"Selected volume: [green]{selected}[/green]")
                    return selected
                else:
                    self.console.print(f"Invalid number. Please choose 1-{len(volumes)}", style="red")
                    continue
            
            # Handle direct volume name
            if choice in volumes:
                self.console.print(f"Selected volume: [green]{choice}[/green]")
                return choice
            
            # Invalid input
            self.console.print(f"Invalid option: '{choice}'. Try again.", style="red")
    
    def _create_new_volume(self) -> Optional[str]:
        """Create a new volume with user input."""
        self.console.print()
        self.console.print("Create New Volume", style="bold green")
        
        while True:
            name = Prompt.ask(
                "Enter new volume name",
                console=self.console
            ).strip()
            
            if not name:
                self.console.print("Volume name cannot be empty", style="red")
                continue
            
            # Validate name (basic Docker volume name rules)
            if not name.replace('-', '').replace('_', '').replace('.', '').isalnum():
                self.console.print("Volume name can only contain letters, numbers, hyphens, underscores, and dots", style="red")
                continue
            
            # Check if volume already exists
            existing_volumes = self.get_existing_volumes()
            if name in existing_volumes:
                self.console.print(f"Volume '{name}' already exists", style="red")
                continue
            
            # Confirm creation
            if Confirm.ask(f"Create volume '[green]{name}[/green]'?", console=self.console):
                if self.create_volume(name):
                    self.console.print(f"Successfully created volume: [green]{name}[/green]")
                    return name
                else:
                    self.console.print(f"Failed to create volume: {name}", style="red")
                    return None
            else:
                self.console.print("Volume creation cancelled", style="yellow")
                return None


def main():
    parser = argparse.ArgumentParser(description="Docker Volume Manager")
    parser.add_argument("--select-only", action="store_true", 
                       help="Select volume and output name to stdout")
    
    args = parser.parse_args()
    
    manager = VolumeManager()
    
    if args.select_only:
        # Interactive selection mode
        selected_volume = manager.select_volume_interactive()
        if selected_volume:
            # Output to stdout for script consumption
            print(selected_volume)
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Default mode - just show available volumes
        volumes = manager.get_existing_volumes()
        manager.display_volumes_table(volumes)


if __name__ == "__main__":
    main()
