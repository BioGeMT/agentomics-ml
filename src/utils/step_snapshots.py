"""
Step-level snapshotting and forking system.

This module provides decorators and utilities for snapshotting workspace state
after each step, enabling forking from any point in the pipeline.
"""

import datetime
import hashlib
import json
import shutil
from contextlib import asynccontextmanager
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from utils.content_store import ContentAddressedStore
from utils.serialization import serialize_object, deserialize_object

console = Console()


class Step(Enum):
    """Predefined steps in the Agentomics pipeline (in execution order)"""
    DATA_EXPLORATION = "data_exploration"
    DATA_SPLIT = "data_split"
    DATA_REPRESENTATION = "data_representation"
    MODEL_ARCHITECTURE = "model_architecture"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    PREDICTION_EXPLORATION = "prediction_exploration"
    
    @classmethod
    def get_step_names(cls) -> List[str]:
        """Get ordered list of step names"""
        return [step.value for step in cls]
    
    @classmethod
    def get_step_index(cls, step_name: str) -> int:
        """Get the index of a step by name"""
        names = cls.get_step_names()
        if step_name not in names:
            raise ValueError(f"Unknown step: {step_name}. Valid steps: {names}")
        return names.index(step_name)


class StepRegistry:
    """Global registry tracking execution context"""
    _current_iteration = None
    _current_config = None
    _run_manager = None
    
    @classmethod
    def set_context(cls, config, iteration, run_manager):
        """Set the current execution context"""
        cls._current_config = config
        cls._current_iteration = iteration
        cls._run_manager = run_manager


def snapshotable_step(name: str, forkable: bool = True):
    """
    Decorator to make a step automatically snapshotable and forkable.
    
    Usage:
        @snapshotable_step("data_exploration")
        async def run_data_exploration_step(...):
            # step implementation
            return messages, output
    
    The decorator automatically:
    - Validates step name against predefined Step enum
    - Snapshots workspace after step completion
    - Enables forking from this step
    - Tracks execution time
    - Handles skipping when forking
    
    Args:
        name: Step name (must be one of Step enum values)
        forkable: Whether this step can be forked from (default: True)
    """
    
    # Validate step name and get index from predefined enum
    try:
        index = Step.get_step_index(name)
    except ValueError as e:
        raise ValueError(f"Invalid step name in decorator: {e}")
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get execution context
            config = StepRegistry._current_config
            iteration = StepRegistry._current_iteration
            run_manager = StepRegistry._run_manager
            
            # If no run manager, just execute normally
            if run_manager is None:
                return await func(*args, **kwargs)
            
            # Check if we should skip this step (already done in fork)
            if run_manager.should_skip_step(name):
                source_run = run_manager.resume_from[0] if run_manager.resume_from else 'N/A'
                console.print(f"\n⏭  [yellow]SKIPPING STEP [{index}]: {name}[/yellow]")
                console.print(f"   [dim]Using cached results from: {source_run}[/dim]\n")
                
                # Return cached return value
                cached_return = run_manager.get_cached_return(iteration, name)
                
                if cached_return is None:
                    raise RuntimeError(
                        f"Step {name} should be skipped but no cached output found!"
                    )
                
                return cached_return
            
            # Execute the step
            # Check if this is the first step after skipping (fork resume point)
            is_resume_point = (
                run_manager.resume_from is not None and 
                run_manager.skip_until_step is not None and
                index == run_manager.skip_until_step
            )
            
            if is_resume_point:
                console.print(Panel(
                    "[bold]Fork point reached - starting fresh execution[/bold]",
                    title="▶ Resuming Execution",
                    border_style="bold green"
                ))
            
            console.print(f"\n⚙  [bold cyan]EXECUTING STEP [{index}]: {name}[/bold cyan]")
            start_time = datetime.datetime.now()
            
            try:
                # Call the actual step function
                result = await func(*args, **kwargs)
                messages, output = result
                
                # Snapshot after completion
                run_manager.snapshot_step(
                    iteration=iteration,
                    step_name=name,
                    files_created=output.files_created if hasattr(output, 'files_created') else [],
                    step_return_value=result
                )
                
                duration = (datetime.datetime.now() - start_time).total_seconds()
                console.print(f"\n✓ [green]COMPLETED STEP [{index}]: {name}[/green]")
                console.print(f"  [dim]Duration: {duration:.2f}s[/dim]\n")
                
                return result
                
            except Exception as e:
                console.print(f"\n✗ [red]FAILED STEP [{index}]: {name}[/red]")
                console.print(f"  [dim]Error: {str(e)[:100]}[/dim]\n")
                raise
        
        # Attach metadata to the wrapped function
        wrapper._step_metadata = {
            'name': name,
            'index': index,
            'forkable': forkable
        }
        
        return wrapper
    
    return decorator


@asynccontextmanager
async def step_execution_context(config, iteration, run_manager: Optional['RunManager'] = None):
    """Context manager for step execution with automatic tracking"""
    StepRegistry.set_context(config, iteration, run_manager)
    try:
        yield run_manager
    finally:
        StepRegistry.set_context(None, None, None)


class RunManager:
    """Manages run lifecycle with step-level snapshotting"""
    
    def __init__(self, config, resume_from: Optional[tuple] = None, source_user_prompt: Optional[str] = None):
        """
        Args:
            config: Configuration object
            resume_from: Optional tuple of (run_id, step_name, iteration) to fork from
            source_user_prompt: Optional user prompt from source run (for lineage tracking)
        """
        self.config = config
        self.run_id = config.agent_id
        self.run_dir = config.runs_dir / self.run_id
        self.content_store = ContentAddressedStore(config.workspace_dir)
        
        # Skip configuration
        self.skip_until_step = None
        self._cached_returns = {}
        
        # Fork info
        self.resume_from = resume_from
        self.source_user_prompt = source_user_prompt
        
        # Lineage tracking (for fork relationships)
        self.lineage = None
        
        # Initialize from fork if specified
        if resume_from:
            # Handle both old format (3-tuple) and new format (4-tuple with user_prompt)
            if len(resume_from) == 4:
                source_run_id, step_name, iteration, source_user_prompt = resume_from
                self.source_user_prompt = source_user_prompt
            else:
                source_run_id, step_name, iteration = resume_from
            self._initialize_from_fork(source_run_id, step_name, iteration)
    
    def _get_ancestor_chain(self, run_id: str) -> list[str]:
        """Get the chain of ancestor runs by parsing fork naming convention.
        
        Supports multiple formats:
        - Old format: 'run_A__fork_step_X__fork_step_Y'
        - Intermediate format: 'run_A_forked_from_step_X_iterN_timestamp'
        - Current format: 'run_A_fork_step_X_iterN_timestamp'
        
        Example: 'run_A_fork_step_X_iter2_20250112_143022' -> ['run_A']
        Example: 'run_A_fork_step_X_iter2_20250112_fork_step_Y_iter3_20250113_144000' -> ['run_A', 'run_A_fork_step_X_iter2_20250112_143022']
        """
        ancestors = []
        
        # Try current format first (_fork_)
        if '_fork_' in run_id and '_forked_from_' not in run_id:
            parts = run_id.split('_fork_')
            if len(parts) > 1:
                # Build chain base -> ... -> immediate parent (exclude current run id)
                current = parts[0]
                ancestors.append(current)
                for i in range(1, len(parts) - 1):
                    current = current + '_fork_' + parts[i]
                    ancestors.append(current)
        # Try intermediate format (_forked_from_)
        elif '_forked_from_' in run_id:
            parts = run_id.split('_forked_from_')
            if len(parts) > 1:
                base = parts[0]
                ancestors.append(base)
                
                # Reconstruct intermediate forks by joining parts progressively
                for i in range(1, len(parts)):
                    intermediate_parts = parts[:i+1]
                    intermediate = '_forked_from_'.join(intermediate_parts)
                    ancestors.append(intermediate)
        # Fall back to old format (__fork_)
        elif '__fork_' in run_id:
            parts = run_id.split('__fork_')
            if len(parts) > 1:
                base = parts[0]
                ancestors.append(base)
                
                # Reconstruct intermediate forks
                for i in range(1, len(parts) - 1):
                    intermediate = base + '__fork_' + '__fork_'.join(parts[1:i+1])
                    ancestors.append(intermediate)
        
        return ancestors

    def _locate_snapshot_file(
        self,
        run_candidates: List[str],
        iteration: int,
        step_name: str,
        step_index: int,
    ) -> tuple[str, Path]:
        """
        Find the snapshot file for a given step by searching the source run
        and, if needed, its ancestors (most recent ancestor first).
        """
        for candidate in run_candidates:
            snapshot_path = (
                self.content_store.snapshots_dir
                / candidate
                / f"iteration_{iteration}"
                / f"{step_index:02d}_{step_name}.json"
            )
            if snapshot_path.exists():
                if candidate != run_candidates[0]:
                    console.print(
                        f"[yellow][FORK][/yellow] Snapshot for step '{step_name}' "
                        f"not found in run '{run_candidates[0]}'. "
                        f"Using ancestor run '{candidate}'."
                    )
                return candidate, snapshot_path
        raise ValueError(
            "Snapshot not found for step "
            f"{step_name} (iteration {iteration}). "
            f"Searched runs: {', '.join(run_candidates)}"
        )
    
    def _initialize_from_fork(self, source_run_id: str, step_name: str, iteration: int):
        """Initialize this run by forking from another run's step"""
        fork_info = f"[bold]Source Run:[/bold]  {source_run_id}\n"
        fork_info += f"[bold]Fork Point:[/bold]  {step_name} (iteration {iteration})"
        
        # Check for ancestor chain (fork-of-fork scenario)
        ancestors = self._get_ancestor_chain(source_run_id)
        if ancestors:
            fork_info += f"\n[bold]Ancestor Chain:[/bold] {' → '.join(ancestors)}"
        
        console.print(Panel(fork_info, title="[bold]Forking Mode Active[/bold]", border_style="bold yellow"))
        
        # Get step index (needed for lineage loading)
        step_index = self._get_step_index(step_name)
        snapshot_search_order = [source_run_id] + list(reversed(ancestors))
        
        # Load snapshot manifest
        _, snapshot_file = self._locate_snapshot_file(
            snapshot_search_order,
            iteration,
            step_name,
            step_index,
        )
        
        with open(snapshot_file) as f:
            snapshot = json.load(f)
        
        metadata = snapshot.get("metadata", {})
        
        # Recursively build full lineage chain by following parent links
        full_lineage = self._build_full_lineage_chain(source_run_id, step_name, iteration)
        
        # Use provided source user prompt
        parent_user_prompt = self.source_user_prompt
        
        # Store lineage for this run
        self.lineage = {
            "parent_run_id": source_run_id,
            "fork_point": {
                "step": step_name,
                "iteration": iteration,
                "user_prompt": self.config.user_prompt  # Current run's prompt
            },
            "full_lineage": full_lineage
        }
        
        # Display lineage tree
        self._display_lineage_tree(full_lineage, self.run_id)
        
        # Restore workspace files
        print(f"[FORK] Restoring workspace from snapshot...")
        manifest = snapshot["manifest"]
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Build list of directories to restore to:
        # 1. Current fork directory
        # 2. Source run directory
        # 3. All ancestor directories (for fork-of-fork scenarios)
        restore_dirs = [self.run_dir]
        
        source_run_dir = self.config.runs_dir / source_run_id
        source_run_dir.mkdir(parents=True, exist_ok=True)
        restore_dirs.append(source_run_dir)
        
        # Add ancestor directories for fork-of-fork support
        for ancestor_id in ancestors:
            ancestor_dir = self.config.runs_dir / ancestor_id
            ancestor_dir.mkdir(parents=True, exist_ok=True)
            restore_dirs.append(ancestor_dir)
        
        restored_count = 0
        for rel_path, file_hash in manifest.items():
            try:
                # Restore to all directories to support hardcoded paths at any level
                for target_dir in restore_dirs:
                    self.content_store.restore_file(file_hash, target_dir / rel_path)
                restored_count += 1
            except Exception as e:
                print(f"  Warning: Could not restore {rel_path}: {e}")
        
        dir_count = len(restore_dirs)
        print(f"[FORK] Restored {restored_count} files to {dir_count} director{'y' if dir_count == 1 else 'ies'}\n")
        
        print(f"[FORK] Loading cached outputs from source run...")
        
        # Load cached return values for steps 0 through step_index
        step_names = Step.get_step_names()
        steps_to_skip = []
        for i in range(step_index):
            step_name = step_names[i]
            _, step_snapshot_file = self._locate_snapshot_file(
                snapshot_search_order,
                iteration,
                step_name,
                i,
            )
            
            with open(step_snapshot_file) as f:
                step_snapshot = json.load(f)
            
            # Check if return_value_hash exists in metadata (for backward compatibility)
            metadata = step_snapshot.get("metadata", {})
            return_value_hash = metadata.get("return_value_hash")
            
            if return_value_hash is None:
                raise ValueError(
                    f"Snapshot for step {step_name} (iteration {iteration}) does not contain return_value_hash.\n"
                    f"This snapshot was created before return value caching was implemented.\n"
                    f"Please re-run the source run to create a new snapshot with return values."
                )
            
            return_obj = self.content_store.objects_dir / return_value_hash[:2] / return_value_hash[2:]
            
            if not return_obj.exists():
                raise ValueError(
                    f"Return value object not found for step {step_name} (iteration {iteration}).\n"
                    f"Expected path: {return_obj}\n"
                    f"Hash: {return_value_hash}\n"
                    f"The snapshot references a return value that doesn't exist in the object store.\n"
                    f"This may indicate corrupted storage or a missing object."
                )
            
            # Deserialize the entire (messages, output) tuple from pickle
            try:
                pickled_data = return_obj.read_bytes()
                step_messages, step_output = deserialize_object(pickled_data)
                
                # Store cached return with iteration 0 (current iteration in new run)
                # even though we loaded it from source iteration, because when we retrieve
                # it later, we'll be looking it up with the current iteration (0)
                self._cached_returns[(0, step_name)] = (step_messages, step_output)
                steps_to_skip.append(f"[{i}] {step_name}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to deserialize cached return value for step {step_name} (iteration {iteration}): {e}\n"
                    f"Object path: {return_obj}"
                ) from e
        
        # Set skip marker: skip everything before the fork point
        self.skip_until_step = step_index
        
        # Print fork summary
        summary_text = f"[bold]Steps to SKIP[/bold] (using cached results):\n"
        for step in steps_to_skip:
            summary_text += f"  • {step}\n"
        
        resume_step_index = step_index
        if resume_step_index < len(step_names):
            summary_text += f"\n[bold]Will RESUME execution at:[/bold]\n"
            summary_text += f"  ▶ [{resume_step_index}] {step_names[resume_step_index]}"
        else:
            summary_text += "\n[bold]All steps already executed in source run.[/bold]"
        
        console.print(Panel(summary_text, title="Fork Summary", border_style="green"))
    
    def _extract_parent_lineage(self, metadata: Dict[str, Any]) -> Optional[List[Dict]]:
        """Extract lineage information from snapshot metadata if present."""
        parent_lineage = metadata.get("lineage")
        if not parent_lineage:
            return None
        
        if "full_lineage" in parent_lineage:
            return parent_lineage["full_lineage"]
        
        if "parent_run_id" in parent_lineage:
            return [{
                "run_id": parent_lineage["parent_run_id"],
                "fork_point": parent_lineage.get("fork_point", {})
            }]
        
        return None
    
    def _build_full_lineage_chain(self, source_run_id: str, step_name: str, iteration: int) -> List[Dict]:
        """Recursively build the full lineage chain by following parent links."""
        # First, try to get full_lineage from the source run's snapshot
        try:
            step_index = self._get_step_index(step_name)
            snapshot_search_order = [source_run_id] + list(reversed(self._get_ancestor_chain(source_run_id)))
            
            _, snapshot_file = self._locate_snapshot_file(
                snapshot_search_order,
                iteration,
                step_name,
                step_index,
            )
            
            with open(snapshot_file) as f:
                snapshot = json.load(f)
            metadata = snapshot.get("metadata", {})
            parent_lineage = self._extract_parent_lineage(metadata)
            
            # If parent has full_lineage, use it and add current fork point
            if parent_lineage:
                full_lineage = list(parent_lineage)  # Copy the list
                full_lineage.append({
                    "run_id": source_run_id,
                    "fork_point": {
                        "step": step_name,
                        "iteration": iteration,
                        "user_prompt": self.source_user_prompt or ""
                    }
                })
                return full_lineage
        except (ValueError, FileNotFoundError, KeyError):
            pass
        
        # Fallback: recursively build lineage by following parent links
        full_lineage = []
        visited = set()
        current_run_id = source_run_id
        current_step = step_name
        current_iter = iteration
        current_prompt = self.source_user_prompt or ""
        
        while current_run_id and current_run_id not in visited:
            visited.add(current_run_id)
            
            # Try to find lineage info for this run
            try:
                step_idx = self._get_step_index(current_step)
                search_order = [current_run_id] + list(reversed(self._get_ancestor_chain(current_run_id)))
                
                # Find any snapshot from this run
                found_snapshot = None
                for candidate in search_order:
                    for test_idx in range(len(Step.get_step_names())):
                        test_step = Step.get_step_names()[test_idx]
                        test_path = (
                            self.content_store.snapshots_dir /
                            candidate /
                            f"iteration_{current_iter}" /
                            f"{test_idx:02d}_{test_step}.json"
                        )
                        if test_path.exists():
                            found_snapshot = test_path
                            break
                    if found_snapshot:
                        break
                
                if found_snapshot:
                    with open(found_snapshot) as f:
                        snap = json.load(f)
                    lineage = snap.get("metadata", {}).get("lineage")
                    
                    if lineage and "parent_run_id" in lineage:
                        # Add current run to lineage
                        full_lineage.insert(0, {
                            "run_id": current_run_id,
                            "fork_point": {
                                "step": current_step,
                                "iteration": current_iter,
                                "user_prompt": current_prompt
                            }
                        })
                        
                        # Move to parent
                        fork_pt = lineage.get("fork_point", {})
                        current_run_id = lineage["parent_run_id"]
                        current_step = fork_pt.get("step", "unknown")
                        current_iter = fork_pt.get("iteration", 0)
                        current_prompt = fork_pt.get("user_prompt", "")
                        continue
            except (ValueError, FileNotFoundError, KeyError):
                pass
            
            # No parent found - this is an original run or we can't find lineage
            full_lineage.insert(0, {
                "run_id": current_run_id,
                "fork_point": {
                    "step": current_step,
                    "iteration": current_iter,
                    "user_prompt": current_prompt
                } if current_step != "unknown" else {}
            })
            break
        
        return full_lineage
    
    def _display_lineage_tree(self, full_lineage: List[Dict], current_run_id: str):
        """Display lineage as a Rich tree showing fork relationships"""
        if not full_lineage:
            return
        
        # Build Rich Tree
        # Original run (first in lineage)
        original = full_lineage[0]
        original_run_id = original["run_id"]
        original_fork = original.get("fork_point", {})
        
        # Determine if original is truly original (no fork_point) or a fork itself
        is_original = not original_fork
        
        if is_original:
            # Truly original run
            root_label = f"[bold]{original_run_id}[/bold] [dim](original)[/dim]"
        else:
            # Original is itself a fork
            root_label = f"[bold]{original_run_id}[/bold]"
        
        tree = Tree(root_label)
        
        # Add fork point info for original if it exists
        if original_fork:
            step = original_fork.get("step", "unknown")
            iter_num = original_fork.get("iteration", "?")
            prompt = original_fork.get("user_prompt")
            fork_info = f"Fork @ [yellow]{step}[/yellow] (iter {iter_num})"
            if prompt:
                prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
                fork_info += f"\n  [dim]Prompt:[/dim] {prompt_short}"
            tree.add(fork_info)
        
        # Build tree structure for intermediate forks
        current_branch = tree
        
        # Display intermediate forks
        for fork in full_lineage[1:]:
            fork_run_id = fork["run_id"]
            fork_point = fork.get("fork_point", {})
            step = fork_point.get("step", "unknown")
            iter_num = fork_point.get("iteration", "?")
            prompt = fork_point.get("user_prompt")
            
            # Add fork node
            fork_node = current_branch.add(f"[bold]{fork_run_id}[/bold]")
            fork_info = f"Fork @ [yellow]{step}[/yellow] (iter {iter_num})"
            if prompt:
                prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
                fork_info += f"\n  [dim]Prompt:[/dim] {prompt_short}"
            fork_node.add(fork_info)
            current_branch = fork_node
        
        # Add current run with its user prompt
        current_label = f"[bold green]{current_run_id}[/bold green] [dim](current)[/dim]"
        current_node = current_branch.add(current_label)
        if self.config.user_prompt:
            prompt_short = self.config.user_prompt[:50] + "..." if len(self.config.user_prompt) > 50 else self.config.user_prompt
            current_node.add(f"[dim]Prompt:[/dim] {prompt_short}")
        
        # Display the tree
        console.print("\n[bold cyan]Fork Lineage Tree:[/bold cyan]")
        console.print(tree)
        console.print()  # Empty line after tree
    
    def snapshot_step(self, iteration: int, step_name: str, files_created: List[str], 
                     step_return_value: tuple):
        """Create a snapshot after a step completes"""
        
        step_index = self._get_step_index(step_name)
        
        print(f"  [SNAPSHOT] Saving step state for future forking...")
        
        # Snapshot workspace files
        manifest = self.content_store.store_directory(
            self.run_dir,
            exclude_patterns=[
                "iteration_",
                "__pycache__",
                ".cache",
                ".agentomics_storage",
                ".conda"  # Conda environments (large, rarely change, handled separately)
            ]
        )
        
        print(f"  [SNAPSHOT] Stored {len(manifest)} files")
        
        # Store return value (messages, output)
        return_value_bytes = self._serialize_return_value(step_return_value)
        return_value_hash = hashlib.sha256(return_value_bytes).hexdigest()
        
        return_obj = self.content_store.objects_dir / return_value_hash[:2] / return_value_hash[2:]
        if not return_obj.exists():
            return_obj.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            return_obj.parent.chmod(0o755)  # Ensure correct permissions
            return_obj.write_bytes(return_value_bytes)
            return_obj.chmod(0o644)  # Make file readable
        
        # Save manifest
        metadata = {
            "step_name": step_name,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "files_created": files_created,
            "return_value_hash": return_value_hash,
        }
        
        # Add lineage information if this is a fork
        if self.lineage is not None:
            metadata["lineage"] = self.lineage
        
        self.content_store.save_snapshot(
            self.run_id,
            iteration,
            f"{step_index:02d}_{step_name}",
            manifest,
            metadata
        )
        
        print(f"  [SNAPSHOT] Step state saved - can fork from this point")
        
        print(f"  Snapshot saved ({len(manifest)} files)")
    
    def should_skip_step(self, step_name: str) -> bool:
        """Check if a step should be skipped"""
        if self.skip_until_step is None:
            return False
        
        step_index = self._get_step_index(step_name)
        return step_index < self.skip_until_step
    
    def get_cached_return(self, iteration: int, step_name: str):
        """Get cached return value for a skipped step"""
        return self._cached_returns.get((iteration, step_name))
    
    def _get_step_index(self, step_name: str) -> int:
        """Get the index of a step by its name"""
        return Step.get_step_index(step_name)
    
    def _serialize_return_value(self, step_return_value: tuple) -> bytes:
        """Serialize (messages, output) tuple to bytes using pickle"""
        return serialize_object(step_return_value)
    


def get_latest_iteration(workspace_dir: Path, run_id: str) -> int:
    """
    Find the latest (most recent) iteration for a run.
    
    Looks in .agentomics_storage/snapshots/{run_id}/ to find iteration directories.
    This works even when forking, since we copy the storage directory.
    """
    # Look in the content-addressed storage snapshots directory
    snapshots_dir = workspace_dir / ".agentomics_storage" / "snapshots" / run_id
    
    if not snapshots_dir.exists():
        raise ValueError(f"No snapshots found for run '{run_id}'. Cannot determine latest iteration.")
    
    # Find all iteration directories
    iteration_dirs = [
        d for d in snapshots_dir.iterdir()
        if d.is_dir() and d.name.startswith("iteration_")
    ]
    
    if not iteration_dirs:
        raise ValueError(f"No iteration snapshots found for run '{run_id}'.")
    
    # Extract iteration numbers and get the maximum
    iteration_numbers = []
    for d in iteration_dirs:
        try:
            iter_num = int(d.name.split("_")[1])
            iteration_numbers.append(iter_num)
        except (IndexError, ValueError):
            continue
    
    if not iteration_numbers:
        raise ValueError(f"Could not parse iteration numbers for run '{run_id}'.")
    
    return max(iteration_numbers)

