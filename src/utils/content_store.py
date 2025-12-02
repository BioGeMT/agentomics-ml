"""
Content-addressed storage for workspace snapshots.

This module implements a git-like object store where files are stored once
by their content hash, and snapshots reference these files via lightweight manifests.

Directory Structure:
    .agentomics_storage/
    ├── objects/          # Content-addressed file storage (deduplicated)
    │   └── {hash[:2]}/
    │       └── {hash[2:]}
    ├── snapshots/        # Step-level snapshot manifests
    │   └── {run_id}/
    │       └── iteration_{N}/
    │           └── {step_index:02d}_{step_name}.json
    └── configs/          # Run configuration files
        └── {run_id}.json
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


class ContentAddressedStore:
    """
    Git-like content-addressed storage for workspace snapshots.
    
    Files are stored once by their SHA-256 hash, and snapshots
    just reference these hashes. This dramatically reduces storage
    through deduplication.
    """
    
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.storage_dir = self.workspace_dir / ".agentomics_storage"
        self.objects_dir = self.storage_dir / "objects"
        self.snapshots_dir = self.storage_dir / "snapshots"
        
        # Create storage directories with readable permissions (755 = rwxr-xr-x)
        self.objects_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        
        # Ensure parent directories also have readable permissions
        self.storage_dir.chmod(0o755)
        self.objects_dir.chmod(0o755)
        self.snapshots_dir.chmod(0o755)
    
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(65536):  # 64KB chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def store_file(self, file_path: Path) -> str:
        """
        Store a file in the object store.
        
        Returns: The hash of the stored file.
        If file already exists (same hash), does nothing (deduplication!).
        """
        # Compute hash
        file_hash = self.compute_file_hash(file_path)
        
        # Object path: objects/a1/23f4d5e6...
        # (First 2 chars as directory, like git)
        obj_dir = self.objects_dir / file_hash[:2]
        obj_path = obj_dir / file_hash[2:]
        
        # If object already exists, we're done (deduplication!)
        if obj_path.exists():
            return file_hash
        
        # Store the file with readable permissions
        obj_dir.mkdir(exist_ok=True, mode=0o755)
        obj_dir.chmod(0o755)  # Ensure it has correct permissions if it already existed
        
        shutil.copy2(file_path, obj_path)
        obj_path.chmod(0o644)  # Make file readable (rw-r--r--)
        
        return file_hash
    
    def store_directory(self, dir_path: Path, exclude_patterns: List[str] = None) -> Dict[str, str]:
        """
        Store all files in a directory.
        
        Returns: A manifest mapping relative paths to file hashes.
        Example:
            {
                "train.csv": "a123f4d5...",
                "validation.csv": "b789c2e1...",
                "models/model.py": "c456d7e8..."
            }
        """
        exclude_patterns = exclude_patterns or []
        manifest = {}
        
        for item in dir_path.rglob("*"):
            # Skip if not a file
            if not item.is_file():
                continue
            
            # Skip excluded patterns
            if any(pattern in str(item) for pattern in exclude_patterns):
                continue
            
            # Get relative path
            rel_path = str(item.relative_to(dir_path))
            
            # Store file and get hash
            file_hash = self.store_file(item)
            manifest[rel_path] = file_hash
        
        return manifest
    
    def save_snapshot(self, run_id: str, iteration: int, step_name: str, 
                     manifest: Dict[str, str], metadata: Dict):
        """
        Save a snapshot manifest.
        
        The manifest is just a JSON file mapping paths to hashes.
        It's tiny (few KB), even if the workspace is huge.
        """
        snapshot_dir = self.snapshots_dir / run_id / f"iteration_{iteration}"
        snapshot_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        
        # Ensure parent directories have correct permissions
        run_dir = self.snapshots_dir / run_id
        if run_dir.exists():
            run_dir.chmod(0o755)
        snapshot_dir.chmod(0o755)
        
        snapshot_file = snapshot_dir / f"{step_name}.json"
        
        snapshot_data = {
            "manifest": manifest,
            "metadata": metadata
        }
        
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
        
        # Make snapshot file readable
        snapshot_file.chmod(0o644)
    
    def load_snapshot(self, run_id: str, iteration: int, step_name: str) -> tuple[Dict[str, str], Dict]:
        """Load a snapshot manifest"""
        snapshot_file = self.snapshots_dir / run_id / f"iteration_{iteration}" / f"{step_name}.json"
        
        if not snapshot_file.exists():
            raise ValueError(f"Snapshot not found: {snapshot_file}")
        
        with open(snapshot_file, 'r') as f:
            snapshot_data = json.load(f)
        
        return snapshot_data["manifest"], snapshot_data["metadata"]
    
    def restore_file(self, file_hash: str, dest_path: Path):
        """Restore a file from the object store to a destination"""
        obj_path = self.objects_dir / file_hash[:2] / file_hash[2:]
        
        if not obj_path.exists():
            raise ValueError(f"Object not found: {file_hash}")
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try copy-on-write, fallback to regular copy
        if not self._try_cow_copy(obj_path, dest_path):
            shutil.copy2(obj_path, dest_path)
    
    def _try_cow_copy(self, src: Path, dest: Path) -> bool:
        """Try copy-on-write, return True if successful"""
        try:
            if sys.platform == 'darwin':
                # macOS APFS clonefile
                subprocess.run(
                    ['cp', '-c', str(src), str(dest)],
                    check=True,
                    capture_output=True
                )
                return True
            elif sys.platform == 'linux':
                # Linux reflink (XFS, BTRFS)
                result = subprocess.run(
                    ['cp', '--reflink=always', str(src), str(dest)],
                    capture_output=True
                )
                return result.returncode == 0
        except:
            pass
        
        return False
    
    def get_storage_stats(self) -> Dict:
        """Get statistics about storage usage"""
        total_objects = sum(1 for _ in self.objects_dir.rglob("*") if _.is_file())
        total_size = sum(f.stat().st_size for f in self.objects_dir.rglob("*") if f.is_file())
        total_snapshots = sum(1 for _ in self.snapshots_dir.rglob("*.json"))
        
        return {
            "total_objects": total_objects,
            "total_size_mb": total_size / (1024 * 1024),
            "total_snapshots": total_snapshots,
        }

