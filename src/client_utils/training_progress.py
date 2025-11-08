from typing import Dict, Optional
from datetime import datetime

class TrainingProgress:
    def __init__(self):
        self._current_epoch_metrics: Dict[str, float] = {}
        self._batch_count = 0
    
    def report_batch(self, epoch: int, batch: int, metrics: Dict[str, float]):
        """Update running averages for the current epoch (no print)."""
        self._batch_count += 1
        for metric_name, value in metrics.items():
            if metric_name not in self._current_epoch_metrics:
                self._current_epoch_metrics[metric_name] = 0.0
            self._current_epoch_metrics[metric_name] += (
                (value - self._current_epoch_metrics[metric_name]) / self._batch_count
            )
    
    def end_epoch(self, epoch: int):
        """Print final metrics for the epoch and reset counters, with a blank line before and after, and force flush."""
        import sys
        print()  # Blank line before epoch summary
        print(f"Epoch {epoch} completed - Average metrics:", end="")
        for metric_name, value in self._current_epoch_metrics.items():
            print(f" {metric_name}: {value:.4f}", end="")
        print("\n")  # Blank line after epoch summary
        sys.stdout.flush()  # Force flush to output
        self._current_epoch_metrics.clear()
        self._batch_count = 0