from __future__ import annotations

import sys


class TrainingReporter:
    """Emit structured training updates.

    Use `report_epoch()` when the trainer naturally produces epoch summaries,
    use `report_batch()` only when the chosen training API already exposes a
    true batch loop or batch callback, and call `report_unavailable()` when
    the chosen training API exposes no real epoch or batch progress hooks.
    If you use report_unavailable you must call `report_epoch()` at the end of training to report final metrics.
    """

    def __init__(self) -> None:
        self._batch_report_interval = 50
        self._current_epoch: int | None = None
        self._batches_seen_in_epoch = 0
        self._batch_loss_total_since_report = 0.0
        self._batches_seen_since_report = 0

    def report_unavailable(self, reason: str) -> None:
        """Emit that no meaningful intermediate progress is available.

        If you use this function you must call report_epoch() at the end of training to report final metrics.

        Args:
            reason: Short concrete reason, ideally one sentence.
        """
        self._emit("unavailable", reason=reason.strip())

    def report_batch(self, epoch: int, batch: int, train_loss: float) -> None:
        """Accumulate batch losses and periodically emit a recent-batch summary.

        Args:
            epoch: Current epoch number.
            batch: Current batch index within the epoch.
            train_loss: Mean loss for this batch.
        """
        if self._current_epoch != epoch:
            self._current_epoch = epoch
            self._batches_seen_in_epoch = 0
            self._batch_loss_total_since_report = 0.0
            self._batches_seen_since_report = 0

        self._batches_seen_in_epoch += 1
        self._batch_loss_total_since_report += float(train_loss)
        self._batches_seen_since_report += 1

        if self._batches_seen_in_epoch % self._batch_report_interval == 0:
            average_batch_loss = self._batch_loss_total_since_report / self._batches_seen_since_report
            self._emit(
                "batch",
                epoch=epoch,
                batch=batch,
                train_loss=average_batch_loss,
            )
            self._batch_loss_total_since_report = 0.0
            self._batches_seen_since_report = 0

    def report_epoch(
        self,
        epoch: int,
        train_loss: float | None = None,
        val_loss: float | None = None,
        val_metric_name: str | None = None,
        val_metric: float | None = None,
        early_stopping_patience_remaining: int | None = None,
    ) -> None:
        """Emit an epoch summary.

        Args:
            epoch: Current epoch number for the summary being reported.
            train_loss: Optional epoch training loss.
            val_loss: Optional epoch validation loss.
            val_metric_name: Optional label for the run's main validation
                metric, such as "AUROC" or "MAE". Supply together with val_metric.
            val_metric: Optional value for val_metric_name.
                Supply together with val_metric_name.
            early_stopping_patience_remaining: Optional not-improving epochs remaining before
                early stopping triggers.
        """
        self._emit(
            "epoch",
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metric_name=val_metric_name,
            val_metric=val_metric,
            early_stopping_patience_remaining=early_stopping_patience_remaining,
        )

    def _emit(self, event: str, **fields: object) -> None:
        parts = [f"{k}={f'{v:.4f}' if isinstance(v, float) else v}" for k, v in fields.items() if v is not None]
        print(f"\nTRAINING_REPORT ({event}): \033[1m{'  '.join(parts)}\033[0m")
        sys.stdout.flush()
