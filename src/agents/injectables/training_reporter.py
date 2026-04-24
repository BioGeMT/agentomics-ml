from __future__ import annotations

import json
import sys


class TrainingReporter:
    """Emit structured training updates.

    Use `report_epoch()` when the trainer naturally produces epoch summaries,
    use `report_batch()` only when the chosen training API already exposes a
    true batch loop or batch callback, and call `report_unavailable()` once
    when the chosen training API exposes no real epoch or batch progress hooks.
    """

    def __init__(self) -> None:
        self._batch_report_interval = 50
        self._current_epoch: int | None = None
        self._batches_seen_in_epoch = 0
        self._batch_loss_total_since_report = 0.0
        self._batches_seen_since_report = 0

    def report_unavailable(self, reason: str) -> None:
        """Emit that no meaningful intermediate progress is available.

        Args:
            reason: Short concrete reason, ideally one sentence.
        """
        self._emit("unavailable", reason=reason.strip())

    def report_batch(self, epoch: int, batch: int, train_loss: float) -> None:
        """Accumulate batch losses and periodically emit a recent-batch summary.

        Args:
            epoch: Current epoch number. Be consistent about 0-based vs 1-based.
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
        validation_loss: float | None = None,
        validation_metric_name: str | None = None,
        validation_metric: float | None = None,
        early_stopping_patience_remaining: int | None = None,
    ) -> None:
        """Emit an epoch summary.

        Args:
            epoch: Current epoch number for the summary being reported.
            train_loss: Optional epoch training loss. Supply it when you
                can compute it.
            validation_loss: Optional epoch validation loss. Supply it when you
                can compute it.
            validation_metric_name: Optional metric label for the run's main
                validation metric, such as `"AUROC"` or `"MAE"`. Supply it when you 
                can compute it and only together with `validation_metric`.
            validation_metric: Optional value for `validation_metric_name`.
                Supply it only together with `validation_metric_name`.
            early_stopping_patience_remaining: Optional number of epochs
                remaining before early stopping would trigger. Supply it only
                when the training code knows this value.
        """
        self._emit(
            "epoch",
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            validation_metric_name=validation_metric_name,
            validation_metric=validation_metric,
            early_stopping_patience_remaining=early_stopping_patience_remaining,
        )

    def _emit(self, event: str, **fields: object) -> None:
        payload: dict[str, object] = {"event": event}
        for field_name, value in fields.items():
            if value is None:
                continue
            if isinstance(value, float):
                value = round(value, 6)
            payload[field_name] = value
        print(f"TRAINING_REPORT: {json.dumps(payload)}")
        sys.stdout.flush()
