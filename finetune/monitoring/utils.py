import datetime
import logging
import sys
import time
from collections.abc import Callable
from typing import Any

from tqdm import tqdm


class DeltaTimeFormatter(logging.Formatter):
    def format(self, record):
        delta = datetime.timedelta(
            seconds=int(record.relativeCreated / 1000)
        )  # no milliseconds
        record.delta = delta
        return super().format(record)


def format_parameter_summary(total: int, trainable: int) -> str:
    frozen = total - trainable
    trainable_percent = 100 * trainable / total if total else 0.0
    return (
        "Model parameters: "
        f"total={total:,}, trainable={trainable:,}, frozen={frozen:,} "
        f"({trainable_percent:.2f}% trainable)."
    )


class TrainingProgress:
    """Rank-zero terminal progress for either exact epochs or training steps."""

    def __init__(
        self,
        max_steps: int,
        epoch_mode: bool,
        make_bar: Callable[..., Any] = tqdm,
    ) -> None:
        self.max_steps = max_steps
        self.epoch_mode = epoch_mode
        self.make_bar = make_bar
        self.bar: Any | None = None

        if not self.epoch_mode:
            self.bar = self.make_bar(desc="Training", total=max_steps, unit="step")

    def start_epoch(self, epoch: int, total: int) -> None:
        if not self.epoch_mode:
            return
        self.close()
        self.bar = self.make_bar(desc=f"Epoch {epoch}", total=total, unit="sample")

    def advance_epoch(self) -> None:
        if self.epoch_mode and self.bar is not None:
            self.bar.update()

    def update_step(
        self, step: int, loss: float, lr: float, eta_seconds: float
    ) -> None:
        if self.bar is None:
            return
        if not self.epoch_mode:
            self.bar.update()
        self.bar.set_postfix(
            step=f"{step}/{self.max_steps}",
            loss=f"{loss:.3f}",
            lr=f"{lr:.1e}",
            eta=self._format_eta(eta_seconds),
        )

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    @staticmethod
    def _format_eta(eta_seconds: float) -> str:
        total_seconds = max(0, round(eta_seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"


def set_logger(level: int = logging.INFO):
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    tz, *_ = time.tzname

    LOGFORMAT = "%(asctime)s - %(delta)s - %(name)s - %(levelname)s - %(message)s"
    TIMEFORMAT = f"%Y-%m-%d %H:%M:%S ({tz})"
    formatter = DeltaTimeFormatter(LOGFORMAT, TIMEFORMAT)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    root.addHandler(handler)
