import json
from pathlib import Path

from finetune.data.dataset import count_samples_per_epoch
from finetune.monitoring.utils import TrainingProgress, format_parameter_summary


def write_jsonl(path: Path, durations: list[float]) -> None:
    path.write_text(
        "".join(
            json.dumps({"path": f"audio-{index}.wav", "duration": duration})
            + "\n"
            for index, duration in enumerate(durations)
        )
    )


def test_count_samples_per_epoch_matches_segment_sharding(tmp_path):
    data_file = tmp_path / "train.jsonl"
    write_jsonl(data_file, [2.0, 5.0, 0.0])

    assert count_samples_per_epoch(data_file, duration_sec=2.0, rank=0, world_size=2) == 2
    assert count_samples_per_epoch(data_file, duration_sec=2.0, rank=1, world_size=2) == 2


def test_training_progress_uses_epoch_bar_and_metrics_postfix():
    bars = []

    class FakeBar:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updated = []
            self.postfix = None
            self.closed = False

        def update(self, amount=1):
            self.updated.append(amount)

        def set_postfix(self, **kwargs):
            self.postfix = kwargs

        def close(self):
            self.closed = True

    def make_bar(**kwargs):
        bar = FakeBar(**kwargs)
        bars.append(bar)
        return bar

    progress = TrainingProgress(max_steps=10, epoch_mode=True, make_bar=make_bar)
    progress.start_epoch(epoch=2, total=3)
    progress.advance_epoch()
    progress.update_step(step=4, loss=1.25, lr=2e-6, eta_seconds=90)
    progress.close()

    assert bars[0].kwargs == {"desc": "Epoch 2", "total": 3, "unit": "sample"}
    assert bars[0].updated == [1]
    assert bars[0].postfix == {"step": "4/10", "loss": "1.250", "lr": "2.0e-06", "eta": "00:01:30"}
    assert bars[0].closed


def test_training_progress_falls_back_to_step_bar():
    bars = []

    class FakeBar:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updated = []
            self.postfix = None

        def update(self, amount=1):
            self.updated.append(amount)

        def set_postfix(self, **kwargs):
            self.postfix = kwargs

        def close(self):
            pass

    def make_bar(**kwargs):
        bar = FakeBar(**kwargs)
        bars.append(bar)
        return bar

    progress = TrainingProgress(max_steps=2, epoch_mode=False, make_bar=make_bar)
    progress.update_step(step=1, loss=0.5, lr=1e-4, eta_seconds=0)

    assert bars[0].kwargs == {"desc": "Training", "total": 2, "unit": "step"}
    assert bars[0].updated == [1]
    assert bars[0].postfix["step"] == "1/2"


def test_parameter_summary_reports_total_trainable_and_frozen():
    assert (
        format_parameter_summary(total=5, trainable=3)
        == "Model parameters: total=5, trainable=3, frozen=2 (60.00% trainable)."
    )
