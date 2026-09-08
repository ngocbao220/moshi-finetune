import itertools
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np
import sphn
import torch.distributed as dist

from finetune.distributed import get_rank

from .interleaver import InterleavedTokenizer, Sample

logger = logging.getLogger("dataset")


AudioChunkPath = tuple[str, float]
_LOADED_DATASETS: dict[Path, list[AudioChunkPath]] = {}


class EpochProgress(Protocol):
    def start_epoch(self, epoch: int, total: int) -> None: ...

    def advance_epoch(self) -> None: ...


def main_logger_info(message: str) -> None:
    if dist.is_initialized() and get_rank() == 0:
        logger.info(message)


def load_file(path: Path, world_size: int, rank: int) -> list[str]:
    lines = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if not idx % world_size == rank:
                continue
            lines.append(line)
    return lines


def maybe_load_local_dataset(
    path: Path, rank: int, world_size: int, instruct_tokenizer: InterleavedTokenizer
) -> list[AudioChunkPath]:
    if path in _LOADED_DATASETS:
        return _LOADED_DATASETS[path]

    duration = instruct_tokenizer.duration_sec
    main_logger_info(f"Loading {path} ...")
    lines: list[str] = load_file(path, rank=rank, world_size=world_size)

    chunks: list[AudioChunkPath] = []
    for line in lines:
        data = json.loads(line)
        start_sec = 0
        while start_sec < data["duration"]:
            chunks.append((data["path"], start_sec))
            start_sec += duration

    main_logger_info(f"{path} loaded and chunked.")
    _LOADED_DATASETS[path] = chunks

    return _LOADED_DATASETS[path]


@dataclass
class DataDir:
    path: Path

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        jsonl_files = list(self.path.rglob("*jsonl"))
        assert len(jsonl_files) > 0, (
            f"{self.path} does not seem to have any files ending with '.jsonl'"
        )
        return jsonl_files


@dataclass
class DataFile:
    path: Path

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        return [self.path]


def parse_data_sources(
    pretrain_data: str,
) -> tuple[list[DataDir | DataFile], list[float]]:
    seen: set[str] = set()
    sources: list[DataDir | DataFile] = []
    weights: list[float] = []

    sample_sources = pretrain_data

    for source in sample_sources.strip().split(","):
        if not source:
            continue

        source_items = source.strip().split(":")
        if len(source_items) == 1:
            path_ = source_items[0]
            weight = 1.0
        elif len(source_items) == 2:
            path_, weight_ = source_items
            weight = float(weight_)
        else:
            raise ValueError(
                f"{source} is not correctly formatted. Make sure to format each data source "
                "as <path/to/data>:<weight> or just <path/to/data>"
            )

        assert path_ not in seen, (
            f"{path_} seems to be duplicated. Make sure to only add it once."
        )
        assert weight > 0, (
            f"Make sure to define strictly positive data sampling weights, not {weight}"
        )

        data: DataDir | DataFile
        if Path(path_).is_dir():
            data = DataDir(path=Path(path_))
        elif Path(path_).is_file():
            data = DataFile(path=Path(path_))
        else:
            raise FileNotFoundError(
                f"The path {path_} does not exist. Make sure {path_} is either a file or directory "
                "that contains training data."
            )

        sources.append(data)
        weights.append(weight)

        seen.add(path_)

    sum_weights = sum(weights)
    n_weights = [weight / sum_weights for weight in weights]

    assert min(n_weights) > 0
    assert abs(1 - sum(n_weights)) < 1e-8, (
        f"Defined data sampling weights {weights} must sum to 1."
    )
    return sources, n_weights


def has_exact_epoch_progress(pretrain_data: str) -> bool:
    sources, _ = parse_data_sources(pretrain_data)
    return len(sources) == 1


def count_samples_per_epoch(
    jsonl_file: Path, duration_sec: float, rank: int, world_size: int
) -> int:
    """Count this rank's padded segments using sphn's sequential sharding rule."""
    sample_index = 0
    count = 0
    with jsonl_file.open() as file:
        for line in file:
            duration = json.loads(line)["duration"]
            num_segments = math.ceil(duration / duration_sec)
            for _ in range(num_segments):
                if sample_index % world_size == rank:
                    count += 1
                sample_index += 1
    return count


def build_dataset(
    pretrain_data: str,
    instruct_tokenizer: InterleavedTokenizer,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
    shuffle_pretrain: bool = False,
    epoch_progress: EpochProgress | None = None,
) -> Iterator[Sample]:
    sources, probabilities = parse_data_sources(pretrain_data=pretrain_data)

    shuffle = not is_eval and shuffle_pretrain

    if len(sources) != 1:
        epoch_progress = None

    dataset_iterators = [
        get_dataset_iterator(
            source,
            instruct_tokenizer=instruct_tokenizer,
            rank=rank,
            world_size=world_size,
            is_finite=is_eval,
            seed=seed,
            shuffle_at_epoch=shuffle,
            epoch_progress=epoch_progress,
        )
        for source in sources
    ]

    if is_eval:
        combined_iterator = itertools.chain.from_iterable(dataset_iterators)
    else:
        # make sure random_seed is different per rank and original seed
        random_seed = np.array((seed, rank))
        rng = np.random.RandomState(seed=random_seed)
        combined_iterator = interleave_iterators(
            dataset_iterators, probabilities=probabilities, rng=rng
        )

    return combined_iterator


def get_rng(seed: int, rank: int) -> np.random.RandomState:
    random_seed = np.array((seed, rank))
    rng = np.random.RandomState(seed=random_seed)
    return rng


def get_dataset_iterator(
    source: DataDir | DataFile,
    instruct_tokenizer: InterleavedTokenizer,
    rank: int,
    world_size: int,
    is_finite: bool,
    seed: int | None,
    shuffle_at_epoch: bool,
    epoch_progress: EpochProgress | None = None,
) -> Iterator[Sample]:
    epoch = 1
    while True:
        jsonl_files = source.jsonl_files
        if epoch_progress is not None:
            epoch_progress.start_epoch(
                epoch,
                sum(
                    count_samples_per_epoch(
                        jsonl_file,
                        duration_sec=instruct_tokenizer.duration_sec,
                        rank=rank,
                        world_size=world_size,
                    )
                    for jsonl_file in jsonl_files
                ),
            )
        for jsonl_file in jsonl_files:
            dataset = sphn.dataset_jsonl(
                str(jsonl_file),
                duration_sec=instruct_tokenizer.duration_sec,
                num_threads=4,
                sample_rate=instruct_tokenizer.mimi.sample_rate,
                pad_last_segment=True,
            )
            if shuffle_at_epoch:
                dataset = dataset.shuffle(
                    with_replacement=False, skip=rank, step_by=world_size, seed=seed
                )
                seed += 1
            else:
                dataset = dataset.seq(skip=rank, step_by=world_size)
            for sample in dataset:
                wav = sample["data"][..., : sample["unpadded_len"]]
                tokenized_sample = instruct_tokenizer(
                    wav, sample["start_time_sec"], sample["path"]
                )
                if epoch_progress is not None:
                    epoch_progress.advance_epoch()
                yield tokenized_sample
        if is_finite:
            break
        main_logger_info(f"Finished epoch {epoch}")
        epoch += 1


def interleave_iterators(iterators: list[Iterator], probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        yield next(iterators[it_id])
