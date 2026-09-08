"""Prepare gated OtoSpeech WebDataset shards for Moshi fine-tuning."""

import argparse
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    import torch


DATASET_ID = "otoearth/otoSpeech-full-duplex-processed-141h"
Alignment = list[list[object]]
TranscribeChannel = Callable[[Any, int], Alignment]


@dataclass(frozen=True)
class PreparationResult:
    prepared: int
    rejected: int
    duration_sec: float


def _load_audio(content: bytes, suffix: str) -> tuple[Any, int]:
    import torchaudio

    return torchaudio.load(io.BytesIO(content), format=suffix.removeprefix("."))


def _save_audio(path: Path, waveform: Any, sample_rate: int) -> None:
    import torchaudio

    torchaudio.save(path, waveform, sample_rate, format="wav")


def _valid_alignments(value: object) -> Alignment | None:
    if not isinstance(value, list):
        return None
    alignments: Alignment = []
    for alignment in value:
        if (
            not isinstance(alignment, list)
            or len(alignment) != 3
            or not isinstance(alignment[0], str)
            or not isinstance(alignment[1], list)
            or len(alignment[1]) != 2
            or not all(isinstance(timestamp, (int, float)) for timestamp in alignment[1])
            or alignment[1][0] >= alignment[1][1]
            or not isinstance(alignment[2], str)
        ):
            return None
        alignments.append(alignment)
    return alignments


def _read_alignments(
    archive: tarfile.TarFile, audio_member: tarfile.TarInfo
) -> Alignment | None:
    metadata_name = str(Path(audio_member.name).with_suffix(".json"))
    try:
        metadata_file = archive.extractfile(metadata_name)
    except KeyError:
        return None
    if metadata_file is None:
        return None
    try:
        metadata = json.load(metadata_file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    return _valid_alignments(metadata.get("alignments"))


def _make_sample_name(audio_member: tarfile.TarInfo, used_names: set[str]) -> str:
    stem = Path(audio_member.name).stem
    name = stem
    suffix = 2
    while name in used_names:
        name = f"{stem}_{suffix}"
        suffix += 1
    used_names.add(name)
    return name


def _default_transcriber(model_name: str) -> TranscribeChannel:
    import torch
    import whisper_timestamped as whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)

    def transcribe_channel(wav: Any, sample_rate: int) -> Alignment:
        if sample_rate != 16_000:
            import torchaudio.functional as audio_functional

            wav = audio_functional.resample(wav, sample_rate, 16_000)
            sample_rate = 16_000
        result = whisper.transcribe(
            model,
            wav.squeeze(0).cpu().numpy(),
            language="en",
            vad="auditok",
            verbose=None,
        )
        alignments: Alignment = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                start, end = word.get("start"), word.get("end")
                text = word.get("text")
                if (
                    isinstance(text, str)
                    and isinstance(start, (int, float))
                    and isinstance(end, (int, float))
                ):
                    alignments.append([text, [start, end], "SPEAKER_MAIN"])
        return alignments

    return transcribe_channel


def _write_rejection(rejections, member_name: str, error: Exception) -> None:
    json.dump({"path": member_name, "error": str(error)}, rejections)
    rejections.write("\n")
    rejections.flush()


def prepare_archives(
    archives: Iterable[Path],
    output_dir: Path,
    *,
    max_samples: int | None = None,
    max_hours: float | None = None,
    moshi_channel: int = 0,
    transcribe_channel: TranscribeChannel | None = None,
) -> PreparationResult:
    """Convert stereo audio in WebDataset tar archives to Moshi's local format."""
    if max_samples is None and max_hours is None:
        raise ValueError("Specify --max-samples or --max-hours to bound preparation.")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive.")
    if max_hours is not None and max_hours <= 0:
        raise ValueError("max_hours must be positive.")
    if moshi_channel not in (0, 1):
        raise ValueError("moshi_channel must be 0 or 1.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir()
    prepared = rejected = 0
    duration_sec = 0.0
    used_names: set[str] = set()
    transcriber = transcribe_channel

    with (output_dir / "train.jsonl").open("w") as manifest, (
        output_dir / "rejected.jsonl"
    ).open("w") as rejections:
        for archive_path in archives:
            with tarfile.open(archive_path) as archive:
                for member in archive:
                    if (
                        not member.isfile()
                        or Path(member.name).suffix.lower() not in {".flac", ".wav"}
                    ):
                        continue
                    if max_samples is not None and prepared >= max_samples:
                        return PreparationResult(prepared, rejected, duration_sec)
                    if max_hours is not None and duration_sec >= max_hours * 3600:
                        return PreparationResult(prepared, rejected, duration_sec)
                    try:
                        source = archive.extractfile(member)
                        if source is None:
                            raise ValueError("Could not read audio member.")
                        waveform, sample_rate = _load_audio(
                            source.read(), Path(member.name).suffix
                        )
                        if waveform.ndim != 2 or waveform.shape[0] != 2:
                            raise ValueError("Moshi training requires stereo audio.")
                        sample_duration = waveform.shape[-1] / sample_rate
                        if (
                            max_hours is not None
                            and duration_sec + sample_duration > max_hours * 3600
                        ):
                            return PreparationResult(prepared, rejected, duration_sec)
                        alignments = _read_alignments(archive, member)
                        if alignments is None:
                            if transcriber is None:
                                raise RuntimeError(
                                    "No transcript available and ASR is disabled."
                                )
                            alignments = transcriber(
                                waveform[moshi_channel : moshi_channel + 1], sample_rate
                            )
                        name = _make_sample_name(member, used_names)
                        wav_path = audio_dir / f"{name}.wav"
                        _save_audio(wav_path, waveform, sample_rate)
                        (audio_dir / f"{name}.json").write_text(
                            json.dumps({"alignments": alignments}, ensure_ascii=False)
                        )
                        json.dump(
                            {"path": f"audio/{name}.wav", "duration": sample_duration},
                            manifest,
                        )
                        manifest.write("\n")
                        manifest.flush()
                        prepared += 1
                        duration_sec += sample_duration
                    except Exception as error:
                        _write_rejection(rejections, member.name, error)
                        rejected += 1
    return PreparationResult(prepared, rejected, duration_sec)


def download_and_prepare(args: argparse.Namespace) -> PreparationResult:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the gated OtoSpeech dataset.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    files = api.list_repo_files(DATASET_ID, repo_type="dataset")
    shard_names = sorted(
        name
        for name in files
        if name.startswith("data/train/") and name.endswith(".tar")
    )
    if not shard_names:
        raise RuntimeError("No OtoSpeech training shards were found.")

    def archives() -> Iterable[Path]:
        for shard_name in shard_names:
            yield Path(
                hf_hub_download(
                    DATASET_ID,
                    shard_name,
                    repo_type="dataset",
                    token=token,
                    cache_dir=args.cache_dir,
                )
            )

    transcriber: TranscribeChannel | None = None

    def transcribe_channel(wav: Any, sample_rate: int) -> Alignment:
        nonlocal transcriber
        if transcriber is None:
            transcriber = _default_transcriber(args.asr_model)
        assert transcriber is not None
        return transcriber(wav, sample_rate)

    result = prepare_archives(
        archives(),
        args.output_dir,
        max_samples=args.max_samples,
        max_hours=args.max_hours,
        moshi_channel=args.moshi_channel,
        transcribe_channel=transcribe_channel,
    )
    if result.prepared == 0:
        raise RuntimeError("No valid OtoSpeech samples were prepared; see rejected.jsonl.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    limit = parser.add_mutually_exclusive_group(required=True)
    limit.add_argument("--max-samples", type=int)
    limit.add_argument("--max-hours", type=float)
    parser.add_argument("--asr-model", default="medium")
    parser.add_argument("--moshi-channel", type=int, choices=(0, 1), default=0)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    result = download_and_prepare(args)
    print(
        f"Prepared {result.prepared} samples ({result.duration_sec / 3600:.2f} h); "
        f"rejected {result.rejected}. Dataset: {args.output_dir / 'train.jsonl'}"
    )


if __name__ == "__main__":
    main()
