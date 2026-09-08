import io
import json
import sys
import tarfile
from pathlib import Path

import finetune.data.otospeech as otospeech


class FakeWaveform:
    ndim = 2

    def __init__(self, channels: int):
        self.shape = (channels, 16_000)

    def __getitem__(self, item):
        return FakeWaveform(item.stop - item.start)


def test_load_audio_uses_sphn_without_torchaudio(monkeypatch):
    class FakeArray:
        pass

    class FakeSphn:
        @staticmethod
        def read(path):
            assert Path(path).suffix == ".flac"
            assert Path(path).read_bytes() == b"flac"
            return FakeArray(), 16_000

    class FakeTorch:
        @staticmethod
        def from_numpy(audio):
            return audio

    monkeypatch.setitem(sys.modules, "sphn", FakeSphn())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torchaudio", None)

    waveform, sample_rate = otospeech._load_audio(b"flac", ".flac")

    assert isinstance(waveform, FakeArray)
    assert sample_rate == 16_000


def _write_archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_prepare_archives_writes_moshi_dataset_with_asr_fallback(
    tmp_path: Path, monkeypatch
):
    archive = tmp_path / "sample.tar"
    _write_archive(archive, {"conversation.wav": b"stereo"})
    output_dir = tmp_path / "prepared"
    monkeypatch.setattr(
        otospeech,
        "_load_audio",
        lambda content, suffix: (FakeWaveform(2), 16_000),
    )
    monkeypatch.setattr(
        otospeech, "_save_audio", lambda path, wav, rate: path.write_bytes(b"wav")
    )

    def transcribe(wav, sample_rate):
        assert wav.shape[0] == 1
        assert sample_rate == 16_000
        return [["hello", [0.0, 0.5], "SPEAKER_MAIN"]]

    result = otospeech.prepare_archives(
        [archive], output_dir, max_samples=1, transcribe_channel=transcribe
    )

    assert result.prepared == 1
    rows = [json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines()]
    assert rows == [{"path": "audio/conversation.wav", "duration": 1.0}]
    assert json.loads((output_dir / "audio" / "conversation.json").read_text()) == {
        "alignments": [["hello", [0.0, 0.5], "SPEAKER_MAIN"]]
    }
    assert (output_dir / "audio" / "conversation.wav").exists()


def test_prepare_archives_records_mono_audio_as_rejected(
    tmp_path: Path, monkeypatch
):
    archive = tmp_path / "sample.tar"
    _write_archive(archive, {"conversation.wav": b"mono"})
    monkeypatch.setattr(
        otospeech,
        "_load_audio",
        lambda content, suffix: (FakeWaveform(1), 16_000),
    )

    result = otospeech.prepare_archives([archive], tmp_path / "prepared", max_samples=1)

    assert result.prepared == 0
    assert result.rejected == 1
    rejection = json.loads((tmp_path / "prepared" / "rejected.jsonl").read_text())
    assert "stereo" in rejection["error"]
