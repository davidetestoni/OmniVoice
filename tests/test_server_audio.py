import numpy as np
import torch

from server import _audio_to_pcm16le_bytes, _audio_to_wav_bytes


def test_audio_to_pcm16le_bytes_accepts_numpy_output() -> None:
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)

    pcm = _audio_to_pcm16le_bytes(audio)

    assert pcm == np.array([0, 16383, -16383], dtype=np.int16).tobytes()


def test_audio_to_pcm16le_bytes_accepts_torch_output() -> None:
    audio = torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float32)

    pcm = _audio_to_pcm16le_bytes(audio)

    assert pcm == np.array([0, 32767, -32767], dtype=np.int16).tobytes()


def test_audio_to_wav_bytes_accepts_numpy_output() -> None:
    audio = np.zeros(240, dtype=np.float32)

    wav = _audio_to_wav_bytes(audio, 24000)

    assert wav[:4] == b"RIFF"
