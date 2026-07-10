"""Audio I/O and dry/wet mixing helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def mix(dry: np.ndarray, wet: np.ndarray, wet_percent: float) -> np.ndarray:
    """Blend dry and wet signals where the VOX mix value is the wet percentage."""
    wet_ratio = clamp(wet_percent / 100.0, 0.0, 1.0)
    return dry * (1.0 - wet_ratio) + wet * wet_ratio


def decode_audio(path: Path, sample_rate: int, channels: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio.reshape((-1, channels)).copy()


def encode_audio(path: Path, audio: np.ndarray, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "-",
        str(path),
    ]
    subprocess.run(command, input=np.asarray(audio, dtype=np.float32).tobytes(), check=True)


def overlay_audio(output: np.ndarray, overlay: np.ndarray, start_sample: int, volume: float) -> None:
    """Mix an audio clip into an output buffer, clipping it to the buffer bounds."""
    output_start = max(0, start_sample)
    overlay_start = max(0, -start_sample)
    if output_start >= len(output) or overlay_start >= len(overlay):
        return
    sample_count = min(len(output) - output_start, len(overlay) - overlay_start)
    if sample_count <= 0:
        return
    output[output_start : output_start + sample_count] += overlay[overlay_start : overlay_start + sample_count] * volume


# Keep the former helper names available to callers during the package move.
_clamp = clamp
_mix = mix
_decode_audio = decode_audio
_encode_audio = encode_audio
