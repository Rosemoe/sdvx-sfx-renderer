"""Compatibility entry point for the relocated SFX renderer.

New code should import :mod:`sfx_renderer` directly.
"""
from sfx_renderer import FXEffects, FXRenderEvent
from sfx_renderer.audio import (
    _clamp,
    _decode_audio,
    _encode_audio,
    _mix,
)
from sfx_renderer.renderer import DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE, main

__all__ = [
    "DEFAULT_CHANNELS",
    "DEFAULT_SAMPLE_RATE",
    "FXEffects",
    "FXRenderEvent",
    "_clamp",
    "_decode_audio",
    "_encode_audio",
    "_mix",
    "main",
]


if __name__ == "__main__":
    main()
