"""Fail-fast resolver for the ffmpeg binary the ingest stage needs.

``ingest.py`` decodes 1080p frames straight into numpy via ffmpeg's stdout
(build brief §5.12 — never accumulate per-frame PNGs to disk), so a usable
ffmpeg must exist before any download. This helper resolves it **once** and
fails fast with a clear, actionable message if it is absent (brief §6):

1. a system ``ffmpeg`` on ``PATH`` (``brew install ffmpeg``), else
2. the portable binary shipped by the optional ``imageio-ffmpeg`` wheel
   (``pip install imageio-ffmpeg``).

CPU-only, no ``gui/`` or training-path imports.
"""

from __future__ import annotations

import shutil
from pathlib import Path


class FFmpegNotFoundError(RuntimeError):
    """Raised when no usable ffmpeg binary can be resolved."""


def resolve_ffmpeg() -> str:
    """Return an absolute path to a usable ffmpeg binary, or fail fast.

    Tries the system ``ffmpeg`` on ``PATH`` first, then the portable binary
    from ``imageio-ffmpeg``. Raises :class:`FFmpegNotFoundError` with install
    guidance if neither is available.
    """
    system = shutil.which("ffmpeg")
    if system is not None:
        return system

    try:
        import imageio_ffmpeg
    except ImportError:
        raise FFmpegNotFoundError(
            "ffmpeg not found. Install a system ffmpeg (`brew install ffmpeg`) "
            "or the portable fallback (`pip install imageio-ffmpeg`)."
        ) from None

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def resolve_ffprobe() -> str:
    """Return an absolute path to a usable ``ffprobe`` binary, or fail fast.

    ``ffprobe`` is needed to read a downloaded stream's *true* native resolution
    before decoding (the resolution firewall — brief §2 / FIX-5 — must key off the
    honest source height, never an upscaled buffer). Resolution order:

    1. a system ``ffprobe`` on ``PATH`` (ships alongside ``brew install ffmpeg``);
    2. the ``ffprobe`` binary sitting next to a resolved ``ffmpeg`` (both a system
       ffmpeg dir and the ``imageio-ffmpeg`` wheel co-locate the two).

    Raises :class:`FFmpegNotFoundError` with install guidance if neither exists.
    """
    system = shutil.which("ffprobe")
    if system is not None:
        return system

    ffmpeg_path = Path(resolve_ffmpeg())
    sibling = ffmpeg_path.with_name(ffmpeg_path.name.replace("ffmpeg", "ffprobe"))
    if sibling.exists() and sibling != ffmpeg_path:
        return str(sibling)

    raise FFmpegNotFoundError(
        "ffprobe not found (needed to read a stream's native resolution before "
        "decoding). Install a system ffmpeg (`brew install ffmpeg`, which ships "
        "ffprobe) so the resolution firewall can key off the honest source height."
    )
