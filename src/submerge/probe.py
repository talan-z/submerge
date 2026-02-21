"""Subtitle track inspection in video files."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Codecs de sous-titres texte supportés
TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text"}
# Codecs de sous-titres image (non supportés)
IMAGE_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvdsub", "pgssub"}


class NoSubtitleTracksError(Exception):
    """No text subtitle tracks found."""


class ProbeError(Exception):
    """Error during video file inspection."""


@dataclass
class SubtitleTrack:
    """Represents a subtitle track."""

    index: int
    codec: str
    language: str | None
    title: str | None
    is_forced: bool
    is_default: bool
    is_text: bool

    @property
    def display_name(self) -> str:
        """Display name of the track."""
        parts = [f"#{self.index}"]
        if self.language:
            parts.append(f"[{self.language}]")
        if self.title:
            parts.append(self.title)
        parts.append(f"({self.codec})")
        flags = []
        if self.is_default:
            flags.append("default")
        if self.is_forced:
            flags.append("forced")
        if flags:
            parts.append(f"[{', '.join(flags)}]")
        return " ".join(parts)


def list_subtitle_tracks(video_path: str | Path) -> list[SubtitleTrack]:
    """List subtitle tracks from a video file.

    Args:
        video_path: Path to video file (MKV, MP4, etc.)

    Returns:
        List of subtitle tracks found

    Raises:
        ProbeError: If ffprobe fails
        NoSubtitleTracksError: If no text subtitle tracks found
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise ProbeError(f"File not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "s",
        str(video_path),
    ]

    logger.debug(f"Executing: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise ProbeError("ffprobe timeout - file may be corrupted")
    except subprocess.CalledProcessError as e:
        raise ProbeError(f"ffprobe failed: {e.stderr}") from e
    except FileNotFoundError:
        raise ProbeError("ffprobe not found. Install ffmpeg: brew install ffmpeg")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise ProbeError(f"JSON parsing error from ffprobe: {e}") from e

    streams = data.get("streams", [])
    if not streams:
        raise NoSubtitleTracksError(
            f"No subtitle tracks found in {video_path.name}"
        )

    tracks: list[SubtitleTrack] = []
    image_tracks_found = False

    for stream in streams:
        codec = stream.get("codec_name", "unknown")
        tags = stream.get("tags", {})
        disposition = stream.get("disposition", {})

        is_text = codec.lower() in TEXT_CODECS
        if codec.lower() in IMAGE_CODECS:
            image_tracks_found = True

        track = SubtitleTrack(
            index=stream.get("index", 0),
            codec=codec,
            language=tags.get("language"),
            title=tags.get("title"),
            is_forced=disposition.get("forced", 0) == 1,
            is_default=disposition.get("default", 0) == 1,
            is_text=is_text,
        )
        tracks.append(track)

    if image_tracks_found:
        logger.warning(
            "Image-based subtitle tracks detected (PGS/VOBSUB). "
            "These formats are not supported and will be ignored."
        )

    text_tracks = [t for t in tracks if t.is_text]
    if not text_tracks:
        raise NoSubtitleTracksError(
            f"No text subtitle tracks found in {video_path.name}. "
            "Only image-based subtitles (PGS/VOBSUB) were detected."
        )

    # Warning if multiple EN tracks
    en_tracks = [t for t in text_tracks if t.language and t.language.lower() in ("eng", "en")]
    if len(en_tracks) > 1:
        logger.warning(
            f"Multiple English tracks detected ({len(en_tracks)}). "
            "Use --track to specify which one to use."
        )

    return tracks


def find_track_by_language(
    tracks: list[SubtitleTrack], language: str
) -> SubtitleTrack | None:
    """Find the first text track matching a language.

    Args:
        tracks: List of tracks
        language: Language code (en, eng, fr, fra, pl, pol, etc.)

    Returns:
        The found track or None
    """
    language = language.lower()
    for track in tracks:
        if not track.is_text:
            continue
        if track.language:
            track_lang = track.language.lower()
            # Match exact code (eng == eng)
            if track_lang == language:
                return track
            # Match 2-char to 3-char (en -> eng) only if lengths are 2 and 3
            if len(language) == 2 and len(track_lang) == 3 and track_lang.startswith(language):
                return track
            # Dirty fix for German (common legacy tag)
            if language == "de" and track_lang in {"ger", "deu"}:
                return track
            # Match 3-char to 2-char (eng -> en) only if lengths are 3 and 2
            if len(language) == 3 and len(track_lang) == 2 and language.startswith(track_lang):
                return track
    return None
