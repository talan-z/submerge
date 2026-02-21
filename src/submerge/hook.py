"""Business logic for Bazarr hook for automatic bilingual subtitle generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout

from .config import SubtoolsSettings, get_settings
from .merge import MergeConfig, merge_bilingual
from .probe import list_subtitle_tracks, find_track_by_language, NoSubtitleTracksError
from .extract import extract_subtitles, SubtitleExtractionError

logger = logging.getLogger(__name__)

# Constantes
LOCK_TIMEOUT = 5  # Secondes


class HookError(Exception):
    """Base error for hook."""


class InvalidLanguageError(HookError):
    """Unsupported language."""


class ProcessingError(HookError):
    """Error during processing."""


@dataclass
class HookResult:
    """Hook result."""

    status: str  # "merged", "waiting", "skipped", "already_processing"
    files: list[str] | None = None
    present: list[str] | None = None
    missing: list[str] | None = None
    reason: str | None = None

def is_temp_embedded_subtitle(path: Path | None) -> bool:
    """Return True if this path is an extracted embedded subtitle temp file."""
    return bool(path and path.name.endswith("tmp.srt"))


def cleanup_temp_subtitle(path: Path | None) -> None:
    """Delete the temporary extracted subtitle if it matches our temp naming."""
    if not is_temp_embedded_subtitle(path):
        return

    try:
        if path.exists():
            path.unlink()
            logger.debug(f"Deleted temporary embedded subtitle: {path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp subtitle {path}: {e}")


def validate_lang(lang: str, settings: SubtoolsSettings | None = None) -> str:
    """Validate and normalize a language.

    Args:
        lang: Language code (e.g., fr, pl, en)
        settings: Settings to get required languages

    Returns:
        Normalized lowercase language

    Raises:
        InvalidLanguageError: If language is not in configured pairs
    """
    settings = settings or get_settings()
    lang_lower = lang.lower().strip()
    if lang_lower not in settings.required_langs:
        raise InvalidLanguageError(
            f"Invalid language: {lang}. Must be one of: {', '.join(sorted(settings.required_langs))}"
        )
    return lang_lower


def find_subtitle_path(video_path: Path, lang: str) -> Path | None:
    """Find subtitle file for a language.
    Priority:
      1) embedded text track (extract if needed, write as *.srt.tmp)
      2) external subtitle files (.srt/.ass and hearing-impaired variants)

    Searches in order: .srt, .ass, .hi.srt, .hi.ass

    Args:
        video_path: Path to video file
        lang: Language code (fr, pl, en)

    Returns:
        Path to subtitle file or None if not found
    """
    video_dir = video_path.parent
    video_stem = video_path.stem

    # --- 1) Try embedded text subtitle tracks first ---
    try:
        tracks = list_subtitle_tracks(video_path)
        track = find_track_by_language(tracks, lang)
    except NoSubtitleTracksError:
        track = None
    except Exception as e:
        logger.warning(f"Subtitle probe failed for {video_path}: {e}")
        track = None

    if track and getattr(track, "is_text", False):
        # Use a temp filename that won't be picked up by other tools
        extracted_path = video_dir / f"{video_stem}.{lang}.embedded.tmp.srt"

        # If already extracted and newer than video, reuse it
        try:
            if extracted_path.exists():
                # optional mtime safety: only reuse if it's newer than the video
                if extracted_path.stat().st_mtime >= video_path.stat().st_mtime:
                    logger.debug(f"Using cached embedded subtitle: {extracted_path}")
                    return extracted_path
                # else fall through to re-extract
        except Exception:
            # ignore mtime edge-cases and attempt extraction
            pass

        # Attempt extraction
        try:
            logger.info(f"Extracting embedded subtitle track {track.index} -> {extracted_path}")
            extract_subtitles(video_path, extracted_path, track_index=track.index)
            return extracted_path
        except SubtitleExtractionError as e:
            logger.warning(f"Embedded subtitle extraction failed for {video_path}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error extracting embedded subtitle for {video_path}: {e}")

    # --- 2) Fallback to external subtitle files ---
    patterns = [
        f"{video_stem}.{lang}.srt",
        f"{video_stem}.{lang}.ass",
        f"{video_stem}.{lang}.hi.srt",  # Hearing impaired
        f"{video_stem}.{lang}.hi.ass",
    ]

    for pattern in patterns:
        path = video_dir / pattern
        if path.exists():
            logger.debug(f"Found: {path}")
            return path

    logger.debug(f"Not found: {lang} for {video_stem}")
    return None


def check_all_languages_present(
    video_path: Path,
    settings: SubtoolsSettings | None = None,
) -> dict[str, Path] | None:
    """Check if all required languages are present.

    Args:
        video_path: Path to video file
        settings: Settings to get required languages

    Returns:
        Dict {lang: path} if all languages present, None otherwise
    """
    settings = settings or get_settings()
    result = {}

    for lang in settings.required_langs:
        logger.debug(f"Checking subtitle availability for: {lang}")

        path = find_subtitle_path(video_path, lang)

        if path is None:
            logger.debug(f"Subtitle missing for language: {lang}")
            continue

        result[lang] = path

    # AFTER loop finishes
    if len(result) == len(settings.required_langs):
        return result

    logger.debug(
        f"Not all required subtitles present. "
        f"Found: {list(result.keys())}, "
        f"Required: {settings.required_langs}"
    )

    return None

def get_present_and_missing(
    video_path: Path,
    settings: SubtoolsSettings | None = None,
) -> tuple[list[str], list[str]]:
    """Return present and missing languages.

    Args:
        video_path: Path to video file
        settings: Settings to get required languages

    Returns:
        Tuple (present_languages, missing_languages)
    """
    settings = settings or get_settings()
    present = []
    missing = []
    for lang in sorted(settings.required_langs):
        if find_subtitle_path(video_path, lang):
            present.append(lang)
        else:
            missing.append(lang)
    return present, missing


def get_output_path(video_path: Path, lang_bottom: str, lang_top: str) -> Path:
    """Generate output path for a language pair.

    Args:
        video_path: Path to video file
        lang_bottom: Language displayed at bottom
        lang_top: Language displayed at top

    Returns:
        Path to output .ass file
    """
    return video_path.parent / f"{video_path.stem}.{lang_bottom}-{lang_top}.ass"


def should_skip_existing(
    video_path: Path,
    sub_paths: dict[str, Path],
    settings: SubtoolsSettings | None = None,
) -> bool:
    """Check if .ass files exist and are newer than sources.

    Args:
        video_path: Path to video file
        sub_paths: Dict {lang: path} of source subtitles
        settings: Settings to get configured pairs

    Returns:
        True if all .ass exist and are newer than their sources
    """
    settings = settings or get_settings()

    for lang_bottom, lang_top in settings.pairs:
        output_path = get_output_path(video_path, lang_bottom, lang_top)

        if not output_path.exists():
            logger.debug(f"Skip check: {output_path} does not exist")
            return False

        output_mtime = output_path.stat().st_mtime

        # Check that .ass is newer than both sources
        for lang in (lang_bottom, lang_top):
            source_path = sub_paths[lang]
            source_mtime = source_path.stat().st_mtime
            if source_mtime > output_mtime:
                logger.debug(f"Skip check: {source_path} newer than {output_path}")
                return False

    logger.info("Existing .ass files are up to date, skipping")
    return True


def get_lock_path(video_path: Path) -> Path:
    """Return the lock file path for a video."""
    return video_path.parent / f".{video_path.stem}.sub-tools.lock"


def process_bilingual_merge(
    video_path: Path,
    sub_paths: dict[str, Path],
    settings: SubtoolsSettings | None = None,
) -> list[Path]:
    """Generate bilingual files for all configured pairs.

    Note: Subtitles are assumed already synchronized (Bazarr does sync).

    Args:
        video_path: Path to video file
        sub_paths: Dict {lang: path} of source subtitles
        settings: Settings to get pairs and styles

    Returns:
        List of created .ass files

    Raises:
        ProcessingError: If processing fails
    """
    settings = settings or get_settings()
    created_files: list[Path] = []

    merge_config = MergeConfig(
        color_bottom=settings.color_bottom,
        color_top=settings.color_top,
        fontsize=settings.fontsize,
        layout=settings.layout,
    )

    try:
        for lang_bottom, lang_top in settings.pairs:
            output_path = get_output_path(video_path, lang_bottom, lang_top)

            logger.info(f"Merging {lang_bottom}-{lang_top}...")
            merge_bilingual(
                sub_paths[lang_bottom],
                sub_paths[lang_top],
                output_path,
                merge_config,
            )
            created_files.append(output_path)
            logger.info(f"Created: {output_path}")

    except Exception as e:
        # Cleanup partially created files
        for f in created_files:
            f.unlink(missing_ok=True)
        raise ProcessingError(f"Error during processing: {e}") from e

    return created_files


def process_hook(
    video_path: Path,
    subtitle_path: Path,
    lang: str,
    settings: SubtoolsSettings | None = None,
) -> HookResult:
    """Main hook entry point.

    Args:
        video_path: Path to video file
        subtitle_path: Path to downloaded subtitle (for logging)
        lang: Language code of downloaded subtitle
        settings: Settings for configuration

    Returns:
        HookResult with status and details

    Raises:
        InvalidLanguageError: If language is invalid
        ProcessingError: If processing fails
    """
    settings = settings or get_settings()

    # Validate language
    lang = validate_lang(lang, settings)
    logger.info(f"Hook called: video={video_path}, subtitle={subtitle_path.name}, lang={lang}")

    # Check that video exists
    if not video_path.exists():
        raise ProcessingError(f"Video file not found: {video_path}")

    # Acquire lock
    lock_path = get_lock_path(video_path)
    lock = FileLock(lock_path, timeout=LOCK_TIMEOUT)

    subtitle_path = find_subtitle_path(video_path, lang)

    try:
        with lock.acquire(timeout=LOCK_TIMEOUT):
            logger.debug(f"Lock acquired: {lock_path}")

            # Check if all languages are present
            sub_paths = check_all_languages_present(video_path, settings)

            if sub_paths is None:
                present, missing = get_present_and_missing(video_path, settings)
                logger.info(f"Missing languages: {missing}")
                return HookResult(
                    status="waiting",
                    present=present,
                    missing=missing,
                )

            # Check if we can skip
            if should_skip_existing(video_path, sub_paths, settings):
                return HookResult(
                    status="skipped",
                    reason="already_exists",
                )

            # Process (no sync - Bazarr already did it)
            created_files = process_bilingual_merge(video_path, sub_paths, settings)

            return HookResult(
                status="merged",
                files=[str(f) for f in created_files],
            )

    except Timeout:
        logger.warning(f"Lock timeout for {video_path}")
        return HookResult(status="already_processing")
    finally:
        #cleanup_temp_subtitle(subtitle_path)
        # Clean up lock file after use
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass  # Ignore if file is still in use
