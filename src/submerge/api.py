"""FastAPI API for Bazarr hook integration."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException

from .config import get_settings
from .hook import (
    InvalidLanguageError,
    ProcessingError,
    process_hook,
)

# Configure logging for the whole app
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("submerge")


class HealthCheckFilter(logging.Filter):
    """Filters out /health request logs to reduce noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "/health" not in message


def setup_logging_filters() -> None:
    """Configure logging filters. Called at startup."""
    health_filter = HealthCheckFilter()
    logging.getLogger("uvicorn.access").addFilter(health_filter)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Validate config at startup (fail fast)
    settings = get_settings()
    if not settings.pairs:
        raise RuntimeError(
            "SUBTOOLS_PAIRS environment variable is required. "
            "Example: SUBTOOLS_PAIRS='fr-pl,en-pl'"
        )

    setup_logging_filters()

    return FastAPI(
        title="SubMerge API",
        description="API for automatic bilingual subtitle generation",
        version="1.0.0",
    )


app = create_app()


def validate_path(path_str: str, param_name: str) -> Path:
    """Validate and resolve a path.

    Args:
        path_str: Path to validate
        param_name: Parameter name (for error messages)

    Returns:
        Resolved and validated Path

    Raises:
        HTTPException: If the path is invalid
    """
    try:
        path = Path(path_str)

        # Must be an absolute path
        if not path.is_absolute():
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": f"{param_name} must be an absolute path"},
            )

        # Resolve to eliminate .. and symlinks
        return path.resolve()

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Invalid path {param_name}={path_str}: {e}")
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": f"Invalid {param_name} path"},
        )


@app.post("/hook")
def hook(
    video: str = Form(..., description="Path to video file"),
    subtitle: str = Form(..., description="Path to downloaded subtitle"),
    lang: str = Form(..., description="Language code (fr, pl, en)"),
) -> dict:
    """Bazarr post-processing hook.

    Receives information about a subtitle downloaded by Bazarr.
    Checks if all required languages are present.
    If yes, generates bilingual .ass files.

    Returns:
        - 200 {"status": "merged", "files": [...]} if merge completed
        - 200 {"status": "waiting", "present": [...], "missing": [...]} if languages missing
        - 200 {"status": "skipped", "reason": "already_exists"} if .ass already present
        - 200 {"status": "already_processing"} if lock busy
        - 400 {"status": "error", "message": "..."} if invalid parameter
        - 500 {"status": "error", "message": "..."} if internal error
    """
    # Validate paths before processing
    video_path = validate_path(video, "video")
    subtitle_path = validate_path(subtitle, "subtitle")

    logger.info(f"Hook request: video={video_path.name}, lang={lang}")

    try:
        result = process_hook(video_path, subtitle_path, lang)

        # Use 200 for all statuses, status is in the body
        response: dict = {"status": result.status}

        if result.files:
            response["files"] = result.files
        if result.present:
            response["present"] = result.present
        if result.missing:
            response["missing"] = result.missing
        if result.reason:
            response["reason"] = result.reason

        return response

    except InvalidLanguageError as e:
        logger.warning(f"Invalid language: {e}")
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": str(e)},
        )
    except ProcessingError as e:
        error_msg = str(e)
        # Don't expose full paths in errors
        if "not found" in error_msg.lower():
            logger.warning(f"File not found: {e}")
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "Video file not found"},
            )
        logger.error(f"Processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Processing failed"},
        )
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Internal server error"},
        )


@app.get("/health")
def health() -> dict:
    """Health check - verifies that ffmpeg and ffprobe are accessible."""
    ffmpeg_available = shutil.which("ffmpeg") is not None
    ffprobe_available = shutil.which("ffprobe") is not None

    all_ok = ffmpeg_available and ffprobe_available

    return {
        "status": "ok" if all_ok else "degraded",
        "ffmpeg": ffmpeg_available,
        "ffprobe": ffprobe_available,
        "version": "custom",
    }
