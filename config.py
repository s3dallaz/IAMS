"""
Configuration module for Intelligent Attendance Monitoring System.

This module contains foundational configuration for the application.
Future phase configurations will be added as needed.
"""

import os
from pathlib import Path
from typing import Final, Optional

# Project root directory
PROJECT_ROOT: Final[Path] = Path(__file__).parent.absolute()

# ============================================================================
# Environment loading (centralized)
# ============================================================================

# Environment file location (project root).
# `.env` holds real secrets and is listed in .gitignore - only `.env.example`
# is tracked by Git, so credentials are never committed.
ENV_FILE: Final[Path] = PROJECT_ROOT / ".env"

# Values that mean "the key has not been filled in yet" (not a real secret).
AI_API_KEY_PLACEHOLDERS: Final[frozenset] = frozenset({
    "",
    "your_api_key_here",
    "your-api-key-here",
    "changeme",
    "todo",
    "none",
    "null",
})

# Environment variable names that must never be logged or sent to the model.
SENSITIVE_ENV_VARS: Final[frozenset] = frozenset({
    "OMNIROUTE_API_KEY",
})


def load_environment() -> bool:
    """
    Load environment variables from the project-root `.env` file.

    Real environment variables always win over `.env` entries
    (`override=False`), so shell / CI configuration takes precedence.

    Loading is idempotent and safe to call more than once. This runs at import
    time so every entry point (app.py, tests, CLI helpers) gets the same values.

    Returns:
        bool: True if a `.env` file was loaded, False if it is missing or
              python-dotenv is unavailable.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        # Keep config importable even if python-dotenv is not installed.
        return False

    if not ENV_FILE.is_file():
        return False

    return bool(load_dotenv(ENV_FILE, override=False))


ENV_FILE_LOADED: bool = load_environment()

# ============================================================================
# Camera startup (Windows Media Foundation backend)
# ============================================================================
# OpenCV's default Windows camera backend (MSMF) spends ~7 s setting up
# hardware-accelerated format conversion when the camera opens, and the same
# again when CameraManager sets the FPS - about 15 s before the first frame
# on the development laptop. With the transforms disabled the camera starts
# in under 1 s at the same resolution and frame rate
# (measured with manual_tests/camera_open_timing.py).
# OpenCV reads this flag from the environment, so it must be set before cv2
# is imported; config is the first project module app.py imports.
# Set OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=1 in .env to restore OpenCV's
# default behaviour.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to `default`."""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to `default`."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Camera frame size for live attendance and enrollment: 720p. The camera is
# asked for this size; frames a camera still delivers larger are shrunk to
# fit (aspect ratio kept) in the camera reader thread, ~2 ms per frame. The
# development laptop's Logitech Brio 100 sends 1920x1080 whatever OpenCV asks
# for, and at 720p every later step (liveness, detection, the preview JPEG)
# handles 2.25x fewer pixels. Set both to 0 in .env for the camera's own size.
CAMERA_WIDTH: int = _env_int("CAMERA_WIDTH", 1280)
CAMERA_HEIGHT: int = _env_int("CAMERA_HEIGHT", 720)


# ============================================================================
# Phase 6: AI Assistant Configuration (OmniRoute)
# ============================================================================

# Defaults mirror the OmniRoute provider settings in opencode.json
OMNIROUTE_API_KEY: str = os.getenv("OMNIROUTE_API_KEY", "")
OMNIROUTE_BASE_URL: str = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
OMNIROUTE_MODEL: str = os.getenv("OMNIROUTE_MODEL", "kr/claude-sonnet-4.5")

# Timeout (seconds) applied to AI provider requests
AI_REQUEST_TIMEOUT: float = _env_float("AI_REQUEST_TIMEOUT", 30.0)

# Sampling temperature for the AI Assistant.
# Low value keeps factual answers deterministic and reduces hallucination.
AI_TEMPERATURE: float = _env_float("AI_TEMPERATURE", 0.0)

# Maximum tokens for a generated answer
AI_MAX_TOKENS: int = int(_env_float("AI_MAX_TOKENS", 800))

# Maximum characters accepted for any single tool argument value.
# Keeps model output bounded and prevents oversized/injected payloads.
AI_MAX_ARG_LENGTH: int = int(_env_float("AI_MAX_ARG_LENGTH", 200))

# Number of prior conversation turns kept as LLM context.
AI_MAX_HISTORY_MESSAGES: int = int(_env_float("AI_MAX_HISTORY_MESSAGES", 12))

# Set to True to skip the natural-language answer step and return raw tool
# results. Useful for debugging and for offline environments.
AI_RAW_MODE: bool = os.getenv("AI_RAW_MODE", "false").lower() == "true"


def get_ai_api_key() -> str:
    """
    Return the OmniRoute API key, read live from the environment.

    Read live (not cached at import) so tests and runtime changes to the
    environment are picked up without reimporting the module.
    """
    return os.getenv("OMNIROUTE_API_KEY", "").strip()


def is_ai_assistant_configured(api_key: Optional[str] = None) -> bool:
    """
    Check whether the AI Assistant has a usable API key.

    Args:
        api_key: Optional explicit key to test. When omitted, the current
                 environment value is used.

    Returns:
        bool: True if a real (non-empty, non-placeholder) key is available.
    """
    if api_key is None:
        api_key = get_ai_api_key()
    return api_key.strip().lower() not in AI_API_KEY_PLACEHOLDERS


# Data directories
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DATABASE_DIR: Final[Path] = DATA_DIR / "database"
FACE_ENCODINGS_DIR: Final[Path] = DATA_DIR / "face_encodings"
ENROLLMENT_IMAGES_DIR: Final[Path] = DATA_DIR / "enrollment_images"
UNKNOWN_FACES_DIR: Final[Path] = DATA_DIR / "unknown_faces"
LOGS_DIR: Final[Path] = DATA_DIR / "logs"
ATTENDANCE_SNAPSHOTS_DIR: Final[Path] = DATA_DIR / "attendance_snapshots"

# Database configuration
DATABASE_NAME: Final[str] = "attendance.db"
DATABASE_PATH: Final[Path] = DATABASE_DIR / DATABASE_NAME

# Application settings
APP_NAME: Final[str] = "Intelligent Attendance Monitoring System"
APP_VERSION: Final[str] = "0.1.0-alpha"
APP_DEBUG: bool = os.getenv("APP_DEBUG", "false").lower() == "true"

# Logging configuration
LOG_LEVEL: str = os.getenv("APP_LOG_LEVEL", "INFO")

# ============================================================================
# Phase 2: Computer Vision Configuration
# ============================================================================

# ============================================================================
# Phase 2B: Face Detection Diagnostics (audit instrumentation)
# ============================================================================
#
# These settings only add observability. They do not change detection,
# recognition, or liveness behaviour unless explicitly overridden in .env.
# See FACE_DIAGNOSTICS below for the usage instructions.

# Master switch for the detection-diagnostics overlay.
# Off by default so normal behaviour and performance are unchanged.
FACE_DIAGNOSTICS_ENABLED: bool = os.getenv("FACE_DIAGNOSTICS_ENABLED", "false").lower() == "true"

# Emit a one-line INFO summary every N frames when diagnostics are enabled.
# 0 or 1 means every frame. Keeps log volume bounded during a live stream.
FACE_DIAGNOSTICS_LOG_EVERY: int = int(_env_float("FACE_DIAGNOSTICS_LOG_EVERY", 30))

# Draw the diagnostic overlay on the displayed video frame.
FACE_DIAGNOSTICS_DRAW_OVERLAY: bool = os.getenv(
    "FACE_DIAGNOSTICS_DRAW_OVERLAY", "true"
).lower() == "true"

# Warn (log) when a detected face is smaller than this many pixels on its
# longer edge. Based on measurement, BlazeFace short-range detection collapses
# below ~100 px, so this surfaces "too far away" before total detection loss.
FACE_DIAGNOSTICS_WARN_FACE_PX: int = int(_env_float("FACE_DIAGNOSTICS_WARN_FACE_PX", 100))

# Optional persistent CSV of per-frame detection measurements. When empty, no
# file is written. Useful for measuring across real distances.
FACE_DIAGNOSTICS_CSV_PATH: str = os.getenv("FACE_DIAGNOSTICS_CSV_PATH", "")

# Face pipeline models (OpenCV Zoo, run with OpenCV's DNN module).
# - YuNet detector: finds faces (also turned ones) and 5 landmarks (eyes, nose,
#   mouth corners) in ~7 ms; "mediapipe" = the previous BlazeFace detector.
# - SFace recogniser: 128-d face embedding of the aligned face in ~5 ms
#   (dlib took ~240 ms and blocked other threads); "dlib" = the previous model.
# Each falls back to the previous model if its file is missing. Embeddings of
# the two recognisers are not compatible: SFace ones are stored next to the
# dlib ones as enc_sface_<id>.npy and created automatically from the saved
# enrollment photos (EnrollmentService.load_active_gallery).
FACE_DETECTOR_BACKEND: str = os.getenv("FACE_DETECTOR_BACKEND", "yunet").lower()
FACE_RECOGNITION_BACKEND: str = os.getenv("FACE_RECOGNITION_BACKEND", "sface").lower()
YUNET_MODEL_PATH: Final[Path] = PROJECT_ROOT / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH: Final[Path] = PROJECT_ROOT / "face_recognition_sface_2021dec.onnx"

# SFace match threshold: Euclidean distance between unit-length embeddings.
# 1.128 is OpenCV's published threshold (= cosine similarity 0.363); lower is
# stricter. The ambiguity margin works like FACE_RECOGNITION_MARGIN below.
SFACE_RECOGNITION_THRESHOLD: float = _env_float("SFACE_RECOGNITION_THRESHOLD", 1.128)
SFACE_RECOGNITION_MARGIN: float = _env_float("SFACE_RECOGNITION_MARGIN", 0.1)

# Face detection confidence threshold (MediaPipe).
# Now overridable via .env so the threshold can be tuned without editing code.
FACE_DETECTION_CONFIDENCE: Final[float] = _env_float("FACE_DETECTION_CONFIDENCE", 0.7)

# Face recognition distance threshold (face_recognition/dlib)
# Lower values = stricter matching
# Set FACE_RECOGNITION_THRESHOLD in .env to override the default of 0.6.
FACE_RECOGNITION_THRESHOLD: Final[float] = _env_float("FACE_RECOGNITION_THRESHOLD", 0.6)

# Minimum distance gap between the best match and the best match of a DIFFERENT
# enrolled identity. When a query face is too close to call (nearly the same
# distance to two students), the match is ambiguous and is reported as
# "Unknown" instead of being assigned to the likely-wrong student.
# Only applies when at least two distinct identities are enrolled.
# Set FACE_RECOGNITION_MARGIN=0.0 in .env to disable the ambiguity check.
FACE_RECOGNITION_MARGIN: Final[float] = _env_float("FACE_RECOGNITION_MARGIN", 0.1)

# ============================================================================
# Phase 3: Liveness Detection Configuration
# ============================================================================

# Liveness detection confidence threshold
# Confidence scores above this threshold indicate a live person
# NOTE: Phase 3A establishes the configuration framework
#       Phase 3B implements blink detection algorithm
LIVENESS_THRESHOLD: Final[float] = 0.75

# Minimum number of frames required for reliable liveness detection
# Temporal analysis across multiple frames is more robust than single-frame detection
LIVENESS_MIN_FRAMES: Final[int] = 10

# Temporal window for liveness analysis (in seconds)
# Frames must be captured within this window for temporal consistency
LIVENESS_TEMPORAL_WINDOW: Final[float] = 2.0

# Eye Aspect Ratio (EAR) default threshold (alias for backward compatibility)
LIVENESS_EAR_THRESHOLD: Final[float] = 0.21

# Eye Aspect Ratio (EAR) threshold for detecting closed eyes (hysteresis lower bound)
# EAR below this value indicates eyes are closing / closed
# Typical range: 0.20-0.23
LIVENESS_EAR_CLOSE_THRESHOLD: Final[float] = 0.21

# Eye Aspect Ratio threshold for detecting eye reopening (hysteresis upper bound)
# EAR above this value indicates eyes are open / reopened
# Higher than close threshold to prevent oscillation at boundary
LIVENESS_EAR_OPEN_THRESHOLD: Final[float] = 0.25

# Number of consecutive frames with EAR below threshold to confirm eye closure
# Set to 1 to reliably capture fast natural human blinks (100-250ms duration)
LIVENESS_EAR_CONSECUTIVE_FRAMES: Final[int] = 1

# Minimum number of blinks required to classify as LIVE
# A blink is a complete eye closure followed by reopening
LIVENESS_MIN_BLINKS: Final[int] = 1

# EAR temporal smoothing window size (frames)
# 0 = no smoothing (required to detect fast natural blinks without attenuation)
# Smoothing would mask rapid blinks by averaging EAR values across frames
LIVENESS_EAR_SMOOTHING_WINDOW: Final[int] = 0

# ============================================================================
# Phase 3C: Active Liveness & Anti-Spoofing Configuration
# ============================================================================

# Enable active challenge-response liveness detection
# Requires user to perform randomized actions (e.g., head turns)
LIVENESS_ENABLE_ACTIVE_CHALLENGES: Final[bool] = True

# Challenge timeout in seconds
# The person must finish the blinks + head turn within this time, otherwise a
# new random challenge starts. A pre-recorded video cannot know the challenge
# in advance; a relayed one (video call) has little time to react.
# Set LIVENESS_CHALLENGE_TIMEOUT=0 in .env to disable the time limit.
LIVENESS_CHALLENGE_TIMEOUT: Final[float] = _env_float("LIVENESS_CHALLENGE_TIMEOUT", 10.0)

# Each challenge asks for a random number of blinks between LIVENESS_MIN_BLINKS
# and this value (and a random head-turn direction), so the required actions
# differ from person to person and cannot be pre-recorded.
LIVENESS_MAX_BLINKS: Final[int] = int(_env_float("LIVENESS_MAX_BLINKS", 3))

# Head turn angle thresholds (in degrees)
# Yaw angle required to complete left/right turn challenges
# Configurable via .env as LIVENESS_HEAD_TURN_ANGLE
# Default 15 degrees (range: 5-35 degrees in 5-degree increments)
LIVENESS_HEAD_TURN_ANGLE: Final[float] = _env_float("LIVENESS_HEAD_TURN_ANGLE", 15.0)

# Head pose return-to-center tolerance (in degrees)
# Maximum deviation from center to consider head "centered"
LIVENESS_HEAD_CENTER_TOLERANCE: Final[float] = 15.0

# Minimum frames the head must be held in target position
# Prevents accidental/brief movements from completing challenges
LIVENESS_HEAD_HOLD_FRAMES: Final[int] = 3

# Require challenge completion along with blink detection for LIVE state
# If False, blink detection alone is sufficient (Phase 3B behavior)
LIVENESS_REQUIRE_CHALLENGE: Final[bool] = os.getenv("LIVENESS_REQUIRE_CHALLENGE", "true").lower() == "true"

# Seconds a face may be lost before liveness restarts. The face detector
# briefly loses a face that turns for the head-turn challenge; resetting at
# once would wipe the progress the challenge asks for.
LIVENESS_FACE_LOST_GRACE: float = _env_float("LIVENESS_FACE_LOST_GRACE", 2.0)

# ----------------------------------------------------------------------------
# Spoof (presentation attack) detection - part of the liveness requirement
# ----------------------------------------------------------------------------
# Blinks and head turns can be replayed from a video, so with liveness required
# a face must also pass these checks (src/computer_vision/spoof_detection.py).
# Each check is skipped, with a warning, if its model file is missing.

# Anti-spoof model (MiniFASNet), judged per processed frame. Calibrated with
# manual_tests/spoof_check_live.py on the development webcam: a real face
# scored >= 0.89 on every frame (mean 1.00), while a face on a phone flips
# between ~0 and ~1 (43-66 % of frames below 0.6), so an average let phones
# pass. Now:
# - a frame scoring below ANTI_SPOOF_THRESHOLD is a "spoof frame";
# - ANTI_SPOOF_BAD_FRAMES spoof frames among the last ANTI_SPOOF_WINDOW block
#   the person until they leave the view;
# - marking needs the last ANTI_SPOOF_MIN_FRAMES frames all above the threshold.
ANTI_SPOOF_ENABLED: bool = os.getenv("ANTI_SPOOF_ENABLED", "true").lower() == "true"
ANTI_SPOOF_THRESHOLD: float = _env_float("ANTI_SPOOF_THRESHOLD", 0.8)
ANTI_SPOOF_BAD_FRAMES: int = int(_env_float("ANTI_SPOOF_BAD_FRAMES", 2))
ANTI_SPOOF_MIN_FRAMES: int = int(_env_float("ANTI_SPOOF_MIN_FRAMES", 5))
ANTI_SPOOF_WINDOW: int = int(_env_float("ANTI_SPOOF_WINDOW", 10))

# Spoof recheck. Low-quality webcams (blur, noise, low light) give a real face
# a few low-scoring frames, which flagged real people as spoofs. After the
# anti-spoof model flags a face, it is checked again: the flag is lifted when
# at most ANTI_SPOOF_RECHECK_MAX_BAD of the next ANTI_SPOOF_RECHECK_FRAMES
# frames score below the threshold (and the last ANTI_SPOOF_MIN_FRAMES are all
# real) within ANTI_SPOOF_RECHECK_TIMEOUT seconds. A photo scores low on
# nearly every frame and a phone on roughly half, so they keep failing.
# Up to ANTI_SPOOF_MAX_RECHECKS rechecks per person in view; after that (or a
# timed-out recheck) the block stays until they leave. 0 rechecks = no recheck.
# A phone / laptop / TV seen covering the face is never rechecked.
ANTI_SPOOF_RECHECK_FRAMES: int = int(_env_float("ANTI_SPOOF_RECHECK_FRAMES", 15))
ANTI_SPOOF_RECHECK_MAX_BAD: int = int(_env_float("ANTI_SPOOF_RECHECK_MAX_BAD", 1))
ANTI_SPOOF_RECHECK_TIMEOUT: float = _env_float("ANTI_SPOOF_RECHECK_TIMEOUT", 10.0)
ANTI_SPOOF_MAX_RECHECKS: int = int(_env_float("ANTI_SPOOF_MAX_RECHECKS", 2))

# Screen detector (MediaPipe EfficientDet-Lite0): a phone, laptop or TV whose
# box covers at least SCREEN_FACE_OVERLAP of the face box blocks marking until
# the person leaves the view.
SCREEN_DETECTION_ENABLED: bool = os.getenv("SCREEN_DETECTION_ENABLED", "true").lower() == "true"
SCREEN_DETECTION_CONFIDENCE: float = _env_float("SCREEN_DETECTION_CONFIDENCE", 0.4)
SCREEN_FACE_OVERLAP: float = _env_float("SCREEN_FACE_OVERLAP", 0.5)
# Seconds between screen checks for the person in view (~37 ms each);
# 0 = on every recognition run (at most RECOGNITION_INTERVAL apart)
SCREEN_CHECK_INTERVAL: float = _env_float("SCREEN_CHECK_INTERVAL", 0.0)

# Log a "SPOOF CHECK" line (anti-spoof scores, screens seen) once per second
# while a face is in view - for calibrating ANTI_SPOOF_THRESHOLD
SPOOF_DIAGNOSTICS: bool = os.getenv("SPOOF_DIAGNOSTICS", "true").lower() == "true"

# Real-time recognition runs at most every RECOGNITION_INTERVAL seconds in the
# vision engine (blinks and head turns are still checked on every frame)
RECOGNITION_INTERVAL: float = _env_float("RECOGNITION_INTERVAL", 0.2)

# Audit snapshot: a small photo of the face is saved with every real-time
# attendance mark (data/attendance_snapshots/session_<id>/) so a teacher can
# review who was marked. These are biometric images and are git-ignored.
ATTENDANCE_SNAPSHOTS_ENABLED: bool = os.getenv("ATTENDANCE_SNAPSHOTS_ENABLED", "true").lower() == "true"

# Number of enrollment images required per student (to be implemented in Phase 2B)
# ENROLLMENT_IMAGES_REQUIRED: int = 5

# ============================================================================
# FUTURE PHASE PLACEHOLDERS (not yet implemented)
# ============================================================================

# Phase 4: Attendance Configuration (PLACEHOLDER)
# LATE_THRESHOLD_MINUTES: int = 15
# AUTO_CLOSE_SESSIONS: bool = True
# REQUIRE_LIVENESS: bool = True

# Phase 6: AI Configuration - implemented above (OMNIROUTE_API_KEY,
# OMNIROUTE_BASE_URL, OMNIROUTE_MODEL, AI_REQUEST_TIMEOUT).
AI_RATE_LIMIT: int = 10  # queries per minute


def ensure_directories() -> None:
    """
    Ensure all required directories exist.
    Creates directories if they don't exist.
    """
    directories = [
        DATA_DIR,
        DATABASE_DIR,
        FACE_ENCODINGS_DIR,
        ENROLLMENT_IMAGES_DIR,
        UNKNOWN_FACES_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def validate_config() -> bool:
    """
    Validate configuration on application startup.

    Returns:
        bool: True if configuration is valid, False otherwise.
    """
    try:
        ensure_directories()
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test configuration
    print(f"{APP_NAME} v{APP_VERSION}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Database Path: {DATABASE_PATH}")
    print(f"Debug Mode: {APP_DEBUG}")
    print(f"Log Level: {LOG_LEVEL}")

    if validate_config():
        print("\n[OK] Configuration is valid")
        print("[OK] All directories created successfully")
    else:
        print("\n[ERROR] Configuration validation failed")
