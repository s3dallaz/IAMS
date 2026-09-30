"""
Configuration module for Intelligent Attendance Monitoring System.

This module contains foundational configuration for the application.
Future phase configurations will be added as needed.
"""

import os
from pathlib import Path
from typing import Final

# Project root directory
PROJECT_ROOT: Final[Path] = Path(__file__).parent.absolute()

# Data directories
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DATABASE_DIR: Final[Path] = DATA_DIR / "database"
FACE_ENCODINGS_DIR: Final[Path] = DATA_DIR / "face_encodings"
ENROLLMENT_IMAGES_DIR: Final[Path] = DATA_DIR / "enrollment_images"
UNKNOWN_FACES_DIR: Final[Path] = DATA_DIR / "unknown_faces"
LOGS_DIR: Final[Path] = DATA_DIR / "logs"

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

# Face detection confidence threshold (MediaPipe)
FACE_DETECTION_CONFIDENCE: Final[float] = 0.7

# Face recognition distance threshold (face_recognition/dlib)
# Lower values = stricter matching
FACE_RECOGNITION_THRESHOLD: Final[float] = 0.6

# ============================================================================
# Phase 3: Liveness Detection Configuration
# ============================================================================

# Liveness detection confidence threshold
# Confidence scores above this threshold indicate a live person
# NOTE: Phase 3A establishes the configuration framework
#       Phase 3B/3C will implement the algorithm and tune these values
LIVENESS_THRESHOLD: Final[float] = 0.75

# Minimum number of frames required for reliable liveness detection
# Temporal analysis across multiple frames is more robust than single-frame detection
LIVENESS_MIN_FRAMES: Final[int] = 10

# Temporal window for liveness analysis (in seconds)
# Frames must be captured within this window for temporal consistency
LIVENESS_TEMPORAL_WINDOW: Final[float] = 2.0

# Number of enrollment images required per student (to be implemented in Phase 2B)
# ENROLLMENT_IMAGES_REQUIRED: int = 5

# ============================================================================
# FUTURE PHASE PLACEHOLDERS (not yet implemented)
# ============================================================================

# Phase 4: Attendance Configuration (PLACEHOLDER)
# LATE_THRESHOLD_MINUTES: int = 15
# AUTO_CLOSE_SESSIONS: bool = True
# REQUIRE_LIVENESS: bool = True

# Phase 6: AI Configuration (PLACEHOLDER)
# OMNIROUTE_API_KEY: str = os.getenv("OMNIROUTE_API_KEY", "")
# AI_RATE_LIMIT: int = 10  # queries per minute


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
