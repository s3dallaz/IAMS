"""
Computer vision layer for face detection, recognition, and liveness detection.

This module contains implementations for:
- Face detection (OpenCV Haar Cascade)
- Face recognition (face_recognition/dlib)
- Liveness detection (MediaPipe Face Mesh + motion analysis)
- CV pipeline orchestration
- Camera management

Phase 2A: CV foundation and dependencies established
Phase 2B: Camera Manager implemented
Phase 2C: Face Detection implemented
Phase 2D: Face Recognition implemented
Phase 3A: Liveness Detection foundation implemented
Phase 3B: Will implement blink detection and temporal analysis
Phase 3C: Will implement complete anti-spoofing logic
"""

__version__ = "0.3.0-alpha"

from src.computer_vision.camera_manager import CameraManager
from src.computer_vision.face_detector import FaceDetector, DetectedFace
from src.computer_vision.liveness_detector import (
    LivenessDetector,
    LivenessResult,
    LivenessState,
    LivenessDetectionError
)

__all__ = [
    "CameraManager",
    "FaceDetector",
    "DetectedFace",
    "LivenessDetector",
    "LivenessResult",
    "LivenessState",
    "LivenessDetectionError"
]
