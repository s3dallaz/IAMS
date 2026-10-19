"""
JSON (de)serialisation of real-time results between the vision engine
process and the Streamlit page.
"""

import math
from typing import Any, Dict, Optional

from src.computer_vision.liveness_detector import LivenessState
from src.services.real_time_attendance_service import FaceProcessingResult, RealTimeAttendanceResult

# FaceProcessingResult constructor arguments, plus the attributes the manager fills in afterwards
_FACE_ARGS = (
    "bbox", "detection_confidence", "identity", "is_recognized", "recognition_distance",
    "student_id", "student_db_id", "student_name", "liveness_state", "liveness_confidence",
    "blink_count", "min_blinks_required", "challenge_type", "challenge_state",
    "challenge_progress", "attendance_marked", "attendance_id", "attendance_status",
    "already_marked", "reason",
)
_FACE_EXTRA = ("spoof_score", "challenge_seconds_left")


def _json_number(value: Optional[float]) -> Optional[float]:
    """inf / nan are not valid JSON."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def face_to_dict(face: FaceProcessingResult) -> Dict[str, Any]:
    data = {name: getattr(face, name) for name in _FACE_ARGS + _FACE_EXTRA}
    data["bbox"] = [int(v) for v in face.bbox]
    data["liveness_state"] = face.liveness_state.value
    data["recognition_distance"] = _json_number(face.recognition_distance)
    for name in ("detection_confidence", "liveness_confidence", "challenge_progress",
                 "spoof_score", "challenge_seconds_left"):
        data[name] = _json_number(data[name])
    return data


def face_from_dict(data: Dict[str, Any]) -> FaceProcessingResult:
    kwargs = {name: data[name] for name in _FACE_ARGS if name in data}
    kwargs["bbox"] = tuple(kwargs.get("bbox", (0, 0, 0, 0)))
    kwargs["liveness_state"] = LivenessState(kwargs.get("liveness_state", LivenessState.UNKNOWN.value))
    if kwargs.get("recognition_distance") is None:
        kwargs["recognition_distance"] = float("inf")
    face = FaceProcessingResult(**kwargs)
    for name in _FACE_EXTRA:
        setattr(face, name, data.get(name))
    return face


def result_to_dict(result: RealTimeAttendanceResult) -> Dict[str, Any]:
    return {
        "session_id": result.session_id,
        "faces_detected": result.faces_detected,
        "processed_faces": [face_to_dict(face) for face in result.processed_faces],
        "session_active": result.session_active,
        "marked_count": result.marked_count,
        "timestamp": result.timestamp,
        "multiple_faces": result.multiple_faces,
        "message": result.message,
    }


def result_from_dict(data: Dict[str, Any]) -> RealTimeAttendanceResult:
    return RealTimeAttendanceResult(
        session_id=data.get("session_id"),
        faces_detected=data.get("faces_detected", 0),
        processed_faces=[face_from_dict(face) for face in data.get("processed_faces", [])],
        session_active=data.get("session_active", True),
        marked_count=data.get("marked_count", 0),
        timestamp=data.get("timestamp"),
        multiple_faces=data.get("multiple_faces", False),
        message=data.get("message", ""),
    )
