"""
Tests for the real-time vision engine (src/realtime): result protocol, the
on-video instruction bar, and the engine's HTTP API with a fake camera and
pipeline (no webcam or models needed).
"""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.computer_vision.liveness_detector import LivenessState
from src.realtime import engine as engine_module
from src.realtime.engine import VisionEngine, draw_preview, instruction_bar, make_handler
from src.realtime.engine_client import EngineController, _free_port
from src.realtime.protocol import result_from_dict, result_to_dict
from src.services.real_time_attendance_service import FaceProcessingResult, RealTimeAttendanceResult


def _face(**overrides):
    fields = dict(bbox=(10, 20, 30, 40), detection_confidence=0.9, identity="2021001",
                  is_recognized=True, student_id="2021001", student_name="Alice",
                  blink_count=1, min_blinks_required=2, challenge_type="turn_left",
                  challenge_state="active", reason="Liveness verification pending / not confirmed")
    fields.update(overrides)
    return FaceProcessingResult(**fields)


def _result(*faces, **overrides):
    fields = dict(session_id=3, faces_detected=len(faces), processed_faces=list(faces), session_active=True)
    fields.update(overrides)
    return RealTimeAttendanceResult(**fields)


# -- protocol -------------------------------------------------------------------

def test_result_survives_the_json_round_trip():
    face = _face(liveness_state=LivenessState.LIVE)
    face.spoof_score = 0.91
    face.challenge_seconds_left = 4.2
    data = json.loads(json.dumps(result_to_dict(_result(face, marked_count=1))))

    back = result_from_dict(data)

    restored = back.processed_faces[0]
    assert back.session_id == 3 and back.marked_count == 1
    assert restored.bbox == (10, 20, 30, 40)
    assert restored.liveness_state == LivenessState.LIVE
    assert restored.recognition_distance == float("inf")  # inf is sent as null
    assert restored.spoof_score == pytest.approx(0.91)
    assert restored.challenge_seconds_left == pytest.approx(4.2)


# -- instruction bar -----------------------------------------------------------------

def test_instruction_bar_asks_for_blinks_and_the_head_turn_together():
    face = _face()
    face.challenge_seconds_left = 6.3
    text, _ = instruction_bar(_result(face), None, time.time())
    assert text == "Alice: Blink 2 times (1/2) + Turn head LEFT  (7s)"


@pytest.mark.parametrize("face_kwargs, multiple, expected", [
    ({"is_recognized": False, "identity": "Unknown"}, False, "Unknown person - not enrolled"),
    ({"already_marked": True}, False, "Alice - already marked"),
    ({"reason": "Spoof suspected: face looks like a photo or screen"}, False, "SPOOF SUSPECTED"),
    ({}, True, "Multiple faces"),
])
def test_instruction_bar_states(face_kwargs, multiple, expected):
    text, _ = instruction_bar(_result(_face(**face_kwargs), multiple_faces=multiple), None, time.time())
    assert text.startswith(expected)


def test_marked_banner_stays_for_a_few_seconds():
    mark = {"name": "Alice", "status": "PRESENT", "time": 100.0}
    assert instruction_bar(_result(), mark, 101.0)[0] == "MARKED: Alice (PRESENT)"
    assert instruction_bar(_result(), mark, 110.0)[0] is None


def test_preview_is_mirrored_and_keeps_its_size():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :40] = 255                                   # bright strip on the left
    view = draw_preview(frame, None, None)
    assert view.shape == frame.shape
    assert view[:, -40:].mean() == 255 and view[:, :40].mean() == 0
    assert draw_preview(frame, "Blink", (0, 200, 255)).shape == frame.shape


# -- engine + HTTP API with a fake camera and pipeline ----------------------------------

@pytest.fixture
def running_engine(monkeypatch):
    """A VisionEngine on a free port, with a fake camera and attendance manager."""
    camera = MagicMock()
    camera.open.return_value = True

    def read_frame():
        time.sleep(0.01)
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    camera.read_frame.side_effect = read_frame
    manager = MagicMock()
    manager.process_frame_async.return_value = _result(_face())
    enrollment = MagicMock()
    enrollment.return_value.load_active_gallery.return_value = []
    monkeypatch.setattr(engine_module, "CameraManager", MagicMock(return_value=camera))
    monkeypatch.setattr(engine_module, "RealTimeAttendanceManager", MagicMock(return_value=manager))
    monkeypatch.setattr(engine_module, "EnrollmentService", enrollment)
    monkeypatch.setattr(engine_module, "GALLERY_RELOAD_SECONDS", 0.05)

    engine = VisionEngine(session_id=3, settings={"require_liveness": True, "require_head_turn": False})
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(engine, "secret"))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    assert engine.start()
    loop = threading.Thread(target=engine.run, daemon=True)
    loop.start()
    yield engine, f"http://127.0.0.1:{port}", manager, camera, enrollment, loop
    engine.stop()
    loop.join(timeout=5)
    server.shutdown()
    server.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read()


def _post(url, data, token="secret"):
    request = urllib.request.Request(url, data=json.dumps(data).encode(), method="POST",
                                     headers={"X-Engine-Token": token, "Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=5)


def test_status_reports_the_running_pipeline(running_engine):
    engine, base, manager, *_ = running_engine
    time.sleep(0.2)
    status = json.loads(_get(base + "/status.json"))
    assert status["state"] == "running"
    assert status["result"]["processed_faces"][0]["identity"] == "2021001"
    assert manager.fast_liveness is True
    manager.enable_background_processing.assert_called_once()
    manager.update_liveness.assert_called()           # liveness on every frame


def test_stream_delivers_jpeg_frames(running_engine):
    _, base, *_ = running_engine
    with urllib.request.urlopen(base + "/stream.mjpg", timeout=5) as response:
        assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
        chunk = response.read(4096)
    assert b"--frame" in chunk and b"Content-Type: image/jpeg" in chunk
    assert b"\xff\xd8" in chunk                        # JPEG start marker


def test_settings_change_the_running_pipeline(running_engine):
    _, base, manager, *_ = running_engine
    assert manager.require_head_turn is False
    _post(base + "/settings", {"require_head_turn": True, "cooldown_seconds": 7.5})
    assert manager.require_head_turn is True
    assert manager.cooldown_seconds == 7.5


def test_control_requests_need_the_token(running_engine):
    engine, base, *_ = running_engine
    with pytest.raises(urllib.error.HTTPError) as error:
        _post(base + "/stop", {}, token="wrong")
    assert error.value.code == 403
    assert not engine.stopped


def test_stop_releases_the_camera(running_engine):
    engine, base, manager, camera, _, loop = running_engine
    _post(base + "/stop", {})
    loop.join(timeout=5)
    assert not loop.is_alive()
    camera.release.assert_called_once()
    manager.disable_background_processing.assert_called_once()


def test_gallery_is_reloaded_while_running(running_engine):
    _, _, manager, _, enrollment, _ = running_engine
    time.sleep(0.3)
    assert manager.set_gallery.call_count >= 2       # students enrolled meanwhile are picked up


def test_camera_failure_is_reported(monkeypatch):
    camera = MagicMock()
    camera.open.return_value = False
    monkeypatch.setattr(engine_module, "CameraManager", MagicMock(return_value=camera))
    engine = VisionEngine(session_id=3, settings={})
    assert engine.start() is False
    assert engine.status()["state"] == "error"
    assert "camera" in engine.status()["error"].lower()


def test_controller_without_an_engine():
    controller = EngineController()
    assert controller.is_running() is False
    assert controller.status() is None
    assert controller.update_settings({}) is False
    controller.stop()  # nothing to stop


def test_spoof_warning_comes_before_identity_on_the_video():
    for face in (_face(is_recognized=False, identity="Unknown"), _face(already_marked=True)):
        face.reason = "Spoof suspected: face looks like a photo or screen"
        assert instruction_bar(_result(face), None, time.time())[0].startswith("SPOOF SUSPECTED")
