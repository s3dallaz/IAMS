"""
Real-time vision engine process.

Runs the camera and the whole attendance pipeline in its own process, so the
Streamlit server cannot slow it down (in the Streamlit process the camera
delivered ~15 fps and each st.image() call took ~57 ms):

    camera reader thread --newest frame--> main loop (every frame):
        per-frame liveness (blinks, head turn)       ~10 ms
        submit frame to the recognition worker       (background thread)
        mirror + draw the instruction bar, JPEG      ~3 ms
        --> MJPEG stream to the browser

    recognition worker thread: detection, recognition, spoof checks, marking
    gallery thread: loads the gallery after start-up (the video runs
        meanwhile; recognition waits for it), then every
        GALLERY_RELOAD_SECONDS checks whether the enrollments changed and,
        only then, loads the new gallery and swaps it in (11,000 samples
        take ~2.6 s without the gallery cache, ~44 s after a restart - in
        start-up that timed the engine out, in the main loop it froze the video)

HTTP API on 127.0.0.1 (the only way in):
    GET  /stream.mjpg   annotated video (multipart MJPEG, shown with <img>)
    GET  /status.json   engine state, frame rates and the latest result
    GET  /health        same as status (used while starting)
    POST /settings      {"require_liveness", "require_head_turn",
                         "cooldown_seconds", "head_turn_angle"}  (token required)
    POST /stop          stop the engine (token required)

Started by engine_client.EngineController:
    python -m src.realtime.engine --port 8765 --session 3 --token ... --settings '{...}'
The engine also stops when its parent process exits (its stdin closes).
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

import config  # noqa: F401 - sets the camera flags before cv2 is imported
import cv2
import numpy as np

from src.computer_vision.camera_manager import CameraManager
from src.computer_vision.liveness_detector import LivenessState
from src.realtime.protocol import result_to_dict
from src.services.enrollment_service import EnrollmentService
from src.services.real_time_attendance_service import (
    SPOOF_RECHECK_PREFIX,
    RealTimeAttendanceManager,
    RealTimeAttendanceResult,
)
from src.utils.logger import get_logger
from src.utils.performance_timer import get_timer, log_performance_summary, time_operation

logger = get_logger(__name__)

JPEG_QUALITY = 80
GALLERY_RELOAD_SECONDS = 30.0   # students enrolled during a session are picked up (background thread)
GALLERY_RETRY_SECONDS = 5.0     # retry of a failed first gallery load
MARK_BANNER_SECONDS = 3.0       # how long "Marked: <name>" stays on the video
PERF_LOG_FRAMES = 150

# Instruction bar colours (BGR)
GREEN, RED, ORANGE, YELLOW, BLUE, WHITE = (60, 180, 75), (40, 40, 220), (0, 140, 255), (0, 200, 255), (220, 140, 40), (255, 255, 255)


def instruction_bar(result: RealTimeAttendanceResult, last_mark: Optional[Dict[str, Any]], now: float):
    """Text and colour of the bar drawn under the video, or (None, None) for nothing."""
    if last_mark and now - last_mark["time"] < MARK_BANNER_SECONDS:
        return f"MARKED: {last_mark['name']} ({last_mark['status']})", GREEN
    if result.multiple_faces:
        return "Multiple faces - please step up one at a time", ORANGE
    if not result.processed_faces:
        return None, None
    face = result.processed_faces[0]
    if face.reason.startswith(SPOOF_RECHECK_PREFIX):
        # "...: look straight at the camera and hold still (8s)" -> "(8s)"
        return f"Rechecking: face camera, hold still {face.reason.rsplit(' ', 1)[-1]}", ORANGE
    if face.reason.startswith("Spoof suspected"):
        return "SPOOF SUSPECTED - photo / screen detected", RED
    if not face.is_recognized:
        return "Unknown person - not enrolled", RED
    name = face.student_name or face.identity
    if face.already_marked:
        return f"{name} - already marked", BLUE
    if face.liveness_state == LivenessState.LIVE:
        return f"{name} - verifying...", YELLOW
    parts = []
    if face.blink_count < face.min_blinks_required:
        times = "time" if face.min_blinks_required == 1 else "times"
        parts.append(f"Blink {face.min_blinks_required} {times} ({face.blink_count}/{face.min_blinks_required})")
    if face.challenge_type != "none" and face.challenge_state != "completed":
        direction = "LEFT" if face.challenge_type == "turn_left" else "RIGHT"
        parts.append(f"Turn head {direction}")
    if not parts:
        return f"{name} - checking...", YELLOW
    text = f"{name}: " + " + ".join(parts)
    if face.challenge_seconds_left is not None:
        text += f"  ({int(np.ceil(face.challenge_seconds_left))}s)"
    return text, YELLOW


def gallery_loading_texts(progress: Optional[Tuple[int, int]]) -> Tuple[str, str]:
    """(video bar text, status panel text) while the gallery loads; progress is (done, total) samples."""
    if not progress or not progress[1]:
        return "Loading students - please wait", "Loading the student gallery..."
    done, total = progress
    percent = done * 100 // total
    return (f"Loading students {percent}% - please wait",
            f"Loading the student gallery: {percent}% ({done:,} / {total:,} face samples not prepared yet). "
            f"Student Registry → Prepare recognition data avoids this wait.")


def draw_preview(frame: np.ndarray, text: Optional[str], color) -> np.ndarray:
    """Mirror the frame (like a mirror / front camera) and draw the instruction bar."""
    view = cv2.flip(frame, 1)
    if text:
        h, w = view.shape[:2]
        bar_h = max(36, h // 12)
        # Darken only the bar strip (blending the whole frame cost ~10 ms)
        strip = view[h - bar_h:h]
        strip[:] = (strip * 0.35 + 13).astype(np.uint8)
        scale = bar_h / 55.0
        cv2.rectangle(view, (0, h - bar_h), (8, h), color, -1)
        cv2.putText(view, text, (18, h - bar_h // 3), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, WHITE, max(1, int(scale * 2)), cv2.LINE_AA)
    return view


class VisionEngine:
    """Camera + attendance pipeline; publishes JPEG frames and a status dict."""

    def __init__(self, session_id: int, settings: Dict[str, Any], camera_index: int = 0):
        self.session_id = session_id
        self.camera_index = camera_index
        self.settings = dict(settings)
        self._stop = threading.Event()

        self._frame_cond = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._jpeg_seq = 0

        self._status_lock = threading.Lock()
        self._status: Dict[str, Any] = {"state": "starting", "session_id": session_id}

        self.camera: Optional[CameraManager] = None
        self.manager: Optional[RealTimeAttendanceManager] = None
        self._last_mark: Optional[Dict[str, Any]] = None
        self._gallery_signature: Optional[str] = None
        self._gallery_ready = threading.Event()
        self._gallery_error: Optional[str] = None
        self._gallery_progress: Optional[Tuple[int, int]] = None  # (done, total) of the first load

    # -- control ---------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()
        with self._frame_cond:
            self._frame_cond.notify_all()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def update_settings(self, settings: Dict[str, Any]) -> None:
        self.settings.update(settings)
        if self.manager is not None:
            self._apply_settings()

    def _apply_settings(self) -> None:
        manager = self.manager
        manager.require_liveness = bool(self.settings.get("require_liveness", True))
        manager.cooldown_seconds = float(self.settings.get("cooldown_seconds", 3.0))
        manager.head_turn_angle = float(self.settings.get("head_turn_angle", config.LIVENESS_HEAD_TURN_ANGLE))
        manager.require_head_turn = bool(self.settings.get("require_head_turn", config.LIVENESS_REQUIRE_CHALLENGE))
        manager._get_liveness_detector()  # push angle / head-turn switch to the detector

    # -- published data ----------------------------------------------------

    def status(self) -> Dict[str, Any]:
        with self._status_lock:
            return dict(self._status)

    def _set_status(self, **fields) -> None:
        with self._status_lock:
            self._status.update(fields, updated_at=time.time())

    def wait_for_frame(self, last_seq: int, timeout: float = 1.0):
        """Next JPEG after `last_seq`: (seq, bytes), or (last_seq, None) on timeout / stop."""
        with self._frame_cond:
            self._frame_cond.wait_for(lambda: self._jpeg_seq != last_seq or self.stopped, timeout)
            if self._jpeg_seq == last_seq or self._jpeg is None:
                return last_seq, None
            return self._jpeg_seq, self._jpeg

    def _publish_frame(self, jpeg: bytes) -> None:
        with self._frame_cond:
            self._jpeg = jpeg
            self._jpeg_seq += 1
            self._frame_cond.notify_all()

    # -- main loop -----------------------------------------------------------

    def start(self) -> bool:
        """
        Open the camera and set up the pipeline. False (status 'error') on
        failure. The gallery loads in the background, so a large one cannot
        delay the start (the app gives the engine START_TIMEOUT_SECONDS).
        """
        self.camera = CameraManager(self.camera_index, width=config.CAMERA_WIDTH or None,
                                    height=config.CAMERA_HEIGHT or None, background_reader=True)
        if not self.camera.open():
            self._set_status(state="error", error="Could not open the camera. Is another app using it?")
            return False
        self.manager = RealTimeAttendanceManager(session_id=self.session_id, marked_by="cv_realtime")
        self._apply_settings()
        self.manager.fast_liveness = True
        self.manager.min_processing_interval = config.RECOGNITION_INTERVAL
        self.manager.enable_background_processing()
        threading.Thread(target=self._gil_probe, name="gil-probe", daemon=True).start()
        threading.Thread(target=self._gallery_loop, name="gallery", daemon=True).start()
        self._set_status(state="running", error=None)
        logger.info(f"Vision engine running for session #{self.session_id}")
        return True

    def run(self) -> None:
        """Process camera frames until stopped."""
        manager, camera = self.manager, self.camera
        frames, fps_frames, fps_start = 0, 0, time.perf_counter()
        preview_fps = 0.0
        try:
            while not self.stopped:
                ok, frame = camera.read_frame()
                if not ok or frame is None:
                    self._set_status(state="running", error="Waiting for the camera...")
                    continue

                gallery_ready = self._gallery_ready.is_set()
                if gallery_ready:
                    with time_operation("liveness_per_frame"):
                        manager.update_liveness(frame)
                    result = manager.process_frame_async(frame, session_id=self.session_id)
                    manager.refresh_liveness_display(result)
                    now = time.time()
                    self._remember_mark(result, now)
                    text, color = instruction_bar(result, self._last_mark, now)
                else:
                    # Recognition waits for the gallery: with an empty one every
                    # student would be shown as "Unknown person - not enrolled"
                    result, color = None, BLUE
                    text, loading_status = gallery_loading_texts(self._gallery_progress)
                with time_operation("preview_encode"):
                    view = draw_preview(frame, text, color)
                    ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    self._publish_frame(jpeg.tobytes())

                frames += 1
                fps_frames += 1
                elapsed = time.perf_counter() - fps_start
                if elapsed >= 1.0:
                    preview_fps = fps_frames / elapsed
                    fps_frames, fps_start = 0, time.perf_counter()
                self._set_status(state="running", preview_fps=round(preview_fps, 1),
                                 error=None if gallery_ready else (self._gallery_error or loading_status),
                                 result=result_to_dict(result) if result is not None else None,
                                 last_mark=self._last_mark)

                if frames % PERF_LOG_FRAMES == 0:
                    logger.info(f"ENGINE PERFORMANCE: preview {preview_fps:.1f} fps")
                    log_performance_summary()
        except Exception as e:
            logger.error(f"Vision engine error: {e}", exc_info=True)
            self._set_status(state="error", error=str(e))
        finally:
            self.shutdown()

    def _gallery_loop(self) -> None:
        """
        Load the gallery, then pick up students enrolled / removed during the
        session - all off the frame loop. A cheap signature query runs every
        GALLERY_RELOAD_SECONDS; the slow full load only when it changed.
        set_gallery() swaps the new gallery in with one assignment.
        """
        enrollment = EnrollmentService()
        while not self.stopped:
            try:
                signature = enrollment.gallery_signature()
                if signature != self._gallery_signature:
                    with time_operation("gallery_load"):
                        gallery = enrollment.load_active_gallery(progress=self._on_gallery_progress)
                    self.manager.set_gallery(gallery)
                    self._gallery_signature = signature
                    self._gallery_ready.set()
            except Exception as e:
                if self._gallery_ready.is_set():
                    logger.warning(f"Gallery reload failed (keeping the current gallery): {e}")
                else:
                    logger.error(f"Gallery load failed (retrying): {e}")
                    self._gallery_error = f"Could not load the student gallery ({e}) - retrying"
            wait = GALLERY_RELOAD_SECONDS if self._gallery_ready.is_set() else GALLERY_RETRY_SECONDS
            if self._stop.wait(wait):
                break

    def _on_gallery_progress(self, done: int, total: int) -> None:
        if not self._gallery_ready.is_set():  # reloads during the session run silently
            self._gallery_progress = (done, total)

    def _gil_probe(self) -> None:
        """
        Diagnostics: how late a 20 ms sleep wakes up. Large values mean some
        thread holds Python's GIL for long stretches and starves the others
        (the camera reader among them). Logged as gil_probe_delay.
        """
        timer = get_timer()
        while not self.stopped:
            start = time.perf_counter()
            time.sleep(0.02)
            timer.record_time("gil_probe_delay", max(0.0, (time.perf_counter() - start - 0.02) * 1000.0))

    def _remember_mark(self, result: RealTimeAttendanceResult, now: float) -> None:
        """Keep the latest new attendance mark for the on-video banner."""
        for face in result.processed_faces:
            if face.attendance_marked and (self._last_mark is None or
                                           self._last_mark.get("attendance_id") != face.attendance_id):
                self._last_mark = {
                    "name": face.student_name or face.identity,
                    "student_id": face.student_id,
                    "status": (face.attendance_status or "present").upper(),
                    "attendance_id": face.attendance_id,
                    "time": now,
                }

    def shutdown(self) -> None:
        self.stop()
        if self.manager is not None:
            self.manager.disable_background_processing()
        if self.camera is not None:
            self.camera.release()
        self._set_status(state="stopped")
        logger.info("Vision engine stopped")


def make_handler(engine: VisionEngine, token: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # keep the console quiet
            pass

        def _send_json(self, data: Dict[str, Any], code: int = 200) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/status.json", "/health"):
                self._send_json(engine.status())
            elif path == "/stream.mjpg":
                self._stream()
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            if self.headers.get("X-Engine-Token") != token:
                self._send_json({"error": "forbidden"}, 403)
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            path = self.path.split("?", 1)[0]
            if path == "/settings":
                engine.update_settings(body)
                self._send_json({"ok": True})
            elif path == "/stop":
                self._send_json({"ok": True})
                engine.stop()
            else:
                self._send_json({"error": "not found"}, 404)

        def _stream(self):
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            seq = -1
            try:
                while not engine.stopped:
                    seq, jpeg = engine.wait_for_frame(seq)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the browser closed the stream

    return Handler


def _stop_when_parent_exits(engine: VisionEngine) -> None:
    """The parent holds our stdin open; EOF means it exited (or closed it to stop us)."""
    try:
        # Raw read of the file descriptor: unlike sys.stdin.read() it holds no
        # Python buffer lock, so the interpreter can still exit while it blocks
        while os.read(sys.stdin.fileno(), 1024):
            pass
    except OSError:
        pass
    engine.stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="IAMS real-time vision engine")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--settings", default="{}", help="JSON settings")
    parser.add_argument("--no-parent-watch", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    engine = VisionEngine(args.session, json.loads(args.settings), camera_index=args.camera)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine, args.token))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="engine-http", daemon=True).start()
    if not args.no_parent_watch:
        threading.Thread(target=_stop_when_parent_exits, args=(engine,), name="parent-watch", daemon=True).start()

    try:
        if engine.start():
            engine.run()
        else:
            # Keep serving the error status briefly so the page can show it
            engine._stop.wait(5.0)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
