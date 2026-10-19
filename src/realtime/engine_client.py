"""
Starts, stops and queries the vision engine process (src/realtime/engine.py)
from the Streamlit app over its localhost HTTP API.
"""

import json
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
START_TIMEOUT_SECONDS = 45.0  # camera open + model loading


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class EngineController:
    """One engine process at a time (the camera can only be opened once)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._token: Optional[str] = None
        self.session_id: Optional[int] = None

    # -- lifecycle -------------------------------------------------------------

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, session_id: int, settings: Dict[str, Any], camera_index: int = 0) -> Tuple[bool, str]:
        """Start the engine for `session_id`; waits until the camera runs. Returns (ok, error)."""
        with self._lock:
            if self.is_running():
                if self.session_id == session_id:
                    return True, ""
                self._stop_locked()
            self._port = _free_port()
            self._token = secrets.token_hex(16)
            self.session_id = session_id
            command = [
                sys.executable, "-m", "src.realtime.engine",
                "--port", str(self._port), "--session", str(session_id),
                "--token", self._token, "--camera", str(camera_index),
                "--settings", json.dumps(settings),
            ]
            logger.info(f"Starting vision engine on port {self._port} for session #{session_id}")
            # stdin stays open: the engine stops when this process exits.
            # stdout/stderr go to this console, like the app's own logs.
            self._proc = subprocess.Popen(command, cwd=str(PROJECT_ROOT), stdin=subprocess.PIPE)

            deadline = time.time() + START_TIMEOUT_SECONDS
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    return False, "The vision engine exited while starting (see the console log)."
                status = self._get_json("/health", timeout=0.5)
                if status:
                    if status.get("state") == "running":
                        return True, ""
                    if status.get("state") == "error":
                        error = status.get("error") or "unknown error"
                        self._stop_locked()
                        return False, error
                time.sleep(0.2)
            self._stop_locked()
            return False, "The vision engine did not start in time."

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            self._post("/stop", {})
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("Vision engine did not stop; terminating it")
                proc.kill()
                proc.wait(timeout=5.0)
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        logger.info("Vision engine stopped")
        self._proc = None
        self.session_id = None

    # -- API -------------------------------------------------------------------

    @property
    def stream_url(self) -> Optional[str]:
        return f"http://127.0.0.1:{self._port}/stream.mjpg" if self._port else None

    def status(self) -> Optional[Dict[str, Any]]:
        if not self.is_running():
            return None
        return self._get_json("/status.json", timeout=0.5)

    def update_settings(self, settings: Dict[str, Any]) -> bool:
        return self.is_running() and self._post("/settings", settings)

    def _get_json(self, path: str, timeout: float) -> Optional[Dict[str, Any]]:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self._port}{path}", timeout=timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def _post(self, path: str, data: Dict[str, Any]) -> bool:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self._port}{path}",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Engine-Token": self._token or ""},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=2.0) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False
