"""
Camera Manager for the Intelligent Attendance Monitoring System.

Provides a clean abstraction for camera lifecycle management using OpenCV.
Handles opening, reading, and releasing camera resources safely.
"""

import logging
import threading
import time
from typing import Optional, Tuple

import config  # noqa: F401 - must precede cv2: sets the MSMF camera startup flag
import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.performance_timer import get_timer, time_operation


class CameraManager:
    """
    Manages camera lifecycle for video capture operations.

    Provides safe camera initialization, frame capture, and resource cleanup.
    Uses OpenCV VideoCapture for camera operations.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        buffer_size: int = 1,
        background_reader: bool = False,
        read_timeout: float = 1.0
    ):
        """
        Initialize the CameraManager.

        Args:
            camera_index: Camera device index (default: 0 for primary camera)
            width: Optional frame width (if None, uses camera default)
            height: Optional frame height (if None, uses camera default)
            buffer_size: Camera buffer size (1 = minimize latency, recommended for real-time)
            background_reader: If True, a daemon thread reads the camera
                continuously and read_frame() returns the newest frame, so
                the caller's processing never makes it wait on the camera and
                every camera frame is available for per-frame analysis
                (blink detection). Used by the Real-Time Attendance page.
            read_timeout: With background_reader, the longest read_frame()
                waits for a new frame (seconds) before reporting a failure
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self._capture: Optional[cv2.VideoCapture] = None
        self._is_open = False
        self.logger = get_logger(__name__)

        # Background reader state (only used when background_reader=True)
        self.background_reader = background_reader
        self.read_timeout = read_timeout
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._frame_cond = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_seq = 0      # frames captured by the reader thread
        self._returned_seq = 0   # last frame handed out by read_frame()

    def open(self) -> bool:
        """
        Open the camera and initialize video capture.

        Returns:
            bool: True if camera opened successfully, False otherwise
        """
        if self._is_open:
            self.logger.warning(f"Camera {self.camera_index} is already open")
            return True

        try:
            self.logger.info(f"Opening camera {self.camera_index}")

            with time_operation("camera_constructor"):
                self._capture = cv2.VideoCapture(self.camera_index)

            with time_operation("camera_isOpened_check"):
                is_opened = self._capture.isOpened()

            if not is_opened:
                self.logger.error(f"Failed to open camera {self.camera_index}")
                self._capture = None
                return False

            with time_operation("camera_configuration"):
                # Set resolution if specified
                if self.width is not None:
                    self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                if self.height is not None:
                    self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                # Optimize camera settings to reduce frame capture latency
                # Based on diagnostic measurements showing 70ms capture spikes
                self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                self._capture.set(cv2.CAP_PROP_FPS, 30)

                # Disable auto-exposure if possible to reduce processing variance
                # Note: Not all cameras support these settings
                try:
                    self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
                except:
                    pass  # Silently ignore if not supported

            self._is_open = True
            if self.background_reader:
                self._start_reader()
            backend_name = self._capture.getBackendName() if hasattr(self._capture, 'getBackendName') else "unknown"
            self.logger.info(f"Camera {self.camera_index} opened successfully (backend: {backend_name})")
            return True

        except Exception as e:
            self.logger.error(f"Error opening camera {self.camera_index}: {e}")
            self._capture = None
            self._is_open = False
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame from the camera.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - success: True if frame captured successfully
                - frame: The captured frame as numpy array, or None if failed
        """
        if not self._is_open or self._capture is None:
            self.logger.error("Cannot read frame: camera is not open")
            return False, None

        if self._reader_thread is not None:
            return self._read_latest_frame()

        try:
            success, frame = self._capture.read()

            if not success:
                self.logger.warning(f"Failed to read frame from camera {self.camera_index}")
                return False, None

            return True, frame

        except Exception as e:
            self.logger.error(f"Error reading frame from camera {self.camera_index}: {e}")
            return False, None

    def release(self) -> None:
        """
        Release the camera and clean up resources.

        Safe to call multiple times. Does nothing if camera is already released.
        """
        self._stop_reader()
        if self._capture is not None:
            try:
                self.logger.info(f"Releasing camera {self.camera_index}")
                self._capture.release()
            except Exception as e:
                self.logger.error(f"Error releasing camera {self.camera_index}: {e}")
            finally:
                self._capture = None
                self._is_open = False

    # ------------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------------

    def _start_reader(self) -> None:
        self._reader_stop.clear()
        self._frame_seq = 0
        self._returned_seq = 0
        self._latest_frame = None
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name=f"camera-{self.camera_index}-reader", daemon=True
        )
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        """Read frames as fast as the camera delivers them; keep only the newest."""
        capture = self._capture
        last_frame_time = None
        while not self._reader_stop.is_set() and capture is not None:
            try:
                success, frame = capture.read()
            except Exception as e:
                self.logger.error(f"Error reading frame from camera {self.camera_index}: {e}")
                success, frame = False, None
            if not success or frame is None:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            if last_frame_time is not None:
                # Time between frames from the camera = its real frame rate
                get_timer().record_time("camera_frame_interval", (now - last_frame_time) * 1000.0)
            last_frame_time = now
            with self._frame_cond:
                self._latest_frame = frame
                self._frame_seq += 1
                self._frame_cond.notify_all()

    def _read_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Newest frame not yet returned; waits up to read_timeout for one."""
        with self._frame_cond:
            got_new = self._frame_cond.wait_for(
                lambda: self._frame_seq > self._returned_seq, timeout=self.read_timeout
            )
            if not got_new or self._latest_frame is None:
                self.logger.warning(f"No new frame from camera {self.camera_index} "
                                    f"within {self.read_timeout:.1f} s")
                return False, None
            self._returned_seq = self._frame_seq
            return True, self._latest_frame

    def _stop_reader(self) -> None:
        """Stop the reader thread before the capture is released."""
        thread = self._reader_thread
        if thread is None:
            return
        self._reader_stop.set()
        thread.join(timeout=2.0)
        if thread.is_alive():
            self.logger.warning(f"Camera {self.camera_index} reader thread did not stop in time")
        self._reader_thread = None

    def is_open(self) -> bool:
        """
        Check if the camera is currently open and active.

        Returns:
            bool: True if camera is open, False otherwise
        """
        return self._is_open and self._capture is not None

    def get_frame_dimensions(self) -> Optional[Tuple[int, int]]:
        """
        Get the current frame dimensions (width, height).

        Returns:
            Optional[Tuple[int, int]]: (width, height) if camera is open, None otherwise
        """
        if not self.is_open() or self._capture is None:
            return None

        try:
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        except Exception as e:
            self.logger.error(f"Error getting frame dimensions: {e}")
            return None

    def __enter__(self):
        """Context manager entry: open the camera."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: release the camera."""
        self.release()
        return False

    def __del__(self):
        """Destructor: ensure camera is released."""
        self.release()
