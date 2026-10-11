"""
Camera Manager for the Intelligent Attendance Monitoring System.

Provides a clean abstraction for camera lifecycle management using OpenCV.
Handles opening, reading, and releasing camera resources safely.
"""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.performance_timer import time_operation


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
        buffer_size: int = 1
    ):
        """
        Initialize the CameraManager.

        Args:
            camera_index: Camera device index (default: 0 for primary camera)
            width: Optional frame width (if None, uses camera default)
            height: Optional frame height (if None, uses camera default)
            buffer_size: Camera buffer size (1 = minimize latency, recommended for real-time)
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self._capture: Optional[cv2.VideoCapture] = None
        self._is_open = False
        self.logger = get_logger(__name__)

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
        if self._capture is not None:
            try:
                self.logger.info(f"Releasing camera {self.camera_index}")
                self._capture.release()
            except Exception as e:
                self.logger.error(f"Error releasing camera {self.camera_index}: {e}")
            finally:
                self._capture = None
                self._is_open = False

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
