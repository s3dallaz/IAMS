"""
Tests for CameraManager.

Tests camera lifecycle management without requiring a physical camera.
Uses mocking to simulate OpenCV VideoCapture behavior.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.computer_vision.camera_manager import CameraManager


class TestCameraManagerInitialization:
    """Test CameraManager initialization and opening."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        manager = CameraManager()
        assert manager.camera_index == 0
        assert manager.width is None
        assert manager.height is None
        assert not manager.is_open()
    
    def test_init_with_custom_camera_index(self):
        """Test initialization with custom camera index."""
        manager = CameraManager(camera_index=1)
        assert manager.camera_index == 1
        assert not manager.is_open()
    
    def test_init_with_resolution(self):
        """Test initialization with custom resolution."""
        manager = CameraManager(width=1280, height=720)
        assert manager.width == 1280
        assert manager.height == 720
        assert not manager.is_open()
    
    @patch('cv2.VideoCapture')
    def test_open_success(self, mock_video_capture):
        """Test successful camera opening."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager(camera_index=0)
        result = manager.open()
        
        assert result is True
        assert manager.is_open()
        mock_video_capture.assert_called_once_with(0)
        mock_capture.isOpened.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_open_failure(self, mock_video_capture):
        """Test failed camera opening."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = False
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager(camera_index=0)
        result = manager.open()
        
        assert result is False
        assert not manager.is_open()
    
    @patch('cv2.VideoCapture')
    def test_open_with_resolution(self, mock_video_capture):
        """Test opening camera with custom resolution."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager(camera_index=0, width=1920, height=1080)
        result = manager.open()
        
        assert result is True
        mock_capture.set.assert_any_call(3, 1920)  # CAP_PROP_FRAME_WIDTH = 3
        mock_capture.set.assert_any_call(4, 1080)  # CAP_PROP_FRAME_HEIGHT = 4
    
    @patch('cv2.VideoCapture')
    def test_open_when_already_open(self, mock_video_capture):
        """Test opening camera when already open."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        
        # Try to open again
        result = manager.open()
        assert result is True
        # VideoCapture should only be called once
        assert mock_video_capture.call_count == 1
    
    @patch('cv2.VideoCapture')
    def test_open_exception_handling(self, mock_video_capture):
        """Test exception handling during camera opening."""
        mock_video_capture.side_effect = Exception("Camera initialization error")
        
        manager = CameraManager()
        result = manager.open()
        
        assert result is False
        assert not manager.is_open()


class TestCameraManagerFrameCapture:
    """Test frame capture functionality."""
    
    @patch('cv2.VideoCapture')
    def test_read_frame_success(self, mock_video_capture):
        """Test successful frame reading."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        
        # Create a mock frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture.read.return_value = (True, mock_frame)
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        
        success, frame = manager.read_frame()
        
        assert success is True
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
    
    @patch('cv2.VideoCapture')
    def test_read_frame_failure(self, mock_video_capture):
        """Test failed frame reading."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_capture.read.return_value = (False, None)
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        
        success, frame = manager.read_frame()
        
        assert success is False
        assert frame is None
    
    def test_read_frame_when_not_open(self):
        """Test reading frame when camera is not open."""
        manager = CameraManager()
        
        success, frame = manager.read_frame()
        
        assert success is False
        assert frame is None
    
    @patch('cv2.VideoCapture')
    def test_read_frame_exception_handling(self, mock_video_capture):
        """Test exception handling during frame reading."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_capture.read.side_effect = Exception("Frame read error")
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        
        success, frame = manager.read_frame()
        
        assert success is False
        assert frame is None


class TestCameraManagerRelease:
    """Test camera release functionality."""
    
    @patch('cv2.VideoCapture')
    def test_release(self, mock_video_capture):
        """Test camera release."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        assert manager.is_open()
        
        manager.release()
        
        assert not manager.is_open()
        mock_capture.release.assert_called_once()
    
    def test_release_when_not_open(self):
        """Test releasing camera when not open."""
        manager = CameraManager()
        
        # Should not raise exception
        manager.release()
        assert not manager.is_open()
    
    @patch('cv2.VideoCapture')
    def test_release_multiple_times(self, mock_video_capture):
        """Test releasing camera multiple times."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        
        manager.release()
        manager.release()  # Should not raise exception
        
        assert not manager.is_open()
        mock_capture.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_release_exception_handling(self, mock_video_capture):
        """Test exception handling during release."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_capture.release.side_effect = Exception("Release error")
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        
        # Should handle exception gracefully
        manager.release()
        assert not manager.is_open()


class TestCameraManagerState:
    """Test camera state management."""
    
    def test_is_open_initially_false(self):
        """Test that camera is not open initially."""
        manager = CameraManager()
        assert not manager.is_open()
    
    @patch('cv2.VideoCapture')
    def test_is_open_after_successful_open(self, mock_video_capture):
        """Test is_open returns True after successful open."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        
        assert manager.is_open()
    
    @patch('cv2.VideoCapture')
    def test_is_open_after_failed_open(self, mock_video_capture):
        """Test is_open returns False after failed open."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = False
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        
        assert not manager.is_open()
    
    @patch('cv2.VideoCapture')
    def test_is_open_after_release(self, mock_video_capture):
        """Test is_open returns False after release."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        manager.release()
        
        assert not manager.is_open()


class TestCameraManagerUtilities:
    """Test utility methods."""
    
    @patch('cv2.VideoCapture')
    def test_get_frame_dimensions_success(self, mock_video_capture):
        """Test getting frame dimensions."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.side_effect = lambda prop: 1920 if prop == 3 else 1080
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        
        dimensions = manager.get_frame_dimensions()
        
        assert dimensions is not None
        assert dimensions == (1920, 1080)
    
    def test_get_frame_dimensions_when_not_open(self):
        """Test getting frame dimensions when camera not open."""
        manager = CameraManager()
        
        dimensions = manager.get_frame_dimensions()
        
        assert dimensions is None
    
    @patch('cv2.VideoCapture')
    def test_get_frame_dimensions_exception_handling(self, mock_video_capture):
        """Test exception handling when getting dimensions."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.side_effect = Exception("Get property error")
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        
        dimensions = manager.get_frame_dimensions()
        
        assert dimensions is None


class TestCameraManagerContextManager:
    """Test context manager functionality."""
    
    @patch('cv2.VideoCapture')
    def test_context_manager_opens_and_closes(self, mock_video_capture):
        """Test context manager properly opens and closes camera."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        
        with manager:
            assert manager.is_open()
        
        assert not manager.is_open()
        mock_capture.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_context_manager_with_exception(self, mock_video_capture):
        """Test context manager closes camera even with exception."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        
        with pytest.raises(ValueError):
            with manager:
                assert manager.is_open()
                raise ValueError("Test exception")
        
        assert not manager.is_open()
        mock_capture.release.assert_called_once()


class TestCameraManagerDestructor:
    """Test destructor behavior."""
    
    @patch('cv2.VideoCapture')
    def test_destructor_releases_camera(self, mock_video_capture):
        """Test that destructor releases camera."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture
        
        manager = CameraManager()
        manager.open()
        
        # Explicitly delete to trigger destructor
        del manager
        
        mock_capture.release.assert_called()


class TestCameraManagerBackgroundReader:
    """The opt-in reader thread used by the Real-Time Attendance page."""

    @staticmethod
    def _camera_at(fps, frames_ok=True):
        """Mock VideoCapture delivering numbered frames at `fps`."""
        import itertools
        import time as _time
        capture = Mock()
        capture.isOpened.return_value = True
        counter = itertools.count(1)

        def read():
            _time.sleep(1.0 / fps)
            if not frames_ok:
                return False, None
            return True, np.full((4, 4, 3), next(counter) % 256, dtype=np.uint8)

        capture.read.side_effect = read
        return capture

    @patch('cv2.VideoCapture')
    def test_read_returns_the_newest_frame(self, mock_video_capture):
        import time as _time
        mock_video_capture.return_value = self._camera_at(fps=200)
        manager = CameraManager(background_reader=True)
        manager.open()
        try:
            _, first = manager.read_frame()
            _time.sleep(0.1)                     # slow consumer: frames pile up
            _, newest = manager.read_frame()
            assert int(newest[0, 0, 0]) > int(first[0, 0, 0]) + 1  # stale frames skipped
        finally:
            manager.release()

    @patch('cv2.VideoCapture')
    def test_each_read_waits_for_a_new_frame(self, mock_video_capture):
        mock_video_capture.return_value = self._camera_at(fps=100)
        manager = CameraManager(background_reader=True)
        manager.open()
        try:
            values = [int(manager.read_frame()[1][0, 0, 0]) for _ in range(5)]
            assert len(set(values)) == 5          # never the same frame twice
        finally:
            manager.release()

    @patch('cv2.VideoCapture')
    def test_read_times_out_when_the_camera_stalls(self, mock_video_capture):
        mock_video_capture.return_value = self._camera_at(fps=100, frames_ok=False)
        manager = CameraManager(background_reader=True, read_timeout=0.1)
        manager.open()
        try:
            assert manager.read_frame() == (False, None)
        finally:
            manager.release()

    @patch('cv2.VideoCapture')
    def test_release_stops_the_reader_before_releasing_the_camera(self, mock_video_capture):
        capture = self._camera_at(fps=100)
        mock_video_capture.return_value = capture
        manager = CameraManager(background_reader=True)
        manager.open()
        thread = manager._reader_thread
        assert thread.is_alive()

        manager.release()

        assert not thread.is_alive()
        capture.release.assert_called_once()
        assert manager.read_frame() == (False, None)


class TestCameraManagerFrameSize:
    """Frames larger than width x height are shrunk (cameras such as the Brio 100 ignore the request)."""

    @staticmethod
    def _camera_delivering(shape):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros(shape, dtype=np.uint8))
        return capture

    @pytest.mark.parametrize("delivered, expected", [
        ((1080, 1920, 3), (720, 1280, 3)),     # 1080p camera -> 720p
        ((720, 1280, 3), (720, 1280, 3)),      # already 720p: untouched
        ((480, 640, 3), (480, 640, 3)),        # smaller: never enlarged
        ((1200, 1600, 3), (720, 960, 3)),      # 4:3 fits by height, aspect kept
    ])
    @patch('cv2.VideoCapture')
    def test_frames_are_shrunk_to_fit(self, mock_video_capture, delivered, expected):
        mock_video_capture.return_value = self._camera_delivering(delivered)
        manager = CameraManager(width=1280, height=720)
        manager.open()
        assert manager.read_frame()[1].shape == expected

    @patch('cv2.VideoCapture')
    def test_without_a_size_frames_keep_the_camera_size(self, mock_video_capture):
        mock_video_capture.return_value = self._camera_delivering((1080, 1920, 3))
        manager = CameraManager()
        manager.open()
        assert manager.read_frame()[1].shape == (1080, 1920, 3)

    @patch('cv2.VideoCapture')
    def test_the_background_reader_shrinks_frames_too(self, mock_video_capture):
        mock_video_capture.return_value = self._camera_delivering((1080, 1920, 3))
        manager = CameraManager(width=1280, height=720, background_reader=True)
        manager.open()
        try:
            assert manager.read_frame()[1].shape == (720, 1280, 3)
        finally:
            manager.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
