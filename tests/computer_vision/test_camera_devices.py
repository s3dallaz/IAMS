"""
Tests for choosing the camera by name: a phone linked to the PC can take
camera number 0, and must not be opened instead of the webcam.
"""

from unittest.mock import patch

import cv2
import pytest

from src.computer_vision import camera_devices
from src.computer_vision.camera_devices import (
    CameraNotFoundError, choose_camera, create_camera, is_virtual_camera,
)

PHONE_FIRST = ["S24 Ultra (Windows Virtual Camera)", "Brio 100"]


def test_phone_is_skipped_when_no_camera_is_named():
    assert choose_camera(PHONE_FIRST) == 1


@pytest.mark.parametrize("name", ["S24 Ultra (Windows Virtual Camera)", "Camo", "DroidCam Source 3",
                                  "Camera (NVIDIA Broadcast)", "OBS Virtual Camera"])
def test_virtual_cameras_are_recognised(name):
    assert is_virtual_camera(name)


def test_real_camera_is_not_virtual():
    assert not is_virtual_camera("Brio 100")
    assert not is_virtual_camera("Integrated Webcam")


@pytest.mark.parametrize("preferred", ["Brio 100", "BRIO100", "brio", "bRiO 100"])
def test_named_camera_is_found_ignoring_case_and_spaces(preferred):
    assert choose_camera(PHONE_FIRST, preferred) == 1


def test_a_named_phone_can_still_be_chosen():
    assert choose_camera(PHONE_FIRST, "S24 Ultra") == 0


def test_missing_named_camera_never_falls_back_to_another_one():
    with pytest.raises(CameraNotFoundError) as error:
        choose_camera(["S24 Ultra (Windows Virtual Camera)"], "Brio 100")
    assert "Brio 100" in str(error.value) and "S24 Ultra" in str(error.value)
    with pytest.raises(CameraNotFoundError):
        choose_camera([], "Brio 100")


def test_only_virtual_cameras_or_unknown_names_use_number_zero():
    assert choose_camera(["Camo"]) == 0
    assert choose_camera([]) == 0
    assert choose_camera(None, "Brio 100") == 0   # names could not be read


def test_create_camera_opens_the_number_with_the_backend_it_came_from():
    with patch.object(camera_devices, "list_cameras", return_value=PHONE_FIRST), \
            patch.object(camera_devices.config, "CAMERA_NAME", ""):
        camera = create_camera(width=1280, height=720)
    assert camera.camera_index == 1
    assert camera.api_preference == cv2.CAP_MSMF
    assert camera.width == 1280


def test_create_camera_without_names_lets_opencv_choose():
    with patch.object(camera_devices, "list_cameras", return_value=None), \
            patch.object(camera_devices.config, "CAMERA_NAME", ""):
        camera = create_camera()
    assert camera.camera_index == 0 and camera.api_preference is None


def test_camera_manager_passes_the_backend_to_opencv():
    from src.computer_vision.camera_manager import CameraManager
    with patch("src.computer_vision.camera_manager.cv2.VideoCapture") as capture:
        capture.return_value.isOpened.return_value = False
        CameraManager(1, api_preference=cv2.CAP_MSMF).open()
    capture.assert_called_once_with(1, cv2.CAP_MSMF)


def test_engine_only_gets_a_camera_number_when_one_is_asked_for():
    from src.realtime.engine_client import EngineController
    controller = EngineController()
    with patch("src.realtime.engine_client.subprocess.Popen") as popen, \
            patch.object(EngineController, "_wait_until_running", return_value=(True, ""), create=True):
        popen.side_effect = RuntimeError("stop here")
        with pytest.raises(RuntimeError):
            controller.start(1, {})
    assert "--camera" not in popen.call_args[0][0]
