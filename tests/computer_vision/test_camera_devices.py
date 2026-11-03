"""
Tests for choosing the camera by name: a phone linked to the PC can take
camera number 0, and must not be opened instead of the webcam.
"""

from unittest.mock import patch

import cv2
import pytest

from src.computer_vision import camera_devices
from src.computer_vision.camera_devices import (
    CameraDevice, CameraNotFoundError, SavedCamera, choose_camera, create_camera, is_virtual_camera,
    load_saved_camera, save_camera,
)
from src.data.repositories.settings_repository import SettingsRepository
from src.database.schema import initialize_database
from src.ui.pages.camera_picker import AUTOMATIC, MISSING, camera_labels, picker_options

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


def _devices(*names):
    return [CameraDevice(i, name, f"id-{i}") for i, name in enumerate(names)]


def _create(devices, saved=None, camera_name="", **kwargs):
    with patch.object(camera_devices, "list_camera_devices", return_value=devices), \
            patch.object(camera_devices, "load_saved_camera", return_value=saved), \
            patch.object(camera_devices.config, "CAMERA_NAME", camera_name):
        return create_camera(**kwargs)


def test_create_camera_opens_the_number_with_the_backend_it_came_from():
    camera = _create(_devices(*PHONE_FIRST), width=1280, height=720)
    assert camera.camera_index == 1
    assert camera.api_preference == cv2.CAP_MSMF
    assert camera.width == 1280


def test_create_camera_without_names_lets_opencv_choose():
    camera = _create(None)
    assert camera.camera_index == 0 and camera.api_preference is None


# -- the camera picked on the page ---------------------------------------------

def test_picked_camera_beats_the_setting_and_automatic():
    devices = _devices("S24 Ultra (Windows Virtual Camera)", "Brio 100", "Other Cam")
    assert _create(devices, SavedCamera("id-2", "Other Cam"), camera_name="Brio 100").camera_index == 2
    assert _create(devices, camera_name="Brio 100").camera_index == 1
    assert _create(devices).camera_index == 1


def test_a_phone_can_be_picked_on_purpose():
    devices = _devices("S24 Ultra (Windows Virtual Camera)", "Brio 100")
    assert _create(devices, SavedCamera("id-0", "S24 Ultra (Windows Virtual Camera)")).camera_index == 0


def test_identical_webcams_are_told_apart_by_device_id():
    devices = _devices("Logitech C920", "Logitech C920")
    assert _create(devices, SavedCamera("id-1", "Logitech C920")).camera_index == 1


def test_camera_moved_to_another_usb_port_is_found_by_name():
    devices = _devices("Phone Link", "Brio 100")
    assert _create(devices, SavedCamera("old-usb-path", "Brio 100")).camera_index == 1


def test_missing_picked_camera_gives_a_clear_error_and_no_other_camera():
    with pytest.raises(CameraNotFoundError) as error:
        _create(_devices("S24 Ultra (Windows Virtual Camera)"), SavedCamera("id-9", "Brio 100"))
    assert "Brio 100" in str(error.value) and "S24 Ultra" in str(error.value)


def test_choice_is_saved_and_cleared(tmp_path):
    db = tmp_path / "settings.db"
    assert initialize_database(db)
    assert load_saved_camera(db) is None
    save_camera(CameraDevice(1, "Brio 100", "id-1"), db)
    assert load_saved_camera(db) == SavedCamera("id-1", "Brio 100")
    save_camera(CameraDevice(0, "Other", "id-0"), db)     # replaces it
    assert load_saved_camera(db) == SavedCamera("id-0", "Other")
    save_camera(None, db)                                 # back to automatic
    assert load_saved_camera(db) is None


def test_a_broken_setting_means_automatic(tmp_path):
    db = tmp_path / "settings.db"
    assert initialize_database(db)
    SettingsRepository(db).set(camera_devices.CAMERA_SETTING_KEY, "not json")
    assert load_saved_camera(db) is None


def test_picker_labels_and_options():
    identical = _devices("Logitech C920", "Logitech C920", "Camo")
    labels = list(camera_labels(identical).values())
    assert labels == ["Logitech C920 (1)", "Logitech C920 (2)", "Camo - phone / virtual"]

    devices = _devices("S24 Ultra (Windows Virtual Camera)", "Brio 100")
    with patch.object(camera_devices.config, "CAMERA_NAME", ""):
        options = picker_options(devices, None)
        assert options[AUTOMATIC] == "Automatic (Brio 100)"
        assert MISSING not in options
        options = picker_options(devices, SavedCamera("gone", "Old Webcam"))
    assert options[MISSING] == "Old Webcam - not connected"


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
