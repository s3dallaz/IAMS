"""
Choosing the camera by name instead of by number.

Windows numbers cameras in an order that can change, and a phone linked to
the PC ("S24 Ultra (Windows Virtual Camera)", Camo, ...) can take number 0,
so "open camera 0" may open the phone instead of the webcam. Here the cameras
are read from Media Foundation - the list OpenCV's MSMF backend numbers its
cameras from - without opening any of them.

Which camera is used, first match wins:
1. The camera picked on the page (saved in the settings table)
2. CAMERA_NAME in .env (part of its name, ignoring case and spaces)
3. The first camera that is not a phone / virtual one
"""

import ctypes
import json
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import config  # noqa: F401 - must precede cv2: sets the MSMF camera startup flag
import cv2

from src.computer_vision.camera_manager import CameraManager
from src.data.repositories.settings_repository import SettingsRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Name parts of cameras that are not a camera attached to this PC
VIRTUAL_CAMERA_MARKERS = (
    "virtual", "camo", "droidcam", "iriun", "ivcam", "epoccam", "phone", "manycam", "xsplit", "snap camera",
    "nvidia broadcast",
)
CAMERA_SETTING_KEY = "camera_choice"


@dataclass(frozen=True)
class CameraDevice:
    """A camera Windows reports. device_id tells two identical webcams apart."""
    index: int        # number to open it with (MSMF)
    name: str
    device_id: str = ""


@dataclass(frozen=True)
class SavedCamera:
    """The camera picked on the page."""
    device_id: str
    name: str


class CameraNotFoundError(Exception):
    """The chosen camera is not connected. The message says what to do."""


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_virtual_camera(name: str) -> bool:
    return any(marker in name.lower() for marker in VIRTUAL_CAMERA_MARKERS)


def choose_camera(names: Optional[Sequence[str]], preferred: str = "") -> int:
    """
    The number of the camera to open when none was picked on the page.

    Args:
        names: Camera names in numbering order; None when they could not be read
        preferred: Part of the wanted camera's name (ignoring case and spaces)

    Raises:
        CameraNotFoundError: A preferred camera is not connected. Never falls
            back to another camera then: it could be someone's phone.
    """
    if names is None:
        return 0
    if preferred:
        wanted = _name_key(preferred)
        for index, name in enumerate(names):
            if wanted in _name_key(name):
                return index
        found = ", ".join(names) if names else "none"
        raise CameraNotFoundError(
            f"Camera '{preferred}' was not found. Check that it is plugged in, or change CAMERA_NAME "
            f"in the .env file. Cameras found: {found}.")
    for index, name in enumerate(names):
        if not is_virtual_camera(name):
            return index
    return 0


def find_saved_camera(devices: Sequence[CameraDevice], saved: SavedCamera) -> Optional[CameraDevice]:
    """
    The picked camera among `devices`: the same device, or else one with the
    same name (a webcam's device ID can change when it moves to another USB port).
    """
    for device in devices:
        if saved.device_id and device.device_id == saved.device_id:
            return device
    for device in devices:
        if device.name == saved.name:
            return device
    return None


def list_camera_devices() -> Optional[List[CameraDevice]]:
    """The cameras in OpenCV's MSMF numbering order, or None if they cannot be read. Opens no camera."""
    if sys.platform != "win32":
        return None
    try:
        return _media_foundation_cameras()
    except Exception as e:
        logger.warning(f"Could not read the camera list: {e}")
        return None


def load_saved_camera(database_path: Optional[Path] = None) -> Optional[SavedCamera]:
    """The camera picked on the page, or None for automatic."""
    try:
        value = SettingsRepository(database_path).get(CAMERA_SETTING_KEY)
        if not value:
            return None
        data = json.loads(value)
        return SavedCamera(device_id=data.get("device_id", ""), name=data["name"])
    except Exception as e:  # a settings problem must not stop attendance
        logger.warning(f"Could not read the saved camera: {e}")
        return None


def save_camera(camera: Optional[CameraDevice], database_path: Optional[Path] = None) -> None:
    """Remember the picked camera; None goes back to automatic."""
    from src.services.activity_log import ActivityLog, Category   # services import this module
    repository = SettingsRepository(database_path)
    if camera is None:
        repository.delete(CAMERA_SETTING_KEY)
    else:
        repository.set(CAMERA_SETTING_KEY, json.dumps({"device_id": camera.device_id, "name": camera.name}),
                       description="Camera picked on the Real-Time Attendance page")
    ActivityLog(database_path).record(Category.SETTINGS, 'camera_chosen',
                                      f"Chose the camera: {camera.name}" if camera
                                      else "Camera set back to automatic")


def automatic_camera(devices: Sequence[CameraDevice]) -> Optional[CameraDevice]:
    """The camera used when none is picked, or None if CAMERA_NAME's camera is not connected / none exists."""
    try:
        index = choose_camera([device.name for device in devices], config.CAMERA_NAME)
    except CameraNotFoundError:
        return None
    return devices[index] if index < len(devices) else None


def create_camera(**kwargs) -> CameraManager:
    """
    A CameraManager for the chosen camera (see the module description).
    Takes CameraManager's other arguments. Does not open it yet.

    Raises:
        CameraNotFoundError: The picked camera, or CAMERA_NAME's, is not connected
    """
    devices = list_camera_devices()
    if devices is None:
        return CameraManager(choose_camera(None), **kwargs)

    saved = load_saved_camera()
    if saved:
        device = find_saved_camera(devices, saved)
        if device is None:
            found = ", ".join(d.name for d in devices) or "none"
            raise CameraNotFoundError(
                f"The chosen camera '{saved.name}' was not found. Plug it in, or choose another one under "
                f"Camera on this page. Cameras found: {found}.")
        index = device.index
    else:
        index = choose_camera([d.name for d in devices], config.CAMERA_NAME)
    if index < len(devices):
        logger.info(f"Using camera {index}: {devices[index].name}")
    # The number comes from the MSMF list, so it must be opened with MSMF
    return CameraManager(index, api_preference=cv2.CAP_MSMF, **kwargs)


class _GUID(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 16)]

    def __init__(self, text: str):
        super().__init__()
        ctypes.memmove(self.data, uuid.UUID(text).bytes_le, 16)


def _com_method(obj, index: int, *argtypes):
    """Method `index` of a COM object's vtable."""
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)(vtable[index])


def _release(obj) -> None:
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])(obj)


def _media_foundation_cameras() -> List[CameraDevice]:
    """MFEnumDeviceSources for video capture devices: what OpenCV's MSMF camera numbers refer to."""
    # IMFAttributes vtable positions
    get_allocated_string, set_guid = 13, 24
    source_type = _GUID("c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3")     # MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE
    video_capture = _GUID("8ac3587a-4ae7-42d8-99e0-0a6013eef90f")   # ..._SOURCE_TYPE_VIDCAP_GUID
    friendly_name = _GUID("60d0e559-52f8-4fa2-bbce-acdb34a8ec01")   # MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME
    symbolic_link = _GUID("58f0aad8-22bf-4f8a-bb3d-d2c4978c6e2f")   # ..._VIDCAP_SYMBOLIC_LINK

    mfplat, mf, ole32 = ctypes.WinDLL("mfplat"), ctypes.WinDLL("mf"), ctypes.WinDLL("ole32")
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    com_started = ole32.CoInitializeEx(None, 0) >= 0   # fails harmlessly if this thread already chose a mode
    mfplat.MFStartup(0x20070, 0)
    attributes = ctypes.c_void_p()

    def read_string(device, key) -> str:
        text = ctypes.c_void_p()
        length = ctypes.c_uint32()
        result = _com_method(device, get_allocated_string, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p),
                             ctypes.POINTER(ctypes.c_uint32))(device, ctypes.byref(key), ctypes.byref(text),
                                                              ctypes.byref(length))
        if result < 0 or not text:
            return ""
        value = ctypes.wstring_at(text)
        ole32.CoTaskMemFree(text)
        return value

    try:
        result = mfplat.MFCreateAttributes(ctypes.byref(attributes), 1)
        if result < 0:
            raise OSError(f"MFCreateAttributes failed: {result:#x}")
        _com_method(attributes, set_guid, ctypes.POINTER(_GUID), ctypes.POINTER(_GUID))(
            attributes, ctypes.byref(source_type), ctypes.byref(video_capture))

        devices = ctypes.POINTER(ctypes.c_void_p)()
        count = ctypes.c_uint32()
        result = mf.MFEnumDeviceSources(attributes, ctypes.byref(devices), ctypes.byref(count))
        if result < 0:
            raise OSError(f"MFEnumDeviceSources failed: {result:#x}")

        cameras = []
        for i in range(count.value):
            device = devices[i]
            cameras.append(CameraDevice(index=i, name=read_string(device, friendly_name).strip(),
                                        device_id=read_string(device, symbolic_link)))
            _release(device)
        if devices:
            ole32.CoTaskMemFree(devices)
        return cameras
    finally:
        if attributes:
            _release(attributes)
        mfplat.MFShutdown()
        if com_started:
            ole32.CoUninitialize()
