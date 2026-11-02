"""
Choosing the camera by name instead of by number.

Windows numbers cameras in an order that can change, and a phone linked to
the PC ("S24 Ultra (Windows Virtual Camera)", Camo, ...) can take number 0,
so "open camera 0" may open the phone instead of the webcam. Here the names
are read from Media Foundation - the list OpenCV's MSMF backend numbers its
cameras from - without opening any camera.

CAMERA_NAME in .env picks the camera (part of its name, ignoring case and
spaces). Without it, the first camera that is not a phone / virtual one is used.
"""

import ctypes
import logging
import re
import sys
import uuid
from typing import List, Optional, Sequence

import config  # noqa: F401 - must precede cv2: sets the MSMF camera startup flag
import cv2

from src.computer_vision.camera_manager import CameraManager

logger = logging.getLogger(__name__)

# Name parts of cameras that are not a camera attached to this PC
VIRTUAL_CAMERA_MARKERS = (
    "virtual", "camo", "droidcam", "iriun", "ivcam", "epoccam", "phone", "manycam", "xsplit", "snap camera",
    "nvidia broadcast",
)


class CameraNotFoundError(Exception):
    """The camera named by CAMERA_NAME is not connected. The message says what to do."""


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_virtual_camera(name: str) -> bool:
    return any(marker in name.lower() for marker in VIRTUAL_CAMERA_MARKERS)


def choose_camera(names: Optional[Sequence[str]], preferred: str = "") -> int:
    """
    The number of the camera to open.

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


def list_cameras() -> Optional[List[str]]:
    """Camera names in OpenCV's MSMF numbering order, or None if they cannot be read. Opens no camera."""
    if sys.platform != "win32":
        return None
    try:
        return _media_foundation_cameras()
    except Exception as e:
        logger.warning(f"Could not read the camera names: {e}")
        return None


def create_camera(**kwargs) -> CameraManager:
    """
    A CameraManager for the configured camera (see the module description).
    Takes CameraManager's other arguments. Does not open it yet.

    Raises:
        CameraNotFoundError: CAMERA_NAME is set but that camera is not connected
    """
    names = list_cameras()
    index = choose_camera(names, config.CAMERA_NAME)
    if names is None:
        return CameraManager(index, **kwargs)
    if index < len(names):
        logger.info(f"Using camera {index}: {names[index]}")
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


def _media_foundation_cameras() -> List[str]:
    """MFEnumDeviceSources for video capture devices: what OpenCV's MSMF camera numbers refer to."""
    # IMFAttributes vtable positions
    get_allocated_string, set_guid = 13, 24
    source_type = _GUID("c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3")     # MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE
    video_capture = _GUID("8ac3587a-4ae7-42d8-99e0-0a6013eef90f")   # ..._SOURCE_TYPE_VIDCAP_GUID
    friendly_name = _GUID("60d0e559-52f8-4fa2-bbce-acdb34a8ec01")   # MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME

    mfplat, mf, ole32 = ctypes.WinDLL("mfplat"), ctypes.WinDLL("mf"), ctypes.WinDLL("ole32")
    com_started = ole32.CoInitializeEx(None, 0) >= 0   # fails harmlessly if this thread already chose a mode
    mfplat.MFStartup(0x20070, 0)
    attributes = ctypes.c_void_p()
    try:
        ctypes.windll.kernel32.SetLastError(0)
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

        names = []
        for i in range(count.value):
            device = devices[i]
            text, length = ctypes.c_wchar_p(), ctypes.c_uint32()
            _com_method(device, get_allocated_string, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_wchar_p),
                        ctypes.POINTER(ctypes.c_uint32))(device, ctypes.byref(friendly_name),
                                                         ctypes.byref(text), ctypes.byref(length))
            names.append(text.value or "")
            ole32.CoTaskMemFree(text)
            _release(device)
        if devices:
            ole32.CoTaskMemFree(devices)
        return names
    finally:
        if attributes:
            _release(attributes)
        mfplat.MFShutdown()
        if com_started:
            ole32.CoUninitialize()
