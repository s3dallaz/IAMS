"""
The "Camera" dropdown shown on the pages that use the camera. Listing the
cameras opens none of them. The choice is saved for this PC's database and
used the next time a camera starts.
"""

from typing import Dict, List, Optional

import streamlit as st

from src.computer_vision.camera_devices import (
    CameraDevice, SavedCamera, automatic_camera, find_saved_camera, is_virtual_camera,
    list_camera_devices, load_saved_camera, save_camera,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

AUTOMATIC = "automatic"
MISSING = "missing"


def camera_labels(devices: List[CameraDevice]) -> Dict[CameraDevice, str]:
    """Dropdown text per camera; identical webcams are numbered, phones / virtual cameras are marked."""
    labels = {}
    seen: Dict[str, int] = {}
    total = {name: sum(d.name == name for d in devices) for name in {d.name for d in devices}}
    for device in devices:
        seen[device.name] = seen.get(device.name, 0) + 1
        label = device.name
        if total[device.name] > 1:
            label += f" ({seen[device.name]})"
        if is_virtual_camera(device.name):
            label += " - phone / virtual"
        labels[device] = label
    return labels


def picker_options(devices: List[CameraDevice], saved: Optional[SavedCamera]) -> Dict[object, str]:
    """{option: text}, starting with Automatic, then the cameras; a picked camera that is gone stays listed."""
    automatic = automatic_camera(devices)
    options: Dict[object, str] = {
        AUTOMATIC: f"Automatic ({automatic.name})" if automatic else "Automatic"
    }
    options.update(camera_labels(devices))
    if saved and find_saved_camera(devices, saved) is None:
        options[MISSING] = f"{saved.name} - not connected"
    return options


def render_camera_picker(key: str, disabled: bool = False) -> None:
    """
    The dropdown. Each page gives its own widget `key`.

    Args:
        key: Widget key
        disabled: True while the camera runs (a change applies at the next start)
    """
    devices = list_camera_devices()   # takes milliseconds, so a camera plugged in shows up at once
    if devices is None:
        st.caption("📷 The camera list is not available here; the default camera is used.")
        return
    if not devices:
        st.warning("📷 No camera found. Plug one in.")
        return

    saved = load_saved_camera()
    options = picker_options(devices, saved)
    if saved is None:
        current = AUTOMATIC
    else:
        current = find_saved_camera(devices, saved) or MISSING

    # The saved choice decides, also if another page or tab changed it
    if st.session_state.get(key) != current:
        st.session_state[key] = current

    st.selectbox(
        "📷 Camera", list(options), format_func=options.get, key=key, disabled=disabled,
        on_change=_save_choice, args=(key,),
        help="Stop the camera to change this." if disabled else
             "Which camera to use. Automatic takes the first real camera and skips phones linked to this PC.",
    )
    if current == MISSING:
        st.warning(f"The chosen camera '{saved.name}' is not connected. Plug it in or choose another one.")


def _save_choice(key: str) -> None:
    """Dropdown callback: remember the choice."""
    choice = st.session_state.get(key)
    if choice == MISSING:
        return
    try:
        save_camera(None if choice == AUTOMATIC else choice)
        logger.info(f"Camera choice saved: {'automatic' if choice == AUTOMATIC else choice.name}")
    except Exception as e:
        logger.error(f"Could not save the camera choice: {e}")
        st.toast(f"Could not save the camera choice: {e}", icon="❌")
