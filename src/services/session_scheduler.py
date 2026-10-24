"""
Background scheduler that starts and closes attendance sessions on time.

Runs as a daemon thread inside the app's server process, so it does not
depend on any open page: run_app.py starts it together with the web server,
and app.py makes sure it runs (starting it is idempotent). Every
POLL_SECONDS it applies SessionService.apply_scheduled_transitions(), which
also catches up on sessions whose start / end time passed while the app was
not running. Status changes are compare-and-set updates, so the scheduler,
the pages and the vision engine can never undo each other's changes.
"""

import threading
from typing import Optional

import config
from src.services.session_service import SessionService
from src.utils.logger import get_logger

logger = get_logger(__name__)

POLL_SECONDS = 5.0  # a session starts at most this long after its start time
THREAD_NAME = "session-scheduler"

_lock = threading.Lock()
_stop = threading.Event()


def _running_thread() -> Optional[threading.Thread]:
    # Looked up by name rather than kept in a module variable: Streamlit can
    # re-import this module when its source changes, and a second scheduler
    # must not start next to the first.
    for thread in threading.enumerate():
        if thread.name == THREAD_NAME and thread.is_alive():
            return thread
    return None


def _run(poll_seconds: float) -> None:
    service = SessionService()
    while not _stop.is_set():
        try:
            if config.DATABASE_PATH.exists():  # app.py creates it on the first start
                service.apply_scheduled_transitions()
        except Exception as e:
            logger.error(f"Session scheduler check failed (retrying): {e}")
        _stop.wait(poll_seconds)


def start_session_scheduler(poll_seconds: float = POLL_SECONDS) -> bool:
    """Start the scheduler thread unless it already runs. True if it was started."""
    with _lock:
        if _running_thread() is not None:
            return False
        _stop.clear()
        threading.Thread(target=_run, args=(poll_seconds,), name=THREAD_NAME, daemon=True).start()
        logger.info(f"Session scheduler started (checks every {poll_seconds:g}s)")
        return True


def stop_session_scheduler(timeout: float = 5.0) -> None:
    """Stop the scheduler thread (tests / shutdown)."""
    _stop.set()
    thread = _running_thread()
    if thread is not None:
        thread.join(timeout)


def is_session_scheduler_running() -> bool:
    return _running_thread() is not None
