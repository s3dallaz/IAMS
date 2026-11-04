"""
The "Checks" panel of a session: check-out and spot checks (CheckService).
Shown on Real-Time Attendance and in Session Details. Refreshes itself, so
a check that opens or closes on time, and who was seen, show up by themselves.
"""

from datetime import datetime
from typing import Any, Callable, Dict, Optional

import streamlit as st

from src.services.check_service import (
    DEFAULT_CHECKOUT_MINUTES, DEFAULT_SPOT_MINUTES, KIND_LABELS, MAX_CHECK_MINUTES, TIME_FORMAT, CheckError,
    CheckKind, CheckService, CheckStatus, VoidReason,
)
from src.ui.pages.login import current_user

REFRESH_SECONDS = 5
STATUS_LABELS = {
    CheckStatus.SCHEDULED: "🕒 Scheduled",
    CheckStatus.OPEN: "🟢 Running",
    CheckStatus.DONE: "✅ Done",
    CheckStatus.VOID: "✖️ Cancelled",
}
VOID_LABELS = {
    VoidReason.CANCELLED: "✖️ Cancelled",
    VoidReason.CAMERA_OFF: "📷 Didn't count (camera was off)",
    VoidReason.SESSION_CLOSED: "✖️ Session closed first",
    VoidReason.REPLACED: "✖️ Replaced by a check-out started by hand",
    VoidReason.SETTING_OFF: "✖️ Check-out was turned off",
}
KIND_ICONS = {CheckKind.CHECKOUT: "🚪", CheckKind.SPOT: "🔎"}
NO_END_TIME_NOTE = ("🚪 Check-out is on, but this session has **no end time**, so it can't start by itself. "
                    "It runs when you **close the session** (you'll be asked), or press **🚪 Start check-out now**.")


def status_text(check: Dict[str, Any]) -> str:
    if check['status'] == CheckStatus.VOID:
        return VOID_LABELS.get(check.get('void_reason'), STATUS_LABELS[CheckStatus.VOID])
    return STATUS_LABELS.get(check['status'], check['status'])


def _hhmm(value: Optional[str]) -> str:
    try:
        return datetime.strptime(str(value)[:19], TIME_FORMAT).strftime('%H:%M')
    except (TypeError, ValueError):
        return "—"


def render_checks_panel(session: Dict[str, Any], key: str) -> None:
    """The checks of `session` (a dict with id, status, end_time); `key` keeps widget keys apart per page."""
    if session['status'] == 'closed':
        _render_check_list(session['id'], show_hidden_times=True)
        return
    _live_panel(session['id'], session['status'], session.get('end_time'), key)


@st.fragment(run_every=REFRESH_SECONDS)
def _live_panel(session_id: int, session_status: str, end_time: Optional[str], key: str) -> None:
    service = CheckService()
    st.markdown("#### 🔎 Check-out & spot checks")
    st.caption("Students who checked in show their face again during a check. Anyone who checked in "
               "before a check and is not seen during it is marked **🚪 Left early**.")

    open_check = service.get_open_check(session_id)
    if open_check:
        _render_running(service, open_check, key)
    elif session_status == 'active':
        if not end_time and service.checkout_still_due(session_id):
            st.info(NO_END_TIME_NOTE)
        _render_start_buttons(service, session_id, end_time, key)
    else:
        st.caption("Checks can be started once the session is running; you can schedule one already.")

    _render_schedule(service, session_id, end_time, key)
    _render_check_list(session_id)


def _render_running(service: CheckService, check: Dict[str, Any], key: str) -> None:
    due = service.repository.count_due(check['session_id'], check['starts_at'])
    label = KIND_LABELS[check['kind']]
    closes = " The session **closes by itself** when it ends." if check.get('close_session_after') else ""
    st.warning(f"{KIND_ICONS[check['kind']]} **{label} running until {_hhmm(check['ends_at'])}** - "
               f"**{check['marked_count']} of {due}** checked. Keep the camera on.{closes}")
    if not check.get('camera_seen_at'):
        st.caption("📷 The camera hasn't picked up this check yet (it takes up to 2 seconds). "
                   "If the camera stays off the whole time, the check doesn't count and nobody is marked Left early.")
    col_finish, col_cancel = st.columns(2)
    if col_finish.button("✅ Finish now", key=f"{key}_finish_{check['id']}", width="stretch",
                         help="End it now; whoever was not seen is marked Left early"):
        try:
            flagged = service.finish_check_now(check['id'])
        except CheckError as e:
            st.error(f"❌ {e}")
            return
        if flagged is None:
            st.toast(f"{label} didn't count: the camera wasn't running during it, so nobody was marked "
                     f"Left early.", icon="📷")
        else:
            st.toast(f"{label} finished: {flagged} marked Left early.", icon="✅")
        st.rerun()
    if col_cancel.button("✖️ Cancel", key=f"{key}_cancel_{check['id']}", width="stretch",
                         help="Stop it without marking anybody Left early"):
        _run(lambda: service.cancel_check(check['id']), f"{label} cancelled.")


def _render_start_buttons(service: CheckService, session_id: int, end_time: Optional[str], key: str) -> None:
    user = current_user() or {}
    col_minutes, col_spot, col_checkout = st.columns([1, 2, 2], vertical_alignment="bottom")
    minutes = int(col_minutes.number_input("Minutes", 1, MAX_CHECK_MINUTES, DEFAULT_SPOT_MINUTES,
                                           key=f"{key}_spot_minutes"))
    if col_spot.button("🔎 Spot check now", key=f"{key}_spot_now", type="primary", width="stretch"):
        _run(lambda: service.start_spot_check(session_id, minutes, user.get('full_name'), user.get('id')),
             "🔎 Spot check started: students show their face again.")
    if col_checkout.button("🚪 Start check-out now", key=f"{key}_checkout_now", width="stretch",
                           help="Until the session's end time; without an end time, for the minutes on the left"):
        _run(lambda: service.start_checkout_now(session_id, None if end_time else minutes,
                                                user.get('full_name'), user.get('id')),
             "🚪 Check-out started.")


def _render_schedule(service: CheckService, session_id: int, end_time: Optional[str], key: str) -> None:
    user = current_user() or {}
    with st.popover("🕒 Schedule a spot check", width="stretch"):
        when = st.radio("When", ["random", "time"], key=f"{key}_when", horizontal=True,
                        format_func=lambda w: "At a random time" if w == "random" else "At a time I choose",
                        help="A random time is not shown, so students can't guess it")
        at_time = None
        if when == "time":
            at_time = st.time_input("Start at", value=None, step=60, key=f"{key}_at")
        elif not end_time:
            st.caption("A random time needs the session's end time.")
        minutes = int(st.number_input("Minutes", 1, MAX_CHECK_MINUTES, DEFAULT_SPOT_MINUTES, key=f"{key}_sched_min"))
        if st.button("Schedule", key=f"{key}_schedule", type="primary", width="stretch"):
            if when == "time" and at_time is None:
                st.error("Choose a time.")
                return
            at = datetime.combine(datetime.now().date(), at_time) if at_time else None
            _run(lambda: service.schedule_spot_check(session_id, at=at, randomly=when == "random", minutes=minutes,
                                                     created_by=user.get('full_name'),
                                                     created_by_user_id=user.get('id')),
                 "🕒 Spot check scheduled" + (" at a random time." if when == "random" else "."))


def _render_check_list(session_id: int, show_hidden_times: bool = False) -> None:
    service = CheckService()
    checks = service.list_checks(session_id)
    if not checks:
        st.caption("No checks in this session.")
        return
    rows = []
    for check in checks:
        hidden = check['is_random'] and check['status'] == CheckStatus.SCHEDULED and not show_hidden_times
        rows.append({
            'Check': f"{KIND_ICONS[check['kind']]} {KIND_LABELS[check['kind']]}"
                     + (" (random)" if check['is_random'] else "")
                     + (" (automatic)" if check['kind'] == CheckKind.CHECKOUT and not check['created_by'] else ""),
            'Time': "hidden until it starts" if hidden else f"{_hhmm(check['starts_at'])} - {_hhmm(check['ends_at'])}",
            'Status': status_text(check),
            'Checked': (f"{check['marked_count']} of {check['due_count']}"
                        if check['status'] in (CheckStatus.OPEN, CheckStatus.DONE) else "—"),
        })
    st.dataframe(rows, hide_index=True, width="stretch")
    scheduled = [c for c in checks if c['status'] == CheckStatus.SCHEDULED]
    if scheduled and not show_hidden_times:
        labels = {c['id']: f"{KIND_LABELS[c['kind']]} #{c['id']}" for c in scheduled}
        col_pick, col_button = st.columns([3, 1], vertical_alignment="bottom")
        chosen = col_pick.selectbox("Scheduled check", list(labels), format_func=labels.get,
                                    key=f"cancel_pick_{session_id}")
        if col_button.button("✖️ Cancel it", key=f"cancel_scheduled_{session_id}", width="stretch"):
            _run(lambda: service.cancel_check(chosen), "Scheduled check cancelled.")


def render_close_session(session: Dict[str, Any], key: str, close_now: Callable[[], None]) -> None:
    """
    The inside of a "Close session" popover. With check-out on and no
    check-out yet, offer to run it first and close by itself afterwards.
    `close_now` closes the session at once (each page has its own).
    """
    service = CheckService()
    session_id = session['id']
    user = current_user() or {}
    running = service.get_open_check(session_id)

    if running and running['kind'] == CheckKind.CHECKOUT:
        if running.get('close_session_after'):
            st.info(f"🚪 The session closes by itself when the check-out ends at **{_hhmm(running['ends_at'])}**.")
        else:
            st.write(f"🚪 A check-out is running until **{_hhmm(running['ends_at'])}**.")
            if st.button("🚪 Close when the check-out ends", key=f"{key}_after", type="primary", width="stretch"):
                _close_after(service, session_id, None, user)
        if st.button("Close now", key=f"{key}_now", width="stretch",
                     help="The check-out ends now: whoever was not seen yet is marked Left early"):
            close_now()
        return

    if service.checkout_still_due(session_id):
        st.write("🚪 **Check-out is on and hasn't run yet.** Run it first: students show their face before "
                 "leaving, and the session closes by itself when it ends. Anyone who doesn't show up is "
                 "marked Left early.")
        minutes = int(st.number_input("Check-out minutes", 1, MAX_CHECK_MINUTES,
                                      int(session.get('checkout_minutes') or DEFAULT_CHECKOUT_MINUTES),
                                      key=f"{key}_minutes"))
        if st.button("🚪 Check-out now, then close", key=f"{key}_checkout", type="primary", width="stretch"):
            _close_after(service, session_id, minutes, user)
        if st.button("Close now, without check-out", key=f"{key}_skip", width="stretch",
                     help="Nobody is marked Left early for the check-out"):
            close_now()
        return

    st.write("Closing ends attendance for this session. It cannot be reopened.")
    if st.button("Close session", key=f"{key}_close", type="primary", width="stretch"):
        close_now()


def _close_after(service: CheckService, session_id: int, minutes: Optional[int], user: Dict[str, Any]) -> None:
    try:
        check = service.start_checkout_then_close(session_id, minutes, user.get('full_name'), user.get('id'))
    except CheckError as e:
        st.error(f"❌ {e}")
        return
    st.toast(f"🚪 Check-out running; the session closes by itself at {_hhmm(check['ends_at'])}.", icon="✅")
    st.rerun()


def _run(action, success: str) -> None:
    try:
        action()
    except CheckError as e:
        st.error(f"❌ {e}")
        return
    st.toast(success, icon="✅")
    st.rerun(scope="fragment")
