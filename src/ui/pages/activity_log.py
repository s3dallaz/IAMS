"""
Activity Log page (Admin and IT): who did what, and when.

Filters by date, person, kind of change and text; the newest entries come
first. Entries can't be edited or deleted. A filtered list can be saved as
CSV. Session Details shows the same log for one session
(render_session_history).
"""

import csv
import io
import json
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List

import streamlit as st

from src.services.activity_log import CATEGORY_LABELS, SYSTEM_ACTOR, ActivityLog
from src.services.auth_service import ROLE_LABELS

SHOWN_ROWS = 1000          # newest entries listed on the page
EXPORT_ROWS = 50000        # most entries in a CSV
DEFAULT_DAYS = 7
UNKNOWN_PERSON = "Unknown (not signed in)"
EVERYONE = "__everyone__"
ALL_KINDS = "__all__"
FAILED_SIGN_IN_ACTIONS = ('sign_in_failed', 'sign_in_refused', 'account_locked')


def _who(entry: Dict[str, Any]) -> str:
    name = entry.get('full_name') or entry.get('username') or "?"
    return f"{name} ({entry['username']})" if entry.get('username') else name


def _rows(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        'Time': entry['at'],
        'Who': _who(entry),
        'What happened': entry['summary'],
        'Kind': CATEGORY_LABELS.get(entry['category'], entry['category']),
        'Role': ROLE_LABELS.get(entry.get('role'), "—"),
    } for entry in entries]


def _height(rows: int, most: int = 560) -> int:
    """Table height: fits a few rows, scrolls when there are many."""
    return min(most, 38 + 35 * max(rows, 1))


def _csv(entries: List[Dict[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Time", "Who", "Username", "Role", "Kind", "What happened", "Details"])
    for entry in entries:
        writer.writerow([entry['at'], entry.get('full_name') or "", entry.get('username') or "",
                         ROLE_LABELS.get(entry.get('role'), ""), CATEGORY_LABELS.get(entry['category'], entry['category']),
                         entry['summary'], entry.get('details') or ""])
    return buffer.getvalue().encode("utf-8-sig")   # Excel shows Arabic names with the BOM


def _render_details(entry: Dict[str, Any]) -> None:
    st.markdown(f"**{entry['at']}** · {_who(entry)} · "
                f"{CATEGORY_LABELS.get(entry['category'], entry['category'])}")
    st.write(entry['summary'])
    if entry.get('details'):
        try:
            st.json(json.loads(entry['details']), expanded=True)
        except ValueError:
            st.code(entry['details'])
    st.caption(f"Entry #{entry['id']} · action `{entry['action']}`"
               + (f" · {entry['target_type']} #{entry['target_id']}" if entry.get('target_type') else ""))


def render_activity_log() -> None:
    log = ActivityLog()
    st.title("🕵️ Activity Log")
    st.caption("Who did what, and when: sign-ins, accounts, students, courses, sessions, attendance changed by "
               "hand, check-outs and spot checks. Entries can't be edited or deleted. The camera's own marks are "
               "not listed here (each attendance record shows the camera made it).")

    # -- filters ---------------------------------------------------------------------
    col_dates, col_who, col_kind = st.columns([2, 2, 2])
    today = date.today()
    dates = col_dates.date_input("Dates", value=(today - timedelta(days=DEFAULT_DAYS - 1), today),
                                 max_value=today, key="activity_dates", format="DD/MM/YYYY")
    start, end = (dates if isinstance(dates, (tuple, list)) and len(dates) == 2
                  else (dates[0] if isinstance(dates, (tuple, list)) and dates else today,) * 2)

    people = {EVERYONE: "Everyone"}
    people.update({p['username']: f"{p['full_name'] or p['username']} ({p['username']})" for p in log.people()})
    people[SYSTEM_ACTOR['full_name']] = SYSTEM_ACTOR['full_name']
    people[UNKNOWN_PERSON] = UNKNOWN_PERSON
    who = col_who.selectbox("Person", list(people), format_func=people.get, key="activity_who")

    kinds = {ALL_KINDS: "All kinds", **CATEGORY_LABELS}
    kind = col_kind.selectbox("Kind", list(kinds), format_func=kinds.get, key="activity_kind")
    search = st.text_input("Search", placeholder="Words in the entry, e.g. a student's name or ID, a session name",
                           key="activity_search").strip()

    filters: Dict[str, Any] = {
        'date_from': datetime.combine(start, time.min).strftime('%Y-%m-%d %H:%M:%S'),
        'date_to': datetime.combine(end, time.max).strftime('%Y-%m-%d %H:%M:%S'),
        'category': None if kind == ALL_KINDS else kind,
        'search': search or None,
    }
    if who in (SYSTEM_ACTOR['full_name'], UNKNOWN_PERSON):
        filters['full_name'] = who
    elif who != EVERYONE:
        filters['username'] = who

    # -- numbers ---------------------------------------------------------------------
    total = log.count(**filters)
    failed = log.count(**{**filters, 'category': None, 'search': None, 'actions': FAILED_SIGN_IN_ACTIONS})
    col_total, col_failed, col_download = st.columns([1, 1, 2], vertical_alignment="bottom")
    col_total.metric("Entries", f"{total:,}")
    col_failed.metric("Failed sign-ins / lockouts", f"{failed:,}",
                      help="Wrong passwords or recovery codes, unknown usernames and locked accounts, "
                           "in the dates and for the person chosen")
    if total:
        col_download.download_button(
            "⬇️ Save as CSV", data=lambda: _csv(log.search(limit=EXPORT_ROWS, **filters)),
            file_name=f"activity_log_{start:%Y%m%d}_{end:%Y%m%d}.csv", mime="text/csv", width="stretch",
            help=f"All matching entries (up to {EXPORT_ROWS:,}), with their details")

    # -- the list --------------------------------------------------------------------
    if not total:
        st.info("Nothing matches these filters.")
        return
    entries = log.search(limit=SHOWN_ROWS, **filters)
    if total > SHOWN_ROWS:
        st.caption(f"Showing the newest {SHOWN_ROWS:,} of {total:,} entries. Narrow the dates or filters, or "
                   f"save them all as CSV.")
    st.caption("Click a row to see its details.")
    picked = st.dataframe(
        _rows(entries), hide_index=True, width="stretch", height=_height(len(entries)), key="activity_table",
        on_select="rerun", selection_mode="single-row",
        column_config={'Time': st.column_config.TextColumn(width="medium"),
                       'Role': st.column_config.TextColumn(width="small"),
                       'What happened': st.column_config.TextColumn(width="large")},
    )
    rows = picked.selection.rows if picked and picked.selection else []
    if rows and rows[0] < len(entries):
        with st.container(border=True):
            _render_details(entries[rows[0]])


def render_session_history(session_id: int, key: str) -> None:
    """Session Details: everything the log has about one session (newest first)."""
    log = ActivityLog()
    total = log.count(session_id=session_id)
    with st.expander(f"🕘 History of this session ({total})", key=f"{key}_open"):
        if not total:
            st.caption("Nothing recorded yet.")
            return
        entries = log.search(limit=SHOWN_ROWS, session_id=session_id)
        st.dataframe([{k: v for k, v in row.items() if k != 'Role'} for row in _rows(entries)],
                     hide_index=True, width="stretch", key=f"{key}_table", height=_height(len(entries), 400),
                     column_config={'What happened': st.column_config.TextColumn(width="large")})
        if total > SHOWN_ROWS:
            st.caption(f"The newest {SHOWN_ROWS:,} of {total:,}; the Activity Log page has them all.")
