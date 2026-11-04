"""
Check-out and spot checks: catching students who check in and then leave.

A check is a short window in which students who already checked in show
their face to the camera again (with the same liveness test):
- check-out: at the end of a session - automatic in its last N minutes when
  the session has check-out on and an end time, or started by hand
- spot check: started by the instructor ("now"), or scheduled at a time,
  or at a random time students can't guess

When a check ends, everyone who was present / late BEFORE it began but was
not seen during it is flagged "left early" (they still count as attended).
The camera reports each check it sees running (camera_seen_at). A check the
camera never saw - it was off the whole time - does not count (void,
reason 'camera_off'), so nobody is flagged for a camera that was off. When
the camera was on, the check counts even if nobody showed up: then everyone
due is flagged. Only one check is open in a session at a time.

Closing a session with check-out on: the instructor can run the check-out
first and let the session close by itself when it ends
(start_checkout_then_close, close_session_after on the check).
The background timer (session_scheduler) calls apply_due() every few seconds.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.repositories.check_repository import CheckRepository
from src.data.repositories.session_repository import SessionRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_SPOT_MINUTES = 5
DEFAULT_CHECKOUT_MINUTES = 10
MAX_CHECK_MINUTES = 60
RANDOM_EARLIEST_MINUTES = 2      # a random spot check starts at least this long from now


class CheckKind:
    CHECKOUT = 'checkout'
    SPOT = 'spot'


class CheckStatus:
    SCHEDULED = 'scheduled'
    OPEN = 'open'
    DONE = 'done'
    VOID = 'void'


class VoidReason:
    """Why a check did not count (attendance_checks.void_reason)."""
    CANCELLED = 'cancelled'            # the instructor cancelled it
    CAMERA_OFF = 'camera_off'          # the camera never ran while it was open
    SESSION_CLOSED = 'session_closed'  # the session closed before it started
    REPLACED = 'replaced'              # an automatic check-out replaced by one started by hand
    SETTING_OFF = 'setting_off'        # check-out switched off / end time removed


KIND_LABELS = {CheckKind.CHECKOUT: "Check-out", CheckKind.SPOT: "Spot check"}


class CheckError(Exception):
    """A check can't be started / scheduled; the message says why."""


def _fmt(moment: datetime) -> str:
    return moment.strftime(TIME_FORMAT)


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:19], TIME_FORMAT)
    except ValueError:
        return None


class CheckService:
    """Schedules, opens and closes checks and records who was seen."""

    def __init__(self, database_path: Optional[Path] = None):
        self.repository = CheckRepository(database_path)
        self.session_repository = SessionRepository(database_path)

    # -- starting / scheduling ---------------------------------------------------------

    def start_spot_check(self, session_id: int, minutes: int = DEFAULT_SPOT_MINUTES,
                         created_by: Optional[str] = None, created_by_user_id: Optional[int] = None,
                         now: Optional[datetime] = None) -> int:
        """A spot check that starts now and lasts `minutes`."""
        now = self._now(now)
        self._require_active(session_id)
        self._require_none_open(session_id)
        return self._create(session_id, CheckKind.SPOT, now, now + timedelta(minutes=self._minutes(minutes)),
                            CheckStatus.OPEN, False, created_by, created_by_user_id)

    def schedule_spot_check(self, session_id: int, at: Optional[datetime] = None, randomly: bool = False,
                            minutes: int = DEFAULT_SPOT_MINUTES, created_by: Optional[str] = None,
                            created_by_user_id: Optional[int] = None, now: Optional[datetime] = None) -> int:
        """
        A spot check at `at`, or with `randomly` at a random moment before the
        session's end (and before its check-out), so students can't guess it.
        """
        now = self._now(now)
        session = self._require_open_session(session_id)
        minutes = self._minutes(minutes)
        end = _parse(session.get('end_time'))
        if randomly:
            if end is None:
                raise CheckError("A random spot check needs the session's end time.")
            latest = end - timedelta(minutes=minutes)
            if session.get('checkout_enabled'):
                latest -= timedelta(minutes=session.get('checkout_minutes') or DEFAULT_CHECKOUT_MINUTES)
            earliest = max(now, _parse(session['start_time']) or now) + timedelta(minutes=RANDOM_EARLIEST_MINUTES)
            if latest <= earliest:
                raise CheckError("Not enough time left before the end for a random spot check.")
            at = earliest + timedelta(seconds=random.randint(0, int((latest - earliest).total_seconds())))
            at = at.replace(microsecond=0)
        if at is None:
            raise CheckError("Choose when the spot check starts.")
        if at <= now:
            raise CheckError("That time has already passed. Use 'Spot check now' instead.")
        if end is not None and at + timedelta(minutes=minutes) > end:
            raise CheckError("The spot check would run past the session's end time.")
        return self._create(session_id, CheckKind.SPOT, at, at + timedelta(minutes=minutes),
                            CheckStatus.SCHEDULED, randomly, created_by, created_by_user_id)

    def start_checkout_now(self, session_id: int, minutes: Optional[int] = None,
                           created_by: Optional[str] = None, created_by_user_id: Optional[int] = None,
                           now: Optional[datetime] = None, close_session_after: bool = False) -> int:
        """
        Check-out from now: until the session's end time, or for `minutes`
        (default: its setting). close_session_after: close the session when it ends.
        """
        now = self._now(now)
        session = self._require_active(session_id)
        self._require_none_open(session_id)
        if minutes is None:
            end = _parse(session.get('end_time'))
            ends_at = end if end and end > now else now + timedelta(
                minutes=session.get('checkout_minutes') or DEFAULT_CHECKOUT_MINUTES)
        else:
            ends_at = now + timedelta(minutes=self._minutes(minutes))
        # An automatic check-out still waiting is replaced by this one
        for check in self.repository.list_checks(session_id):
            if check['kind'] == CheckKind.CHECKOUT and check['status'] == CheckStatus.SCHEDULED:
                self.repository.set_status(check['id'], CheckStatus.VOID, void_reason=VoidReason.REPLACED)
        return self._create(session_id, CheckKind.CHECKOUT, now, ends_at, CheckStatus.OPEN, False,
                            created_by, created_by_user_id, close_session_after)

    def checkout_still_due(self, session_id: int) -> bool:
        """True when the session has check-out on and no check-out has run (or is running) yet."""
        session = self.session_repository.get_by_id(session_id)
        if not session or not session.get('checkout_enabled'):
            return False
        return not any(c['kind'] == CheckKind.CHECKOUT and c['status'] in (CheckStatus.OPEN, CheckStatus.DONE)
                       for c in self.repository.list_checks(session_id))

    def start_checkout_then_close(self, session_id: int, minutes: Optional[int] = None,
                                  created_by: Optional[str] = None, created_by_user_id: Optional[int] = None,
                                  now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        "Close the session" with check-out on: run the check-out now (for
        `minutes`, default the session's setting) and close the session by
        itself when it ends. A check-out already running just gets
        "close after". Returns the check-out.

        Raises:
            CheckError: The session isn't running, or a spot check is running
        """
        session = self._require_active(session_id)
        running = self.repository.get_open_check(session_id)
        if running and running['kind'] == CheckKind.CHECKOUT:
            self.repository.set_close_session_after(running['id'])
            return self.repository.get_check(running['id'])
        if running:
            raise CheckError("A spot check is running. Finish or cancel it first.")
        minutes = minutes or session.get('checkout_minutes') or DEFAULT_CHECKOUT_MINUTES
        check_id = self.start_checkout_now(session_id, minutes, created_by, created_by_user_id, now,
                                           close_session_after=True)
        logger.info(f"Session {session_id} closes when its check-out #{check_id} ends")
        return self.repository.get_check(check_id)

    def cancel_check(self, check_id: int) -> None:
        """Cancel a scheduled check (or stop an open one without flagging anybody)."""
        check = self.repository.get_check(check_id)
        if not check or check['status'] not in (CheckStatus.SCHEDULED, CheckStatus.OPEN):
            raise CheckError("Only a scheduled or running check can be cancelled.")
        self.repository.set_status(check_id, CheckStatus.VOID, void_reason=VoidReason.CANCELLED)
        logger.info(f"{KIND_LABELS[check['kind']]} #{check_id} of session {check['session_id']} cancelled")

    def finish_check_now(self, check_id: int, now: Optional[datetime] = None) -> Optional[int]:
        """
        End an open check now (e.g. everyone is through). Returns how many
        were flagged, or None when it did not count (the camera never saw it).
        """
        check = self.repository.get_check(check_id)
        if not check or check['status'] != CheckStatus.OPEN:
            raise CheckError("Only a running check can be finished.")
        return self._finalize(check, _fmt(self._now(now)))

    # -- the camera ---------------------------------------------------------------------

    def get_open_check(self, session_id: int) -> Optional[Dict[str, Any]]:
        return self.repository.get_open_check(session_id)

    def checked_student_ids(self, check_id: int) -> set:
        return self.repository.marked_student_ids(check_id)

    def camera_saw(self, check_id: int, now: Optional[datetime] = None) -> None:
        """The camera is running while this check is open, so the check counts."""
        self.repository.set_camera_seen(check_id, _fmt(self._now(now)))

    def record_mark(self, check_id: int, student_id: int, detection_confidence: Optional[float] = None,
                    liveness_score: Optional[float] = None, marked_by: str = 'system',
                    now: Optional[datetime] = None) -> bool:
        """
        A student was seen during an open check. True if newly recorded.

        Raises:
            CheckError: The check is not open (anymore)
        """
        check = self.repository.get_check(check_id)
        if not check or check['status'] != CheckStatus.OPEN:
            raise CheckError("This check is not running anymore.")
        return self.repository.add_mark(check_id, student_id, _fmt(self._now(now)),
                                        detection_confidence, liveness_score, marked_by)

    # -- listing ------------------------------------------------------------------------

    def list_checks(self, session_id: int) -> List[Dict[str, Any]]:
        """The session's checks with marked_count and due_count (who has to show up)."""
        checks = self.repository.list_checks(session_id)
        for check in checks:
            check['due_count'] = self.repository.count_due(session_id, check['starts_at'])
        return checks

    # -- the background timer ------------------------------------------------------------

    def apply_due(self, now: Optional[datetime] = None) -> Dict[str, List[int]]:
        """
        Keep automatic check-outs in step with their sessions, open checks
        whose time came, close those whose time is over (flagging who
        missed them), and wind down checks of closed / deleted sessions.
        Cheap when nothing is due.
        """
        now = self._now(now)
        changed: Dict[str, List[int]] = {'opened': [], 'closed': [], 'voided': []}
        self._sync_automatic_checkouts(now)

        for check in self.repository.checks_with_status(CheckStatus.SCHEDULED):
            if check['session_status'] == 'closed':
                self.repository.set_status(check['id'], CheckStatus.VOID, void_reason=VoidReason.SESSION_CLOSED)
                changed['voided'].append(check['id'])
            elif check['session_status'] == 'active' and _parse(check['starts_at']) <= now:
                if self.repository.get_open_check(check['session_id']):
                    continue   # another check is still running; this one opens after it
                self.repository.set_status(check['id'], CheckStatus.OPEN)
                changed['opened'].append(check['id'])
                logger.info(f"{KIND_LABELS[check['kind']]} #{check['id']} of session "
                            f"{check['session_id']} opened")

        for check in self.repository.checks_with_status(CheckStatus.OPEN):
            if check['session_status'] == 'closed':   # closed early: the check ends with it
                self._finalize(check, min(check['ends_at'], _fmt(now)))
                changed['closed'].append(check['id'])
            elif _parse(check['ends_at']) <= now:
                self._finalize(check, check['ends_at'])
                changed['closed'].append(check['id'])
        return changed

    def _sync_automatic_checkouts(self, now: datetime) -> None:
        """One automatic check-out per active session that wants one, in its last N minutes."""
        for row in self.repository.sessions_with_checkout():
            end = _parse(row['end_time'])
            if end is None:
                continue
            start = end - timedelta(minutes=row['checkout_minutes'] or DEFAULT_CHECKOUT_MINUTES)
            if row['check_id'] is None:
                if end > now:
                    self._create(row['session_id'], CheckKind.CHECKOUT, max(start, now), end,
                                 CheckStatus.SCHEDULED, False, None, None)
            elif row['check_status'] == CheckStatus.SCHEDULED and (row['starts_at'], row['ends_at']) != (
                    _fmt(start), _fmt(end)):
                self.repository.set_times(row['check_id'], _fmt(max(start, now)), _fmt(end))
        for check_id in self.repository.scheduled_checkouts_without_setting():
            self.repository.set_status(check_id, CheckStatus.VOID, void_reason=VoidReason.SETTING_OFF)

    def _finalize(self, check: Dict[str, Any], ends_at: str) -> Optional[int]:
        """
        Close a check. The camera never saw it (off the whole time) -> void,
        nobody flagged, returns None. Otherwise flag who missed it and return
        how many. A check-out with "close after" then closes its session.
        """
        label = f"{KIND_LABELS[check['kind']]} #{check['id']} of session {check['session_id']}"
        flagged: Optional[int] = None
        if not check.get('camera_seen_at') and not self.repository.marked_student_ids(check['id']):
            self.repository.set_status(check['id'], CheckStatus.VOID, ends_at=ends_at,
                                       void_reason=VoidReason.CAMERA_OFF)
            logger.warning(f"{label} ended without the camera running; it does not count, nobody flagged")
        else:
            self.repository.set_status(check['id'], CheckStatus.DONE, ends_at=ends_at)
            flagged = self.repository.flag_missed(check['id'])
            logger.info(f"{label} closed; {flagged} student(s) flagged as left early")
        if check.get('close_session_after'):
            session = self.session_repository.get_by_id(check['session_id'])
            if session and session['status'] == 'active':
                self.session_repository.close_session(check['session_id'])
                logger.info(f"Session {check['session_id']} closed after its check-out")
        return flagged

    # -- helpers ---------------------------------------------------------------------

    def _create(self, session_id: int, kind: str, starts: datetime, ends: datetime, status: str,
                is_random: bool, created_by: Optional[str], created_by_user_id: Optional[int],
                close_session_after: bool = False) -> int:
        check_id = self.repository.create_check(session_id, kind, _fmt(starts), _fmt(ends), status, is_random,
                                                created_by, created_by_user_id, close_session_after)
        logger.info(f"{KIND_LABELS[kind]} #{check_id} for session {session_id}: {status}, "
                    f"{_fmt(starts)} - {_fmt(ends)}" + (" (random time)" if is_random else ""))
        return check_id

    @staticmethod
    def _now(now: Optional[datetime]) -> datetime:
        return (now or datetime.now()).replace(microsecond=0)

    @staticmethod
    def _minutes(minutes: int) -> int:
        if not isinstance(minutes, int) or not 1 <= minutes <= MAX_CHECK_MINUTES:
            raise CheckError(f"A check lasts 1 to {MAX_CHECK_MINUTES} minutes.")
        return minutes

    def _require_open_session(self, session_id: int) -> Dict[str, Any]:
        session = self.session_repository.get_by_id(session_id)
        if not session:
            raise CheckError("Session not found.")
        if session['status'] == 'closed':
            raise CheckError("This session is closed.")
        return session

    def _require_active(self, session_id: int) -> Dict[str, Any]:
        session = self._require_open_session(session_id)
        if session['status'] != 'active':
            raise CheckError("The session must be running (active) for a check.")
        return session

    def _require_none_open(self, session_id: int) -> None:
        if self.repository.get_open_check(session_id):
            raise CheckError("A check is already running in this session. Finish or cancel it first.")
