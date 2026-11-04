"""
Accounts, login and permissions.

Roles, lowest to highest: instructor < admin < it.
- instructor: runs their own classes (sessions, attendance, own reports)
- admin: all data, and instructor accounts
- it: everything, including admin / IT accounts

Passwords are stored as scrypt hashes (Python's hashlib, no extra package):
a random salt per password, so equal passwords give different hashes, and
the password itself is never stored. After MAX_FAILED_ATTEMPTS wrong
passwords in a row the account is locked for LOCK_MINUTES (an admin / IT
user can unlock it sooner).

Forgotten password: every account can have a one-time recovery code (shown
once, stored as an scrypt hash like passwords). The owner enters it with a
new password on the login page; the code is then used up and a new one is
issued. Wrong codes count towards the same lockout as wrong passwords.
"""

import base64
import hashlib
import hmac
import math
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.repositories.course_repository import CourseRepository
from src.data.repositories.user_repository import UserRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Role:
    INSTRUCTOR = 'instructor'
    ADMIN = 'admin'
    IT = 'it'


ROLE_LEVELS = {Role.INSTRUCTOR: 1, Role.ADMIN: 2, Role.IT: 3}
ROLE_LABELS = {Role.INSTRUCTOR: "Instructor", Role.ADMIN: "Admin", Role.IT: "IT"}

MAX_FAILED_ATTEMPTS = 5
LOCK_MINUTES = 15
MIN_PASSWORD_LENGTH = 8
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,29}$")
DB_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'   # UTC, like the database's CURRENT_TIMESTAMP
RECOVERY_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"   # no look-alikes (0/O, 1/I/L)
RECOVERY_CODE_LENGTH = 16                                    # shown as 4 groups of 4

# scrypt cost: ~50 ms per check - cheap for one login, slow for guessing
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1


class AuthError(Exception):
    """Base exception for account errors; the message can be shown as it is."""


class InvalidCredentialsError(AuthError):
    """Wrong username or password (deliberately not saying which)."""


class AccountLockedError(AuthError):
    """Too many wrong passwords; locked until the time in the message."""

    def __init__(self, message: str, locked_until: datetime):
        super().__init__(message)
        self.locked_until = locked_until


class AccountInactiveError(AuthError):
    """The account was deactivated."""


class InvalidUserDataError(AuthError):
    """A username, name, role or password is not acceptable."""


class PermissionDeniedError(AuthError):
    """The signed-in user may not do this."""


def hash_password(password: str) -> str:
    """'scrypt$n$r$p$salt$hash' with a new random salt."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode('utf-8'), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    encode = lambda b: base64.b64encode(b).decode('ascii')
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${encode(salt)}${encode(digest)}"


def verify_password(password: str, stored: Optional[str]) -> bool:
    """True if `password` matches the stored hash (constant-time comparison)."""
    try:
        scheme, n, r, p, salt, digest = (stored or "").split('$')
        if scheme != 'scrypt':
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(password.encode('utf-8'), salt=base64.b64decode(salt),
                                n=int(n), r=int(r), p=int(p), dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


# A hash checked for unknown usernames, so they take as long as known ones
_DUMMY_HASH = hash_password(secrets.token_hex(8))


def generate_recovery_code() -> str:
    """A new random code like 'K7QF-2MXP-9TRA-4WNE' (about 79 bits: not guessable)."""
    raw = "".join(secrets.choice(RECOVERY_CODE_ALPHABET) for _ in range(RECOVERY_CODE_LENGTH))
    return "-".join(raw[i:i + 4] for i in range(0, RECOVERY_CODE_LENGTH, 4))


def normalize_recovery_code(code: str) -> str:
    """'k7qf 2mxp-9tra4wne' -> 'K7QF2MXP9TRA4WNE': case, spaces and dashes don't matter."""
    return re.sub(r"[^A-Z0-9]", "", (code or "").upper())


def has_role(user: Optional[Dict[str, Any]], role: str) -> bool:
    """True if `user` has `role` or a higher one."""
    return bool(user) and ROLE_LEVELS.get(user.get('role'), 0) >= ROLE_LEVELS[role]


def can_manage(actor: Optional[Dict[str, Any]], target_role: str) -> bool:
    """Admins manage instructor accounts; IT manages every account."""
    if has_role(actor, Role.IT):
        return True
    return has_role(actor, Role.ADMIN) and target_role == Role.INSTRUCTOR


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def _parse_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:19], DB_TIME_FORMAT)
    except ValueError:
        return None


def validate_password(password: str, username: str = "") -> None:
    """Raises InvalidUserDataError for a password that is too weak."""
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise InvalidUserDataError(f"The password must have at least {MIN_PASSWORD_LENGTH} characters.")
    if username and password.lower() == username.lower():
        raise InvalidUserDataError("The password must not be the same as the username.")
    if password.isdigit() or password.isalpha():
        raise InvalidUserDataError("The password must mix letters with digits or symbols.")


class AuthService:
    """Accounts, login and permission checks. Does not contain SQL."""

    def __init__(self, database_path: Optional[Path] = None):
        self.repository = UserRepository(database_path)

    # -- setup / login -------------------------------------------------------

    def needs_first_account(self) -> bool:
        """True until the first account exists (the app then asks for an IT account)."""
        return self.repository.count_users() == 0

    def create_first_account(self, username: str, full_name: str, password: str) -> int:
        """The first IT account. Refused once any account exists."""
        if not self.needs_first_account():
            raise PermissionDeniedError("The first account already exists. Ask IT to create your account.")
        return self._create(username, full_name, Role.IT, password, must_change_password=False, created_by=None)

    def authenticate(self, username: str, password: str, now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Check a login. Returns the user (without the password hash).

        Raises:
            InvalidCredentialsError: Wrong username or password
            AccountLockedError: Too many wrong passwords (also for a right one while locked)
            AccountInactiveError: The account is deactivated
        """
        now = now or _now_utc()
        user = self.repository.get_by_username((username or "").strip())
        if user is None:
            verify_password(password or "", _DUMMY_HASH)   # same time as a known username
            raise InvalidCredentialsError("Wrong username or password.")

        locked_until = _parse_utc(user['locked_until'])
        if locked_until and locked_until > now:
            raise AccountLockedError(self._locked_message(locked_until, now), locked_until)

        if not verify_password(password or "", self.repository.get_password_hash(user['id'])):
            self._count_failure(user, locked_until, now, "Wrong username or password.")

        if not user['is_active']:
            raise AccountInactiveError("This account is deactivated. Ask an administrator.")

        self.repository.record_login(user['id'], now.strftime(DB_TIME_FORMAT))
        logger.info(f"User '{user['username']}' logged in")
        return self.repository.get_by_id(user['id'])

    def _count_failure(self, user: Dict[str, Any], locked_until: Optional[datetime], now: datetime,
                       message: str) -> None:
        """Count a wrong password / recovery code; lock the account at MAX_FAILED_ATTEMPTS. Always raises."""
        attempts = (0 if locked_until else user['failed_attempts'] or 0) + 1   # an expired lock starts anew
        if attempts >= MAX_FAILED_ATTEMPTS:
            until = now + timedelta(minutes=LOCK_MINUTES)
            self.repository.record_failed_login(user['id'], attempts, until.strftime(DB_TIME_FORMAT))
            logger.warning(f"Account '{user['username']}' locked after {attempts} wrong tries")
            raise AccountLockedError(self._locked_message(until, now), until)
        self.repository.record_failed_login(user['id'], attempts, None)
        left = MAX_FAILED_ATTEMPTS - attempts
        raise InvalidCredentialsError(
            message + (f" {left} {'try' if left == 1 else 'tries'} left before the account is locked."
                       if left <= 2 else ""))

    @staticmethod
    def _locked_message(until: datetime, now: datetime) -> str:
        minutes = max(1, math.ceil((until - now).total_seconds() / 60))
        return (f"Too many wrong passwords: this account is locked for {minutes} more minute(s). "
                f"An administrator can unlock it sooner.")

    def current_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """The user, freshly read; None if deleted or deactivated (they are then logged out)."""
        user = self.repository.get_by_id(user_id)
        return user if user and user['is_active'] else None

    # -- own account -------------------------------------------------------------

    def change_own_password(self, user_id: int, current_password: str, new_password: str) -> None:
        """
        Raises:
            InvalidCredentialsError: The current password is wrong
            InvalidUserDataError: The new password is too weak or unchanged
        """
        user = self._require(user_id)
        if not verify_password(current_password or "", self.repository.get_password_hash(user_id)):
            raise InvalidCredentialsError("The current password is wrong.")
        if current_password == new_password:
            raise InvalidUserDataError("The new password must be different from the current one.")
        validate_password(new_password, user['username'])
        self.repository.set_password(user_id, hash_password(new_password), must_change_password=False)
        logger.info(f"User '{user['username']}' changed their password")

    # -- recovery codes --------------------------------------------------------------

    def issue_recovery_code(self, user_id: int) -> str:
        """
        A new recovery code for the account (the old one stops working).
        Returned once to show the owner; only its hash is stored.
        """
        user = self._require(user_id)
        code = generate_recovery_code()
        self.repository.set_recovery_code_hash(user_id, hash_password(normalize_recovery_code(code)))
        logger.info(f"New recovery code issued for '{user['username']}'")
        return code

    def renew_recovery_code(self, user_id: int, current_password: str) -> str:
        """A new code for a signed-in owner, after checking their password (e.g. the old code was lost)."""
        if not verify_password(current_password or "", self.repository.get_password_hash(user_id)):
            raise InvalidCredentialsError("The current password is wrong.")
        return self.issue_recovery_code(user_id)

    def recover_password(self, username: str, recovery_code: str, new_password: str,
                         now: Optional[datetime] = None) -> Tuple[Dict[str, Any], str]:
        """
        Forgotten password: the recovery code proves who it is. Sets the new
        password, unlocks the account, uses the code up and issues a new one.

        Returns:
            (user, new recovery code to show once)

        Raises:
            InvalidCredentialsError: Unknown username, wrong code or no code (one message for all)
            AccountLockedError: Too many wrong tries (wrong codes count like wrong passwords)
            AccountInactiveError: The account is deactivated
            InvalidUserDataError: The new password is too weak
        """
        now = now or _now_utc()
        message = "Wrong username or recovery code."
        user = self.repository.get_by_username((username or "").strip())
        if user is None:
            verify_password(recovery_code or "", _DUMMY_HASH)   # same time as a known username
            raise InvalidCredentialsError(message)
        locked_until = _parse_utc(user['locked_until'])
        if locked_until and locked_until > now:
            raise AccountLockedError(self._locked_message(locked_until, now), locked_until)
        if not verify_password(normalize_recovery_code(recovery_code),
                               self.repository.get_recovery_code_hash(user['id'])):
            self._count_failure(user, locked_until, now, message)
        if not user['is_active']:
            raise AccountInactiveError("This account is deactivated. Ask an administrator.")
        validate_password(new_password, user['username'])

        self.repository.set_password(user['id'], hash_password(new_password), must_change_password=False)
        new_code = self.issue_recovery_code(user['id'])
        self.repository.record_login(user['id'], now.strftime(DB_TIME_FORMAT))
        logger.warning(f"Password of '{user['username']}' reset with its recovery code")
        return self.repository.get_by_id(user['id']), new_code

    def active_it_accounts(self) -> int:
        """How many active IT accounts exist (the app warns while there is only one)."""
        return self.repository.count_active_with_role(Role.IT)

    # -- managing accounts ---------------------------------------------------------

    def list_users(self, actor: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not has_role(actor, Role.ADMIN):
            raise PermissionDeniedError("Only administrators can see the accounts.")
        return self.repository.list_users()

    def create_user(self, actor: Dict[str, Any], username: str, full_name: str, role: str,
                    temporary_password: str) -> int:
        """
        A new account with a temporary password; its owner must change it at the first login.

        Raises:
            PermissionDeniedError, InvalidUserDataError
        """
        if not can_manage(actor, role):
            raise PermissionDeniedError(f"You can't create {ROLE_LABELS.get(role, role)} accounts.")
        return self._create(username, full_name, role, temporary_password, must_change_password=True,
                            created_by=actor['username'])

    def reset_password(self, actor: Dict[str, Any], user_id: int, temporary_password: str) -> None:
        """New temporary password (also unlocks); the owner must change it at the next login."""
        user = self._require_manageable(actor, user_id)
        validate_password(temporary_password, user['username'])
        self.repository.set_password(user_id, hash_password(temporary_password), must_change_password=True)
        logger.info(f"'{actor['username']}' reset the password of '{user['username']}'")

    def unlock(self, actor: Dict[str, Any], user_id: int) -> None:
        user = self._require_manageable(actor, user_id)
        self.repository.unlock(user_id)
        logger.info(f"'{actor['username']}' unlocked '{user['username']}'")

    def update_user(self, actor: Dict[str, Any], user_id: int, full_name: str, role: str) -> None:
        user = self._require_manageable(actor, user_id)
        if role not in ROLE_LEVELS:
            raise InvalidUserDataError("Unknown role.")
        if not can_manage(actor, role):
            raise PermissionDeniedError(f"You can't give the {ROLE_LABELS[role]} role.")
        if user['id'] == actor['id'] and role != user['role']:
            raise PermissionDeniedError("You can't change your own role.")
        self._keep_one_it(user, new_role=role)
        full_name = self._clean_name(full_name)
        self.repository.update_user(user_id, full_name, role)
        if full_name != user['full_name']:   # sections show their instructor's name
            CourseRepository(self.repository.database_path).rename_instructor(user_id, full_name)

    def set_active(self, actor: Dict[str, Any], user_id: int, active: bool) -> None:
        user = self._require_manageable(actor, user_id)
        if not active:
            if user['id'] == actor['id']:
                raise PermissionDeniedError("You can't deactivate your own account.")
            self._keep_one_it(user, new_role=None)
        self.repository.set_active(user_id, active)
        logger.info(f"'{actor['username']}' {'activated' if active else 'deactivated'} '{user['username']}'")

    # -- helpers ---------------------------------------------------------------------

    def _create(self, username: str, full_name: str, role: str, password: str,
                must_change_password: bool, created_by: Optional[str]) -> int:
        username = (username or "").strip()
        if not USERNAME_PATTERN.match(username):
            raise InvalidUserDataError("The username must have 3-30 letters, digits, '.', '_' or '-'.")
        if role not in ROLE_LEVELS:
            raise InvalidUserDataError("Unknown role.")
        full_name = self._clean_name(full_name)
        validate_password(password, username)
        if self.repository.get_by_username(username):
            raise InvalidUserDataError(f"The username '{username}' is already taken.")
        try:
            user_id = self.repository.create_user(username, full_name, role, hash_password(password),
                                                  must_change_password, created_by)
        except sqlite3.IntegrityError:
            raise InvalidUserDataError(f"The username '{username}' is already taken.")
        logger.info(f"Account '{username}' ({role}) created" + (f" by '{created_by}'" if created_by else ""))
        return user_id

    @staticmethod
    def _clean_name(full_name: str) -> str:
        name = re.sub(r"\s+", " ", full_name or "").strip()
        if not name:
            raise InvalidUserDataError("The full name cannot be empty.")
        if len(name) > 100:
            raise InvalidUserDataError("The full name is longer than 100 characters.")
        return name

    def _require(self, user_id: int) -> Dict[str, Any]:
        user = self.repository.get_by_id(user_id)
        if not user:
            raise InvalidUserDataError("Account not found.")
        return user

    def _require_manageable(self, actor: Dict[str, Any], user_id: int) -> Dict[str, Any]:
        user = self._require(user_id)
        if not can_manage(actor, user['role']):
            raise PermissionDeniedError(f"You can't change {ROLE_LABELS[user['role']]} accounts.")
        return user

    def _keep_one_it(self, user: Dict[str, Any], new_role: Optional[str]) -> None:
        """Never leave the system without an active IT account."""
        if (user['role'] == Role.IT and user['is_active'] and new_role != Role.IT
                and self.repository.count_active_with_role(Role.IT) <= 1):
            raise PermissionDeniedError("This is the last active IT account; create another IT account first.")
