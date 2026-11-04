"""
Tests for accounts and login: password hashing, the first IT account, the
lockout after wrong passwords, who may manage which accounts, the page
access per role, temporary passwords and the recovery script.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src.database.schema import initialize_database
from src.services.auth_service import (
    LOCK_MINUTES, MAX_FAILED_ATTEMPTS, AccountInactiveError, AccountLockedError, AuthService,
    InvalidCredentialsError, InvalidUserDataError, PermissionDeniedError, Role, can_manage, has_role,
    hash_password, verify_password,
)
from src.ui.access import can_open, pages_for

NOW = datetime(2026, 11, 3, 9, 0, 0)
PASSWORD = "Secret-123"


@pytest.fixture
def auth():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    initialize_database(db_path)
    yield AuthService(db_path)
    db_path.unlink()


@pytest.fixture
def people(auth):
    """An IT account (the first one), an admin and an instructor, all signed in once."""
    it_id = auth.create_first_account("it.boss", "IT Boss", PASSWORD)
    it = auth.repository.get_by_id(it_id)
    admin = auth.repository.get_by_id(auth.create_user(it, "admin1", "Admin One", Role.ADMIN, PASSWORD))
    teacher = auth.repository.get_by_id(auth.create_user(admin, "sara", "Dr. Sara Ali", Role.INSTRUCTOR,
                                                         PASSWORD))
    return {'it': it, 'admin': admin, 'teacher': teacher}


# -- passwords -------------------------------------------------------------------

def test_password_hash_is_salted_and_verifies():
    first, second = hash_password(PASSWORD), hash_password(PASSWORD)
    assert first != second and PASSWORD not in first
    assert verify_password(PASSWORD, first) and verify_password(PASSWORD, second)
    assert not verify_password("Secret-124", first)
    assert not verify_password(PASSWORD, "garbage") and not verify_password(PASSWORD, None)


@pytest.mark.parametrize("password", ["short1!", "onlyletters", "1234567890", "it.boss"])
def test_weak_passwords_are_refused(auth, password):
    with pytest.raises(InvalidUserDataError):
        auth.create_first_account("it.boss", "IT Boss", password)


def test_password_is_never_stored_as_text(auth, people):
    stored = auth.repository.get_password_hash(people['it']['id'])
    assert stored.startswith("scrypt$") and PASSWORD not in stored
    assert 'password_hash' not in people['it']


# -- first account / login ---------------------------------------------------------

def test_first_account_is_it_and_only_once(auth):
    assert auth.needs_first_account()
    user = auth.repository.get_by_id(auth.create_first_account("it.boss", "IT Boss", PASSWORD))
    assert user['role'] == Role.IT and not user['must_change_password']
    assert not auth.needs_first_account()
    with pytest.raises(PermissionDeniedError):
        auth.create_first_account("other", "Other", PASSWORD)


def test_login_success_records_the_time_and_ignores_username_case(auth, people):
    user = auth.authenticate("IT.BOSS", PASSWORD, now=NOW)
    assert user['id'] == people['it']['id'] and user['last_login_at'] == "2026-11-03 09:00:00"


def test_unknown_user_and_wrong_password_get_the_same_message(auth, people):
    with pytest.raises(InvalidCredentialsError) as unknown:
        auth.authenticate("nobody", PASSWORD, now=NOW)
    with pytest.raises(InvalidCredentialsError) as wrong:
        auth.authenticate("it.boss", "Wrong-123", now=NOW)
    assert str(unknown.value) == str(wrong.value) == "Wrong username or password."


def test_lockout_after_too_many_wrong_passwords(auth, people):
    for attempt in range(MAX_FAILED_ATTEMPTS - 1):
        with pytest.raises(InvalidCredentialsError):
            auth.authenticate("sara", "Wrong-123", now=NOW)
    with pytest.raises(AccountLockedError):
        auth.authenticate("sara", "Wrong-123", now=NOW)
    # Even the right password is refused while locked
    with pytest.raises(AccountLockedError):
        auth.authenticate("sara", PASSWORD, now=NOW + timedelta(minutes=LOCK_MINUTES - 1))
    # After the lock time it works, and the count starts again
    assert auth.authenticate("sara", PASSWORD, now=NOW + timedelta(minutes=LOCK_MINUTES + 1))
    assert auth.repository.get_by_id(people['teacher']['id'])['failed_attempts'] == 0


def test_a_good_login_resets_the_count(auth, people):
    for _ in range(MAX_FAILED_ATTEMPTS - 1):
        with pytest.raises(InvalidCredentialsError):
            auth.authenticate("sara", "Wrong-123", now=NOW)
    auth.authenticate("sara", PASSWORD, now=NOW)
    with pytest.raises(InvalidCredentialsError):   # not locked: a fresh count
        auth.authenticate("sara", "Wrong-123", now=NOW)


def test_admin_unlocks_an_instructor(auth, people):
    for _ in range(MAX_FAILED_ATTEMPTS):
        with pytest.raises((InvalidCredentialsError, AccountLockedError)):
            auth.authenticate("sara", "Wrong-123")
    auth.unlock(people['admin'], people['teacher']['id'])
    assert auth.authenticate("sara", PASSWORD)


def test_deactivated_account_cannot_sign_in_and_is_signed_out(auth, people):
    auth.set_active(people['admin'], people['teacher']['id'], False)
    with pytest.raises(AccountInactiveError):
        auth.authenticate("sara", PASSWORD, now=NOW)
    assert auth.current_user(people['teacher']['id']) is None


# -- temporary passwords --------------------------------------------------------------

def test_new_accounts_must_choose_their_own_password(auth, people):
    assert people['teacher']['must_change_password']
    with pytest.raises(InvalidCredentialsError):
        auth.change_own_password(people['teacher']['id'], "Wrong-123", "MyOwn-456")
    with pytest.raises(InvalidUserDataError):
        auth.change_own_password(people['teacher']['id'], PASSWORD, PASSWORD)
    auth.change_own_password(people['teacher']['id'], PASSWORD, "MyOwn-456")
    assert not auth.repository.get_by_id(people['teacher']['id'])['must_change_password']
    assert auth.authenticate("sara", "MyOwn-456")


def test_reset_password_gives_a_temporary_one_and_unlocks(auth, people):
    for _ in range(MAX_FAILED_ATTEMPTS):
        with pytest.raises((InvalidCredentialsError, AccountLockedError)):
            auth.authenticate("sara", "Wrong-123")
    auth.reset_password(people['admin'], people['teacher']['id'], "Temp-7890")
    user = auth.authenticate("sara", "Temp-7890")
    assert user['must_change_password']


def test_generated_temporary_passwords_are_valid_and_different():
    from src.ui.pages.users import generate_temporary_password
    from src.services.auth_service import validate_password
    passwords = {generate_temporary_password() for _ in range(20)}
    assert len(passwords) == 20
    for password in passwords:
        validate_password(password)
        assert len(password) == 12 and not set(password) & set("0OoIl1")


# -- who may manage whom -----------------------------------------------------------

def test_role_order_and_management_rules(people):
    it, admin, teacher = people['it'], people['admin'], people['teacher']
    assert has_role(it, Role.ADMIN) and has_role(admin, Role.ADMIN) and not has_role(teacher, Role.ADMIN)
    assert has_role(teacher, Role.INSTRUCTOR) and not has_role(None, Role.INSTRUCTOR)
    assert can_manage(it, Role.IT) and can_manage(it, Role.ADMIN)
    assert can_manage(admin, Role.INSTRUCTOR) and not can_manage(admin, Role.ADMIN)
    assert not can_manage(teacher, Role.INSTRUCTOR)


def test_admin_cannot_create_or_change_admins(auth, people):
    with pytest.raises(PermissionDeniedError):
        auth.create_user(people['admin'], "admin2", "Admin Two", Role.ADMIN, PASSWORD)
    with pytest.raises(PermissionDeniedError):
        auth.reset_password(people['admin'], people['it']['id'], "Temp-7890")
    with pytest.raises(PermissionDeniedError):
        auth.update_user(people['admin'], people['teacher']['id'], "Dr. Sara", Role.ADMIN)


def test_instructors_cannot_manage_accounts(auth, people):
    with pytest.raises(PermissionDeniedError):
        auth.list_users(people['teacher'])
    with pytest.raises(PermissionDeniedError):
        auth.create_user(people['teacher'], "x.y", "X Y", Role.INSTRUCTOR, PASSWORD)


def test_the_last_it_account_and_own_account_are_protected(auth, people):
    it = people['it']
    with pytest.raises(PermissionDeniedError):
        auth.set_active(it, it['id'], False)                       # yourself
    with pytest.raises(PermissionDeniedError):
        auth.update_user(it, it['id'], "IT Boss", Role.ADMIN)      # own role
    with pytest.raises(PermissionDeniedError, match="last active IT"):
        auth._keep_one_it(it, new_role=None)                       # it is the only IT account
    second = auth.repository.get_by_id(auth.create_user(it, "it.two", "IT Two", Role.IT, PASSWORD))
    auth._keep_one_it(it, new_role=None)                           # another IT exists: fine
    auth.update_user(it, second['id'], "IT Two", Role.ADMIN)       # so demoting one is allowed
    assert auth.repository.get_by_id(second['id'])['role'] == Role.ADMIN


def test_duplicate_username_is_refused(auth, people):
    with pytest.raises(InvalidUserDataError, match="taken"):
        auth.create_user(people['it'], "SARA", "Another Sara", Role.INSTRUCTOR, PASSWORD)


# -- pages per role -----------------------------------------------------------------

def test_pages_per_role(people):
    teacher_pages = pages_for(people['teacher'])
    assert teacher_pages == ["Home", "My Classes", "Real-Time Attendance", "Session Management",
                             "Statistics & Reports"]
    assert "Users" in pages_for(people['admin']) and "Student Registry" in pages_for(people['admin'])
    assert pages_for(people['it']) == pages_for(people['admin'])
    assert "My Classes" not in pages_for(people['admin'])                  # no sections linked to them
    assert "My Classes" in pages_for(people['admin'], teaches=True)       # an admin who also teaches
    assert not can_open(people['teacher'], "Users") and not can_open(None, "Home")


# -- recovery script -------------------------------------------------------------------

def test_recovery_script_resets_a_forgotten_password(auth, people):
    import scripts.manage_accounts as script
    with patch.object(script, "AuthService", return_value=auth), \
            patch.object(script, "initialize_database"), \
            patch.object(script.getpass, "getpass", return_value="Recovered-42"):
        assert script.main(["reset-password", "it.boss"]) == 0
    assert auth.authenticate("it.boss", "Recovered-42")


def test_lock_messages_count_minutes_and_tries_correctly(auth, people):
    messages = []
    for _ in range(MAX_FAILED_ATTEMPTS):
        with pytest.raises((InvalidCredentialsError, AccountLockedError)) as error:
            auth.authenticate("sara", "Wrong-123", now=NOW)
        messages.append(str(error.value))
    assert messages[-2].endswith("1 try left before the account is locked.")
    assert f"locked for {LOCK_MINUTES} more minute(s)" in messages[-1]


# -- recovery codes --------------------------------------------------------------------

from src.services.auth_service import generate_recovery_code, normalize_recovery_code  # noqa: E402


def test_recovery_codes_look_right_and_are_random():
    codes = {generate_recovery_code() for _ in range(50)}
    assert len(codes) == 50
    for code in codes:
        assert len(code) == 19 and code.count("-") == 3
        assert not set(code) & set("01OIL")


def test_recovery_code_is_stored_only_as_a_hash(auth, people):
    code = auth.issue_recovery_code(people['it']['id'])
    stored = auth.repository.get_recovery_code_hash(people['it']['id'])
    assert stored.startswith("scrypt$") and normalize_recovery_code(code) not in stored
    assert auth.repository.get_by_id(people['it']['id'])['has_recovery_code'] == 1
    assert auth.repository.get_by_id(people['admin']['id'])['has_recovery_code'] == 0


def test_forgotten_password_is_reset_with_the_code(auth, people):
    code = auth.issue_recovery_code(people['it']['id'])
    typed = code.lower().replace("-", " ")             # case, spaces and dashes don't matter
    user, new_code = auth.recover_password("it.boss", typed, "Brand-New-1", now=NOW)
    assert user['id'] == people['it']['id'] and new_code != code
    assert auth.authenticate("it.boss", "Brand-New-1")
    with pytest.raises(InvalidCredentialsError):        # the old code is used up
        auth.recover_password("it.boss", code, "Again-New-2", now=NOW)
    auth.recover_password("it.boss", new_code, "Again-New-2", now=NOW)   # the new one works


def test_recovery_unlocks_and_clears_a_temporary_password(auth, people):
    code = auth.issue_recovery_code(people['teacher']['id'])
    for _ in range(MAX_FAILED_ATTEMPTS):
        with pytest.raises((InvalidCredentialsError, AccountLockedError)):
            auth.authenticate("sara", "Wrong-123", now=NOW)
    later = NOW + timedelta(minutes=LOCK_MINUTES + 1)
    user, _ = auth.recover_password("sara", code, "Mine-Again-3", now=later)
    assert not user['must_change_password'] and user['failed_attempts'] == 0


def test_wrong_codes_count_towards_the_lockout(auth, people):
    auth.issue_recovery_code(people['it']['id'])
    for _ in range(MAX_FAILED_ATTEMPTS - 1):
        with pytest.raises(InvalidCredentialsError, match="recovery code"):
            auth.recover_password("it.boss", "AAAA-BBBB-CCCC-DDDD", "Brand-New-1", now=NOW)
    with pytest.raises(AccountLockedError):
        auth.recover_password("it.boss", "AAAA-BBBB-CCCC-DDDD", "Brand-New-1", now=NOW)


def test_no_code_unknown_user_and_weak_password(auth, people):
    with pytest.raises(InvalidCredentialsError):        # the admin has no code yet
        auth.recover_password("admin1", "AAAA-BBBB-CCCC-DDDD", "Brand-New-1", now=NOW)
    with pytest.raises(InvalidCredentialsError, match="Wrong username or recovery code"):
        auth.recover_password("nobody", "AAAA-BBBB-CCCC-DDDD", "Brand-New-1", now=NOW)
    code = auth.issue_recovery_code(people['it']['id'])
    with pytest.raises(InvalidUserDataError):
        auth.recover_password("it.boss", code, "weak", now=NOW)
    assert auth.authenticate("it.boss", PASSWORD)       # nothing changed


def test_deactivated_account_cannot_recover(auth, people):
    code = auth.issue_recovery_code(people['teacher']['id'])
    auth.set_active(people['admin'], people['teacher']['id'], False)
    with pytest.raises(AccountInactiveError):
        auth.recover_password("sara", code, "Brand-New-1", now=NOW)


def test_renewing_the_code_needs_the_password(auth, people):
    old = auth.issue_recovery_code(people['it']['id'])
    with pytest.raises(InvalidCredentialsError):
        auth.renew_recovery_code(people['it']['id'], "Wrong-123")
    new = auth.renew_recovery_code(people['it']['id'], PASSWORD)
    with pytest.raises(InvalidCredentialsError):
        auth.recover_password("it.boss", old, "Brand-New-1", now=NOW)
    auth.recover_password("it.boss", new, "Brand-New-1", now=NOW)


def test_counting_it_accounts(auth, people):
    assert auth.active_it_accounts() == 1
    auth.create_user(people['it'], "it.two", "IT Two", Role.IT, PASSWORD)
    assert auth.active_it_accounts() == 2


def test_old_users_table_gets_the_recovery_column(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL, role TEXT NOT NULL, password_hash TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1, must_change_password BOOLEAN DEFAULT 0,
                    failed_attempts INTEGER DEFAULT 0, locked_until TIMESTAMP, last_login_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, created_by TEXT)""")
    conn.execute("INSERT INTO users (username, full_name, role, password_hash) VALUES ('it.old', 'Old', 'it', ?)",
                 (hash_password(PASSWORD),))
    conn.commit()
    conn.close()
    assert initialize_database(db)
    old_auth = AuthService(db)
    user = old_auth.authenticate("it.old", PASSWORD)
    assert user['has_recovery_code'] == 0
    code = old_auth.issue_recovery_code(user['id'])
    assert old_auth.recover_password("it.old", code, "Brand-New-1")[0]['username'] == "it.old"
