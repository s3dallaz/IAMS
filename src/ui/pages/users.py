"""
Users page (Admin and IT): create accounts, change name / role, reset a
password, unlock, deactivate. Admins manage instructor accounts; IT
manages every account (AuthService enforces it).

New accounts and reset passwords get a temporary password that is shown
once; its owner must choose their own at the next login.
"""

import secrets
import string
from datetime import datetime, timezone
from typing import Any, Dict

import streamlit as st

from src.services.auth_service import (
    DB_TIME_FORMAT, ROLE_LABELS, ROLE_LEVELS, AuthError, AuthService, Role, can_manage, has_role, validate_password,
)
from src.ui.pages.login import NO_SUGGESTIONS, current_user
from src.utils.logger import get_logger

logger = get_logger(__name__)

FLASH_KEY = "users_flash"
TABLE_VERSION_KEY = "users_table_version"


def generate_temporary_password() -> str:
    """A random 12-character password that passes validate_password (no look-alike characters)."""
    alphabet = "".join(c for c in string.ascii_letters + string.digits if c not in "0OoIl1")
    while True:
        password = "".join(secrets.choice(alphabet) for _ in range(12))
        try:
            validate_password(password)
            return password
        except AuthError:
            continue


def render_second_it_warning(auth: AuthService, user: Dict[str, Any]) -> None:
    """IT users see this while only one IT account exists: if it is lost, nobody can manage admins / IT."""
    if has_role(user, Role.IT) and auth.active_it_accounts() < 2:
        st.warning("⚠️ **Only one IT account exists.** If its password and recovery code are both lost, "
                   "nobody can manage Admin and IT accounts. Create a **second IT account** for another "
                   "trusted person (Users → Create an account → role IT).")


def account_status(user: Dict[str, Any], now: datetime) -> str:
    if not user['is_active']:
        return "⏸️ Deactivated"
    locked_until = user.get('locked_until')
    if locked_until and datetime.strptime(str(locked_until)[:19], DB_TIME_FORMAT) > now:
        return "🔒 Locked"
    if user['must_change_password']:
        return "🔑 Must set password"
    return "✅ Active"


def render_users():
    """Main function to render the Users page."""
    actor = current_user()
    auth = AuthService()
    st.title("👥 Users")
    st.markdown("Accounts that can sign in. **Admins** manage instructor accounts; **IT** manages every account.")
    st.markdown("---")
    _show_flash()
    render_second_it_warning(auth, actor)

    manageable_roles = [role for role in ROLE_LEVELS if can_manage(actor, role)]
    with st.expander("➕ Create an account", expanded=False):
        with st.form(f"create_user_form_{st.session_state.get(TABLE_VERSION_KEY, 0)}"):
            c1, c2 = st.columns(2)
            full_name = c1.text_input("Full name *", placeholder="e.g. Dr. Sara Ali", autocomplete=NO_SUGGESTIONS)
            username = c2.text_input("Username *", placeholder="e.g. sara.ali", autocomplete=NO_SUGGESTIONS,
                                     help="3-30 letters, digits, '.', '_' or '-'")
            role = st.selectbox("Role", manageable_roles, format_func=ROLE_LABELS.get)
            if st.form_submit_button("Create account", type="primary"):
                password = generate_temporary_password()
                try:
                    auth.create_user(actor, username, full_name, role, password)
                except AuthError as e:
                    st.error(f"❌ {e}")
                else:
                    _flash_password(f"✅ Account **{username.strip()}** created.", password)
                    st.rerun()

    users = auth.list_users(actor)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    st.caption(f"{len(users)} account(s). **Select a row** to change it.")
    event = st.dataframe(
        [{
            'Name': u['full_name'],
            'Username': u['username'],
            'Role': ROLE_LABELS.get(u['role'], u['role']),
            'Status': account_status(u, now),
            'Last login': _local_time(u['last_login_at']),
        } for u in users],
        width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row",
        key=f"users_table_{st.session_state.get(TABLE_VERSION_KEY, 0)}",
    )
    rows = event.selection.rows if event else []
    if rows and rows[0] < len(users):
        _render_user_actions(auth, actor, users[rows[0]], now)


def _render_user_actions(auth: AuthService, actor: Dict[str, Any], user: Dict[str, Any], now: datetime):
    st.markdown(f"#### ⚙️ {user['full_name']} (`{user['username']}`)")
    if not can_manage(actor, user['role']):
        st.info(f"Only IT can change {ROLE_LABELS[user['role']]} accounts.")
        return
    is_me = user['id'] == actor['id']

    b1, b2, b3 = st.columns(3)
    if b1.button("🔑 Reset password", key=f"reset_{user['id']}", width="stretch",
                 help="Gives a new temporary password and unlocks the account"):
        password = generate_temporary_password()
        _run(lambda: auth.reset_password(actor, user['id'], password),
             f"🔑 New temporary password for **{user['username']}**.", password)
    if account_status(user, now) == "🔒 Locked":
        if b2.button("🔓 Unlock", key=f"unlock_{user['id']}", width="stretch"):
            _run(lambda: auth.unlock(actor, user['id']), f"🔓 **{user['username']}** unlocked.")
    if user['is_active']:
        if b3.button("⏸️ Deactivate", key=f"deact_{user['id']}", width="stretch", disabled=is_me,
                     help="You can't deactivate yourself" if is_me else "They can no longer sign in"):
            _run(lambda: auth.set_active(actor, user['id'], False), f"⏸️ **{user['username']}** deactivated.")
    elif b3.button("▶️ Activate", key=f"act_{user['id']}", width="stretch"):
        _run(lambda: auth.set_active(actor, user['id'], True), f"▶️ **{user['username']}** activated.")

    with st.form(f"edit_user_{user['id']}"):
        c1, c2 = st.columns(2)
        full_name = c1.text_input("Full name", value=user['full_name'], autocomplete=NO_SUGGESTIONS)
        roles = [role for role in ROLE_LEVELS if can_manage(actor, role)]
        role = c2.selectbox("Role", roles, index=roles.index(user['role']), format_func=ROLE_LABELS.get,
                            disabled=is_me, help="You can't change your own role" if is_me else None)
        if st.form_submit_button("Save changes"):
            _run(lambda: auth.update_user(actor, user['id'], full_name, role), "✅ Saved.")


def _run(action, message: str, password: str = None):
    try:
        action()
    except AuthError as e:
        st.error(f"❌ {e}")
        return
    if password:
        _flash_password(message, password)
    else:
        st.session_state[FLASH_KEY] = (message, None)
        st.session_state[TABLE_VERSION_KEY] = st.session_state.get(TABLE_VERSION_KEY, 0) + 1
    st.rerun()


def _flash_password(message: str, password: str):
    st.session_state[FLASH_KEY] = (message, password)
    st.session_state[TABLE_VERSION_KEY] = st.session_state.get(TABLE_VERSION_KEY, 0) + 1


def _show_flash():
    flash = st.session_state.pop(FLASH_KEY, None)
    if not flash:
        return
    message, password = flash
    st.success(message)
    if password:
        st.markdown("**Temporary password** (shown only now - give it to the person; they must choose "
                    "their own at the first login):")
        st.code(password, language=None)


def _local_time(value) -> str:
    """UTC database time -> local 'YYYY-MM-DD HH:MM'."""
    if not value:
        return "—"
    try:
        utc = datetime.strptime(str(value)[:19], DB_TIME_FORMAT).replace(tzinfo=timezone.utc)
        return utc.astimezone().strftime('%Y-%m-%d %H:%M')
    except ValueError:
        return str(value)
