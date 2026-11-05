"""
Login screens and the signed-in account.

- First start (no accounts yet): create the first IT account
- Login form; wrong passwords count towards the lockout (AuthService)
- An account given a temporary password must choose its own first
- Recovery codes: shown once (first account, first own password, after a
  recovery, or renewed in the sidebar); "Forgot your password?" on the
  login page uses one to set a new password
- Sidebar: who is signed in, change password, recovery code, log out

Login lasts for this browser tab: refreshing or closing it logs out. The
signed-in user is re-read from the database on every page load, so a
deactivated account or a changed role takes effect at once.
"""

import sys
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st

from src.services.auth_service import (
    MIN_PASSWORD_LENGTH, ROLE_LABELS, AuthError, AuthService, InvalidUserDataError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

AUTH_KEY = "auth_user_id"       # session state: the signed-in user's ID
USER_KEY = "auth_user"          # session state: the signed-in user (refreshed each page load)
PENDING_CODE_KEY = "pending_recovery_code"   # session state: a new recovery code not yet shown
FORGOT_KEY = "show_forgot_password"

# Browser suggestion lists ("autocomplete") are off for names, usernames and
# codes: on a shared PC they would show the previous person's username, and a
# broken saved entry made Brave close itself while typing. Password fields only
# tell password managers whether they hold a current or a new password.
NO_SUGGESTIONS = "off"
CURRENT_PASSWORD = "current-password"
NEW_PASSWORD = "new-password"
REALTIME_PAGE_MODULE = "src.ui.pages.realtime_attendance"


def current_user() -> Optional[Dict[str, Any]]:
    """The signed-in user for this page load (set by app.py), or None."""
    return st.session_state.get(USER_KEY)


def acting_user() -> Optional[Dict[str, Any]]:
    """
    For the activity log: the signed-in user of the page now running, or
    None in a background thread (the scheduler), which has no page.
    """
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx(suppress_warning=True) is None:
        return None
    return current_user()


def load_signed_in_user(auth: AuthService) -> Optional[Dict[str, Any]]:
    """The signed-in user, freshly read; logs out a deleted / deactivated account."""
    user_id = st.session_state.get(AUTH_KEY)
    user = auth.current_user(user_id) if user_id else None
    if user is None:
        st.session_state.pop(AUTH_KEY, None)
        st.session_state.pop(USER_KEY, None)
        return None
    st.session_state[USER_KEY] = user
    return user


def _password_rules() -> str:
    return f"At least {MIN_PASSWORD_LENGTH} characters, mixing letters with digits or symbols."


def _centered():
    return st.columns([1, 2, 1])[1]


def render_first_setup(auth: AuthService) -> None:
    """No accounts yet: create the first IT account (who then creates the others)."""
    with _centered():
        st.title("🎓 Welcome")
        st.markdown("No accounts exist yet. Create the first **IT account**. It can create every other "
                    "account (admins and instructors).")
        with st.form("first_account_form"):
            full_name = st.text_input("Full name", autocomplete=NO_SUGGESTIONS)
            username = st.text_input("Username", help="3-30 letters, digits, '.', '_' or '-'",
                                     autocomplete=NO_SUGGESTIONS)
            password = st.text_input("Password", type="password", help=_password_rules(), autocomplete=NEW_PASSWORD)
            confirm = st.text_input("Repeat password", type="password", autocomplete=NEW_PASSWORD)
            submitted = st.form_submit_button("Create IT account", type="primary", width="stretch")
        if not submitted:
            return
        if password != confirm:
            st.error("❌ The two passwords are different.")
            return
        try:
            user_id = auth.create_first_account(username, full_name, password)
        except AuthError as e:
            st.error(f"❌ {e}")
            return
        st.session_state[AUTH_KEY] = user_id
        st.session_state[PENDING_CODE_KEY] = auth.issue_recovery_code(user_id)
        st.rerun()


def render_login(auth: AuthService) -> None:
    if st.session_state.get(FORGOT_KEY):
        _render_forgot_password(auth)
        return
    with _centered():
        st.title("🎓 Sign in")
        st.caption("Intelligent Attendance Monitoring System")
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Username", autocomplete=NO_SUGGESTIONS)
            password = st.text_input("Password", type="password", autocomplete=CURRENT_PASSWORD)
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            try:
                user = auth.authenticate(username, password)
            except AuthError as e:
                st.error(f"❌ {e}")
                return
            st.session_state[AUTH_KEY] = user['id']
            st.rerun()
        st.button("Forgot your password?", key="forgot_password_link", type="tertiary",
                  on_click=lambda: st.session_state.update({FORGOT_KEY: True}))
        st.caption("Use your recovery code, or ask an administrator to reset your password.")


def _render_forgot_password(auth: AuthService) -> None:
    """Set a new password with the account's recovery code."""
    with _centered():
        st.title("🔑 Forgot your password?")
        st.markdown("Enter your username and the **recovery code** you saved when you set your password. "
                    "After this the code is used up, and you get a new one.")
        with st.form("recover_password_form", clear_on_submit=True):
            username = st.text_input("Username", autocomplete=NO_SUGGESTIONS)
            code = st.text_input("Recovery code", placeholder="XXXX-XXXX-XXXX-XXXX", autocomplete=NO_SUGGESTIONS,
                                 help="Capital or small letters, with or without dashes")
            new = st.text_input("New password", type="password", help=_password_rules(), autocomplete=NEW_PASSWORD)
            confirm = st.text_input("Repeat new password", type="password", autocomplete=NEW_PASSWORD)
            submitted = st.form_submit_button("Set new password", type="primary", width="stretch")
        if submitted:
            if new != confirm:
                st.error("❌ The two new passwords are different.")
            else:
                try:
                    user, new_code = auth.recover_password(username, code, new)
                except AuthError as e:
                    st.error(f"❌ {e}")
                else:
                    st.session_state.pop(FORGOT_KEY, None)
                    st.session_state[AUTH_KEY] = user['id']
                    st.session_state[PENDING_CODE_KEY] = new_code
                    st.rerun()
        st.button("⬅️ Back to sign in", key="forgot_back",
                  on_click=lambda: st.session_state.pop(FORGOT_KEY, None))
        st.caption("No recovery code? Ask an administrator (instructors) or IT (admins and IT) to reset "
                   "your password.")


def has_pending_recovery_code() -> bool:
    return bool(st.session_state.get(PENDING_CODE_KEY))


def render_recovery_code_screen(user: Dict[str, Any]) -> None:
    """Show a new recovery code once; the owner confirms they saved it before going on."""
    code = st.session_state.get(PENDING_CODE_KEY)
    with _centered():
        st.title("🧾 Save your recovery code")
        st.markdown("If you ever **forget your password**, this code lets you set a new one yourself "
                    "(**Forgot your password?** on the sign-in page). It is shown **only now**.")
        st.code(code, language=None)
        st.download_button(
            "⬇️ Download as a text file",
            f"Attendance system recovery code\n\nUsername: {user['username']}\nCode: {code}\n"
            f"Created: {datetime.now():%Y-%m-%d %H:%M}\n\nKeep it somewhere safe and private. It works "
            f"once; after using it you get a new one.\n",
            file_name=f"recovery_code_{user['username']}.txt", mime="text/plain", key="download_recovery_code",
        )
        st.warning("Print it or keep the file somewhere safe and private - not on a shared desktop. "
                   "Anyone with this code and your username can set a new password for your account.")
        saved = st.checkbox("I have saved my recovery code", key="recovery_code_saved")
        if st.button("Continue", type="primary", disabled=not saved, width="stretch", key="recovery_code_done"):
            st.session_state.pop(PENDING_CODE_KEY, None)
            st.session_state.pop("recovery_code_saved", None)
            st.rerun()


def render_forced_password_change(auth: AuthService, user: Dict[str, Any]) -> None:
    """The account has a temporary password: its owner chooses their own before going on."""
    with _centered():
        st.title("🔑 Choose your password")
        st.markdown(f"Welcome, **{user['full_name']}**. Your account has a temporary password; "
                    f"choose your own to continue.")
        if _password_form(auth, user, "forced_password_form"):
            st.session_state[PENDING_CODE_KEY] = auth.issue_recovery_code(user['id'])
            st.rerun()
        if st.button("Log out", key="forced_logout"):
            logout()


def render_account_box(auth: AuthService, user: Dict[str, Any]) -> None:
    """Sidebar: who is signed in, change password, log out."""
    st.sidebar.markdown(f"👤 **{user['full_name']}**  \n{ROLE_LABELS.get(user['role'], user['role'])} · "
                        f"`{user['username']}`")
    col_password, col_logout = st.sidebar.columns(2)
    with col_password.popover("🔑 Password", width="stretch"):
        if _password_form(auth, user, "change_password_form"):
            st.success("✅ Password changed.")
    if col_logout.button("🚪 Log out", width="stretch", key="logout_button"):
        logout()
    if not user.get('has_recovery_code'):
        st.sidebar.warning("You have no recovery code yet: without it, a forgotten password needs an "
                           "administrator. Create one below.")
    with st.sidebar.popover("🧾 New recovery code", width="stretch"):
        st.caption("Makes a new recovery code; the old one stops working. Use it if you lost yours.")
        with st.form("renew_recovery_code_form", clear_on_submit=True):
            current = st.text_input("Current password", type="password", autocomplete=CURRENT_PASSWORD)
            if st.form_submit_button("Make a new recovery code", type="primary", width="stretch"):
                try:
                    st.session_state[PENDING_CODE_KEY] = auth.renew_recovery_code(user['id'], current)
                except AuthError as e:
                    st.error(f"❌ {e}")
                else:
                    st.rerun()


def _password_form(auth: AuthService, user: Dict[str, Any], key: str) -> bool:
    """Current + new password form. True once the password was changed."""
    with st.form(key, clear_on_submit=True):
        current = st.text_input("Current password", type="password", autocomplete=CURRENT_PASSWORD)
        new = st.text_input("New password", type="password", help=_password_rules(), autocomplete=NEW_PASSWORD)
        confirm = st.text_input("Repeat new password", type="password", autocomplete=NEW_PASSWORD)
        submitted = st.form_submit_button("Change password", type="primary", width="stretch")
    if not submitted:
        return False
    if new != confirm:
        st.error("❌ The two new passwords are different.")
        return False
    try:
        auth.change_own_password(user['id'], current, new)
    except (AuthError, InvalidUserDataError) as e:
        st.error(f"❌ {e}")
        return False
    return True


def logout() -> None:
    """Sign out: stop a camera this tab started, forget everything this tab knew."""
    if st.session_state.get('att_camera_active'):
        realtime = sys.modules.get(REALTIME_PAGE_MODULE)
        if realtime is not None:
            try:
                realtime._get_engine().stop()
            except Exception as e:
                logger.warning(f"Could not stop the camera at logout: {e}")
    user = st.session_state.get(USER_KEY)
    st.session_state.clear()
    if user:
        logger.info(f"User '{user['username']}' logged out")
        AuthService().record_sign_out(user)
    st.rerun()
