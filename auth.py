"""auth.py - Supabase Authentication & Session Identity for ReWA.AI

ReWA.AI supports two modes of access:
1. Anonymous Visitors (Zero Configuration required):
   - Visitors can use every feature (photo classification, bin sorting, Eco Points,
     Bengaluru municipal guidance, and digital twin simulation).
   - Each visitor receives an isolated per-session UUID stored in st.session_state.
   - Visitors never share or overwrite each other's records.
2. Authenticated Users (via Supabase Free Tier):
   - Email + Password authentication (Sign Up, Log In, Log Out, Password Reset).
   - Logged-in accounts use their Supabase User UUID as user_id across SQLite tables.
   - Session data accumulated while anonymous is automatically merged upon sign-in.

Security & Zero-Cost Constraints:
- Uses ONLY the public 'anon' key from st.secrets (never the service_role key).
- Passwords are never logged, stored in code, or passed in URLs.
- Gracefully degrades to anonymous mode if Supabase credentials are not configured.
"""
import logging
import re
import sqlite3
import time
import uuid
from typing import Optional, Tuple

import streamlit as st

try:
    from supabase import Client, create_client
    from supabase_auth.errors import AuthApiError, AuthError
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None
    create_client = None
    AuthApiError = Exception
    AuthError = Exception

ALLOW_ANONYMOUS = True
PUBLIC_USER_ID = "public"  # Kept for backward compatibility references
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8

# Session keys preserved across user transitions
_KEEP = {
    "supabase_session",
    "supabase_user",
    "active_user",
    "anon_user_id",
    "intro_played",
    "auth_flash_message",
    "auth_flash_type",
}

# Styles for the login panel
CORNER_CSS = """
<style>
/* Hide Streamlit's own Deploy button */
[data-testid="stAppDeployButton"] { display: none !important; }

.st-key-auth_corner {
    max-width: 480px;
    margin: 0 0 14px 0;
    position: relative;
    z-index: 5;
}

.st-key-auth_corner [data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 12px;
}

.st-key-auth_corner summary {
    font-weight: 700;
}

.st-key-auth_corner div.stButton > button {
    pointer-events: auto !important;
}

.st-key-skip_intro_btn {
    position: fixed;
    top: 18px;
    right: 24px;
    z-index: 100001;
}

.st-key-skip_intro_btn button {
    background: rgba(6, 36, 26, 0.90) !important;
    color: #E4FBC2 !important;
    border: 1px solid #C9A24B !important;
    border-radius: 999px !important;
    padding: 6px 14px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    pointer-events: auto !important;
}
</style>
"""


# =====================================================================
# SUPABASE CLIENT SETUP & CREDENTIALS
# =====================================================================
def get_supabase_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Retrieve SUPABASE_URL and SUPABASE_ANON_KEY from st.secrets safely."""
    try:
        # Check flat top-level keys
        url = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase_url")
        key = st.secrets.get("SUPABASE_ANON_KEY") or st.secrets.get("supabase_anon_key")
        if url and key:
            return str(url).strip(), str(key).strip()

        # Check nested [supabase] section
        sub = st.secrets.get("supabase", {})
        if isinstance(sub, dict):
            url = sub.get("url") or sub.get("SUPABASE_URL")
            key = sub.get("anon_key") or sub.get("SUPABASE_ANON_KEY") or sub.get("key")
            if url and key:
                return str(url).strip(), str(key).strip()
    except Exception:
        pass
    return None, None


def is_supabase_configured() -> bool:
    """Return True only if valid, non-placeholder credentials are present."""
    if not SUPABASE_AVAILABLE:
        return False
    url, key = get_supabase_credentials()
    if not url or not key:
        return False
    # Avoid placeholder dummy entries
    if "your-project" in url.lower() or "your-anon-key" in key.lower():
        return False
    return True


@st.cache_resource(show_spinner=False)
def get_supabase_client() -> Optional[Client]:
    """
    Cached Supabase client initialization.
    Uses ONLY the public anon key. Never uses or requests service_role key.
    """
    if not is_supabase_configured():
        return None
    url, key = get_supabase_credentials()
    try:
        return create_client(url, key)
    except Exception as exc:
        logging.warning("Failed to initialize Supabase client: %s", exc)
        return None


# =====================================================================
# VALIDATION & ERROR TRANSLATION HELPERS
# =====================================================================
def validate_email(email: str) -> Tuple[bool, str]:
    """Validate email format and length."""
    if not email or not isinstance(email, str):
        return False, "Please enter your email address."
    clean = email.strip()
    if not EMAIL_REGEX.match(clean):
        return False, "Please enter a valid email address (e.g., user@example.com)."
    return True, ""


def validate_password(password: str, confirm_password: Optional[str] = None) -> Tuple[bool, str]:
    """Validate password length and confirmation match."""
    if not password:
        return False, "Please enter a password."
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
    if confirm_password is not None and password != confirm_password:
        return False, "Passwords do not match. Please re-enter."
    return True, ""


def format_auth_error(exc: Exception) -> str:
    """Translate technical Supabase/network errors into clear, friendly guidance."""
    msg = str(exc)
    msg_lower = msg.lower()

    if "invalid login credentials" in msg_lower or "invalid_grant" in msg_lower or "wrong password" in msg_lower:
        return "Incorrect email or password. Please verify your credentials and try again."
    if "email not confirmed" in msg_lower:
        return (
            "Email not confirmed yet. Please check your inbox (and spam folder) for the confirmation link. "
            "(For demos/testing, email confirmation can be disabled in Supabase: Project Settings -> Auth -> Email)."
        )
    if "user already registered" in msg_lower or "already exists" in msg_lower:
        return "An account with this email already exists. Please log in or use Forgot Password."
    if "password should be at least" in msg_lower or "weak_password" in msg_lower:
        return f"Password is too weak. Please choose a password with at least {MIN_PASSWORD_LENGTH} characters."
    if "rate limit" in msg_lower or "429" in msg_lower or "too many requests" in msg_lower:
        return "Too many requests in a short time. Please wait a moment before trying again."
    if "network" in msg_lower or "connection" in msg_lower or "failed to establish" in msg_lower:
        return "Network connection error. Could not connect to the Supabase server. Please check your connection."
    return f"Authentication error: {msg}"


# =====================================================================
# DATA MERGING & SESSION REFRESH
# =====================================================================
def merge_anon_data_to_user(anon_id: str, new_user_id: str, db_name: Optional[str] = None):
    """
    Merge items and rewards saved during an anonymous session into the user's permanent account.
    """
    if not anon_id or not new_user_id or anon_id == new_user_id:
        return
    try:
        if not db_name:
            try:
                import data_base
                db_name = getattr(data_base, "DB_NAME", "recycling.db")
            except Exception:
                db_name = "recycling.db"
        conn = sqlite3.connect(db_name)
        with conn:
            conn.execute(
                "UPDATE recycling_events SET user_id = ? WHERE user_id = ?",
                (new_user_id, anon_id)
            )
            conn.execute(
                "UPDATE reward_events SET user_id = ? WHERE user_id = ?",
                (new_user_id, anon_id)
            )
            conn.execute(
                "UPDATE daily_activity SET user_id = ? WHERE user_id = ?",
                (new_user_id, anon_id)
            )
            conn.execute(
                "UPDATE ai_corrections SET user_id = ? WHERE user_id = ?",
                (new_user_id, anon_id)
            )
        conn.close()
    except Exception as exc:
        logging.warning("Failed to merge anonymous data into user account: %s", exc)


def refresh_session_if_needed(client: Optional[Client] = None):
    """Check session expiration and refresh if expiring within 60 seconds."""
    session = st.session_state.get("supabase_session")
    if not session:
        return

    expires_at = getattr(session, "expires_at", None)
    if not expires_at:
        return

    now = time.time()
    # If session expires within 60 seconds or already expired
    if (expires_at - now) < 60:
        client = client or get_supabase_client()
        refresh_token = getattr(session, "refresh_token", None)
        if client and refresh_token:
            try:
                refreshed = client.auth.refresh_session(refresh_token)
                if refreshed and refreshed.session:
                    st.session_state.supabase_session = refreshed.session
                    st.session_state.supabase_user = refreshed.user
            except Exception:
                # If refresh fails, log out gracefully to prevent corrupted state
                _clear_auth_state()


def _clear_auth_state():
    """Clear all user credentials and session storage."""
    st.session_state.pop("supabase_session", None)
    st.session_state.pop("supabase_user", None)
    st.session_state.pop("active_user", None)
    st.session_state.pop("anon_user_id", None)


# =====================================================================
# USER IDENTITY RESOLUTION
# =====================================================================
def get_user():
    """Return the authenticated Supabase user dict, or None if not logged in."""
    refresh_session_if_needed()
    user_obj = st.session_state.get("supabase_user")
    if user_obj:
        user_id = getattr(user_obj, "id", None)
        email = getattr(user_obj, "email", None)
        if user_id:
            display_name = email.split("@")[0] if email else "Supabase User"
            return {
                "id": str(user_id),
                "name": display_name,
                "email": email,
                "provider": "supabase",
                "is_guest": False,
            }
    return None


def resolve_user():
    """
    Return the active user dict. Never blocks the page.
    If not logged in, allocates an isolated per-session anonymous UUID.
    """
    user = get_user()

    if user is None:
        anon_id = st.session_state.get("anon_user_id")
        if not anon_id:
            anon_id = f"anon:{uuid.uuid4().hex}"
            st.session_state.anon_user_id = anon_id
        user = {
            "id": anon_id,
            "name": "Visitor",
            "email": None,
            "provider": "public",
            "is_guest": False,
        }

    # If identity on this browser tab switched, clear old session state to prevent leaks
    if st.session_state.get("active_user") != user["id"]:
        for key in list(st.session_state.keys()):
            if key not in _KEEP:
                del st.session_state[key]
        st.session_state.active_user = user["id"]

    return user


# =====================================================================
# AUTH FLOW ACTIONS
# =====================================================================
def sign_in_user(email: str, password: str, client: Optional[Client] = None) -> Tuple[bool, str]:
    """Execute Supabase email + password sign in."""
    valid_e, err_e = validate_email(email)
    if not valid_e:
        return False, err_e
    valid_p, err_p = validate_password(password)
    if not valid_p:
        return False, err_p

    client = client or get_supabase_client()
    if not client:
        return False, "Supabase client is not configured."

    try:
        response = client.auth.sign_in_with_password({
            "email": email.strip(),
            "password": password,
        })
        if response and response.user and response.session:
            pre_login_anon_id = st.session_state.get("anon_user_id")
            st.session_state.supabase_session = response.session
            st.session_state.supabase_user = response.user
            if pre_login_anon_id:
                merge_anon_data_to_user(pre_login_anon_id, str(response.user.id))
            return True, "Successfully logged in!"
        elif response and response.user and not response.session:
            return False, "Login requires email confirmation. Please check your inbox."
        return False, "Unable to sign in. Please try again."
    except Exception as exc:
        return False, format_auth_error(exc)


def sign_up_user(email: str, password: str, confirm_password: str, client: Optional[Client] = None) -> Tuple[bool, str]:
    """Execute Supabase email + password registration."""
    valid_e, err_e = validate_email(email)
    if not valid_e:
        return False, err_e
    valid_p, err_p = validate_password(password, confirm_password)
    if not valid_p:
        return False, err_p

    client = client or get_supabase_client()
    if not client:
        return False, "Supabase client is not configured."

    try:
        response = client.auth.sign_up({
            "email": email.strip(),
            "password": password,
        })
        if response and response.session and response.user:
            pre_login_anon_id = st.session_state.get("anon_user_id")
            st.session_state.supabase_session = response.session
            st.session_state.supabase_user = response.user
            if pre_login_anon_id:
                merge_anon_data_to_user(pre_login_anon_id, str(response.user.id))
            return True, "Account created successfully and signed in!"
        elif response and response.user:
            return True, "Sign up successful! Please check your email to confirm your account."
        return False, "Registration could not be completed. Please try again."
    except Exception as exc:
        return False, format_auth_error(exc)


def reset_password(email: str, client: Optional[Client] = None) -> Tuple[bool, str]:
    """Send Supabase password reset link to user's email."""
    valid_e, err_e = validate_email(email)
    if not valid_e:
        return False, err_e

    client = client or get_supabase_client()
    if not client:
        return False, "Supabase client is not configured."

    try:
        client.auth.reset_password_for_email(email.strip())
        return True, "Password reset link sent! Check your inbox."
    except Exception as exc:
        return False, format_auth_error(exc)


def sign_out_user(client: Optional[Client] = None):
    """Sign out the current user and flush session state."""
    client = client or get_supabase_client()
    if client:
        try:
            client.auth.sign_out()
        except Exception as exc:
            logging.info("Supabase sign out notice: %s", exc)
    _clear_auth_state()


# =====================================================================
# UI COMPONENTS
# =====================================================================
def _render_unconfigured_panel():
    """Friendly setup guidance shown when Supabase secrets are not configured."""
    st.info(
        "👋 **Anonymous mode active**: You can use every feature (photo classification, "
        "Eco Points, Bengaluru rules, and digital twin simulation) right now with full privacy.\n\n"
        "To enable permanent multi-device accounts, configure **Supabase Auth** (100% Free):"
    )
    with st.expander("🛠️ How to configure Supabase Auth (Free Tier)", expanded=False):
        st.markdown(
            """
            1. Create a free account at [supabase.com](https://supabase.com) and start a new project.
            2. Go to **Project Settings** -> **API**.
            3. Copy your **Project URL** and the public **`anon` key** (NEVER copy the `service_role` key).
            4. In `.streamlit/secrets.toml` (locally) or Streamlit Community Cloud Settings:
            ```toml
            SUPABASE_URL = "https://your-project.supabase.co"
            SUPABASE_ANON_KEY = "your-public-anon-key"
            ```
            5. Restart the application.
            """
        )


def _login_panel():
    """Render the Log In / Sign Up / Forgot Password tabs."""
    if not is_supabase_configured():
        _render_unconfigured_panel()
        return

    tab_login, tab_signup, tab_forgot = st.tabs(["Log in", "Sign up", "Forgot password"])

    with tab_login:
        st.caption("Sign in to sync your Eco Points, streaks, and recycling history.")
        login_email = st.text_input("Email", key="sb_login_email", placeholder="you@example.com")
        login_pass = st.text_input("Password", type="password", key="sb_login_pass")

        if st.button("Log in", use_container_width=True, key="sb_btn_login"):
            success, msg = sign_in_user(login_email, login_pass)
            if success:
                st.success(msg)
                time.sleep(0.3)
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        st.caption("Create a free account to permanently save your recycling journey.")
        signup_email = st.text_input("Email", key="sb_signup_email", placeholder="you@example.com")
        signup_pass = st.text_input("Password (min 8 characters)", type="password", key="sb_signup_pass")
        signup_confirm = st.text_input("Confirm password", type="password", key="sb_signup_confirm")

        if st.button("Create account", use_container_width=True, key="sb_btn_signup"):
            success, msg = sign_up_user(signup_email, signup_pass, signup_confirm)
            if success:
                st.success(msg)
                if st.session_state.get("supabase_user"):
                    time.sleep(0.3)
                    st.rerun()
            else:
                st.error(msg)

    with tab_forgot:
        st.caption("Enter your email to receive a password reset link.")
        forgot_email = st.text_input("Email", key="sb_forgot_email", placeholder="you@example.com")

        if st.button("Send reset link", use_container_width=True, key="sb_btn_forgot"):
            success, msg = reset_password(forgot_email)
            if success:
                st.success(msg)
            else:
                st.error(msg)


def account_corner(user):
    """Account panel in the upper section of the page."""
    with st.container(key="auth_corner"):
        if user.get("provider") == "public" or user.get("is_guest"):
            with st.expander("🔐 Log in (optional)", expanded=False):
                _login_panel()
        else:
            display_name = user.get("name") or "Account"
            with st.expander(f"👤 {display_name}", expanded=False):
                email = user.get("email") or "User"
                st.markdown(f"**Signed in as:** `{email}`")
                st.caption(f"User ID: `{user.get('id')}`")
                st.caption("Provider: Supabase Auth (Free Tier)")
                if st.button("Sign out", use_container_width=True, key="sb_btn_signout"):
                    sign_out_user()
                    st.rerun()
