"""auth.py - sign-in for ReWA.AI

Two ways to sign in:
  1. Google account  -> Streamlit's built-in OpenID Connect (st.login)
  2. Phone number    -> SMS one-time code via Twilio Verify (optional)

Usage in app.py:
    import auth
    current_user = auth.require_login()   # stops the page until signed in
    auth.account_bar(current_user)        # shows "Signed in as ..." + Sign out

current_user = {"id": "google:1234" | "phone:+9198...", "name": ..., "provider": ...}
Use current_user["id"] as the key that separates each person's records.
"""
import re

import requests
import streamlit as st

OTP_COOLDOWN_SECONDS = 30
PHONE_RE = re.compile(r"^\+[1-9]\d{7,14}$")  # E.164, e.g. +919876543210

# Session keys that survive a change of user
_KEEP = {"phone_user", "active_user", "intro_played"}


# ---------- config helpers ----------
def _google_enabled():
    try:
        return "auth" in st.secrets and "client_id" in st.secrets["auth"]
    except Exception:
        return False


def _twilio():
    """Return the [twilio] secrets, or None if phone login is not configured."""
    try:
        cfg = st.secrets["twilio"]
        if cfg["account_sid"] and cfg["auth_token"] and cfg["verify_sid"]:
            return cfg
    except Exception:
        pass
    return None


# ---------- Twilio Verify (plain REST, no extra package) ----------
def _send_code(phone):
    cfg = _twilio()
    r = requests.post(
        f"https://verify.twilio.com/v2/Services/{cfg['verify_sid']}/Verifications",
        data={"To": phone, "Channel": "sms"},
        auth=(cfg["account_sid"], cfg["auth_token"]),
        timeout=15,
    )
    r.raise_for_status()


def _check_code(phone, code):
    cfg = _twilio()
    r = requests.post(
        f"https://verify.twilio.com/v2/Services/{cfg['verify_sid']}/VerificationCheck",
        data={"To": phone, "Code": code},
        auth=(cfg["account_sid"], cfg["auth_token"]),
        timeout=15,
    )
    return r.ok and r.json().get("status") == "approved"


# ---------- current user ----------
def get_user():
    """Return the signed-in user dict, or None."""
    if _google_enabled() and getattr(st.user, "is_logged_in", False):
        uid = getattr(st.user, "sub", None) or st.user.email
        return {
            "id": f"google:{uid}",
            "name": getattr(st.user, "name", None) or st.user.email,
            "email": st.user.email,
            "provider": "google",
        }
    phone = st.session_state.get("phone_user")
    if phone:
        return {"id": f"phone:{phone}", "name": phone, "phone": phone, "provider": "phone"}
    return None


def _login_screen():
    st.markdown(
        """
        <style>
        .stApp { background: #F7F3E7; }
        .login-card { max-width: 460px; margin: 8vh auto 18px auto; text-align: center; }
        .login-card h1 { font-size: 2.4rem; color: #14432F; margin: 0; }
        .login-card p { color: #566B5E; }
        </style>
        <div class="login-card">
          <div style="font-size:3rem">♻️</div>
          <h1>ReWA.AI</h1>
          <p>Sign in to keep your recycling history, Eco Points and streaks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        if _google_enabled():
            if st.button("🔐 Continue with Google", use_container_width=True, key="g_login"):
                st.login()

        if _twilio():
            st.markdown("**or use your phone number**")
            _phone_form()

        if not _google_enabled() and not _twilio():
            st.error("No sign-in method is configured yet. See the setup notes for secrets.toml.")


def _phone_form():
    import time

    phone = st.text_input(
        "Phone number with country code",
        placeholder="+919876543210",
        key="phone_input",
    ).strip().replace(" ", "")

    sent_to = st.session_state.get("otp_sent_to")
    last = st.session_state.get("otp_sent_at", 0)
    wait = int(OTP_COOLDOWN_SECONDS - (time.time() - last))

    if st.button(
        "Send code" if not sent_to else "Resend code",
        use_container_width=True,
        disabled=wait > 0 and sent_to is not None,
        key="otp_send",
    ):
        if not PHONE_RE.match(phone):
            st.error("Enter the number in international format, e.g. +919876543210.")
        else:
            try:
                _send_code(phone)
                st.session_state.otp_sent_to = phone
                st.session_state.otp_sent_at = time.time()
                st.rerun()
            except Exception:
                st.error("Could not send the code. Check the number and try again.")

    if sent_to:
        st.caption(f"Code sent to {sent_to}. Resend available in {max(wait, 0)}s.")
        code = st.text_input("6-digit code", max_chars=6, key="otp_code")
        if st.button("Verify & sign in", type="primary", use_container_width=True, key="otp_verify"):
            try:
                if _check_code(sent_to, code.strip()):
                    st.session_state.phone_user = sent_to
                    st.session_state.pop("otp_sent_to", None)
                    st.rerun()
                else:
                    st.error("That code is wrong or has expired.")
            except Exception:
                st.error("Could not verify the code right now. Please try again.")


def require_login():
    """Block the page until the visitor is signed in; return the user dict."""
    user = get_user()
    if user is None:
        _login_screen()
        st.stop()

    # A different person signed in on this browser tab: wipe the previous
    # person's batch, bins, advice etc. so nothing leaks between accounts.
    if st.session_state.get("active_user") != user["id"]:
        for key in list(st.session_state.keys()):
            if key not in _KEEP:
                del st.session_state[key]
        st.session_state.active_user = user["id"]
    return user


def account_bar(user):
    """Small 'Signed in as ... [Sign out]' row."""
    left, right = st.columns([4, 1])
    with left:
        label = user.get("email") or user.get("phone") or user["name"]
        st.caption(f"👤 Signed in as **{label}** ({user['provider']})")
    with right:
        if st.button("Sign out", use_container_width=True, key="sign_out"):
            st.session_state.pop("phone_user", None)
            st.session_state.pop("active_user", None)
            if user["provider"] == "google":
                st.logout()
            else:
                st.rerun()