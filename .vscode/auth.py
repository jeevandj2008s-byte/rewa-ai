"""auth.py - sign-in for ReWA.AI

Everyone can open the website. A visitor who has not logged in is a "guest"
(they can try the classifier, but nothing is saved for them). A "Log in"
panel sits at the top of the page; once logged in it turns into an account
panel with a Sign out button.

DESIGN NOTE (why it is built this way): the login panel is a normal block in
the page flow (an expander). It is NOT pinned with position:fixed and it is
NOT a pop-up dialog, because those two tricks were what could be covered,
blocked or broken by the page's heavy CSS/animations. A normal block cannot
be covered, so the button always responds.

Ways to log in (each one is switched on by its own block in
.streamlit/secrets.toml):
  1. Google account  -> Streamlit's built-in OpenID Connect  [auth]
  2. Phone number    -> SMS one-time code via Twilio Verify   [twilio]
  3. Demo login      -> LOCAL TESTING ONLY, no verification   [dev]

Usage in app.py:
    import auth
    current_user = auth.resolve_user()     # never blocks; guests are allowed
    data_base.set_current_user(current_user["id"])
    auth.account_corner(current_user)      # Log in panel / account panel

current_user = {
    "id": "google:123" | "phone:+9198..." | "demo:name" | "guest:abc...",
    "name": ..., "provider": ..., "is_guest": True | False,
}
Use current_user["id"] as the key that separates each person's records.

Requires: streamlit>=1.42, Authlib (for Google), requests (for Twilio).
"""
import re
import time
import uuid

import requests
import streamlit as st

# True  -> everyone can use EVERY feature without logging in. Visitors share
#          one saved "public" account (items, Eco Points, streaks, history).
# False -> visitors who are not logged in are guests (nothing is saved).
ALLOW_ANONYMOUS = True
PUBLIC_USER_ID = "public"

OTP_COOLDOWN_SECONDS = 30   # wait between "Send code" presses
OTP_MAX_ATTEMPTS = 5        # wrong codes allowed before a new code is needed
PHONE_RE = re.compile(r"^\+[1-9]\d{7,14}$")  # E.164, e.g. +919876543210
CODE_RE = re.compile(r"^\d{4,10}$")

# Session keys that survive a change of user
_KEEP = {"phone_user", "demo_user", "active_user", "guest_id", "intro_played"}

# Styles for the login panel. app.py appends this to its theme CSS.
# Deliberately plain: no fixed positioning, nothing that can cover a click.
CORNER_CSS = """
<style>
/* Hide Streamlit's own Deploy button */
[data-testid="stAppDeployButton"] { display: none !important; }

.st-key-auth_corner {
    max-width: 460px;
    margin: 0 0 14px 0;
    position: relative;
    z-index: 5;
}

.st-key-auth_corner [data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.92);
}

.st-key-auth_corner summary {
    font-weight: 700;
}

.st-key-auth_corner div.stButton > button {
    pointer-events: auto !important;
}
</style>
"""


# ---------- config helpers ----------
def _google_provider():
    """Return (enabled, provider_name).

    Streamlit supports two secrets layouts:
      [auth] client_id = ...           -> default provider (name None)
      [auth.google] client_id = ...    -> named provider ("google")
    """
    if not hasattr(st, "login"):
        return False, None
    try:
        cfg = st.secrets["auth"]
        if cfg.get("client_id"):
            return True, None
        named = []
        for name in cfg.keys():
            sub = cfg[name]
            if hasattr(sub, "get") and sub.get("client_id"):
                named.append(name)
        if named:
            return True, ("google" if "google" in named else named[0])
    except Exception:
        pass
    return False, None


def _google_enabled():
    return _google_provider()[0]


def _twilio():
    """Return the [twilio] secrets, or None if phone login is not configured."""
    try:
        cfg = st.secrets["twilio"]
        if cfg["account_sid"] and cfg["auth_token"] and cfg["verify_sid"]:
            return cfg
    except Exception:
        pass
    return None


def _demo_enabled():
    """Demo login is OFF unless [dev] enabled = true is in secrets.toml."""
    try:
        return bool(st.secrets["dev"]["enabled"])
    except Exception:
        return False


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
    """Return the logged-in user dict, or None if nobody is logged in."""
    if _google_enabled() and getattr(st.user, "is_logged_in", False):
        email = getattr(st.user, "email", None)
        uid = getattr(st.user, "sub", None) or email
        if uid:
            return {
                "id": f"google:{uid}",
                "name": getattr(st.user, "name", None) or email or "Google user",
                "email": email,
                "provider": "google",
            }

    phone = st.session_state.get("phone_user")
    if phone:
        return {"id": f"phone:{phone}", "name": phone, "phone": phone, "provider": "phone"}

    demo = st.session_state.get("demo_user")
    if demo and _demo_enabled():
        return {"id": f"demo:{demo}", "name": demo, "provider": "demo"}

    return None


def resolve_user():
    """Return the current user, or a temporary guest. Never blocks the page."""
    user = get_user()

    if user is None and ALLOW_ANONYMOUS:
        user = {
            "id": PUBLIC_USER_ID,
            "name": "Visitor",
            "provider": "public",
            "is_guest": False,
        }
    elif user is None:
        guest_id = st.session_state.get("guest_id")
        if not guest_id:
            guest_id = uuid.uuid4().hex
            st.session_state.guest_id = guest_id
        user = {
            "id": f"guest:{guest_id}",
            "name": "Guest",
            "provider": "guest",
            "is_guest": True,
        }
    else:
        user["is_guest"] = False

    # A different identity on this browser tab (guest -> user, user -> guest,
    # or user A -> user B): wipe the previous identity's batch, bins, advice
    # etc. so nothing leaks between accounts.
    if st.session_state.get("active_user") != user["id"]:
        for key in list(st.session_state.keys()):
            if key not in _KEEP:
                del st.session_state[key]
        st.session_state.active_user = user["id"]

    return user


# ---------- small display helpers ----------
def _mask_phone(phone):
    phone = phone or ""
    return phone[:3] + "•" * max(len(phone) - 7, 0) + phone[-4:] if len(phone) > 7 else phone


def _short_name(user):
    if user["provider"] == "phone":
        return _mask_phone(user["name"])
    first = (user.get("name") or "").split("@")[0].split(" ")[0]
    return first[:14] or "Account"


# ---------- login forms ----------
def _start_google_login():
    """Button callback. Runs BEFORE the page reruns (same as the original
    working pattern). Any error is saved and shown instead of vanishing."""
    st.session_state.pop("login_error", None)
    _, provider = _google_provider()
    try:
        if provider:
            st.login(provider)
        else:
            st.login()
    except Exception as error:
        st.session_state["login_error"] = str(error)


def _google_form():
    st.button(
        "🔐 Continue with Google",
        on_click=_start_google_login,
        use_container_width=True,
        key="g_login",
    )
    error = st.session_state.get("login_error")
    if error:
        st.error(
            "Google login is not set up correctly. Check the [auth] block in "
            ".streamlit/secrets.toml (redirect_uri, cookie_secret, client_id, "
            "client_secret, server_metadata_url) and that Authlib is "
            "installed (pip install Authlib)."
        )
        st.caption(f"Technical details: {error}")


def _phone_form():
    st.markdown("**Log in with your phone**")

    raw = st.text_input(
        "Phone number with country code",
        placeholder="+919876543210",
        key="phone_input",
    )
    phone = re.sub(r"[\s\-()]", "", raw or "")

    sent_to = st.session_state.get("otp_sent_to")
    last = st.session_state.get("otp_sent_at", 0.0)
    wait = int(OTP_COOLDOWN_SECONDS - (time.time() - last))

    if st.button(
        "Resend code" if sent_to else "Send code",
        use_container_width=True,
        disabled=bool(sent_to) and wait > 0,
        key="otp_send",
    ):
        if not PHONE_RE.match(phone):
            st.error("Enter the number in international format, e.g. +919876543210.")
        else:
            sent = False
            try:
                _send_code(phone)
                sent = True
            except Exception:
                st.error(
                    "Could not send the code. Check the number and your "
                    "Twilio settings (trial accounts can only text numbers "
                    "verified in Twilio) and try again."
                )
            if sent:
                st.session_state.otp_sent_to = phone
                st.session_state.otp_sent_at = time.time()
                st.session_state.otp_attempts = 0
                st.rerun()

    if sent_to:
        note = f"Code sent to {_mask_phone(sent_to)}."
        if wait > 0:
            note += f" You can ask for a new one in {wait}s."
        st.caption(note)

        code = st.text_input("Verification code", max_chars=10, key="otp_code")

        if st.button("Verify and log in", use_container_width=True, key="otp_verify"):
            attempts = st.session_state.get("otp_attempts", 0)
            code = (code or "").strip()

            if attempts >= OTP_MAX_ATTEMPTS:
                st.error("Too many wrong attempts. Please request a new code.")
            elif not CODE_RE.match(code):
                st.error("Enter the digits from the SMS.")
            else:
                approved = None
                try:
                    approved = _check_code(sent_to, code)
                except Exception:
                    st.error("Could not verify the code right now. Please try again.")

                if approved is True:
                    st.session_state.phone_user = sent_to
                    st.rerun()
                elif approved is False:
                    st.session_state.otp_attempts = attempts + 1
                    st.error("That code is wrong or has expired.")


def _demo_form():
    st.warning(
        "Demo login is for local testing only. It does not verify who you are."
    )
    name = st.text_input("Pick any display name", max_chars=24, key="demo_name")
    if st.button("Continue as demo user", use_container_width=True, key="demo_go"):
        clean = re.sub(r"[^a-z0-9_-]", "", (name or "").strip().lower())
        if len(clean) < 2:
            st.error("Use at least 2 letters or numbers.")
        else:
            st.session_state.demo_user = clean
            st.rerun()


def _login_panel():
    google = _google_enabled()
    phone = _twilio() is not None
    demo = _demo_enabled()

    # Status line: tells you instantly which methods this app can see.
    st.caption(
        f"Streamlit {st.__version__} • "
        f"Google: {'ready' if google else 'not set up'} • "
        f"Phone: {'ready' if phone else 'not set up'} • "
        f"Demo: {'on' if demo else 'off'}"
    )

    if not (google or phone or demo):
        st.error(
            "No login method is set up yet. Create .streamlit/secrets.toml "
            "next to app.py with the Google ([auth]), Twilio ([twilio]) or "
            "demo ([dev]) settings, then fully restart the app."
        )
        return

    st.caption("Log in to save your recycling history, Eco Points and streaks.")

    if google:
        _google_form()
        if phone or demo:
            st.markdown("**— or —**")

    if phone:
        _phone_form()
        if demo:
            st.markdown("**— or —**")

    if demo:
        _demo_form()


# ---------- the panel ----------
def _sign_out(user):
    st.session_state.pop("phone_user", None)
    st.session_state.pop("demo_user", None)
    st.session_state.pop("active_user", None)
    if user["provider"] == "google":
        st.logout()
    else:
        st.rerun()


def account_corner(user):
    """Login panel for guests, account panel otherwise (a normal page block)."""
    with st.container(key="auth_corner"):
        if user["is_guest"] or user["provider"] == "public":
            label = "🔐 Log in (optional)" if user["provider"] == "public" else "🔐 Log in"
            with st.expander(label, expanded=False):
                _login_panel()
        else:
            with st.expander(f"👤 {_short_name(user)}", expanded=False):
                label = user.get("email") or (
                    _mask_phone(user["name"]) if user["provider"] == "phone"
                    else user["name"]
                )
                st.markdown(f"**{label}**")
                st.caption(f"Logged in with {user['provider']}")
                if st.button("Sign out", use_container_width=True, key="sign_out"):
                    _sign_out(user)