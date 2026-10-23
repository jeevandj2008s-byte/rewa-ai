"""Tests for rewa_logic.py.

Run with either:
    python -m unittest -v test_rewa_logic
    python -m pytest -q          (if you have pytest installed)
"""
import io
import logging
import os
import tempfile
import unittest
import uuid

# Suppress harmless Streamlit bare-mode context warnings during test execution
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

from PIL import Image

import data_base
import rewards
import rewa_logic as L


# ---------- helpers ----------
class FakeRewards:
    """Mimics the real rewards DB: one reward per event key."""

    def __init__(self, fail=False):
        self.keys = set()
        self.points = 0
        self.fail = fail

    def award(self, uid, label, category):
        if self.fail:
            raise RuntimeError("database is locked")
        if uid in self.keys:
            return False
        self.keys.add(uid)
        self.points += 10
        return True


def make_env(accept=True, rewards=None):
    rewards = rewards or FakeRewards()
    bins = []

    def record_fn(category, label):
        if not accept:
            return False
        bins.append((category, label))
        return True

    env = {
        "recorded_uids": set(),
        "recorded_bins": {},
        "pending_rewards": {},
        "record_fn": record_fn,
        "award_fn": rewards.award,
    }
    return env, rewards, bins


def run(env, uid="a", label="plastic bottle", category="Plastic"):
    return L.record_and_reward(uid, label, category, **env)


def png_bytes(size=(20, 20), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 120, 60)).save(buf, format=fmt)
    return buf.getvalue()


class RecordingAndRewards(unittest.TestCase):
    def test_record_awards_points_once(self):
        env, rewards, bins = make_env()
        self.assertEqual(run(env), ("ok", ""))
        self.assertEqual(rewards.points, 10)
        self.assertEqual(bins, [("Plastic", "plastic bottle")])
        self.assertEqual(env["recorded_bins"]["a"], "Plastic")

    def test_double_click_does_not_double_record_or_reward(self):
        env, rewards, bins = make_env()
        run(env)
        self.assertEqual(run(env)[0], "duplicate")
        self.assertEqual(rewards.points, 10)
        self.assertEqual(len(bins), 1)

    def test_full_bin_gives_no_points_and_is_not_marked_recorded(self):
        env, rewards, _ = make_env(accept=False)
        self.assertEqual(run(env)[0], "full")
        self.assertEqual(rewards.points, 0)
        self.assertNotIn("a", env["recorded_uids"])

    def test_record_crash_gives_no_points(self):
        env, rewards, _ = make_env()

        def boom(category, label):
            raise OSError("disk error")

        env["record_fn"] = boom
        self.assertEqual(run(env)[0], "record_err")
        self.assertEqual(rewards.points, 0)
        self.assertNotIn("a", env["recorded_uids"])

    def test_points_failure_is_remembered_and_retry_recovers(self):
        rewards = FakeRewards(fail=True)
        env, _, _ = make_env(rewards=rewards)
        self.assertEqual(run(env)[0], "points_err")
        self.assertIn("a", env["pending_rewards"])
        self.assertEqual(rewards.points, 0)

        rewards.fail = False
        recovered = L.retry_pending_rewards(env["pending_rewards"], env["award_fn"])
        self.assertEqual(recovered, 1)
        self.assertEqual(rewards.points, 10)
        self.assertEqual(env["pending_rewards"], {})

    def test_retry_never_double_rewards(self):
        env, rewards, _ = make_env()
        run(env)
        env["pending_rewards"]["a"] = ("plastic bottle", "Plastic")  # stale entry
        L.retry_pending_rewards(env["pending_rewards"], env["award_fn"])
        self.assertEqual(rewards.points, 10)

    def test_category_correction_uses_confirmed_bin(self):
        env, _, bins = make_env()
        run(env, label="glass jar", category="Plastic")  # user corrected the bin
        self.assertEqual(bins, [("Plastic", "glass jar")])
        self.assertEqual(env["recorded_bins"]["a"], "Plastic")

    def test_same_photo_cannot_be_recorded_twice(self):
        env, rewards, bins = make_env()
        env["recorded_fingerprints"] = set()
        env["fingerprint"] = "photo-1"
        self.assertEqual(run(env, uid="a")[0], "ok")
        # same picture, classified again -> brand new uid, same fingerprint
        self.assertEqual(run(env, uid="b")[0], "same_photo")
        self.assertEqual(rewards.points, 10)
        self.assertEqual(len(bins), 1)

    def test_different_photos_are_both_recorded(self):
        env, rewards, _ = make_env()
        env["recorded_fingerprints"] = set()
        env["fingerprint"] = "photo-1"
        run(env, uid="a")
        env["fingerprint"] = "photo-2"
        self.assertEqual(run(env, uid="b")[0], "ok")
        self.assertEqual(rewards.points, 20)

    def test_failed_record_does_not_burn_the_photo(self):
        env, rewards, _ = make_env(accept=False)
        env["recorded_fingerprints"] = set()
        env["fingerprint"] = "photo-1"
        self.assertEqual(run(env, uid="a")[0], "full")
        self.assertEqual(env["recorded_fingerprints"], set())

    def test_reward_gap(self):
        self.assertEqual(L.reward_gap(5, 3), 2)
        self.assertEqual(L.reward_gap(3, 5), 0)


class Uncertainty(unittest.TestCase):
    def test_low_score_needs_manual_category(self):
        self.assertTrue(L.needs_manual_category(0.30, "Plastic"))
        self.assertTrue(L.needs_manual_category(0.90, None))
        self.assertFalse(L.needs_manual_category(0.90, "Plastic"))


class ImageValidation(unittest.TestCase):
    def test_valid_png_loads_as_rgb(self):
        self.assertEqual(L.load_validated_image(png_bytes()).mode, "RGB")

    def test_empty_file_rejected(self):
        with self.assertRaises(ValueError):
            L.load_validated_image(b"")

    def test_text_file_rejected(self):
        with self.assertRaises(ValueError):
            L.load_validated_image(b"hello, I am not an image")

    def test_oversized_file_rejected(self):
        with self.assertRaises(ValueError):
            L.load_validated_image(png_bytes(), max_bytes=10)

    def test_unsupported_format_rejected(self):
        with self.assertRaises(ValueError):
            L.load_validated_image(png_bytes(fmt="BMP"))

    def test_large_image_is_shrunk(self):
        img = L.load_validated_image(png_bytes(size=(3000, 1000)), max_side=1600)
        self.assertLessEqual(max(img.size), 1600)


class RewardExtras(unittest.TestCase):
    def test_no_badges_when_nothing_recorded(self):
        badges = L.badge_progress({"total_events": 0, "longest_streak": 0}, [])
        self.assertFalse(any(b["earned"] for b in badges))

    def test_badges_follow_summary(self):
        rows = [{"category": "Plastic", "points": 10},
                {"category": "Paper", "points": 10},
                {"category": "Hazardous", "points": 10}]
        badges = {b["key"]: b for b in L.badge_progress(
            {"total_events": 5, "longest_streak": 3}, rows)}
        self.assertTrue(badges["first"]["earned"])
        self.assertTrue(badges["five"]["earned"])
        self.assertTrue(badges["streak3"]["earned"])
        self.assertFalse(badges["streak7"]["earned"])
        self.assertTrue(badges["explorer"]["earned"])
        self.assertTrue(badges["hazard"]["earned"])
        self.assertFalse(badges["twentyfive"]["earned"])

    def test_garden_stages(self):
        self.assertEqual(L.garden_stage(0)["name"], "Empty plot")
        self.assertEqual(L.garden_stage(1)["name"], "Seed")
        self.assertEqual(L.garden_stage(5)["name"], "Sprout")
        self.assertEqual(L.garden_stage(14)["name"], "Sprout")
        self.assertEqual(L.garden_stage(15)["name"], "Sapling")
        top = L.garden_stage(500)
        self.assertEqual(top["name"], "Grove")
        self.assertIsNone(top["next_name"])

    def test_garden_progress_between_stages(self):
        stage = L.garden_stage(3)  # Seed(1) -> Sprout(5)
        self.assertEqual(stage["remaining"], 2)
        self.assertAlmostEqual(stage["progress"], 0.5)

    def test_missions(self):
        m = L.daily_missions(2, 3, 1, 0)
        self.assertEqual([x["done"] for x in m], [False, False, False])
        m = L.daily_missions(5, 3, 2, 1)
        self.assertTrue(all(x["done"] for x in m))
        self.assertEqual(m[0]["current"], 3)  # capped at the target

    def test_passport_stamps(self):
        stamps = L.passport_stamps([{"category": "Glass", "points": 10}],
                                   ["Plastic", "Glass"])
        self.assertEqual(stamps, [{"bin": "Plastic", "stamped": False},
                                  {"bin": "Glass", "stamped": True}])


class SessionIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_recycling.db")
        self.orig_db = getattr(data_base, "DB_NAME", "recycling.db")
        data_base.DB_NAME = self.db_path
        data_base.create_data_base()
        rewards.init_rewards_db()

    def tearDown(self):
        data_base.DB_NAME = self.orig_db
        self.temp_dir.cleanup()

    def test_events_isolated_between_sessions(self):
        session_1 = f"anon:{uuid.uuid4().hex}"
        session_2 = f"anon:{uuid.uuid4().hex}"

        # Session 1 records an event
        data_base.set_current_user(session_1)
        data_base.save_event("Plastic", "Water Bottle", "2026-10-23T10:00:00")
        s1_events = data_base.get_events()
        self.assertEqual(len(s1_events), 1)
        self.assertEqual(s1_events[0]["item_name"], "Water Bottle")

        # Session 2 must see zero events and zero totals from Session 1
        data_base.set_current_user(session_2)
        s2_events = data_base.get_events()
        self.assertEqual(len(s2_events), 0)
        s2_totals = data_base.get_bin_totals()
        self.assertEqual(s2_totals, {})

        # Session 2 records its own event
        data_base.save_event("Paper", "Cardboard Box", "2026-10-23T10:05:00")
        s2_events = data_base.get_events()
        self.assertEqual(len(s2_events), 1)
        self.assertEqual(s2_events[0]["item_name"], "Cardboard Box")

        # Session 1 still only sees Session 1's event
        data_base.set_current_user(session_1)
        s1_events = data_base.get_events()
        self.assertEqual(len(s1_events), 1)
        self.assertEqual(s1_events[0]["item_name"], "Water Bottle")

    def test_rewards_isolated_between_sessions(self):
        session_1 = f"anon:{uuid.uuid4().hex}"
        session_2 = f"anon:{uuid.uuid4().hex}"

        # Session 1 earns rewards
        data_base.set_current_user(session_1)
        awarded = rewards.award_event("evt_1", "plastic bottle", "Plastic")
        self.assertTrue(awarded)
        s1_summary = rewards.get_summary()
        self.assertEqual(s1_summary["total_points"], 10)
        self.assertEqual(s1_summary["total_events"], 1)

        # Session 2 must see 0 points and 0 events
        data_base.set_current_user(session_2)
        s2_summary = rewards.get_summary()
        self.assertEqual(s2_summary["total_points"], 0)
        self.assertEqual(s2_summary["total_events"], 0)
        self.assertEqual(rewards.get_points_by_category(), [])

        # Session 2 earns rewards
        awarded_2 = rewards.award_event("evt_2", "newspaper", "Paper")
        self.assertTrue(awarded_2)
        s2_summary = rewards.get_summary()
        self.assertEqual(s2_summary["total_points"], 10)

        # Session 1 remains unaffected
        data_base.set_current_user(session_1)
        s1_summary = rewards.get_summary()
        self.assertEqual(s1_summary["total_points"], 10)

    def test_corrections_logging_and_stats(self):
        session_1 = f"anon:{uuid.uuid4().hex}"
        session_2 = f"anon:{uuid.uuid4().hex}"

        # Session 1 logs an accepted prediction
        data_base.set_current_user(session_1)
        data_base.log_correction("test.jpg", "plastic bottle", "plastic bottle", "Plastic", "Plastic")
        stats_1 = data_base.get_correction_stats()
        self.assertEqual(stats_1["total"], 1)
        self.assertEqual(stats_1["corrections"], 0)
        self.assertEqual(stats_1["accepted"], 1)
        self.assertEqual(stats_1["correction_rate"], 0.0)

        # Session 1 logs a human correction
        data_base.log_correction("test2.jpg", "plastic wrapper", "paper sheet", "Plastic", "Paper")
        stats_1 = data_base.get_correction_stats()
        self.assertEqual(stats_1["total"], 2)
        self.assertEqual(stats_1["corrections"], 1)
        self.assertEqual(stats_1["correction_rate"], 50.0)

        # Session 2 must have zero corrections and zero totals
        data_base.set_current_user(session_2)
        stats_2 = data_base.get_correction_stats()
        self.assertEqual(stats_2["total"], 0)
        self.assertEqual(stats_2["corrections"], 0)
        self.assertEqual(stats_2["correction_rate"], 0.0)


class BinRulesTests(unittest.TestCase):
    def test_all_candidate_labels_have_mapping(self):
        import bin_rules
        for label in bin_rules.CANDIDATE_LABELS:
            self.assertIn(
                label,
                bin_rules.ITEM_TO_BIN,
                f"Candidate label '{label}' is missing from ITEM_TO_BIN mapping"
            )
            if label != "other waste":
                target_bin = bin_rules.resolve_bin(label)
                self.assertIn(
                    target_bin,
                    bin_rules.CANONICAL_BINS,
                    f"Label '{label}' mapped to '{target_bin}', not in canonical bins"
                )

    def test_bengaluru_rules_complete(self):
        import bin_rules
        for label in bin_rules.CANDIDATE_LABELS:
            rule = bin_rules.get_bengaluru_rule(label)
            self.assertIsNotNone(rule)
            self.assertIn("bengaluru_category", rule)
            self.assertIn("guidance", rule)
            self.assertIn("source", rule)


class SupabaseAuthTests(unittest.TestCase):
    """Unit tests for Supabase Authentication, validation, and session identity."""

    def setUp(self):
        import streamlit as st
        # Reset session state for test isolation
        for k in list(st.session_state.keys()):
            del st.session_state[k]

        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_auth_recycling.db")
        self.orig_db = getattr(data_base, "DB_NAME", "recycling.db")
        data_base.DB_NAME = self.db_path
        data_base.create_data_base()
        rewards.init_rewards_db()

    def tearDown(self):
        data_base.DB_NAME = self.orig_db
        self.temp_dir.cleanup()

    def test_validate_email(self):
        import auth
        # Valid emails
        ok, _ = auth.validate_email("test@example.com")
        self.assertTrue(ok)
        ok, _ = auth.validate_email("user.name+tag@sub.domain.org")
        self.assertTrue(ok)

        # Invalid emails
        ok, msg = auth.validate_email("")
        self.assertFalse(ok)
        ok, msg = auth.validate_email("notanemail")
        self.assertFalse(ok)
        ok, msg = auth.validate_email("user@")
        self.assertFalse(ok)
        ok, msg = auth.validate_email("@domain.com")
        self.assertFalse(ok)
        ok, msg = auth.validate_email(None)
        self.assertFalse(ok)

    def test_validate_password(self):
        import auth
        # Valid password
        ok, _ = auth.validate_password("securepassword123")
        self.assertTrue(ok)

        # Matching confirmation
        ok, _ = auth.validate_password("securepassword123", "securepassword123")
        self.assertTrue(ok)

        # Too short
        ok, msg = auth.validate_password("short")
        self.assertFalse(ok)
        self.assertIn("at least 8 characters", msg)

        # Mismatch
        ok, msg = auth.validate_password("securepassword123", "differentpassword")
        self.assertFalse(ok)
        self.assertIn("do not match", msg)

    def test_format_auth_error_messages(self):
        import auth
        self.assertIn(
            "Incorrect email or password",
            auth.format_auth_error(Exception("Invalid login credentials")),
        )
        self.assertIn(
            "Email not confirmed yet",
            auth.format_auth_error(Exception("Email not confirmed")),
        )
        self.assertIn(
            "An account with this email already exists",
            auth.format_auth_error(Exception("User already registered")),
        )
        self.assertIn(
            "Too many requests",
            auth.format_auth_error(Exception("Rate limit exceeded 429")),
        )
        self.assertIn(
            "Network connection error",
            auth.format_auth_error(Exception("Failed to establish a connection")),
        )

    def test_two_anonymous_sessions_never_share_id(self):
        import streamlit as st
        import auth

        # Session A
        st.session_state.clear()
        user_a = auth.resolve_user()
        id_a = user_a["id"]
        self.assertTrue(id_a.startswith("anon:"))

        # Session B
        st.session_state.clear()
        user_b = auth.resolve_user()
        id_b = user_b["id"]
        self.assertTrue(id_b.startswith("anon:"))

        self.assertNotEqual(id_a, id_b, "Two anonymous sessions must have unique UUIDs")

    def test_mock_sign_in_and_identity_resolution(self):
        from unittest.mock import MagicMock
        import streamlit as st
        import auth

        mock_client = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "supabase-uuid-999"
        mock_user.email = "ecouser@example.com"

        mock_session = MagicMock()
        mock_session.access_token = "fake-jwt-token"
        mock_session.refresh_token = "fake-refresh-token"
        mock_session.expires_at = 9999999999

        mock_response = MagicMock()
        mock_response.user = mock_user
        mock_response.session = mock_session
        mock_client.auth.sign_in_with_password.return_value = mock_response

        # Execute sign in
        success, msg = auth.sign_in_user(
            "ecouser@example.com", "ValidPassword123!", client=mock_client
        )
        self.assertTrue(success)
        mock_client.auth.sign_in_with_password.assert_called_once_with({
            "email": "ecouser@example.com",
            "password": "ValidPassword123!",
        })

        # Identity resolution should now report the logged-in Supabase user
        resolved = auth.resolve_user()
        self.assertEqual(resolved["id"], "supabase-uuid-999")
        self.assertEqual(resolved["email"], "ecouser@example.com")
        self.assertEqual(resolved["provider"], "supabase")
        self.assertFalse(resolved["is_guest"])

        # Sign out should flush credentials
        auth.sign_out_user(client=mock_client)
        mock_client.auth.sign_out.assert_called_once()
        self.assertIsNone(auth.get_user())

    def test_mock_sign_up_flow(self):
        from unittest.mock import MagicMock
        import auth

        mock_client = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "supabase-uuid-new"
        mock_user.email = "newbie@example.com"

        mock_response = MagicMock()
        mock_response.user = mock_user
        mock_response.session = None  # Requires email verification
        mock_client.auth.sign_up.return_value = mock_response

        success, msg = auth.sign_up_user(
            "newbie@example.com", "SecretPass123!", "SecretPass123!", client=mock_client
        )
        self.assertTrue(success)
        self.assertIn("confirm your account", msg)

    def test_mock_reset_password(self):
        from unittest.mock import MagicMock
        import auth

        mock_client = MagicMock()
        success, msg = auth.reset_password("forgot@example.com", client=mock_client)
        self.assertTrue(success)
        mock_client.auth.reset_password_for_email.assert_called_once_with("forgot@example.com")

    def test_merge_anon_data_to_user(self):
        import sqlite3
        import auth

        anon_id = f"anon:merge_test_{uuid.uuid4().hex[:8]}"
        supabase_id = f"supabase_uuid_{uuid.uuid4().hex[:8]}"

        # Insert test records under anon_id
        conn = sqlite3.connect(self.db_path)
        with conn:
            conn.execute(
                "INSERT INTO recycling_events (material, item_name, timestamp, user_id) VALUES (?, ?, ?, ?)",
                ("Plastic", "plastic bottle", "2026-01-01 12:00:00", anon_id)
            )
            conn.execute(
                "INSERT OR IGNORE INTO reward_events (event_key, item_label, bin_name, points, activity_date, user_id) VALUES (?, ?, ?, ?, ?, ?)",
                ("evt_merge_1", "plastic bottle", "Plastic", 10, "2026-01-01", anon_id)
            )
        conn.close()

        # Merge
        auth.merge_anon_data_to_user(anon_id, supabase_id, db_name=self.db_path)

        # Verify records moved to supabase_id
        conn = sqlite3.connect(self.db_path)
        ev_count = conn.execute(
            "SELECT COUNT(*) FROM recycling_events WHERE user_id = ?", (supabase_id,)
        ).fetchone()[0]
        rw_count = conn.execute(
            "SELECT COUNT(*) FROM reward_events WHERE user_id = ?", (supabase_id,)
        ).fetchone()[0]
        old_count = conn.execute(
            "SELECT COUNT(*) FROM recycling_events WHERE user_id = ?", (anon_id,)
        ).fetchone()[0]
        conn.close()

        self.assertGreaterEqual(ev_count, 1)
        self.assertGreaterEqual(rw_count, 1)
        self.assertEqual(old_count, 0)


class MultiObjectDetectorTests(unittest.TestCase):
    """Unit tests for YOLO detector, cropping, NMS, overlap merging, and fallback."""

    def test_box_clamping(self):
        import detector
        # Out-of-bounds negative coordinates
        clamped = detector.clamp_box([-50, -20, 150, 250], 100, 200)
        self.assertEqual(clamped, [0, 0, 100, 200])

        # Exceeding width and height
        clamped2 = detector.clamp_box([10, 20, 500, 600], 100, 200)
        self.assertEqual(clamped2, [10, 20, 100, 200])

        # Degenerate inverted box
        clamped3 = detector.clamp_box([80, 80, 50, 50], 100, 100)
        self.assertGreater(clamped3[2], clamped3[0])
        self.assertGreater(clamped3[3], clamped3[1])

    def test_box_padding(self):
        import detector
        # Interior box with 10% padding
        padded = detector.pad_box([20, 20, 60, 60], 100, 100, padding_ratio=0.10)
        # width = 40, dx = 4 -> [16, 16, 64, 64]
        self.assertEqual(padded, [16, 16, 64, 64])

        # Box near border: should not exceed [0, 0, 100, 100]
        padded_border = detector.pad_box([2, 2, 98, 98], 100, 100, padding_ratio=0.20)
        self.assertEqual(padded_border, [0, 0, 100, 100])

    def test_calculate_iou_and_containment(self):
        import detector
        # Non-overlapping
        self.assertEqual(detector.calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertEqual(detector.calculate_containment([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

        # Identical boxes
        self.assertEqual(detector.calculate_iou([0, 0, 50, 50], [0, 0, 50, 50]), 1.0)
        self.assertEqual(detector.calculate_containment([0, 0, 50, 50], [0, 0, 50, 50]), 1.0)

        # Nested box (box2 completely inside box1)
        b1 = [0, 0, 100, 100]  # area 10000
        b2 = [20, 20, 40, 40]   # area 400
        self.assertEqual(detector.calculate_containment(b1, b2), 1.0)
        self.assertLess(detector.calculate_iou(b1, b2), 0.1)

    def test_merge_overlapping_boxes(self):
        import detector
        # Two heavily overlapping boxes (e.g. bottle body + label)
        b1 = {"box": [10, 10, 60, 60], "conf": 0.85}
        b2 = {"box": [15, 15, 65, 65], "conf": 0.70}
        merged = detector.merge_overlapping_boxes([b1, b2], iou_thresh=0.60)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["box"], [10, 10, 65, 65])
        self.assertEqual(merged[0]["conf"], 0.85)

        # Two distinct non-overlapping boxes
        b_distinct1 = {"box": [10, 10, 40, 40], "conf": 0.9}
        b_distinct2 = {"box": [70, 70, 90, 90], "conf": 0.8}
        merged_distinct = detector.merge_overlapping_boxes([b_distinct1, b_distinct2])
        self.assertEqual(len(merged_distinct), 2)

    def test_filter_and_cap_boxes(self):
        import detector
        img_w, img_h = 200, 200
        boxes = [
            {"box": [0, 0, 5, 5], "conf": 0.9},      # Tiny noise: 5x5 (< 24px)
            {"box": [100, 10, 150, 80], "conf": 0.75}, # Valid object 2
            {"box": [20, 10, 70, 80], "conf": 0.80},   # Valid object 1
        ]
        filtered = detector.filter_and_cap_boxes(boxes, img_w, img_h, min_size=20, max_objects=6)
        self.assertEqual(len(filtered), 2)
        # Verify left-to-right sorting (x1 ascending)
        self.assertEqual(filtered[0]["box"][0], 20)
        self.assertEqual(filtered[1]["box"][0], 100)

        # Cap enforcement
        many_boxes = [{"box": [i * 10, 10, i * 10 + 30, 60], "conf": 0.5 + i * 0.05} for i in range(10)]
        capped = detector.filter_and_cap_boxes(many_boxes, 400, 200, max_objects=4)
        self.assertEqual(len(capped), 4)

    def test_mock_detect_objects(self):
        from unittest.mock import MagicMock
        import torch
        import detector
        from PIL import Image

        img = Image.new("RGB", (200, 200), color="white")
        mock_model = MagicMock()

        # Mock boxes returned by Ultralytics YOLO
        mock_box_data = MagicMock()
        mock_box_data.__len__.return_value = 2
        mock_box_data.xyxy = torch.tensor([[10.0, 10.0, 80.0, 120.0], [110.0, 20.0, 180.0, 140.0]])
        mock_box_data.conf = torch.tensor([0.88, 0.76])

        mock_result = MagicMock()
        mock_result.boxes = mock_box_data
        mock_model.predict.return_value = [mock_result]

        detections = detector.detect_objects(img, model=mock_model)
        self.assertEqual(len(detections), 2)
        self.assertIn("crop", detections[0])
        self.assertIn("box", detections[0])
        self.assertIsInstance(detections[0]["crop"], Image.Image)

    def test_mock_classify_objects_multi_and_fallback(self):
        from unittest.mock import MagicMock
        import detector
        from PIL import Image

        img = Image.new("RGB", (200, 200), color="white")
        candidate_labels = ["plastic bottle", "paper sheet", "metal can"]

        # Mock classifier returning dummy CLIP predictions
        mock_classifier = MagicMock()
        mock_classifier.side_effect = lambda crop, candidate_labels: [
            {"label": "plastic bottle", "score": 0.85},
            {"label": "metal can", "score": 0.10},
            {"label": "paper sheet", "score": 0.05},
        ]

        # 1. Multi-object path with mock detector returning 2 items
        mock_detector = MagicMock()
        mock_box_data = MagicMock()
        mock_box_data.__len__.return_value = 2
        import torch
        mock_box_data.xyxy = torch.tensor([[10.0, 10.0, 60.0, 80.0], [90.0, 20.0, 160.0, 120.0]])
        mock_box_data.conf = torch.tensor([0.90, 0.80])
        mock_result = MagicMock()
        mock_result.boxes = mock_box_data
        mock_detector.predict.return_value = [mock_result]

        results, is_fallback = detector.classify_objects(
            img,
            classifier=mock_classifier,
            candidate_labels=candidate_labels,
            detector_model=mock_detector,
            multi_object_enabled=True,
        )
        self.assertFalse(is_fallback)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["best_label"], "plastic bottle")
        self.assertGreaterEqual(results[0]["best_score"], 0.8)

        # 2. Fallback path when YOLO finds no objects (or toggle OFF)
        mock_detector.predict.return_value = []
        results_fallback, is_fallback_flag = detector.classify_objects(
            img,
            classifier=mock_classifier,
            candidate_labels=candidate_labels,
            detector_model=mock_detector,
            multi_object_enabled=True,
        )
        self.assertTrue(is_fallback_flag)
        self.assertEqual(len(results_fallback), 1)
        self.assertEqual(results_fallback[0]["box"], [0, 0, 200, 200])

    def test_draw_bounding_boxes(self):
        import detector
        from PIL import Image

        img = Image.new("RGB", (300, 300), color="white")
        detections = [
            {"box": [20, 20, 100, 150], "best_label": "plastic bottle"},
            {"box": [120, 30, 220, 180], "best_label": "cardboard box"},
        ]
        annotated = detector.draw_bounding_boxes(img, detections)
        self.assertEqual(annotated.size, (300, 300))
        self.assertEqual(annotated.mode, "RGB")

    def test_downstream_recording_once_per_object(self):
        import rewa_logic
        # Simulate two objects from the same photo
        env, rewards_mock, bins = make_env()

        # Object 1 recorded
        status1, _ = rewa_logic.record_and_reward(
            uid="photo_obj_1",
            label="plastic bottle",
            category="Plastic",
            fingerprint="fp_obj_1",
            **env
        )
        self.assertEqual(status1, "ok")
        self.assertEqual(rewards_mock.points, 10)

        # Object 2 from same photo recorded (distinct UID and fingerprint)
        status2, _ = rewa_logic.record_and_reward(
            uid="photo_obj_2",
            label="plastic bag",
            category="Plastic",
            fingerprint="fp_obj_2",
            **env
        )
        self.assertEqual(status2, "ok")
        self.assertEqual(rewards_mock.points, 20)

        # Double click on Object 1 rejected as duplicate
        status_dup, _ = rewa_logic.record_and_reward(
            uid="photo_obj_1",
            label="plastic bottle",
            category="Plastic",
            fingerprint="fp_obj_1",
            **env
        )
        self.assertEqual(status_dup, "duplicate")
        self.assertEqual(rewards_mock.points, 20)


if __name__ == "__main__":
    unittest.main()
