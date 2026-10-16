"""Tests for rewa_logic.py.

Run with either:
    python -m unittest -v test_rewa_logic
    python -m pytest -q          (if you have pytest installed)
"""
import io
import unittest

from PIL import Image

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


if __name__ == "__main__":
    unittest.main()