"""Tests for motion_prompts.py (Shot Motion Prompts). No GPU, no network."""

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import motion_prompts as mp  # noqa: E402

PLAN = {"title": "Run Free", "style": "cinematic", "shots": [
    {"id": 1, "description": "A runner laces her shoes at dawn", "camera": "close-up, slow push-in",
     "duration": 5, "image_prompt": "", "video_prompt": ""},
    {"id": 2, "description": "She sprints across an empty bridge", "camera": "tracking shot",
     "duration": 3, "image_prompt": "", "video_prompt": ""},
    {"id": 3, "description": "", "camera": "wide",
     "duration": 2, "image_prompt": "", "video_prompt": "Product spins on a plinth"},
]}


class GridSnapTests(unittest.TestCase):
    def test_wan_grid(self):
        # 5 s * 16 fps = 80 -> nearest 4k+1 is 81
        self.assertEqual(mp.snap_frames(5, "wan2.1"), (81, 81 / 16))
        self.assertEqual(mp.snap_frames(3, "wan2.1")[0], 49)        # 48 -> 49
        self.assertEqual(mp.snap_frames(2, "wan2.2")[0], 33)        # 32 -> 33
        self.assertEqual(mp.snap_frames(10, "wan2.1")[0], 81)       # clamped to the max
        self.assertEqual(mp.snap_frames(0.1, "wan2.1")[0], 5)       # at least one grid step
        for s in (1, 2.2, 3.7, 4.4, 7):
            f, _ = mp.snap_frames(s, "wan2.2")
            self.assertEqual((f - 1) % 4, 0)

    def test_minimax_h3_grid(self):
        for s in (1, 2, 3, 4.5, 6, 9):
            f, secs = mp.snap_frames(s, "minimax-h3")
            self.assertEqual((f - 5) % 17, 0, f)
            self.assertAlmostEqual(secs, f / 24, places=3)
        self.assertEqual(mp.snap_frames(4, "minimax-h3")[0], 90)    # 96 -> 17*5+5 = 90
        self.assertEqual(mp.snap_frames(5, "minimax-h3")[0], 124)   # 120 -> 17*7+5 = 124

    def test_ltx_grid(self):
        self.assertEqual(mp.snap_frames(5, "ltx")[0], 121)          # 120 -> 121
        self.assertEqual(mp.snap_frames(3, "ltx")[0], 73)           # 72 -> 73
        for s in (1.3, 2.9, 6.1):
            self.assertEqual((mp.snap_frames(s, "ltx")[0] - 1) % 8, 0)

    def test_api_engines(self):
        self.assertEqual(mp.snap_frames(3, "kling"), (120, 5.0))
        self.assertEqual(mp.snap_frames(8, "kling")[1], 10.0)
        self.assertEqual(mp.snap_frames(7.5, "kling")[1], 10.0)     # tie -> longer
        self.assertEqual(mp.snap_frames(2, "seedance")[1], 4.0)
        self.assertEqual(mp.snap_frames(6.4, "seedance")[1], 6.0)
        self.assertEqual(mp.snap_frames(20, "seedance")[1], 12.0)
        self.assertEqual(mp.snap_frames(3, "hailuo")[1], 6.0)
        self.assertEqual(mp.snap_frames(9, "hailuo")[1], 10.0)

    def test_fps_override(self):
        f, s = mp.snap_frames(5, "wan2.1", fps=24)                 # 120 -> clamped to 81
        self.assertEqual(f, 81)
        f, s = mp.snap_frames(3, "ltx", fps=30)                     # 90 -> 89
        self.assertEqual(f, 89)

    def test_bad_input(self):
        with self.assertRaises(mp.StageError):
            mp.snap_frames(3, "nope")
        with self.assertRaises(mp.StageError):
            mp.snap_frames(0, "wan2.1")


class CameraTests(unittest.TestCase):
    def test_vocabulary(self):
        self.assertEqual(mp.camera_move("push-in"), "slow dolly push-in")
        self.assertEqual(mp.camera_move("Push in", "subtle"), "very slow dolly push-in")
        self.assertEqual(mp.camera_move("pull back"), "slow dolly pull-back")
        self.assertEqual(mp.camera_move("pan left", "dynamic"), "fast pan left")
        self.assertEqual(mp.camera_move("tracking shot following the subject"),
                         "slow tracking shot following the subject")
        self.assertEqual(mp.camera_move("orbit"), "slow orbit around the subject")
        self.assertEqual(mp.camera_move("close-up, shallow depth of field, static"),
                         "close-up, locked-off static camera")
        self.assertEqual(mp.camera_move("wide establishing shot, slow push-in"),
                         "wide shot, slow dolly push-in")
        self.assertEqual(mp.camera_move("top-down overhead shot, slow rotation"),
                         "slow overhead rotation")
        self.assertEqual(mp.camera_move("handheld"), "subtle handheld drift")

    def test_framing_only_and_empty(self):
        self.assertEqual(mp.camera_move("wide"), "wide shot, slow dolly push-in")
        self.assertEqual(mp.camera_move("", "subtle"), "almost static camera with a gentle drift")
        with self.assertRaises(mp.StageError):
            mp.camera_move("pan", "wild")


class BuildTests(unittest.TestCase):
    def test_prompts_and_lists(self):
        r = mp.build_motion_prompts(json.dumps(PLAN), "wan2.1", "natural", 0, "35mm film")
        self.assertEqual(len(r["prompts"]), 3)
        self.assertEqual(r["frame_counts"], [81, 49, 33])
        self.assertEqual(r["fps"], 16)
        self.assertIn("A runner laces her shoes at dawn.", r["prompts"][0])
        self.assertIn("Camera: close-up, slow dolly push-in.", r["prompts"][0])
        self.assertTrue(r["prompts"][0].endswith("35mm film."))
        self.assertIn("Product spins on a plinth", r["prompts"][2])     # video_prompt fallback
        self.assertIn("morphing", r["negative_prompt"])
        self.assertAlmostEqual(r["seconds"][0], 81 / 16, places=3)

    def test_durations_override(self):
        r = mp.build_motion_prompts(PLAN, "ltx", "subtle", 0, "", "2, 4")
        self.assertEqual(r["frame_counts"], [49, 97, 49])                # 3rd from the plan (2 s)
        r = mp.build_motion_prompts(PLAN, "kling", "dynamic", 0, "", "9")
        self.assertEqual(r["seconds"], [10.0, 5.0, 5.0])
        self.assertNotIn("static frame", r["negative_prompt"])
        with self.assertRaises(mp.StageError):
            mp.build_motion_prompts(PLAN, "ltx", "natural", 0, "", "2, x")

    def test_fps_override_output(self):
        r = mp.build_motion_prompts(PLAN, "minimax-h3", "natural", 25)
        self.assertEqual(r["fps"], 25)
        for f in r["frame_counts"]:
            self.assertEqual((f - 5) % 17, 0)

    def test_bad_plan(self):
        with self.assertRaises(mp.StageError):
            mp.build_motion_prompts("", "wan2.1")
        with self.assertRaises(mp.StageError):
            mp.build_motion_prompts("{not json", "wan2.1")
        with self.assertRaises(mp.StageError):
            mp.build_motion_prompts('{"shots": []}', "wan2.1")


if __name__ == "__main__":
    unittest.main()
