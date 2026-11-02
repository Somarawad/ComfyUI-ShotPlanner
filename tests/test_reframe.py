"""Tests for reframe.py (Ad Reframe, Ad Cutdown). numpy + PIL only."""

import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import reframe as rf  # noqa: E402


def moving_square_mask(n, h, w, size=20, x_from=10, x_to=None, y=None):
    x_to = w - size - 10 if x_to is None else x_to
    y = h // 2 - size // 2 if y is None else y
    m = np.zeros((n, h, w), np.float32)
    for i in range(n):
        x = int(round(x_from + (x_to - x_from) * i / max(1, n - 1)))
        m[i, y:y + size, x:x + size] = 1
    return m


class CropTests(unittest.TestCase):
    def test_crop_size(self):
        self.assertEqual(rf.crop_size(1920, 1080, "9:16"), (608, 1080))
        self.assertEqual(rf.crop_size(1920, 1080, "1:1"), (1080, 1080))
        self.assertEqual(rf.crop_size(1920, 1080, "4:5"), (864, 1080))
        self.assertEqual(rf.crop_size(1080, 1920, "16:9"), (1080, 608))
        self.assertEqual(rf.output_size(1080, "9:16"), (1080, 1920))
        with self.assertRaises(rf.StageError):
            rf.crop_size(10, 10, "wide")

    def test_boxes_stay_inside(self):
        W, H = 320, 180
        cents = [(-500, 90), (0, 0), (160, 90), (319, 179), (5000, -40)]
        for ratio in ("9:16", "1:1", "4:5"):
            for x0, y0, cw, ch in rf.crop_boxes(cents, W, H, ratio, smoothing=0):
                self.assertGreaterEqual(x0, 0)
                self.assertGreaterEqual(y0, 0)
                self.assertLessEqual(x0 + cw, W)
                self.assertLessEqual(y0 + ch, H)
        boxes = rf.crop_boxes([(0, 90), (160, 90), (320, 90)], W, H, "9:16", 0)
        cw = boxes[0][2]
        self.assertEqual([b[0] for b in boxes], [0, 160 - cw // 2, W - cw])

    def test_mask_follow(self):
        n, H, W = 12, 90, 160
        mask = moving_square_mask(n, H, W)
        cents = rf.mask_centers(mask, W, H)
        self.assertAlmostEqual(cents[0][0], 20.0, places=4)          # square 10..30
        self.assertAlmostEqual(cents[-1][0], W - 20.0, places=4)
        self.assertAlmostEqual(cents[0][1], H / 2.0, delta=0.6)
        # mask at half resolution maps to frame pixels
        half = mask[:, ::2, ::2]
        self.assertAlmostEqual(rf.mask_centers(half, W, H)[0][0], cents[0][0], delta=1.5)
        frames = np.random.default_rng(0).random((n, H, W, 3)).astype(np.float32)
        out, boxes = rf.reframe(frames, "9:16", 64, mask, smoothing=0)
        self.assertEqual(out.shape, (n, 114, 64, 3))
        self.assertLess(boxes[0][0], boxes[-1][0])                  # crop moves with the subject
        x0, y0, cw, ch = boxes[n // 2]
        cx = cents[n // 2][0]
        self.assertLessEqual(x0, cx)
        self.assertGreaterEqual(x0 + cw, cx)

    def test_smoothing(self):
        jitter = [100 + (15 if i % 2 else -15) for i in range(40)]
        raw = rf.smooth_path(jitter, 0)
        smooth = rf.smooth_path(jitter, 0.7)
        fixed = rf.smooth_path(jitter, 1.0)
        self.assertEqual(list(raw), jitter)
        self.assertLess(np.std(np.diff(smooth)), np.std(np.diff(raw)) * 0.3)
        self.assertAlmostEqual(float(np.mean(smooth)), 100, delta=2)
        self.assertTrue(np.allclose(fixed, np.mean(jitter)))
        ramp = rf.smooth_path(np.arange(50.0), 0.5)                 # zero-phase: no lag mid-way
        self.assertAlmostEqual(ramp[25], 25, delta=0.5)

    def test_empty_mask_frames_and_mismatch(self):
        n, H, W = 6, 60, 100
        mask = moving_square_mask(n, H, W, size=10)
        mask[2:4] = 0
        cents = rf.fill_missing(rf.mask_centers(mask, W, H), (50, 30))
        self.assertEqual(len(cents), n)
        self.assertTrue(all(c is not None for c in cents))
        self.assertEqual(rf.fill_missing([None, None], (5, 5)), [(5, 5), (5, 5)])
        with self.assertRaises(rf.StageError):
            rf.reframe(np.zeros((n, H, W, 3), np.float32), "1:1", 64, mask[:3])

    def test_saliency_fallback(self):
        H, W = 90, 160
        f = np.zeros((H, W, 3), np.float32)
        f[30:60, 100:130] = 1.0                                     # bright object right of centre
        x, y = rf.saliency_center(f)
        self.assertGreater(x, W / 2 + 10)
        self.assertAlmostEqual(y, 45, delta=8)
        cx, cy = rf.saliency_center(np.zeros((H, W, 3), np.float32))
        self.assertEqual((cx, cy), (W / 2, H / 2))
        out, boxes = rf.reframe(np.repeat(f[None], 4, 0), "9:16", 32, None, 0.5)
        self.assertEqual(out.shape[1:], (56, 32, 3))
        self.assertGreater(boxes[0][0], (W - boxes[0][2]) / 2)     # crop shifted right


TIMING = {"fps": 10.0, "total_frames": 100, "total_seconds": 10.0, "shots": [
    {"shot": 1, "start_frame": 0, "end_frame": 30},
    {"shot": 2, "start_frame": 30, "end_frame": 50},
    {"shot": 3, "start_frame": 50, "end_frame": 70},
    {"shot": 4, "start_frame": 70, "end_frame": 100},
], "width": 64, "height": 36}
TOTAL = 120              # 100 shot frames + 20 end card frames


def verdicts(scores):
    return json.dumps({"shots": [{"shot": i + 1, "score": s} for i, s in enumerate(scores)]})


class CutdownTests(unittest.TestCase):
    def test_six_seconds_keeps_end_card(self):
        p = rf.plan_cutdown(TOTAL, json.dumps(TIMING), 6.0, "auto", verdicts([9, 5, 8, 9]), 1.3)
        self.assertEqual(p["total_frames"], 60)
        self.assertEqual(p["ranges"][-1], (100, 120))                 # end card intact, last
        self.assertEqual(p["timing"]["end_card_frames"], 20)
        self.assertEqual(p["timing"]["total_frames"], 40)
        kept = [s["source_shot"] for s in p["timing"]["shots"]]
        self.assertEqual(kept, [1, 3, 4])                             # 4 s / 1.3 s -> 3 shots; 2 scores worst
        self.assertEqual(sum(b - a for a, b in p["ranges"]), 60)
        # first shot keeps its head, last its tail
        self.assertEqual(p["ranges"][0][0], 0)
        self.assertEqual(p["ranges"][2][1], 100)

    def test_auto_shot_limit(self):
        p = rf.plan_cutdown(TOTAL, TIMING, 4.0, "auto", None, min_shot_s=1.0)  # 2 s of shots
        self.assertEqual([s["source_shot"] for s in p["timing"]["shots"]], [1, 4])
        self.assertEqual(p["total_frames"], 40)

    def test_explicit_keep_and_fifteen(self):
        p = rf.plan_cutdown(TOTAL, TIMING, 15.0, "2, 4")
        self.assertEqual([s["source_shot"] for s in p["timing"]["shots"]], [2, 4])
        self.assertEqual(p["ranges"], [(30, 50), (70, 100), (100, 120)])   # whole shots fit
        self.assertEqual(p["total_frames"], 70)
        self.assertIn("shorter than the target", p["report"])
        new = p["timing"]["shots"]
        self.assertEqual((new[1]["start_frame"], new[1]["end_frame"]), (20, 50))

    def test_no_end_card(self):
        p = rf.plan_cutdown(100, TIMING, 6.0)
        self.assertEqual(p["total_frames"], 60)
        self.assertEqual(p["timing"]["end_card_frames"], 0)

    def test_allocation(self):
        self.assertEqual(rf.allocate_frames([30, 20, 30], 100), [30, 20, 30])
        a = rf.allocate_frames([30, 20, 30], 41)
        self.assertEqual(sum(a), 41)
        self.assertTrue(all(1 <= x <= l for x, l in zip(a, [30, 20, 30])))

    def test_choose_shots(self):
        self.assertEqual(rf.choose_shots(5, "auto", 3, {2: 1, 3: 9, 4: 5}), [1, 3, 5])
        self.assertEqual(rf.choose_shots(5, "auto", 2), [1, 5])
        self.assertEqual(rf.choose_shots(1, "auto", 3), [1])
        self.assertEqual(rf.choose_shots(4, "auto", 1), [4])
        self.assertEqual(rf.choose_shots(5, "4,1"), [1, 4])
        with self.assertRaises(rf.StageError):
            rf.choose_shots(3, "1, 9")

    def test_errors(self):
        with self.assertRaises(rf.StageError):
            rf.plan_cutdown(TOTAL, TIMING, 1.5)             # end card alone is 2 s
        with self.assertRaises(rf.StageError):
            rf.plan_cutdown(90, TIMING, 6.0)                # fewer frames than the timing
        with self.assertRaises(rf.StageError):
            rf.plan_cutdown(TOTAL, None, 6.0)
        with self.assertRaises(rf.StageError):
            rf.plan_cutdown(TOTAL, TIMING, 6.0, "auto", "{not json")


if __name__ == "__main__":
    unittest.main()
