"""Tests for video_qa.py (Shot Video QA, Clip Reroll Merge, clip utilities).

Synthetic clips only; Gemini is mocked. No GPU, no network.
"""

import json
import os
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stage_utils as su  # noqa: E402
import video_qa as vq  # noqa: E402

H, W, N = 72, 96, 24


def texture(h=H, w=W, seed=0, blur=8):
    rng = np.random.default_rng(seed)
    img = Image.fromarray((rng.random((h, w, 3)) * 255).astype(np.uint8))
    a = np.asarray(img.filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255.0
    return (a - a.min()) / (a.max() - a.min() + 1e-6)


def pan(n=N, step=1, seed=0):
    big = texture(H, W + n * step + 1, seed)
    return np.stack([big[:, i * step:i * step + W] for i in range(n)])


def frozen(n=N):
    return np.repeat(texture()[None], n, axis=0)


def flicker(n=N, amp=0.15):
    g = np.array([1 + amp if i % 2 else 1 - amp for i in range(n)], np.float32)
    return np.clip(pan(n) * g[:, None, None, None], 0, 1)


def hard_cut(n=N):
    return np.concatenate([pan(n, seed=1)[: n // 2], pan(n, seed=2)[n // 2:]])


def melting(n=N):
    c = pan(n)
    c[:, 8:24, 8:24] = np.random.default_rng(5).random((n, 16, 16, 3))
    return c


def gemini_reply(obj_or_text):
    text = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text)
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


GOOD = {"score": 8.5, "issues": [], "fixed_motion_prompt": ""}
BAD = {"score": 4, "issues": [{"type": "hands", "frame": 12, "note": "six fingers"},
                              {"type": "weird", "frame": "x", "note": "?"}],
       "fixed_motion_prompt": "Hands stay still on the cup. Camera: slow dolly push-in."}


def setUpModule():
    su.sleep = lambda s: None       # no waiting on retries


class MetricTests(unittest.TestCase):
    def test_smooth_pan_passes(self):
        m, heat = vq.clip_metrics(pan(), pan()[:1])
        self.assertFalse(any(m["flags"].values()), m)
        self.assertEqual(vq.offline_score(m)[0], 10.0)
        self.assertEqual(heat.shape, (8, 8))

    def test_frozen(self):
        m, _ = vq.clip_metrics(frozen())
        self.assertTrue(m["flags"]["frozen"])
        self.assertTrue(vq.clip_metrics(frozen(1))[0]["flags"]["frozen"])   # single frame
        score, reasons = vq.offline_score(m)
        self.assertLess(score, 7)
        self.assertIn("frozen", reasons[0])

    def test_flicker(self):
        m, _ = vq.clip_metrics(flicker())
        self.assertTrue(m["flags"]["flicker"])
        self.assertFalse(m["flags"]["jump"])
        self.assertLess(vq.clip_metrics(pan())[0]["flicker"], vq.THRESHOLDS["flicker"])

    def test_hard_cut_is_a_jump(self):
        m, _ = vq.clip_metrics(hard_cut())
        self.assertTrue(m["flags"]["jump"])
        self.assertEqual(m["jump_frames"], [N // 2])
        self.assertFalse(m["flags"]["flicker"])        # one cut is not flicker
        self.assertFalse(m["flags"]["frozen"])

    def test_drift_from_keyframe(self):
        c = pan()
        self.assertFalse(vq.clip_metrics(c, c[:1])[0]["flags"]["drift"])
        other = texture(seed=9)[None]
        m, _ = vq.clip_metrics(c, other)
        self.assertTrue(m["flags"]["drift"])
        self.assertGreater(m["drift"], vq.clip_metrics(c, c[:1])[0]["drift"])
        same = vq.drift_score(c[0], c[0])
        self.assertAlmostEqual(same["drift"], 0.0, places=3)

    def test_melting_region_heatmap(self):
        m, heat = vq.clip_metrics(melting())
        self.assertTrue(m["flags"]["melt"])
        i, j = np.unravel_index(np.argmax(heat), heat.shape)
        # the noisy square covers rows/cols 8-24 of 72x96 -> top-left cells
        self.assertLessEqual(i, 2)
        self.assertLessEqual(j, 2)
        ov = vq.heat_overlay(melting()[0], heat)
        self.assertEqual(ov.shape, (H, W, 3))
        self.assertGreater(ov[12, 12, 0] - ov[12, 12, 1], ov[60, 80, 0] - ov[60, 80, 1])

    def test_sample_indices(self):
        self.assertEqual(vq.sample_indices(24, 6), [0, 5, 9, 14, 18, 23])
        self.assertEqual(vq.sample_indices(3, 6), [0, 1, 2])
        self.assertEqual(vq.sample_indices(10, 1), [5])

    def test_bad_shape(self):
        with self.assertRaises(vq.StageError):
            vq.clip_metrics(np.zeros((4, 4)))


class VerdictTests(unittest.TestCase):
    def _shot(self, clip, llm=None):
        m, _ = vq.clip_metrics(clip)
        s, r = vq.offline_score(m)
        return {"metrics": m, "offline_score": s, "offline_reasons": r, "llm": llm}

    def test_combine(self):
        per = [self._shot(pan()), self._shot(frozen()), self._shot(pan(), vq.parse_review(json.dumps(BAD))),
               self._shot(pan(), vq.parse_review(json.dumps(GOOD)))]
        v = vq.combine_verdicts(per, 7, seed=42, motion_prompts=["p1", "p2", "p3", "p4"])
        self.assertEqual([s["route"] for s in v["shots"]], ["pass", "reroll", "reroll", "pass"])
        self.assertEqual(v["reroll_shots"], [2, 3])
        self.assertEqual(v["shots"][2]["score"], 4.0)            # min(offline 10, llm 4)
        self.assertEqual(v["shots"][3]["score"], 8.5)
        self.assertIsNone(v["shots"][0]["reroll_seed"])
        seeds = [v["shots"][1]["reroll_seed"], v["shots"][2]["reroll_seed"]]
        self.assertNotEqual(seeds[0], seeds[1])
        self.assertNotIn(42, seeds)
        shots_s, prompts, seeds2 = vq.reroll_lists(v)
        self.assertEqual(shots_s, "2, 3")
        self.assertEqual(prompts, ["p2", BAD["fixed_motion_prompt"]])
        self.assertEqual(seeds2, seeds)
        self.assertTrue(any("hands @frame 12: six fingers" in r for r in v["shots"][2]["reasons"]))
        # deterministic
        self.assertEqual(vq.combine_verdicts(per, 7, 42)["shots"][1]["reroll_seed"], seeds[0])
        self.assertIn("REROLL", vq.report_text(v))

    def test_frozen_always_rerolls(self):
        v = vq.combine_verdicts([self._shot(frozen())], pass_score=0)
        self.assertEqual(v["shots"][0]["route"], "reroll")

    def test_parse_review(self):
        r = vq.parse_review("Sure! ```json\n" + json.dumps(BAD) + "\n```")
        self.assertEqual(r["score"], 4.0)
        self.assertEqual([i["type"] for i in r["issues"]], ["hands", "other"])
        self.assertIsNone(r["issues"][1]["frame"])
        self.assertEqual(vq.parse_review('{"score": 15}')["score"], 10.0)
        with self.assertRaises(vq.StageError):
            vq.parse_review("no json here")
        with self.assertRaises(vq.StageError):
            vq.parse_review('{"issues": []}')


class GeminiTests(unittest.TestCase):
    FR = [("Frame 0:", "abc")]

    def test_good_json(self):
        post = mock.Mock(return_value=gemini_reply(GOOD))
        r, model, _ = vq.gemini_review(self.FR, "p", "KEY", seed=3, post=post)
        self.assertEqual((r["score"], model), (8.5, "gemini-flash-lite-latest"))
        url, body, headers, _ = post.call_args[0]
        self.assertIn("gemini-flash-lite-latest:generateContent", url)
        self.assertEqual(headers, {"x-goog-api-key": "KEY"})
        self.assertNotIn("KEY", url)
        parts = body["contents"][0]["parts"]
        self.assertEqual(parts[2]["inline_data"], {"mime_type": "image/jpeg", "data": "abc"})
        self.assertEqual(body["generationConfig"]["seed"], 3)

    def test_bad_json_then_good(self):
        post = mock.Mock(side_effect=[gemini_reply("not json at all"), gemini_reply(GOOD)])
        r, model, log = vq.gemini_review(self.FR, "p", "K", post=post)
        self.assertEqual((r["score"], model, post.call_count), (8.5, "gemini-flash-lite-latest", 2))
        self.assertIn("bad reply", log[0])

    def test_503_falls_back_to_next_model(self):
        busy = su.HttpError("busy", 503)
        post = mock.Mock(side_effect=[busy, busy, gemini_reply(GOOD)])
        r, model, log = vq.gemini_review(self.FR, "p", "K", post=post)
        self.assertEqual(model, "gemini-flash-latest")
        self.assertEqual(post.call_count, 3)
        self.assertIn("gemini-flash-latest:generateContent", post.call_args[0][0])

    def test_all_fail_raises(self):
        post = mock.Mock(side_effect=su.HttpError("busy", 429))
        with self.assertRaises(vq.StageError) as cm:
            vq.gemini_review(self.FR, "p", "SECRET", post=post)
        self.assertEqual(post.call_count, 4)
        self.assertNotIn("SECRET", str(cm.exception))

    def test_auth_error_stops(self):
        post = mock.Mock(side_effect=su.HttpError("denied", 403))
        with self.assertRaises(vq.StageError):
            vq.gemini_review(self.FR, "p", "K", post=post)
        self.assertEqual(post.call_count, 1)

    def test_run_with_llm(self):
        post = mock.Mock(side_effect=[gemini_reply(GOOD), gemini_reply(BAD)])
        r = vq.run_video_qa([pan(), pan(seed=3)], [pan()[:1], pan(seed=3)[:1]], "check logo",
                            7, True, "", 4, 11, "logo upright", "K", ["m1", "m2"], post=post)
        self.assertEqual(r["reroll_shots"], "2")
        self.assertEqual(r["reroll_prompts"], [BAD["fixed_motion_prompt"]])
        self.assertEqual(len(r["reroll_seeds"]), 1)
        body = post.call_args_list[0][0][1]
        prompt = body["contents"][0]["parts"][0]["text"]
        self.assertIn("check logo", prompt)
        self.assertIn("logo upright", prompt)
        self.assertIn("m1", prompt)
        self.assertEqual(body["contents"][0]["parts"][1]["text"], "KEYFRAME (approved still):")
        self.assertEqual(r["verdicts"]["shots"][0]["model"], "gemini-flash-lite-latest")

    def test_missing_key(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}):
            with self.assertRaises(vq.StageError):
                vq.run_video_qa([pan()], use_llm=True)


class RunTests(unittest.TestCase):
    def test_offline_run_and_contact_sheet(self):
        clips = [pan(), frozen(), hard_cut()]
        kfs = [c[:1] for c in clips]
        r = vq.run_video_qa(clips, kfs, frames_per_clip=4, seed=1)
        self.assertEqual(r["reroll_shots"], "2, 3")
        self.assertEqual(len(r["reroll_prompts"]), 2)
        sheet = r["contact_sheet"]
        self.assertEqual(sheet.ndim, 4)
        self.assertEqual(sheet.shape[0], 1)
        self.assertEqual(sheet.shape[1], 3 * 128 + 2 * 4)
        # label cells: green for pass, red for reroll
        self.assertGreater(sheet[0, 60, 2, 1], sheet[0, 60, 2, 0])
        self.assertGreater(sheet[0, 128 + 4 + 60, 2, 0], sheet[0, 128 + 4 + 60, 2, 1])
        json.dumps(r["verdicts"])          # serialisable

    def test_keyframe_batch_and_mismatch(self):
        clips = [pan(), pan(seed=4)]
        batch = np.concatenate([clips[0][:1], clips[1][:1]])
        r = vq.run_video_qa(clips, [batch])
        self.assertEqual(r["reroll_shots"], "")
        with self.assertRaises(vq.StageError):
            vq.run_video_qa(clips, [clips[0][:1], clips[0][:1], clips[0][:1]])
        with self.assertRaises(vq.StageError):
            vq.run_video_qa([])


class MergeTests(unittest.TestCase):
    def verdicts(self, routes):
        return json.dumps({"shots": [{"shot": i + 1, "route": r} for i, r in enumerate(routes)]})

    def test_nothing_failed(self):
        v = self.verdicts(["pass", "pass"])
        self.assertFalse(vq.needs_rerolled(v, None))
        self.assertFalse(vq.needs_rerolled(v, (None,)))
        out, rep = vq.merge_clips(["a", "b"], v, None)
        self.assertEqual(out, ["a", "b"])
        self.assertIn("nothing replaced", rep)

    def test_with_rerolls(self):
        v = self.verdicts(["pass", "reroll", "pass", "reroll"])
        self.assertTrue(vq.needs_rerolled(v, None))
        self.assertTrue(vq.needs_rerolled(v, (None,)))
        self.assertTrue(vq.needs_rerolled(v, [None]))
        self.assertFalse(vq.needs_rerolled(v, ["x2", "x4"]))
        out, _ = vq.merge_clips(["a", "b", "c", "d"], v, ["B", "D"])
        self.assertEqual(out, ["a", "B", "c", "D"])

    def test_errors(self):
        v = self.verdicts(["reroll", "pass"])
        with self.assertRaises(vq.StageError):
            vq.merge_clips(["a", "b"], v, (None,))
        with self.assertRaises(vq.StageError):
            vq.merge_clips(["a", "b"], v, ["A", "B"])          # too many
        with self.assertRaises(vq.StageError):
            vq.merge_clips(["a"], v, ["A"])                    # clip count mismatch
        with self.assertRaises(vq.StageError):
            vq.merge_clips(["a"], "{}", None)


class UtilityTests(unittest.TestCase):
    def test_split_batch(self):
        self.assertEqual(vq.split_batch(10, [3, 3, 4]), [(0, 3), (3, 6), (6, 10)])
        self.assertEqual(vq.equal_counts(10, 3), [4, 3, 3])
        with self.assertRaises(vq.StageError):
            vq.split_batch(10, [3, 3])
        with self.assertRaises(vq.StageError):
            vq.equal_counts(2, 3)

    def test_resample_counts(self):
        self.assertEqual(vq.resample_count(81, 16, 24), 122)   # 121.5 rounds up
        self.assertEqual(vq.resample_count(80, 16, 24), 120)
        self.assertEqual(vq.resample_count(49, 16, 24), 74)
        self.assertEqual(vq.resample_count(120, 24, 16), 80)
        self.assertEqual(vq.resample_fps(pan(32), 16, 24).shape[0], 48)
        with self.assertRaises(vq.StageError):
            vq.resample_count(10, 0, 24)

    def test_resample_blend_values(self):
        clip = np.arange(5, dtype=np.float32).reshape(5, 1, 1, 1).repeat(3, axis=3) / 10
        out = vq.resample_fps(clip, 2, 3)                      # 5 -> 8 frames (7.5 rounds up)
        self.assertEqual(out.shape[0], 8)
        self.assertAlmostEqual(float(out[1, 0, 0, 0]), 2 / 3 / 10, places=5)   # between 0 and 1
        self.assertAlmostEqual(float(out[-1, 0, 0, 0]), 0.4, places=5)         # clamped to last
        near = vq.resample_fps(clip, 2, 3, "nearest")
        self.assertTrue(set(np.round(near[:, 0, 0, 0] * 10).tolist()) <= {0, 1, 2, 3, 4})

    def test_jpeg(self):
        import base64
        b = base64.b64decode(vq.frame_to_jpeg_b64(texture(600, 900), max_side=256))
        self.assertEqual(b[:2], b"\xff\xd8")
        import io
        self.assertEqual(max(Image.open(io.BytesIO(b)).size), 256)


if __name__ == "__main__":
    unittest.main()
