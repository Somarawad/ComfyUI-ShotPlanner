"""Smoke tests of the video / voice / beat / reframe node wrappers (needs torch, no ComfyUI).

The pack is imported as a package so the relative imports and the `if __package__:`
registration in __init__.py run exactly as in ComfyUI.
"""

import importlib.util
import json
import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PKG = "shotplanner_stage_pkg"
NODE_MODULES = {
    "video_nodes": ("VIDEO_NODE_CLASS_MAPPINGS", "Shot Planner/Video"),
    "video_qa_nodes": ("VIDEO_QA_NODE_CLASS_MAPPINGS", "Shot Planner/Video"),
    "voice_nodes": ("VOICE_NODE_CLASS_MAPPINGS", "Shot Planner/Audio"),
    "beat_nodes": ("BEAT_NODE_CLASS_MAPPINGS", "Shot Planner/Audio"),
    "reframe_nodes": ("REFRAME_NODE_CLASS_MAPPINGS", "Shot Planner/Ad"),
}

PLAN = {"title": "T", "style": "s", "shots": [
    {"id": 1, "description": "A cup steams", "camera": "push-in", "duration": 2},
    {"id": 2, "description": "Hands lift it", "camera": "static", "duration": 3}]}


def load_pkg():
    if PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(PKG, os.path.join(ROOT, "__init__.py"),
                                                      submodule_search_locations=[ROOT])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[PKG] = pkg
        spec.loader.exec_module(pkg)
    return sys.modules[PKG]


def mod(name):
    load_pkg()
    return sys.modules[f"{PKG}.{name}"]


def pan_clip(n=16, h=36, w=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    big = torch.nn.functional.avg_pool2d(torch.rand(1, 3, h, w + n + 1, generator=g), 5, 1, 2)[0]
    big = big.permute(1, 2, 0)
    big = (big - big.min()) / (big.max() - big.min())
    return torch.stack([big[:, i:i + w] for i in range(n)])


class RegistrationTests(unittest.TestCase):
    def test_all_registered_with_tooltips(self):
        pkg = load_pkg()
        self.assertIn("ShotPlanner", pkg.NODE_CLASS_MAPPINGS)        # original nodes
        self.assertIn("AdTimeline", pkg.NODE_CLASS_MAPPINGS)         # ad assembler nodes
        count = 0
        for name, (attr, category) in NODE_MODULES.items():
            mapping = getattr(mod(name), attr)
            for key, cls in mapping.items():
                count += 1
                self.assertIs(pkg.NODE_CLASS_MAPPINGS[key], cls)
                self.assertIn(key, pkg.NODE_DISPLAY_NAME_MAPPINGS)
                self.assertEqual(cls.CATEGORY, category, key)
                self.assertEqual(len(cls.OUTPUT_TOOLTIPS), len(cls.RETURN_TYPES), key)
                if hasattr(cls, "OUTPUT_IS_LIST"):
                    self.assertEqual(len(cls.OUTPUT_IS_LIST), len(cls.RETURN_TYPES), key)
                for group in cls.INPUT_TYPES().values():
                    for inp, (typ, *opts) in group.items():
                        self.assertTrue(opts and opts[0].get("tooltip"), f"{key}.{inp} has no tooltip")
                self.assertTrue(callable(getattr(cls(), cls.FUNCTION)))
        self.assertEqual(count, 11)


class VideoNodeTests(unittest.TestCase):
    def test_motion_prompts(self):
        n = mod("video_nodes").ShotMotionPrompts()
        prompts, frames, secs, neg, fps = n.build(json.dumps(PLAN), "wan2.1", "natural", 0)
        self.assertEqual((frames, fps), ([33, 49], 16))
        self.assertEqual(len(prompts), 2)
        with self.assertRaises(RuntimeError):
            n.build("", "wan2.1", "natural", 0)

    def test_qa_merge_chain(self):
        m = mod("video_qa_nodes")
        good, frozen = pan_clip(), pan_clip()[:1].repeat(16, 1, 1, 1)
        report, sheet, vj, shots, prompts, seeds = m.ShotVideoQA().check(
            [good, frozen], [good[:1], frozen[:1]], [""], [7.0], [False], ["gemini"], [""],
            [4], [5], motion_prompts=["p1", "p2"])
        self.assertEqual(shots, "2")
        self.assertEqual(prompts, ["p2"])
        self.assertEqual(len(seeds), 1)
        self.assertEqual(sheet.ndim, 4)
        self.assertIsInstance(sheet, torch.Tensor)
        merge = m.ClipRerollMerge()
        self.assertEqual(merge.check_lazy_status(clips=[good, frozen], verdicts_json=[vj],
                                                 rerolled=None), ["rerolled"])
        self.assertEqual(merge.check_lazy_status(clips=[good, frozen], verdicts_json=[vj],
                                                 rerolled=(None,)), ["rerolled"])
        fixed = pan_clip(seed=3)
        out, rep = merge.merge([good, frozen], [vj], [fixed])
        self.assertIs(out[1], fixed)
        self.assertIs(out[0], good)
        # nothing failed -> rerolled is never requested
        _, _, vj2, *_ = m.ShotVideoQA().check([good], [good[:1]], [""], [7.0], [False],
                                              ["gemini"], [""], [4], [0])
        self.assertEqual(merge.check_lazy_status(clips=[good], verdicts_json=[vj2], rerolled=None), [])
        self.assertEqual(merge.merge([good], [vj2], None)[0], [good])
        with self.assertRaises(RuntimeError):
            merge.merge([good, frozen], [vj], (None,))

    def test_qa_with_mocked_gemini(self):
        m = mod("video_qa_nodes")
        reply = {"candidates": [{"content": {"parts": [{"text": json.dumps(
            {"score": 9, "issues": [], "fixed_motion_prompt": ""})}]}}]}
        with mock.patch.object(sys.modules[f"{PKG}.stage_utils"], "http_post_json",
                               return_value=reply) as post:
            report, *_ = m.ShotVideoQA().check([pan_clip()], [pan_clip()[:1]], [""], [7.0], [True],
                                               ["gemini"], [""], [3], [0], api_key=["k"])
        self.assertIn("LLM 9", report)
        self.assertEqual(post.call_count, 1)

    def test_clip_utilities(self):
        m = mod("video_qa_nodes")
        batch = pan_clip(20)
        (clips,) = m.ClipsFromBatch().split([batch], ["5, 15"], [1])
        self.assertEqual([c.shape[0] for c in clips], [5, 15])
        (clips,) = m.ClipsFromBatch().split([batch], [""], [1], frame_counts=[10, 10])
        self.assertEqual([c.shape[0] for c in clips], [10, 10])
        (clips,) = m.ClipsFromBatch().split([batch], [""], [3])
        self.assertEqual([c.shape[0] for c in clips], [7, 7, 6])
        with self.assertRaises(RuntimeError):
            m.ClipsFromBatch().split([batch], ["5, 5"], [1])
        (last,) = m.ClipLastFrame().last(clips)
        self.assertTrue(torch.equal(last[0][0], clips[0][-1]))
        self.assertEqual(last[0].shape[0], 1)
        out, fps = m.ClipResampleFPS().resample([pan_clip(81), pan_clip(16)], [16.0], [24.0], ["blend"])
        self.assertEqual(([c.shape[0] for c in out], fps), ([122, 24], 24.0))
        out, _ = m.ClipResampleFPS().resample([pan_clip(16)], [16.0], [24.0], ["nearest"])
        self.assertEqual(out[0].shape[0], 24)


class AudioNodeTests(unittest.TestCase):
    def test_voiceover_and_fit(self):
        v = mod("voice_nodes")
        voice = sys.modules[f"{PKG}.voice"]
        timing = {"fps": 24, "total_seconds": 5.0, "shots": [
            {"shot": 1, "start_s": 0.0, "end_s": 2.0}, {"shot": 2, "start_s": 2.0, "end_s": 5.0}]}
        sr = 24000
        tone = (0.4 * np.sin(2 * np.pi * 220 * np.arange(int(2.2 * sr)) / sr)).astype(np.float32)[None]
        wav = voice.encode_wav16(tone, sr)
        with mock.patch.object(sys.modules[f"{PKG}.stage_utils"], "http_post",
                               return_value=(wav, "audio/wav")):
            audio, vt = v.AdVoiceover().speak("Line one.\nLine two.", "openai", "", "", "", 1.0,
                                              json.dumps(timing), api_key="k")
        self.assertEqual(audio["sample_rate"], 48000)
        self.assertEqual(tuple(audio["waveform"].shape), (1, 2, 5 * 48000))
        self.assertEqual([l["start_s"] for l in json.loads(vt)["lines"]], [0.3, 2.3])
        fitted, report, _ = v.AdVoiceFit().fit(audio, vt, json.dumps(timing), 0.12)
        lines = report.split("\n")
        self.assertIn("overlap", lines[0])                 # line 1 (0.3-2.5 s) runs into line 2
        self.assertIn("OVERRUN", lines[1])                 # too long for its 1.65 s slot
        self.assertIn("fits", lines[2])
        self.assertEqual(fitted["waveform"].shape, audio["waveform"].shape)

    def test_beat_sync(self):
        b = mod("beat_nodes")
        sr = 22050
        x = np.zeros(sr * 12, np.float32)
        for k in range(24):                       # 120 bpm from 0.4 s
            a = int((0.4 + 0.5 * k) * sr)
            x[a:a + 200] = (1.0 if k % 4 == 0 else 0.5) * np.hanning(200)
        music = {"waveform": torch.from_numpy(x)[None, None].repeat(1, 2, 1), "sample_rate": sr}
        durs, bpm, beats_json, offset, aligned, total = b.AdBeatSync().sync(music, 4, 8.0, 1.5, "auto")
        self.assertAlmostEqual(bpm, 120, delta=1)
        self.assertAlmostEqual(offset, 0.4, delta=0.03)
        self.assertEqual(len(durs.split(",")), 4)
        self.assertEqual(aligned["waveform"].shape[-1], x.size - int(round(offset * sr)))
        self.assertIn("cut_times", json.loads(beats_json))
        self.assertAlmostEqual(total, 8.0, delta=0.5)


class SocialNodeTests(unittest.TestCase):
    def test_reframe_and_cutdown(self):
        r = mod("reframe_nodes")
        frames = torch.rand(30, 36, 64, 3)
        mask = torch.zeros(30, 36, 64)
        mask[:, 10:20, 40:50] = 1
        (out,) = r.AdReframe().reframe(frames, "9:16", 36, 0.5, mask)
        self.assertEqual(tuple(out.shape), (30, 64, 36, 3))
        (out,) = r.AdReframe().reframe(frames, "1:1", 32, 0.5)
        self.assertEqual(tuple(out.shape), (30, 32, 32, 3))
        timing = {"fps": 5.0, "shots": [{"shot": 1, "start_frame": 0, "end_frame": 10},
                                         {"shot": 2, "start_frame": 10, "end_frame": 25}]}
        cut, tj, count, secs, rep = r.AdCutdown().cut(frames, json.dumps(timing), 4.0, "auto")
        self.assertEqual((count, secs), (20, 4.0))
        self.assertTrue(torch.equal(cut[-5:], frames[-5:]))          # end card kept
        self.assertEqual(json.loads(tj)["end_card_frames"], 5)
        with self.assertRaises(RuntimeError):
            r.AdCutdown().cut(frames, json.dumps(timing), 0.5, "auto")


if __name__ == "__main__":
    unittest.main()
