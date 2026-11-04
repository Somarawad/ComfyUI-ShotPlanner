"""Unit tests for ad_assembly.py (and a smoke test of the ComfyUI node wrappers).

Run from this folder:  python -m unittest      (or from the repo root: python -m unittest discover tests)
No GPU and no ComfyUI needed; only numpy, torch and Pillow.
"""

import importlib.util
import json
import math
import os
import sys
import unittest

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ad_assembly as aa  # noqa: E402

SR = 8000


def solid(color, h=36, w=64, b=1):
    return torch.tensor(color, dtype=torch.float32).view(1, 1, 1, 3).repeat(b, h, w, 1)


def ramp_clip(n, h=36, w=64):
    """Clip whose frame k has value k/100 everywhere (frame index is recoverable)."""
    return torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1).repeat(1, h, w, 3) / 100.0


def tone(seconds, sr=SR, freq=220.0, amp=0.5, channels=1):
    t = torch.arange(int(seconds * sr)) / sr
    w = amp * torch.sin(2 * math.pi * freq * t)
    return w.unsqueeze(0).repeat(channels, 1)


def audio(wave, sr=SR):
    return {"waveform": wave.unsqueeze(0), "sample_rate": sr}


PLAN = {"title": "Run Free", "style": "cinematic", "shots": [
    {"id": 1, "description": "A runner laces her shoes at dawn, city still asleep", "camera": "close-up",
     "duration": 4, "image_prompt": "", "video_prompt": ""},
    {"id": 2, "description": "She sprints across an empty bridge", "camera": "wide",
     "duration": 2, "image_prompt": "", "video_prompt": ""},
    {"id": 3, "description": "Product shot of the shoe", "camera": "macro",
     "duration": 3, "image_prompt": "", "video_prompt": ""},
]}


class DurationTests(unittest.TestCase):
    def test_aspect_size(self):
        self.assertEqual(aa.aspect_size(1920, "16:9"), (1920, 1080))
        self.assertEqual(aa.aspect_size(1080, "9:16"), (1080, 1920))
        self.assertEqual(aa.aspect_size(1080, "1:1"), (1080, 1080))
        self.assertEqual(aa.aspect_size(1080, "4:5"), (1080, 1350))
        w, h = aa.aspect_size(1001, "16:9")
        self.assertEqual((w % 2, h % 2), (0, 0))
        with self.assertRaises(aa.AdAssemblyError):
            aa.aspect_size(1920, "wide")

    def test_parse_durations(self):
        self.assertEqual(aa.parse_durations("3, 2.5 4;1s"), [3.0, 2.5, 4.0, 1.0])
        self.assertEqual(aa.parse_durations(""), [])
        with self.assertRaises(aa.AdAssemblyError):
            aa.parse_durations("3, two")
        with self.assertRaises(aa.AdAssemblyError):
            aa.parse_durations("0")

    def test_durations_from_shot_list(self):
        self.assertEqual(aa.durations_from_shot_list(json.dumps(PLAN)), [4.0, 2.0, 3.0])
        self.assertEqual(aa.durations_from_shot_list({"shots": [{"duration": "x"}, {}]}), [3.0, 3.0])
        self.assertEqual(aa.durations_from_shot_list(""), [])
        with self.assertRaises(aa.AdAssemblyError):
            aa.durations_from_shot_list("not json")

    def test_resolve_durations_priority(self):
        self.assertEqual(aa.resolve_durations(4, json.dumps(PLAN), "1.5"), [1.5, 2.0, 3.0, 3.0])
        self.assertEqual(aa.resolve_durations(2), [3.0, 3.0])

    def test_seconds_to_frames(self):
        self.assertEqual(aa.seconds_to_frames(3, 24), 72)
        self.assertEqual(aa.seconds_to_frames(2.5, 24), 60)
        self.assertEqual(aa.seconds_to_frames(0.001, 24), 1)

    def test_plan_timeline_total_is_sum_for_every_transition(self):
        counts = [72, 48, 96]
        for tr in aa.TRANSITIONS:
            lay = aa.plan_timeline(counts, tr, 6)
            self.assertEqual(lay["total_frames"], 216, tr)
            self.assertEqual([s["end"] - s["start"] for s in lay["shots"]], counts)

    def test_plan_timeline_crossfade_overlap(self):
        lay = aa.plan_timeline([10, 10], "crossfade", 6)
        a, b = lay["shots"]
        self.assertEqual(lay["transition_frames"], 6)
        overlap = range(max(a["render_start"], b["render_start"]), min(a["render_end"], b["render_end"]))
        self.assertEqual(len(overlap), 6)                       # crossfade length
        self.assertEqual((a["render_end"], b["render_start"]), (13, 7))  # centred on the cut at 10
        self.assertEqual(aa.plan_timeline([10, 4], "crossfade", 6)["transition_frames"], 4)  # clamp
        self.assertEqual(aa.plan_timeline([10, 10], "cut", 6)["transition_frames"], 0)
        with self.assertRaises(aa.AdAssemblyError):
            aa.plan_timeline([10], "spin")

    def test_timing_dict_and_load_timing(self):
        lay = aa.plan_timeline([48, 24], "cut", 0)
        t = aa.timing_dict(lay, 24, 64, 36, ["still", "clip"])
        self.assertEqual(t["total_seconds"], 3.0)
        self.assertEqual((t["shots"][1]["start_s"], t["shots"][1]["end_s"]), (2.0, 3.0))
        self.assertEqual(t["shots"][0]["source"], "still")
        self.assertEqual(aa.load_timing(json.dumps(t))["fps"], 24.0)
        self.assertIsNone(aa.load_timing(""))
        with self.assertRaises(aa.AdAssemblyError):
            aa.load_timing('{"x": 1}')


class ImageTests(unittest.TestCase):
    def test_center_crop_box(self):
        self.assertEqual(aa.center_crop_box(1920, 1080, 1080, 1920), (656.25, 0.0, 607.5, 1080.0))
        x0, y0, w, h = aa.center_crop_box(1000, 1000, 1920, 1080)
        self.assertAlmostEqual(w / h, 16 / 9)
        self.assertEqual(x0, 0.0)

    def test_center_crop_resize_keeps_centre(self):
        # 16:9 image with red left third, green centre, blue right third -> 9:16 crop is all green
        img = torch.zeros(1, 90, 160, 3)
        img[:, :, :53, 0] = 1
        img[:, :, 53:107, 1] = 1
        img[:, :, 107:, 2] = 1
        out = aa.center_crop_resize(img, 18, 32)
        self.assertEqual(tuple(out.shape), (1, 32, 18, 3))
        self.assertGreater(out[..., 1].mean().item(), 0.99)
        self.assertLess(out[..., 0].max().item() + out[..., 2].max().item(), 0.01)

    def test_clip_frame_index_trim_and_hold(self):
        self.assertEqual([aa.clip_frame_index(i, 3) for i in range(5)], [0, 1, 2, 2, 2])

    def test_ken_burns_scale(self):
        self.assertEqual(aa.ken_burns_scale(0, 10, 0.1), 1.0)
        self.assertAlmostEqual(aa.ken_burns_scale(9, 10, 0.1), 1.1)
        self.assertEqual(aa.ken_burns_scale(0, 1, 0.1), 1.0)

    def test_zoom_frame(self):
        img = torch.zeros(40, 40, 3)
        img[10:30, 10:30] = 1.0                     # white square = half the width
        self.assertTrue(torch.allclose(aa.zoom_frame(img, 1.0, 40, 40), img, atol=1e-5))
        z = aa.zoom_frame(img, 2.0, 40, 40)         # 2x zoom: square fills the frame
        self.assertGreater(z.mean().item(), 0.95)

    def test_shot_source_still_and_clip(self):
        still = aa.ShotSource(solid([0.2, 0.4, 0.6], 90, 160), 32, 18, 10, 0.1)
        self.assertEqual(still.kind, "still")
        self.assertEqual(tuple(still.frame(0).shape), (18, 32, 3))
        clip = aa.ShotSource(ramp_clip(4), 32, 18, 10)
        self.assertEqual(clip.kind, "clip")
        vals = [round(clip.frame(i).mean().item() * 100) for i in range(6)]
        self.assertEqual(vals, [0, 1, 2, 3, 3, 3])          # plays, then holds the last frame

    def test_still_push_in_moves(self):
        img = torch.zeros(1, 36, 64, 3)
        img[:, 16:20, 30:34] = 1.0
        src = aa.ShotSource(img, 64, 36, 20, 0.5)
        self.assertGreater(src.frame(19).sum().item(), src.frame(0).sum().item() * 1.5)


class TransitionTests(unittest.TestCase):
    def test_crossfade(self):
        a, b = torch.zeros(2, 2, 3), torch.ones(2, 2, 3)
        self.assertTrue(torch.allclose(aa.crossfade(a, b, 0.25), torch.full((2, 2, 3), 0.25)))

    def test_transition_progress(self):
        ps = [aa.transition_progress(j, 3) for j in range(3)]
        self.assertEqual(ps, [0.25, 0.5, 0.75])

    def test_dip_gain(self):
        g = [aa.dip_gain(o, 4) for o in range(-3, 3)]
        self.assertEqual(g, [1.0, 0.5, 0.0, 1 / 3, 2 / 3, 1.0])
        self.assertEqual(aa.dip_gain(-1, 0), 1.0)

    def test_horizontal_box_blur(self):
        img = torch.zeros(4, 21, 3)
        img[:, 10] = 1.0
        out = aa.horizontal_box_blur(img, 2)
        self.assertEqual(tuple(out.shape), (4, 21, 3))
        self.assertAlmostEqual(out[0, 8:13, 0].sum().item(), 1.0, places=5)
        self.assertAlmostEqual(out[0, 10, 0].item(), 0.2, places=5)
        self.assertTrue(torch.equal(aa.horizontal_box_blur(img, 0), img))

    def test_whip_pan(self):
        a, b = solid([1, 0, 0])[0], solid([0, 0, 1])[0]
        mid = aa.whip_pan(a, b, 0.5)
        self.assertEqual(tuple(mid.shape), tuple(a.shape))
        self.assertGreater(mid[..., 0].mean().item(), 0.3)
        self.assertGreater(mid[..., 2].mean().item(), 0.3)
        self.assertTrue(torch.allclose(aa.whip_pan(a, b, 1.0), b))


class AssembleTests(unittest.TestCase):
    def test_cut_length_and_content(self):
        frames, t = aa.assemble_timeline([solid([1, 0, 0]), solid([0, 1, 0])], [1.0, 0.5], 10,
                                         32, 18, "cut", 6, 0.0)
        self.assertEqual(tuple(frames.shape), (15, 18, 32, 3))
        self.assertEqual(t["total_frames"], 15)
        self.assertTrue(torch.allclose(frames[9], solid([1, 0, 0], 18, 32)[0]))
        self.assertTrue(torch.allclose(frames[10], solid([0, 1, 0], 18, 32)[0]))
        self.assertEqual([s["source"] for s in t["shots"]], ["still", "still"])

    def test_crossfade_frames(self):
        frames, t = aa.assemble_timeline([solid([1, 0, 0]), solid([0, 1, 0])], [1, 1], 10,
                                         32, 18, "crossfade", 4, 0.0)
        self.assertEqual(frames.shape[0], 20)
        red = frames[:, 0, 0, 0]
        blended = [f for f in range(20) if 0.01 < red[f].item() < 0.99]
        self.assertEqual(blended, [8, 9, 10, 11])           # 4 frames centred on the cut at 10
        self.assertTrue(torch.allclose(frames[:, 0, 0, 0] + frames[:, 0, 0, 1], torch.ones(20)))

    def test_dip_to_black(self):
        frames, _ = aa.assemble_timeline([solid([1, 1, 1]), solid([1, 1, 1])], [1, 1], 10,
                                         32, 18, "dip-to-black", 4, 0.0)
        lum = [round(frames[f].mean().item(), 3) for f in range(7, 13)]
        self.assertEqual(lum, [1.0, 0.5, 0.0, 0.333, 0.667, 1.0])

    def test_clip_trim_and_hold(self):
        frames, t = aa.assemble_timeline([ramp_clip(30), ramp_clip(3)], [1, 1], 10, 64, 36, "cut")
        idx = [round(frames[f].mean().item() * 100) for f in range(20)]
        self.assertEqual(idx[:10], list(range(10)))         # long clip trimmed to 10 frames
        self.assertEqual(idx[10:], [0, 1, 2] + [2] * 7)     # short clip held on its last frame
        self.assertEqual(t["shots"][0]["source"], "clip")

    def test_whip_pan_timeline(self):
        frames, _ = aa.assemble_timeline([solid([1, 0, 0]), solid([0, 0, 1])], [1, 1], 10,
                                         32, 18, "whip-pan blur", 6, 0.0)
        self.assertEqual(frames.shape[0], 20)
        self.assertTrue(torch.allclose(frames[0], solid([1, 0, 0], 18, 32)[0]))
        self.assertTrue(torch.allclose(frames[-1], solid([0, 0, 1], 18, 32)[0]))

    def test_aspect_crop_in_timeline(self):
        frames, t = aa.assemble_timeline([solid([0.5, 0.5, 0.5], 90, 160)], [0.5], 24,
                                         *aa.aspect_size(18, "9:16"), "cut")
        self.assertEqual(tuple(frames.shape[1:]), (32, 18, 3))
        self.assertEqual((t["width"], t["height"]), (18, 32))

    def test_mismatched_durations(self):
        with self.assertRaises(aa.AdAssemblyError):
            aa.assemble_timeline([solid([1, 1, 1])], [1, 2], 24, 32, 18)


class TextTests(unittest.TestCase):
    def test_load_font(self):
        f = aa.load_font(40)
        self.assertGreater(aa._text_width(f, "Hello"), 20)
        self.assertIsNotNone(aa.load_font(20, "/no/such/font.ttf"))   # falls back

    def test_wrap_text(self):
        f = aa.load_font(20)
        wrapped = aa.wrap_text("one two three four five six seven eight", f, 120)
        self.assertGreater(len(wrapped.split("\n")), 1)
        for line in wrapped.split("\n"):
            self.assertLessEqual(aa._text_width(f, line), 120)
        self.assertEqual(aa.wrap_text("a\\nb", f, 1000), "a\nb")

    def test_parse_hex_color(self):
        self.assertEqual(aa.parse_hex_color("#FF8000"), (255, 128, 0))
        self.assertEqual(aa.parse_hex_color("fff"), (255, 255, 255))
        self.assertEqual(aa.parse_hex_color("", (1, 2, 3)), (1, 2, 3))
        with self.assertRaises(aa.AdAssemblyError):
            aa.parse_hex_color("#GG0000")

    def test_render_text_layer_positions_and_styles(self):
        rows = {}
        for pos in aa.SUPER_POSITIONS:
            layer = aa.render_text_layer(320, 180, "Hello", 24, pos)
            ys = np.nonzero(layer[..., 3])[0]
            rows[pos] = ys.mean()
        self.assertLess(rows["top"], rows["center"])
        self.assertLess(rows["center"], rows["lower third"])
        for style in aa.SUPER_STYLES:
            layer = aa.render_text_layer(320, 180, "Hello", 24, "center", style)
            self.assertEqual(layer.shape, (180, 320, 4))
            self.assertGreater(layer[..., 3].max(), 200)
        box = aa.render_text_layer(320, 180, "Hi", 24, "center", "black box")
        plain = aa.render_text_layer(320, 180, "Hi", 24, "center", "outline")
        self.assertGreater((box[..., 3] > 0).sum(), (plain[..., 3] > 0).sum())
        with self.assertRaises(aa.AdAssemblyError):
            aa.render_text_layer(320, 180, "x", 24, "left")

    def test_composite_layer(self):
        frame = torch.zeros(10, 10, 3)
        layer = np.zeros((10, 10, 4), np.uint8)
        layer[2:4, 2:4] = (255, 255, 255, 255)
        aa.composite_layer(frame, layer, 0.5)
        self.assertAlmostEqual(frame[2, 2, 0].item(), 0.5, places=4)
        self.assertEqual(frame[0, 0, 0].item(), 0.0)


class SupersTests(unittest.TestCase):
    timing = aa.timing_dict(aa.plan_timeline([48, 24, 72], "cut"), 24)

    def test_parse_supers_shots_and_ranges(self):
        sups = aa.parse_supers("# comment\n2 | Fast\n\n1.5-4s | Claim\\nline two\n3|End", self.timing)
        self.assertEqual(sups[0], {"start_s": 2.0, "end_s": 3.0, "text": "Fast"})
        self.assertEqual(sups[1], {"start_s": 1.5, "end_s": 4.0, "text": "Claim\nline two"})
        self.assertEqual((sups[2]["start_s"], sups[2]["end_s"]), (3.0, 6.0))

    def test_parse_supers_errors(self):
        for bad in ("no pipe", "9 | no such shot", "4-2 | backwards", "soon | x"):
            with self.assertRaises(aa.AdAssemblyError, msg=bad):
                aa.parse_supers(bad, self.timing)
        with self.assertRaises(aa.AdAssemblyError):
            aa.parse_supers("1 | needs timing", None)
        self.assertEqual(len(aa.parse_supers("0-1 | ok", None)), 1)

    def test_super_frame_range(self):
        self.assertEqual(aa.super_frame_range({"start_s": 1, "end_s": 2}, 24, 100), (24, 48))
        self.assertEqual(aa.super_frame_range({"start_s": 3, "end_s": 9}, 24, 100), (72, 100))

    def test_fade_alpha(self):
        a = [aa.fade_alpha(f, 10, 20, 2, 2) for f in range(9, 21)]
        self.assertEqual(a, [0.0, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.0])
        self.assertEqual(aa.fade_alpha(10, 10, 12, 0, 0), 1.0)

    def test_apply_supers_only_in_window(self):
        frames = torch.zeros(48, 90, 160, 3)
        out = aa.apply_supers(frames, [{"start_s": 0.5, "end_s": 1.0, "text": "SALE"}], 24,
                              font_size=24, fade_in=0, fade_out=0)
        changed = [f for f in range(48) if out[f].abs().sum() > 0]
        self.assertEqual(changed, list(range(12, 24)))
        self.assertEqual(frames.abs().sum().item(), 0.0)    # input not modified


class EndCardTests(unittest.TestCase):
    def test_blur_frame(self):
        img = torch.zeros(20, 20, 3)
        img[:, 10:] = 1.0
        out = aa.blur_frame(img, 3, darken=0.0)
        self.assertEqual(tuple(out.shape), (20, 20, 3))
        self.assertTrue(0.05 < out[10, 9, 0].item() < 0.95)
        self.assertLess(aa.blur_frame(torch.ones(4, 4, 3), 0, 0.5).max().item(), 0.51)

    def test_logo_rgba_mask(self):
        logo = torch.ones(4, 6, 3)
        mask = torch.zeros(1, 4, 6)
        mask[:, :, :3] = 1.0                    # left half transparent (Load Image convention)
        rgba = np.array(aa.logo_rgba(logo, mask))
        self.assertEqual(rgba.shape, (4, 6, 4))
        self.assertEqual((rgba[0, 0, 3], rgba[0, 5, 3]), (0, 255))
        inv = np.array(aa.logo_rgba(logo, mask, invert_mask=True))
        self.assertEqual((inv[0, 0, 3], inv[0, 5, 3]), (255, 0))
        self.assertTrue((np.array(aa.logo_rgba(logo))[..., 3] == 255).all())

    def test_render_end_card(self):
        logo = Image.new("RGBA", (40, 20), (255, 0, 0, 255))
        card = aa.render_end_card(160, 90, None, (0, 0, 0), logo, "Tagline", "novara.it", 0.3)
        self.assertEqual(tuple(card.shape), (90, 160, 3))
        self.assertGreater(card[..., 0].max().item(), 0.9)          # logo drawn
        self.assertAlmostEqual(card[0, 0].sum().item(), 0.0)         # solid black corner
        bg = torch.full((90, 160, 3), 0.5)
        card2 = aa.render_end_card(160, 90, bg, tagline="", cta="")
        self.assertTrue(torch.allclose(card2, bg, atol=1 / 255))

    def test_append_end_card_frame_count_and_fade(self):
        frames = torch.zeros(30, 18, 32, 3)
        card = torch.ones(18, 32, 3)
        out = aa.append_end_card(frames, card, 2.5, 24, 4)
        self.assertEqual(out.shape[0], 30 + 60)
        self.assertEqual([round(out[30 + j].mean().item(), 2) for j in range(5)],
                         [0.2, 0.4, 0.6, 0.8, 1.0])
        self.assertTrue(torch.equal(out[:30], frames))


class AudioTests(unittest.TestCase):
    def test_db_to_gain(self):
        self.assertAlmostEqual(aa.db_to_gain(0), 1.0)
        self.assertAlmostEqual(aa.db_to_gain(-20), 0.1)

    def test_to_stereo(self):
        self.assertEqual(tuple(aa.to_stereo(torch.zeros(1, 5)).shape), (2, 5))
        self.assertEqual(tuple(aa.to_stereo(torch.zeros(5)).shape), (2, 5))
        self.assertEqual(tuple(aa.to_stereo(torch.zeros(6, 5)).shape), (2, 5))

    def test_resample(self):
        w = tone(1.0, 44100, 440)
        r = aa.resample(w, 44100, 48000)
        self.assertEqual(r.shape[-1], 48000)
        self.assertEqual(aa.resample(w, 44100, 22050).shape[-1], 22050)
        self.assertIs(aa.resample(w, 44100, 44100), w)
        # frequency preserved: zero crossings per second stay ~2*440
        crossings = ((r[0, :-1] < 0) & (r[0, 1:] >= 0)).sum().item()
        self.assertAlmostEqual(crossings, 440, delta=3)

    def test_fit_length(self):
        w = torch.ones(2, 10)
        self.assertEqual(aa.fit_length(w, 5).sum().item(), 10)
        out = aa.fit_length(w, 20, 15)
        self.assertEqual((out[:, :15].sum().item(), out[:, 15:].sum().item()), (0, 10))
        self.assertEqual(aa.fit_length(w, 20, -4).sum().item(), 12)

    def test_prepare_audio(self):
        self.assertIsNone(aa.prepare_audio(None, SR))
        p = aa.prepare_audio(audio(tone(1.0, 16000), 16000), SR)
        self.assertEqual(tuple(p.shape), (2, SR))

    def test_rms_envelope_and_speech_activity(self):
        w = torch.cat([torch.zeros(1, SR), tone(1.0, amp=0.5)], dim=1)
        env, hop = aa.rms_envelope(w, SR)
        self.assertEqual(hop, SR // 100)
        self.assertEqual(env.numel(), 200)
        self.assertLess(env[50].item(), 1e-6)
        self.assertAlmostEqual(env[150].item(), 0.5 / math.sqrt(2), places=2)
        act = aa.speech_activity(env)
        self.assertFalse(bool(act[:95].any()))
        self.assertTrue(bool(act[105:].all()))
        self.assertEqual(aa.speech_activity(torch.zeros(0)).numel(), 0)

    def test_smooth_gain(self):
        target = torch.tensor([1.0] * 5 + [0.1] * 50 + [1.0] * 50)
        g = aa.smooth_gain(target, 0.01, 0.15, 0.15)
        self.assertEqual(g[0].item(), 1.0)
        self.assertTrue(0.1 < g[10].item() < 1.0)            # still moving after 50 ms
        self.assertAlmostEqual(g[54].item(), 0.1, delta=0.05)  # settled
        self.assertTrue(bool((g[5:55][1:] <= g[5:55][:-1] + 1e-7).all()))

    def test_ducking_gain(self):
        n = 4 * SR
        voice = torch.zeros(2, n)
        voice[:, SR:2 * SR] = tone(1.0, amp=0.5)
        g = aa.ducking_gain(voice, SR, n, -10)
        self.assertEqual(g.numel(), n)
        self.assertAlmostEqual(g[int(1.5 * SR)].item(), aa.db_to_gain(-10), delta=0.01)
        self.assertAlmostEqual(g[int(0.2 * SR)].item(), 1.0, delta=0.01)
        self.assertAlmostEqual(g[int(3.5 * SR)].item(), 1.0, delta=0.01)
        self.assertLess(g[SR].item(), 0.6)                   # look-ahead: already ducked at onset
        self.assertTrue(torch.equal(aa.ducking_gain(None, SR, 10), torch.ones(10)))

    def test_fade_out(self):
        w = torch.ones(2, 100)
        out = aa.fade_out(w, 100, 0.5)
        self.assertEqual(out[0, 49].item(), 1.0)
        self.assertEqual(out[0, -1].item(), 0.0)
        self.assertEqual(w.min().item(), 1.0)                # input untouched

    def test_soft_limit(self):
        w = torch.linspace(-3, 3, 1001).unsqueeze(0)
        out = aa.soft_limit(w, -1.0)
        ceiling = aa.db_to_gain(-1.0)
        self.assertLessEqual(out.abs().max().item(), ceiling + 1e-6)
        quiet = torch.full((1, 4), 0.3)
        self.assertTrue(torch.equal(aa.soft_limit(quiet), quiet))
        self.assertTrue(bool((out[0, 1:] >= out[0, :-1]).all()))   # monotonic, no fold-over

    def test_mix_length_equals_total(self):
        for secs, sr in ((15.0, 48000), (2.37, 8000)):
            mix = aa.mix_ad_audio(audio(tone(1.0)), audio(tone(5.0, channels=2)), None,
                                  secs, 0.2, sample_rate=sr)
            self.assertEqual(tuple(mix.shape), (2, int(round(secs * sr))))
        silent = aa.mix_ad_audio(total_seconds=1.0, sample_rate=SR)
        self.assertEqual(tuple(silent.shape), (2, SR))
        self.assertEqual(silent.abs().max().item(), 0.0)

    def test_mix_ducks_music_under_voice(self):
        music = audio(tone(6.0, freq=110, amp=0.8))
        vo_wave = tone(2.0, freq=330, amp=0.5)
        no_vo = aa.mix_ad_audio(None, music, None, 6.0, 2.0, -14, -10, 0.0, SR)
        ducked = aa.mix_ad_audio(audio(vo_wave), music, None, 6.0, 2.0, -14, -10, 0.0, SR)
        # isolate the music in the ducked mix by removing the (known) VO track
        vo_track = aa.fit_length(aa.to_stereo(vo_wave), 6 * SR, 2 * SR)
        music_part = ducked - vo_track
        rms = lambda x: x.pow(2).mean().sqrt().item()  # noqa: E731
        during = slice(int(2.5 * SR), int(3.5 * SR))
        before = slice(int(0.5 * SR), int(1.5 * SR))
        ratio_db = 20 * math.log10(rms(music_part[:, during]) / rms(no_vo[:, during]))
        self.assertAlmostEqual(ratio_db, -10, delta=0.5)
        self.assertAlmostEqual(rms(music_part[:, before]), rms(no_vo[:, before]), delta=1e-4)

    def test_mix_peaks_limited_and_fade(self):
        loud = audio(tone(2.0, amp=1.0))
        mix = aa.mix_ad_audio(loud, loud, loud, 2.0, 0.0, 0.0, 0.0, 0.5, SR)
        self.assertLessEqual(mix.abs().max().item(), aa.db_to_gain(-1) + 1e-6)
        self.assertLess(mix[:, -40:].abs().max().item(), 0.05)


class ScriptTests(unittest.TestCase):
    def test_word_budget(self):
        self.assertEqual(aa.word_budget(4), 10)
        self.assertEqual(aa.word_budget(2, 2.5), 5)
        self.assertEqual(aa.word_budget(0.1), 1)

    def test_fit_words(self):
        self.assertEqual(aa.fit_words("she runs across the bridge", 4), "She runs across.")
        self.assertEqual(aa.fit_words("Hello world!", 5), "Hello world!")
        self.assertEqual(aa.fit_words("", 3), "")

    def test_first_clause(self):
        self.assertEqual(aa._first_clause("A runner laces up, city asleep"), "A runner laces up")
        self.assertEqual(aa._first_clause("Wide shot. Then more"), "Wide shot.")

    def test_script_fits_timing(self):
        vo, supers = aa.script_from_shots(json.dumps(PLAN), "Novara", "Run free")
        lines = vo.split("\n")
        self.assertEqual(len(lines), 3)
        for line, shot in zip(lines, PLAN["shots"]):
            self.assertLessEqual(len(line.split()), aa.word_budget(shot["duration"]))
        self.assertEqual(lines[-1], "Novara. Run free.")
        parsed = aa.parse_supers(supers, aa.timing_dict(aa.plan_timeline([96, 48, 72]), 24))
        self.assertEqual([s["text"] for s in parsed], ["Run Free", "Run free"])
        _, every = aa.script_from_shots(PLAN, "Novara", "Run free", supers_mode="every shot")
        self.assertEqual(len(every.split("\n")), 3)
        with self.assertRaises(aa.AdAssemblyError):
            aa.script_from_shots("{}")


def _load_nodes():
    """Import the pack as a package (ad_nodes uses relative imports)."""
    spec = importlib.util.spec_from_file_location("shotplanner_pkg", os.path.join(ROOT, "__init__.py"),
                                                  submodule_search_locations=[ROOT])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["shotplanner_pkg"] = pkg
    spec.loader.exec_module(pkg)
    return pkg, sys.modules["shotplanner_pkg.ad_nodes"]


class NodeSmokeTests(unittest.TestCase):
    def test_registration_and_tooltips(self):
        pkg, nodes = _load_nodes()
        for key in nodes.AD_NODE_CLASS_MAPPINGS:
            self.assertIn(key, pkg.NODE_CLASS_MAPPINGS)
            self.assertIn(key, pkg.NODE_DISPLAY_NAME_MAPPINGS)
            cls = nodes.AD_NODE_CLASS_MAPPINGS[key]
            self.assertEqual(cls.CATEGORY, "Shot Planner/Ad")
            spec = cls.INPUT_TYPES()
            for group in spec.values():
                for name, (typ, *opts) in group.items():
                    self.assertTrue(opts and opts[0].get("tooltip"), f"{key}.{name} has no tooltip")
        self.assertIn("ShotPlanner", pkg.NODE_CLASS_MAPPINGS)   # original nodes still there

    def test_full_chain(self):
        _, n = _load_nodes()
        tl = n.AdTimeline().build([solid([1, 0, 0]), ramp_clip(5)], [12], [64], ["16:9"],
                                  ["crossfade"], [4], [0.05], shot_list_json=[json.dumps(PLAN)],
                                  durations=["1, 0.5"])
        frames, count, fps, timing, total = tl
        self.assertEqual((count, fps, total), (18, 12.0, 1.5))
        self.assertEqual(tuple(frames.shape), (18, 36, 64, 3))
        (sup,) = n.AdSupers().draw(frames, "1 | Hello\n0.5-1.5 | World", 12, "lower third",
                                   "black box", 2, 2, timing_json=timing)
        self.assertEqual(sup.shape, frames.shape)
        logo, mask = torch.ones(1, 10, 20, 3), torch.zeros(1, 10, 20)
        ec, ec_count, ec_secs = n.AdEndCard().append(sup, "Tag", "novara.it", "blurred last frame",
                                                     "#111111", 1.0, fps, 0.3, 3, logo, mask)
        self.assertEqual((ec_count, ec_secs), (30, 2.5))
        (mix,) = n.AdAudioMix().mix(ec_secs, 0.2, -14, -10, 0.5, SR,
                                    voiceover=audio(tone(1.0)), music=audio(tone(3.0)))
        self.assertEqual(tuple(mix["waveform"].shape), (1, 2, int(2.5 * SR)))
        vo, supers = n.AdScriptFromShots().write(json.dumps(PLAN), "Novara", "Run free", 2.5,
                                                 "first and last")
        self.assertTrue(vo and supers)

    def test_split_single_batch(self):
        _, n = _load_nodes()
        out = n.AdTimeline().build([solid([0.5, 0.5, 0.5], b=3)], [10], [32], ["1:1"], ["cut"],
                                   [0], [0.0], split_single_batch=[True])
        self.assertEqual(json.loads(out[3])["shots"][2]["source"], "still")
        self.assertEqual(out[1], 90)


if __name__ == "__main__":
    unittest.main()
