"""Tests for beat_sync.py (Ad Beat Sync) on synthetic click tracks. numpy only."""

import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import beat_sync as bs  # noqa: E402

SR = 44100


def click_track(bpm, seconds=20.0, sr=SR, first=0.5, accent=True, lead_beats=0, seed=0):
    """Noise-burst clicks every beat from `first`; beat 0, 4, 8... (downbeats) are louder.
    lead_beats > 0 adds unaccented pickup beats before `first`."""
    x = np.zeros(int(seconds * sr), np.float32)
    rng = np.random.default_rng(seed)
    p = 60.0 / bpm
    n = int(0.01 * sr)
    env = np.exp(-np.arange(n) / (0.002 * sr)).astype(np.float32)
    k = -lead_beats
    while True:
        t = first + k * p
        if t > seconds - 0.05:
            break
        if t >= 0:
            amp = 1.0 if (k % 4 == 0 or not accent) else 0.45
            a = int(round(t * sr))
            x[a:a + n] += amp * rng.standard_normal(n).astype(np.float32) * env
        k += 1
    return np.stack([x, x])            # stereo


class TempoTests(unittest.TestCase):
    def test_bpm_within_one(self):
        for bpm in (90, 120, 128):
            r = bs.detect_beats(click_track(bpm), SR)
            self.assertAlmostEqual(r["bpm"], bpm, delta=1.0, msg=f"{bpm} bpm")
            self.assertAlmostEqual(r["period_s"], 60.0 / bpm, delta=0.005)

    def test_other_sample_rate_and_mono(self):
        r = bs.detect_beats(click_track(100, sr=48000)[0], 48000)
        self.assertAlmostEqual(r["bpm"], 100, delta=1.0)

    def test_beat_times_and_downbeat(self):
        bpm = 120
        r = bs.detect_beats(click_track(bpm, first=0.5 + 2 * 0.5, lead_beats=2), SR)
        # true clicks at 0.5, 1.0, 1.5 ...; the accented downbeat is at 1.5 s
        diffs = [min(abs(t - (0.5 + 0.5 * k)) for k in range(40)) for t in r["beat_times"][:20]]
        self.assertLess(max(diffs), 0.02)
        self.assertAlmostEqual(r["downbeat_s"], 1.5, delta=0.02)

    def test_errors(self):
        with self.assertRaises(bs.StageError):
            bs.detect_beats(np.zeros((2, 1000), np.float32), SR)          # too short
        with self.assertRaises(bs.StageError):
            bs.detect_beats(np.zeros((2, SR * 5), np.float32), SR)        # silence


class CutTests(unittest.TestCase):
    def test_preferred_durations(self):
        plan = {"shots": [{"duration": 2}, {"duration": 4}, {"duration": 2}]}
        self.assertEqual(bs.preferred_durations(3, 16, json.dumps(plan)), [4.0, 8.0, 4.0])
        self.assertEqual(bs.preferred_durations(2, 10), [5.0, 5.0])
        p = bs.preferred_durations(4, 15, json.dumps(plan))           # 4th shot = mean
        self.assertAlmostEqual(sum(p), 15)
        with self.assertRaises(bs.StageError):
            bs.preferred_durations(0, 10)

    def test_cuts_on_beats(self):
        period = 0.5                                          # 120 bpm
        r = bs.beat_cut_durations(period, 5, 15.0, 1.5, "2", fps=24)
        self.assertEqual(len(r["durations"]), 5)
        for c in r["cut_times"]:
            self.assertAlmostEqual((c / (2 * period)) % 1, 0, places=6)   # every 2 beats
        self.assertAlmostEqual(r["total_seconds"], 15.0, places=6)
        self.assertGreaterEqual(min(r["durations"]), 1.5)
        self.assertAlmostEqual(sum(r["durations"]), r["total_seconds"], places=6)

    def test_preferences_followed(self):
        prefs = [2.0, 6.0, 2.0, 5.0]
        r = bs.beat_cut_durations(0.5, 4, 15.0, 1.5, "1", prefs=prefs)
        self.assertEqual([round(d, 3) for d in r["durations"]], prefs)
        r = bs.beat_cut_durations(60 / 128, 4, 15.0, 1.5, "4", prefs=prefs)
        bar = 4 * 60 / 128
        for d in r["durations"]:
            self.assertAlmostEqual(d / bar, round(d / bar), delta=0.03)    # frame rounding only

    def test_auto_grid_and_frames(self):
        r = bs.beat_cut_durations(0.5, 6, 18.0, 1.5, "auto", fps=24)
        self.assertEqual(r["beats_per_cut"], 2)
        for d in r["durations"]:
            self.assertAlmostEqual(d * 24, round(d * 24), places=6)        # whole frames
        r = bs.beat_cut_durations(60 / 90, 3, 12.0, 1.5, "auto")
        self.assertIn(r["beats_per_cut"], (2, 4))

    def test_infeasible(self):
        with self.assertRaises(bs.StageError):
            bs.beat_cut_durations(0.5, 10, 10.0, 1.5)
        with self.assertRaises(bs.StageError):
            bs.beat_cut_durations(0.5, 5, 10.0, 1.5, "8")              # 4 s bars, 5 shots in 10 s

    def test_format(self):
        self.assertEqual(bs.format_durations([3.0, 2.5, 1.0 / 3]), "3, 2.5, 0.333")


class FullTests(unittest.TestCase):
    def test_beat_sync(self):
        music = click_track(128, seconds=24, first=0.3)
        r = bs.beat_sync(music, SR, 5, 15.0, 1.5, "auto",
                         json.dumps({"shots": [{"duration": d} for d in (3, 2, 3, 2, 5)]}), 24)
        self.assertAlmostEqual(r["bpm"], 128, delta=1.0)
        self.assertAlmostEqual(r["music_offset_s"], 0.3, delta=0.02)
        self.assertEqual(r["aligned_beat_times"][0], 0.0)
        durs = [float(x) for x in r["durations"].split(",")]
        self.assertEqual(len(durs), 5)
        self.assertAlmostEqual(sum(durs), r["total_seconds"], places=2)
        self.assertLess(abs(r["total_seconds"] - 15.0), 60 / 128 * 2)
        # every cut lands on a detected (aligned) beat, within a frame
        beats = np.array(r["aligned_beat_times"])
        for c in np.cumsum(durs)[:-1]:
            self.assertLess(np.min(np.abs(beats - c)), 1 / 24 + 0.01)


if __name__ == "__main__":
    unittest.main()
