"""Tests for palette.py (palette extraction, Lab merging, names, WCAG contrast). numpy only."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import palette as pal  # noqa: E402


def two_colour(h=120, w=200, split=0.7, a=(59, 42, 32), b=(230, 200, 150)):
    im = np.zeros((h, w, 3), np.uint8)
    k = int(round(h * split))
    im[:k] = a
    im[k:] = b
    return im


class HexTests(unittest.TestCase):
    def test_round_trip(self):
        self.assertEqual(pal.hex_to_rgb("#C8743A"), (200, 116, 58))
        self.assertEqual(pal.hex_to_rgb("abc"), (170, 187, 204))
        self.assertEqual(pal.rgb_to_hex((200, 116, 58)), "#C8743A")
        self.assertEqual(pal.hex_to_rgb("nope", (1, 2, 3)), (1, 2, 3))
        with self.assertRaises(ValueError):
            pal.hex_to_rgb("#12")

    def test_lab_round_trip(self):
        for c in [(0, 0, 0), (255, 255, 255), (200, 116, 58), (10, 200, 100), (30, 60, 220)]:
            back = pal.lab_to_rgb(pal.rgb_to_lab(np.array(c, float)))
            np.testing.assert_allclose(back, c, atol=0.6)
        L, a, b = pal.rgb_to_lab(np.array([255.0, 255, 255]))
        self.assertAlmostEqual(L, 100.0, places=2)
        self.assertAlmostEqual(a, 0.0, places=2)


class ExtractionTests(unittest.TestCase):
    def test_two_colours_two_swatches_with_shares(self):
        sw = pal.extract_palette([two_colour()])
        self.assertEqual(len(sw), 2)
        self.assertEqual(sw[0]["hex"], "#3B2A20")
        self.assertEqual(sw[1]["hex"], "#E6C896")
        self.assertAlmostEqual(sw[0]["share"], 0.7, delta=0.01)
        self.assertAlmostEqual(sw[1]["share"], 0.3, delta=0.01)
        self.assertAlmostEqual(sum(s["share"] for s in sw), 1.0, places=3)
        self.assertEqual(sw[0]["name"], "espresso brown")

    def test_float_frames_and_several_images(self):
        a = two_colour(split=1.0).astype(np.float32) / 255.0          # all brown
        b = two_colour(split=0.0).astype(np.float32) / 255.0          # all sand
        sw = pal.extract_palette([a, b, b, b])
        self.assertEqual(len(sw), 2)
        self.assertAlmostEqual(sw[0]["share"], 0.75, delta=0.01)    # sand first: 3 of 4 frames

    def test_near_duplicates_merge(self):
        im = two_colour(a=(120, 60, 40), b=(124, 62, 41), split=0.5)
        sw = pal.extract_palette([im])
        self.assertEqual(len(sw), 1)
        self.assertAlmostEqual(sw[0]["share"], 1.0)

    def test_merge_clusters_lab(self):
        labs = [[50, 10, 10], [52, 11, 9], [80, -20, 30]]
        out, w = pal.merge_clusters(labs, [3, 1, 2], threshold=5)
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(w.tolist()), [2.0, 4.0])
        merged = out[int(np.argmax(w))]
        np.testing.assert_allclose(merged, [50.5, 10.25, 9.75])      # size-weighted mean
        out2, _ = pal.merge_clusters(labs, [3, 1, 2], threshold=1)  # nothing that close
        self.assertEqual(len(out2), 3)

    def test_colors_limit_and_determinism(self):
        rng = np.random.default_rng(1)
        im = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        a = pal.extract_palette([im], colors=4)
        b = pal.extract_palette([im], colors=4)
        self.assertLessEqual(len(a), 4)
        self.assertEqual(a, b)
        shares = [s["share"] for s in a]
        self.assertEqual(shares, sorted(shares, reverse=True))

    def test_downsample_keeps_mean(self):
        im = two_colour(h=1000, w=1600)
        small = pal.downsample(im)
        self.assertLessEqual(max(small.shape[:2]), pal.SAMPLE_SIDE)

    def test_unique_names(self):
        sw = [{"name": "cocoa", "lab": [40, 5, 5]}, {"name": "cocoa", "lab": [20, 5, 5]}]
        pal._unique_names(sw)
        self.assertEqual({s["name"] for s in sw}, {"deep cocoa", "pale cocoa"})

    def test_pick_accent_prefers_colourful(self):
        sw = pal.extract_palette([two_colour(a=(30, 30, 30), b=(220, 90, 30), split=0.8)])
        self.assertEqual(pal.pick_accent(sw)["hex"], sw[1]["hex"])

    def test_names(self):
        self.assertEqual(pal.color_name("#3B2A20"), "espresso brown")
        self.assertEqual(pal.color_name((29, 42, 74)), "navy")
        self.assertIn("white", pal.color_name("#FAFAF7"))


class ContrastTests(unittest.TestCase):
    def test_ratio(self):
        self.assertAlmostEqual(pal.contrast_ratio("#000000", "#FFFFFF"), 21.0, places=3)
        self.assertAlmostEqual(pal.contrast_ratio("#777777", "#777777"), 1.0)
        self.assertTrue(pal.meets_aa("#767676", "#FFFFFF"))        # the classic AA grey
        self.assertFalse(pal.meets_aa("#777777", "#FFFFFF"))
        self.assertTrue(pal.meets_aa("#949494", "#FFFFFF", large=True))

    def test_pick_text_color_aa_on_any_swatch(self):
        rng = np.random.default_rng(0)
        for c in rng.integers(0, 256, (200, 3)):
            t = pal.pick_text_color(tuple(int(v) for v in c))
            self.assertGreaterEqual(pal.contrast_ratio(t, tuple(int(v) for v in c)), 4.5 - 1e-6)
        self.assertEqual(pal.pick_text_color("#FFFFFF"), "#111111")
        self.assertEqual(pal.pick_text_color("#101010"), "#FFFFFF")

    def test_ensure_contrast(self):
        for fg, bg in (("#C8743A", "#13110F"), ("#E8C27A", "#F3EEE5"), ("#333333", "#222222")):
            out = pal.ensure_contrast(fg, bg, 4.5)
            self.assertGreaterEqual(pal.contrast_ratio(out, bg), 4.5 - 1e-6)
        self.assertEqual(pal.ensure_contrast("#000000", "#FFFFFF"), "#000000")  # already fine


if __name__ == "__main__":
    unittest.main()
