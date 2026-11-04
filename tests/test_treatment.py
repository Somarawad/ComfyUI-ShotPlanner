"""Tests for treatment.py and treatment_nodes.py (no GPU, no network).

Layout tests render at low dpi: every layout is in points, so the geometry is the same as
at 300 dpi, only cheaper. The node tests import the pack as a package (like ComfyUI does)
with a fake folder_paths module.
"""

import importlib.util
import itertools
import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import treatment as tr  # noqa: E402

LONG = "An extraordinarily long description of a shot that keeps on going " * 8


def frames(n, w=160, h=90):
    out = []
    for i in range(n):
        a = np.zeros((h, w, 3), np.float32)
        a[..., 0] = np.linspace(0, 1, w)
        a[..., 1] = (i % 5) / 5
        a[h // 3:2 * h // 3, w // 4 + i % 40:w // 2 + i % 40] = 1.0
        out.append(a)
    return out


def plan(n, long=False):
    return json.dumps({
        "title": ("A Very Long Title That Keeps Going " * 6) if long else "The First Light",
        "style": "warm natural light",
        "shots": [{"id": i + 1, "description": LONG if long else f"A cup steams, shot {i + 1}",
                   "camera": ["slow push-in", "pan left", "static", "tilt down"][i % 4],
                   "duration": 1.5 + i % 3} for i in range(n)]})


def project(n, long=False, bare=False, **extra):
    if bare:
        return tr.build_project(frames(n), plan(n, long), **extra)
    kw = dict(
        brand="NOVARA", client_line="Spec commercial · 15s" * (12 if long else 1),
        director="Somar Awad", logline=LONG if long else "One minute of light.",
        tone_words="warm, unhurried, Milanese morning" * (4 if long else 1),
        supers_text="\n".join(f"{i + 1} | {'Super text ' * (20 if long else 1)}" for i in range(n)),
        voiceover_text="\n".join(f"{i + 1} | {'Line ' * (30 if long else 3)}" for i in range(n)),
        verdicts_json=json.dumps([{"shot": i + 1, "score": 6 + i % 4, "pass": i % 3 != 0,
                                   "issues": ["soft hands " * (15 if long else 1)]}
                                  for i in range(n)]),
        brand_rules="\n".join(["- " + "rule words " * (20 if long else 4)] * (10 if long else 3)),
        date_text="2 November 2026")
    kw.update(extra)
    return tr.build_project(frames(n), plan(n, long), **kw)


def problems(doc):
    """Overlapping content boxes and boxes outside the page, per page."""
    out = []
    W, H = doc.size_pt
    for name, boxes in zip(doc.names, doc.boxes):
        for layer, (x0, y0, x1, y1), tag in boxes:
            if x0 < -0.5 or y0 < -0.5 or x1 > W + 0.5 or y1 > H + 0.5:
                out.append((name, "outside", tag))
        content = [b for b in boxes if b[0] == "content"]
        for a, b in itertools.combinations(content, 2):
            (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = a[1], b[1]
            if min(ax1, bx1) - max(ax0, bx0) > 0.5 and min(ay1, by1) - max(ay0, by0) > 0.5:
                out.append((name, "overlap", a[2], b[2]))
    return out


def width_fn(per_char=1.0):
    return lambda s: len(s) * per_char


# =========================================================================== text engine

class WrapTests(unittest.TestCase):
    def test_greedy_wrap_fits(self):
        w = width_fn()
        lines = tr.wrap_text("the quick brown fox jumps over the lazy dog", w, 15)
        self.assertTrue(all(w(l) <= 15 for l in lines))
        self.assertEqual(" ".join(lines), "the quick brown fox jumps over the lazy dog")

    def test_no_hyphens_long_word_broken_between_letters(self):
        w = width_fn()
        lines = tr.wrap_text("a supercalifragilistic word", w, 8)
        self.assertTrue(all(w(l) <= 8 for l in lines))
        self.assertFalse(any(l.endswith("-") for l in lines))
        self.assertEqual("".join(lines).replace(" ", ""), "asupercalifragilisticword")

    def test_widow_control(self):
        w = width_fn()
        text = "one two three four five six seven"          # greedy: ... / "seven"
        greedy = tr.wrap_text(text, w, 27, widow=False)
        self.assertEqual(greedy[-1], "seven")
        fixed = tr.wrap_text(text, w, 27)
        self.assertGreaterEqual(len(fixed[-1].split()), 2)
        self.assertEqual(" ".join(fixed), text)
        self.assertTrue(all(w(l) <= 27 for l in fixed))

    def test_widow_not_forced_when_it_would_overflow(self):
        w = width_fn()
        lines = tr.wrap_text("aaaa bbbb cccc dddddddd", w, 14)
        self.assertTrue(all(w(l) <= 14 for l in lines))

    def test_paragraphs_kept(self):
        lines = tr.wrap_text("first para\n\nsecond", width_fn(), 50)
        self.assertEqual(lines, ["first para", "", "second"])

    def test_balance(self):
        w = width_fn()
        text = "A title of several words that wraps"
        greedy = tr.wrap_text(text, w, 30)
        bal = tr.balance_lines(text, w, 30)
        self.assertEqual(len(bal), len(greedy))
        self.assertLessEqual(max(map(w, bal)) - min(map(w, bal)),
                             max(map(w, greedy)) - min(map(w, greedy)))

    def test_ellipsize(self):
        w = width_fn()
        s = tr.ellipsize("one two three four", w, 10)
        self.assertTrue(s.endswith(tr.ELLIPSIS))
        self.assertLessEqual(w(s), 10)


class FitTests(unittest.TestCase):
    U = 1.0

    def test_title_shrinks_to_fit(self):
        st = tr.TextStyle(family="serif", size=60)
        short = tr.fit_title("Hi", st, 400, 2, self.U, 20)
        self.assertEqual(short.style.size, 60)
        long = tr.fit_title("A much longer title that cannot fit at sixty points", st, 400, 2,
                            self.U, 20)
        self.assertLess(long.style.size, 60)
        self.assertGreaterEqual(long.style.size, 20)
        self.assertLessEqual(len(long.lines), 2)
        self.assertTrue(all(w <= 400 + 1e-6 for w in long.widths))
        self.assertFalse(long.truncated)

    def test_ellipsis_only_as_last_resort(self):
        st = tr.TextStyle(size=12)
        blk = tr.layout_text(LONG, st, 200, self.U, max_lines=2, min_size=10)
        self.assertEqual(blk.style.size, 10)            # shrank first, to the minimum
        self.assertTrue(blk.truncated)
        self.assertEqual(len(blk.lines), 2)
        self.assertTrue(blk.lines[-1].endswith(tr.ELLIPSIS))

    def test_height_limit(self):
        st = tr.TextStyle(size=10, leading=1.5)
        blk = tr.layout_text(LONG, st, 150, self.U, max_h=60)
        self.assertLessEqual(blk.height, 60 + 1e-6)

    def test_leading_on_baseline_grid(self):
        for size in (7, 9.3, 13, 42):
            blk = tr.layout_text("a b", tr.TextStyle(size=size), 100, self.U)
            self.assertAlmostEqual(blk.leading / tr.BASELINE, round(blk.leading / tr.BASELINE))


# =========================================================================== images

class CropTests(unittest.TestCase):
    def test_crop_box_bounds_and_aspect(self):
        for (iw, ih) in ((1920, 1080), (1080, 1920), (500, 500), (37, 911)):
            for aspect in (0.5, 0.8, 1.0, 1.78, 2.39):
                for focal in ((0, 0), (1, 1), (0.5, 0.5), (0.1, 0.9), (-3, 5)):
                    for zoom in (1, 2.3):
                        x0, y0, x1, y1 = tr.crop_box(iw, ih, aspect, focal, zoom)
                        self.assertGreaterEqual(x0, 0)
                        self.assertGreaterEqual(y0, 0)
                        self.assertLessEqual(x1, iw)
                        self.assertLessEqual(y1, ih)
                        self.assertAlmostEqual((x1 - x0) / (y1 - y0), aspect,
                                               delta=aspect * 0.05 + 2.0 / (y1 - y0))

    def test_crop_follows_focal(self):
        left = tr.crop_box(1920, 1080, 1.0, (0.0, 0.5))
        right = tr.crop_box(1920, 1080, 1.0, (1.0, 0.5))
        self.assertEqual(left[0], 0)
        self.assertEqual(right[2], 1920)

    def test_focal_point_finds_detail(self):
        a = np.zeros((90, 160, 3), np.float32)
        a[10:30, 120:150] = 1.0                       # bright, detailed patch top right
        fx, fy = tr.focal_point(a)
        self.assertGreater(fx, 0.6)
        self.assertLess(fy, 0.45)

    def test_cover_never_stretches(self):
        a = np.zeros((90, 160, 3), np.float32)
        a[:, :, 0] = np.linspace(0, 1, 160)
        im = tr.cover(a, 50, 100)
        self.assertEqual(im.size, (50, 100))

    def test_fit_rect(self):
        x, y, w, h = tr.fit_rect(16 / 9, 0, 0, 100, 100)
        self.assertAlmostEqual(w / h, 16 / 9)
        self.assertLessEqual(w, 100)


# =========================================================================== data

class DataTests(unittest.TestCase):
    def test_timecode(self):
        self.assertEqual(tr.timecode(0), "00:00:00:00")
        self.assertEqual(tr.timecode(2.5, 24), "00:00:02:12")
        self.assertEqual(tr.timecode(61.04, 25), "00:01:01:01")
        self.assertEqual(tr.timecode(3725.5, 23.976), "01:02:05:12")
        self.assertEqual(tr.short_duration(3.0), "3 s")
        self.assertEqual(tr.short_duration(2.5), "2.5 s")

    def test_camera_mapping(self):
        cases = {
            "slow push-in": ("push-in", "frames-in"), "Dolly in on the cup": ("push-in", "frames-in"),
            "pull back": ("pull-out", "frames-out"), "pan left to right": ("pan-right", "arrow-right"),
            "pan right to left": ("pan-left", "arrow-left"), "whip pan left": ("pan-left", "arrow-left"),
            "pan": ("pan-right", "arrow-right"), "tilt down": ("tilt-down", "arrow-down"),
            "tilt": ("tilt-up", "arrow-up"), "crane up": ("crane-up", "arrow-up"),
            "tracking shot to the left": ("track-left", "arrow-left"), "orbit": ("orbit", "arc"),
            "static": ("static", None), "locked-off wide": ("static", None), "": ("static", None),
            "handheld": ("handheld", None),
        }
        for text, (move, arrow) in cases.items():
            self.assertEqual(tr.camera_move(text), move, text)
            self.assertEqual(tr.camera_arrow(text), arrow, text)
        self.assertEqual(set(tr.CAMERA_ARROWS), set(tr.CAMERA_MOVES))

    def test_verdicts_any_shape(self):
        v = tr.parse_verdicts(json.dumps([{"shot": 2, "score": 0.86, "route": "reroll"},
                                          {"shot": 3, "score": 72}, {}]))
        self.assertEqual(v[2]["score"], 8.6)
        self.assertFalse(v[2]["pass"])
        self.assertEqual(v[2]["route"], "reroll")
        self.assertEqual(v[2]["issues"], [])
        self.assertEqual(tr.parse_verdicts(json.dumps({"verdicts": [{"shot": 1, "pass": True}]}))
                         [1]["pass"], True)
        self.assertEqual(tr.parse_verdicts(json.dumps({"3": {"score": 9, "issues": "blur"}}))
                         [3]["issues"], ["blur"])
        self.assertEqual(tr.parse_verdicts(""), {})
        with self.assertRaises(tr.TreatmentError):
            tr.parse_verdicts("{not json")

    def test_supers_and_vo(self):
        self.assertEqual(tr.parse_numbered_lines("1 | Hello\n3 | End\n9 | nope", 3),
                         {1: "Hello", 3: "End"})
        self.assertEqual(tr.parse_numbered_lines("a\nb", 3, sequential=True), {1: "a", 2: "b"})
        P = project(3, supers_text="0.5-1 | Early", voiceover_text="")
        self.assertEqual(P.shots[0].super_text, "Early")

    def test_timing_from_json_and_shot_list(self):
        t = {"fps": 25, "total_seconds": 7, "shots": [{"shot": 1, "start_s": 0, "end_s": 3},
                                                       {"shot": 2, "start_s": 3, "end_s": 6}]}
        P = project(2, timing_json=json.dumps(t))
        self.assertEqual((P.fps, P.total_seconds, P.shots[1].start), (25.0, 7.0, 3.0))
        self.assertTrue(P.has_timing)
        P2 = project(3, bare=True)
        self.assertEqual([s.start for s in P2.shots], [0.0, 1.5, 4.0])
        self.assertAlmostEqual(P2.total_seconds, 7.5)

    def test_missing_inputs(self):
        P = tr.build_project(frames(2), "")
        self.assertEqual(len(P.shots), 2)
        self.assertTrue(P.approach)
        P = tr.build_project([], plan(3))              # no keyframes: placeholders
        self.assertEqual(len(P.shots), 3)
        self.assertIsNone(P.shots[0].image)
        self.assertEqual(len(tr.render_treatment(P, "agency minimal", dpi=24).pages),
                         tr.treatment_page_count(3, "agency minimal"))
        with self.assertRaises(tr.TreatmentError):
            tr.build_project([], "")
        with self.assertRaises(tr.TreatmentError):
            tr.get_theme("neon")

    def test_approach_is_plain_sentences(self):
        P = project(6)
        self.assertTrue(P.approach.endswith("one idea."))
        self.assertIn("Six shots", P.approach)
        one = project(1)
        self.assertIn("single, held image", one.approach)

    def test_logo_mask(self):
        rgb = np.zeros((20, 40, 3), np.float32)
        mask = np.ones((20, 40), np.float32)
        mask[5:15, 10:30] = 0                          # ComfyUI: 0 = visible
        logo = tr.logo_from_arrays(rgb, mask)
        self.assertEqual(logo.size, (20, 10))          # trimmed to visible pixels
        self.assertEqual(logo.mode, "RGBA")

    def test_accent_override_and_contrast(self):
        P = project(2, accent_hex="#c8743a")
        self.assertEqual(P.accent, "#C8743A")
        ctx = tr.make_ctx(P, tr.get_theme("editorial dark"), "doc", 595, 842)
        self.assertGreaterEqual(tr.pal.contrast_ratio(ctx.accent_text, ctx.T.bg), 3.0 - 1e-6)
        self.assertGreaterEqual(tr.pal.contrast_ratio(ctx.on_accent, ctx.accent), 4.5 - 1e-6)


# =========================================================================== layouts

class LayoutTests(unittest.TestCase):
    """Every page with 1, 4 and 12 shots, full / missing / very long inputs, in every
    template: no overlapping content, nothing off the page."""

    def check(self, doc, label):
        p = problems(doc)
        self.assertEqual(p, [], f"{label}: {p[:5]}")

    def test_treatment_pages(self):
        for n in (1, 4, 12):
            for variant in ("full", "bare", "long"):
                P = project(n, long=variant == "long", bare=variant == "bare")
                for t in tr.TEMPLATE_NAMES:
                    for size in ("A4", "Letter"):
                        doc = tr.render_treatment(P, t, size, dpi=36)
                        self.assertEqual(len(doc.pages), tr.treatment_page_count(n, t))
                        self.check(doc, f"treatment {t} {size} n={n} {variant}")

    def test_deck_slides(self):
        for n in (1, 4, 12):
            for variant in ("full", "bare", "long"):
                P = project(n, long=variant == "long", bare=variant == "bare")
                for t in tr.TEMPLATE_NAMES:
                    doc = tr.render_deck(P, t, 480)
                    self.assertEqual(len(doc.pages), tr.deck_slide_count(n))
                    self.assertEqual(doc.pages[0].size, (480, 270))
                    self.check(doc, f"deck {t} n={n} {variant}")

    def test_page_counts(self):
        self.assertEqual(tr.treatment_page_count(1, "editorial dark"), 9)
        self.assertEqual(tr.treatment_page_count(4, "editorial dark"), 10)
        self.assertEqual(tr.treatment_page_count(12, "editorial light"), 12)
        self.assertEqual(tr.treatment_page_count(12, "agency minimal"), 14)
        self.assertEqual(tr.deck_slide_count(12), 18)

    def test_page_names_and_size(self):
        doc = tr.render_treatment(project(4), "editorial dark", "A4", dpi=36)
        self.assertEqual(doc.names, ["cover", "idea", "mood", "palette", "story", "shots-1",
                                     "shots-2", "brand", "notes", "closing"])
        self.assertEqual(doc.pages[0].size, (298, 421))       # A4 at 36 dpi
        letter = tr.render_treatment(project(1), "editorial dark", "Letter", dpi=36)
        self.assertEqual(letter.pages[0].size, (306, 396))

    def test_templates_differ_in_layout_not_only_colour(self):
        P = project(4)
        boxes = {}
        for t in tr.TEMPLATE_NAMES:
            doc = tr.render_treatment(P, t, dpi=24)
            cover = doc.boxes[0]
            boxes[t] = sorted(tag for _, _, tag in cover)
            self.assertIn("title", boxes[t])
        sizes = {t: tr.get_theme(t).scale["cover"] for t in tr.TEMPLATE_NAMES}
        self.assertEqual(len(set(sizes.values())), 3)
        self.assertEqual(len({tr.get_theme(t).cover for t in tr.TEMPLATE_NAMES}), 3)
        self.assertEqual(len({tr.get_theme(t).shots for t in tr.TEMPLATE_NAMES}), 3)

    def test_storyboard(self):
        for paper in tr.STORYBOARD_PAPERS:
            for n in (1, 4, 12):
                doc = tr.render_storyboard(project(n), paper, 4, dpi=36)
                self.assertGreaterEqual(len(doc.pages), 1)
                self.check(doc, f"storyboard {paper} n={n}")
                arrows = sum(1 for b in doc.boxes[0] if b[2] == "camera-arrow")
                expected = sum(1 for s in project(n).shots[:arrows or 1]
                               if tr.camera_arrow(s.camera))
                self.assertGreater(arrows + expected, 0)
        doc = tr.render_storyboard(project(4), "A4 landscape", 4, dpi=24)
        tags = [b[2] for b in doc.boxes[0]]
        self.assertEqual(tags.count("panel"), 4)
        self.assertEqual(tags.count("camera-arrow"), 3)        # shots 1, 2, 4 move; 3 static
        many = tr.render_storyboard(project(12), "A4 landscape", 2, dpi=24)
        self.assertEqual(sum([b[2] for b in bx].count("panel") for bx in many.boxes), 12)

    def test_storyboard_layout_math(self):
        pw, ph, rows, sheets = tr.storyboard_layout(12, 4, 780, 500, 16 / 9, 12, 40)
        self.assertEqual(sheets, 1)
        self.assertLessEqual(rows * (ph + 40) + (rows - 1) * 12, 500)
        pw, ph, rows, sheets = tr.storyboard_layout(30, 2, 780, 500, 16 / 9, 12, 40)
        self.assertGreater(sheets, 1)

    def test_palette_card(self):
        P = project(3)
        card = tr.render_palette_card(P.palette, P.accent, 800)
        self.assertEqual(card.size, (800, 450))


class PdfTests(unittest.TestCase):
    def test_pdf_written_and_reopened(self):
        P = project(4)
        with tempfile.TemporaryDirectory() as d:
            doc = tr.render_treatment(P, "editorial light", "A4", dpi=36)
            path = tr.save_pdf(doc, os.path.join(d, "sub", "t.pdf"))
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(tr.pdf_page_count(path), len(doc.pages))
            info = tr.pdf_info(path)
            self.assertEqual(info.get("Title"), "The First Light")
            self.assertEqual(info.get("Author"), "Somar Awad")
            with open(path, "rb") as f:
                head = f.read(5)
            self.assertEqual(head, b"%PDF-")
            deck = tr.render_deck(P, "agency minimal", 480)
            p2 = tr.save_pdf(deck, os.path.join(d, "deck.pdf"))
            self.assertEqual(tr.pdf_page_count(p2), tr.deck_slide_count(4))

    def test_page_size_in_points(self):
        P = project(1)
        with tempfile.TemporaryDirectory() as d:
            doc = tr.render_treatment(P, "editorial dark", "A4", dpi=36)
            path = tr.save_pdf(doc, os.path.join(d, "a4.pdf"))
            with open(path, "rb") as f:
                data = f.read()
            self.assertIn(b"/MediaBox [ 0 0 596", data)        # 210 mm = 595.3 pt

    def test_previews_same_width(self):
        doc = tr.render_treatment(project(2), "editorial dark", dpi=36)
        pv = tr.previews(doc.pages, 200)
        self.assertEqual({p.size for p in pv}, {(200, 283)})


# =========================================================================== nodes

PKG = "shotplanner_treatment_pkg"


def load_nodes(output_dir):
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: output_dir

    def get_save_image_path(prefix, base):
        sub = os.path.dirname(prefix)
        return os.path.join(base, sub), os.path.basename(prefix), 1, sub, prefix
    fp.get_save_image_path = get_save_image_path
    sys.modules["folder_paths"] = fp
    if PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(PKG, os.path.join(ROOT, "__init__.py"),
                                                      submodule_search_locations=[ROOT])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[PKG] = pkg
        spec.loader.exec_module(pkg)
    return sys.modules[PKG], sys.modules[f"{PKG}.treatment_nodes"]


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
class NodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pkg, cls.m = load_nodes(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        sys.modules.pop("folder_paths", None)

    def kf(self, n=3):
        return [self.torch.from_numpy(np.stack(frames(n)))]       # one batch, as a list

    def common(self, n=3, **kw):
        d = dict(keyframes=self.kf(n), shot_list_json=[plan(n)], brand=["NOVARA"],
                 client_line=["Spec"], director=["SA"], logline=[""],
                 tone_words=["warm, slow"], template=["agency minimal"])
        d.update(kw)
        return d

    def test_registered_with_tooltips(self):
        for key, cls in self.m.TREATMENT_NODE_CLASS_MAPPINGS.items():
            self.assertIs(self.pkg.NODE_CLASS_MAPPINGS[key], cls)
            self.assertIn(key, self.pkg.NODE_DISPLAY_NAME_MAPPINGS)
            self.assertEqual(cls.CATEGORY, "Shot Planner/Treatment")
            self.assertEqual(len(cls.OUTPUT_TOOLTIPS), len(cls.RETURN_TYPES))
            for group in cls.INPUT_TYPES().values():
                for inp, (typ, *opts) in group.items():
                    self.assertTrue(opts and opts[0].get("tooltip"), f"{key}.{inp}")
        self.assertEqual(len(self.m.TREATMENT_NODE_CLASS_MAPPINGS), 4)

    def test_treatment_builder(self):
        node = self.m.TreatmentBuilder()
        path, pages, count = node.build(["Letter"], ["proj/treat"], **self.common(
            3, verdicts_json=[json.dumps([{"shot": 1, "score": 9}])],
            logo=[self.torch.ones(1, 8, 16, 3)], logo_mask=[self.torch.zeros(1, 8, 16)]))
        self.assertIsInstance(path, str)
        self.assertTrue(path.startswith(os.path.join(self.tmp.name, "proj")))
        self.assertTrue(path.endswith("treat_00001.pdf"))
        self.assertIsInstance(count, int)
        self.assertEqual(count, tr.treatment_page_count(3, "agency minimal"))
        self.assertIsInstance(pages, self.torch.Tensor)
        self.assertEqual(tuple(pages.shape[::3]), (count, 3))
        self.assertEqual(pages.shape[2], tr.PREVIEW_WIDTH)
        self.assertEqual(tr.pdf_page_count(path), count)
        # a second run does not overwrite the first
        path2, _, _ = node.build(["A4"], ["proj/treat"], **self.common(1))
        self.assertNotEqual(path, path2)

    def test_pitch_deck_builder(self):
        path, slides, count = self.m.PitchDeckBuilder().build(
            ["1920x1080"], ["deck"], **self.common(2, template=["editorial light"]))
        self.assertEqual(count, tr.deck_slide_count(2))
        self.assertEqual(tuple(slides.shape), (count, 1080, 1920, 3))
        self.assertTrue(os.path.isfile(path))

    def test_storyboard_sheet(self):
        sheet, path = self.m.StoryboardSheet().draw(self.kf(4), [plan(4)], [3], ["16:9"],
                                                    [False], ["sb"])
        self.assertEqual(path, "")
        self.assertEqual(tuple(sheet.shape[1:]), (2160, 3840, 3))
        sheet, path = self.m.StoryboardSheet().draw(self.kf(2), [plan(2)], [2],
                                                    ["A4 landscape"], [True], ["sb"])
        self.assertTrue(path.endswith(".pdf") and os.path.isfile(path))

    def test_palette_from_shots(self):
        img, pj, acc, on = self.m.PaletteFromShots().extract(self.kf(2), [5])
        data = json.loads(pj)
        self.assertLessEqual(len(data["swatches"]), 5)
        self.assertEqual(acc, data["accent_hex"])
        self.assertTrue(tr.pal.meets_aa(on, acc))
        self.assertEqual(tuple(img.shape), (1, 900, 1600, 3))
        _, _, acc2, _ = self.m.PaletteFromShots().extract(self.kf(1), [3], ["#123456"])
        self.assertEqual(acc2, "#123456")
        with self.assertRaises(RuntimeError):
            self.m.PaletteFromShots().extract([], [3])

    def test_errors_are_runtime_errors(self):
        with self.assertRaises(RuntimeError):
            self.m.TreatmentBuilder().build(["A4"], ["x"], **self.common(
                2, verdicts_json=["{broken"]))

    def test_output_path_without_comfy(self):
        saved = sys.modules.pop("folder_paths")
        cwd = os.getcwd()
        try:
            os.chdir(self.tmp.name)
            p = self.m.output_path("nested/name")
            self.assertTrue(p.endswith(os.path.join("output", "nested", "name_00001.pdf")))
        finally:
            os.chdir(cwd)
            sys.modules["folder_paths"] = saved


if __name__ == "__main__":
    unittest.main()
