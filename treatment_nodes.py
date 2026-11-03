"""ComfyUI nodes for the Treatment & Pitch Deck stage (Shot Planner/Treatment).

All layout and rendering lives in treatment.py and palette.py; these classes only unwrap
ComfyUI inputs (IMAGE tensors, lists) and wrap outputs. Every node uses INPUT_IS_LIST so
keyframes arrive the same way whether they come as one batch or as a list of images
(e.g. from Shot Critic / Detail Fixer with different sizes per shot).
"""

import json
import os

import numpy as np
import torch

from . import palette as pal
from . import treatment as tr

CATEGORY = "Shot Planner/Treatment"


# --------------------------------------------------------------------------- helpers

def _one(v, default=None):
    if isinstance(v, (list, tuple)):
        v = v[0] if v else default
    return default if v is None else v


def _frames(images):
    """IMAGE batch / list of batches -> list of [H,W,3] float numpy arrays."""
    out = []
    if images is None:
        return out
    if not isinstance(images, (list, tuple)):
        images = [images]
    for t in images:
        if t is None:
            continue
        a = t.detach().cpu().float().numpy() if hasattr(t, "detach") else np.asarray(t, np.float32)
        if a.ndim == 3:
            a = a[None]
        out.extend(a[i, ..., :3] for i in range(a.shape[0]))
    return out


def _to_tensor(pages):
    arrs = [np.asarray(p.convert("RGB"), dtype=np.float32) / 255.0 for p in pages]
    return torch.from_numpy(np.stack(arrs, 0))


def _logo(logo, logo_mask, invert):
    imgs = _frames(logo)
    if not imgs:
        return None
    mask = None
    m = _one(logo_mask)
    if m is not None:
        mask = m.detach().cpu().float().numpy() if hasattr(m, "detach") else np.asarray(m)
    return tr.logo_from_arrays(imgs[0], mask, bool(invert))


def output_path(prefix, ext="pdf"):
    """Next free <output>/<prefix>_00001.pdf, using ComfyUI's folder_paths when present
    (so subfolders in the prefix work as in Save Image), else ./output."""
    prefix = (prefix or "treatment").strip() or "treatment"
    try:
        import folder_paths
        base = folder_paths.get_output_directory()
        folder, name, counter, _, _ = folder_paths.get_save_image_path(prefix, base)
    except ImportError:
        base = os.path.abspath("output")
        folder = os.path.join(base, os.path.dirname(prefix))
        name = os.path.basename(prefix)
        counter = 1
    os.makedirs(folder, exist_ok=True)
    while True:
        path = os.path.join(folder, f"{name}_{counter:05}.{ext}")
        if not os.path.exists(path):
            return path
        counter += 1


def _err(node, e):
    return RuntimeError(f"{node}: {e}")


TIP = {
    "keyframes": "Approved keyframes, one still per shot in shot order (after Shot Critic / "
                 "Detail Fixer). A batch or a list; sizes may differ.",
    "shot_list_json": "Shot Planner plan: title, style and per-shot description, camera, "
                      "duration.",
    "brand": "Brand name; used as a wordmark when no logo is connected.",
    "client_line": "One line under the title, e.g. \"Spec commercial · 15s · 16:9\".",
    "director": "Director's name on the cover and closing page.",
    "logline": "One or two sentences that sell the idea. Empty = built from the shot "
               "list's title and style.",
    "tone_words": "Comma-separated tone words, e.g. \"warm, unhurried, Milanese morning\".",
    "template": "editorial dark: full-bleed cinematic, serif. editorial light: warm paper, "
                "framed images, big numerals. agency minimal: white Swiss grid, sans "
                "headlines, two shots per page.",
    "filename_prefix": "PDF name inside ComfyUI's output folder (subfolders allowed).",
    "verdicts_json": "Shot Critic (or Shot Video QA) verdicts: adds QA badges, critic notes "
                     "and a score chart. All fields optional.",
    "timing_json": "Ad Timeline timing: real in/out timecodes and the timeline ruler. "
                   "Without it, shot list durations are used.",
    "supers_text": "On-screen text as \"shot_number | text\" lines (Ad Script From Shots).",
    "voiceover_text": "VO as \"shot_number | line\" lines or one line per shot.",
    "brand_rules": "One rule per line for the brand page. Empty = sensible defaults.",
    "pipeline_notes": "\"Key: value\" items separated by ; or new lines, shown on the "
                      "production notes page.",
    "logo": "Brand logo (first image is used).",
    "logo_mask": "Logo transparency, e.g. the MASK from Load Image (1 = transparent).",
    "invert_mask": "Turn on if your mask is 1 where the logo is visible.",
    "accent_hex": "Accent colour, e.g. #C8743A. Empty = picked from the keyframe palette.",
    "date": "Date shown on the documents. Empty = today.",
}


def _common_required():
    return {
        "keyframes": ("IMAGE", {"tooltip": TIP["keyframes"]}),
        "shot_list_json": ("STRING", {"forceInput": True, "tooltip": TIP["shot_list_json"]}),
        "brand": ("STRING", {"default": "NOVARA", "tooltip": TIP["brand"]}),
        "client_line": ("STRING", {"default": "Spec commercial · 15s · 16:9",
                                   "tooltip": TIP["client_line"]}),
        "director": ("STRING", {"default": "Somar Awad", "tooltip": TIP["director"]}),
        "logline": ("STRING", {"default": "", "multiline": True, "tooltip": TIP["logline"]}),
        "tone_words": ("STRING", {"default": "warm, unhurried, Milanese morning",
                                  "tooltip": TIP["tone_words"]}),
        "template": (tr.TEMPLATE_NAMES, {"default": "editorial dark", "tooltip": TIP["template"]}),
    }


def _common_optional():
    return {
        "verdicts_json": ("STRING", {"forceInput": True, "tooltip": TIP["verdicts_json"]}),
        "timing_json": ("STRING", {"forceInput": True, "tooltip": TIP["timing_json"]}),
        "supers_text": ("STRING", {"forceInput": True, "tooltip": TIP["supers_text"]}),
        "voiceover_text": ("STRING", {"forceInput": True, "tooltip": TIP["voiceover_text"]}),
        "brand_rules": ("STRING", {"default": "", "multiline": True, "tooltip": TIP["brand_rules"]}),
        "pipeline_notes": ("STRING", {"default": tr.DEFAULT_PIPELINE, "multiline": True,
                                      "tooltip": TIP["pipeline_notes"]}),
        "logo": ("IMAGE", {"tooltip": TIP["logo"]}),
        "logo_mask": ("MASK", {"tooltip": TIP["logo_mask"]}),
        "invert_mask": ("BOOLEAN", {"default": False, "tooltip": TIP["invert_mask"]}),
        "accent_hex": ("STRING", {"default": "", "tooltip": TIP["accent_hex"]}),
        "date": ("STRING", {"default": "", "tooltip": TIP["date"]}),
    }


def _project(keyframes, shot_list_json, brand, client_line, director, logline, tone_words,
             verdicts_json=None, timing_json=None, supers_text=None, voiceover_text=None,
             brand_rules=None, pipeline_notes=None, logo=None, logo_mask=None,
             invert_mask=None, accent_hex=None, date=None, **_):
    return tr.build_project(
        _frames(keyframes), _one(shot_list_json, ""), brand=_one(brand, ""),
        client_line=_one(client_line, ""), director=_one(director, ""),
        logline=_one(logline, ""), tone_words=_one(tone_words, ""),
        verdicts_json=_one(verdicts_json), timing_json=_one(timing_json),
        supers_text=_one(supers_text, ""), voiceover_text=_one(voiceover_text, ""),
        brand_rules=_one(brand_rules, ""), logo=_logo(logo, logo_mask, _one(invert_mask, False)),
        accent_hex=_one(accent_hex, ""),
        pipeline_notes=_one(pipeline_notes, tr.DEFAULT_PIPELINE), date_text=_one(date, ""))


# --------------------------------------------------------------------------- nodes

class TreatmentBuilder:
    """Director's treatment PDF (portrait A4 / Letter, 300 dpi) plus page previews."""

    @classmethod
    def INPUT_TYPES(cls):
        req = _common_required()
        req["page_size"] = (list(tr.PAGE_SIZES_PT), {"default": "A4",
                            "tooltip": "A4 (210 x 297 mm) or US Letter (8.5 x 11 in)."})
        req["filename_prefix"] = ("STRING", {"default": "treatment/treatment",
                                             "tooltip": TIP["filename_prefix"]})
        return {"required": req, "optional": _common_optional()}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "IMAGE", "INT")
    RETURN_NAMES = ("pdf_path", "pages", "page_count")
    OUTPUT_TOOLTIPS = (
        "Path of the PDF written to ComfyUI's output folder.",
        f"Every page as an image, {tr.PREVIEW_WIDTH} px wide (connect Preview Image).",
        "Number of pages.",
    )
    FUNCTION = "build"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Lays out a director's treatment: cover, idea, mood board, palette, story "
                   "at a glance, shot-by-shot pages, brand, production notes and closing page.")

    def build(self, page_size, filename_prefix, **kw):
        try:
            P = _project(**kw)
            doc = tr.render_treatment(P, _one(kw.get("template"), "editorial dark"),
                                      _one(page_size, "A4"))
            path = tr.save_pdf(doc, output_path(_one(filename_prefix, "treatment")))
        except tr.TreatmentError as e:
            raise _err("Treatment Builder", e) from e
        return (path, _to_tensor(tr.previews(doc.pages)), len(doc.pages))


class PitchDeckBuilder:
    """16:9 pitch deck PDF plus slide images."""

    @classmethod
    def INPUT_TYPES(cls):
        req = _common_required()
        req["resolution"] = (list(tr.DECK_RESOLUTIONS), {"default": "1920x1080",
                             "tooltip": "Slide size in pixels (the PDF page is 13.33 x 7.5 in)."})
        req["filename_prefix"] = ("STRING", {"default": "treatment/pitch_deck",
                                             "tooltip": TIP["filename_prefix"]})
        return {"required": req, "optional": _common_optional()}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "IMAGE", "INT")
    RETURN_NAMES = ("pdf_path", "slides", "slide_count")
    OUTPUT_TOOLTIPS = (
        "Path of the PDF written to ComfyUI's output folder.",
        "Every slide at full resolution.",
        "Number of slides.",
    )
    FUNCTION = "build"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Builds a 16:9 pitch deck: cover, idea, mood, palette, one slide per shot, "
                   "timeline and end card.")

    def build(self, resolution, filename_prefix, **kw):
        try:
            P = _project(**kw)
            doc = tr.render_deck(P, _one(kw.get("template"), "editorial dark"),
                                 _one(resolution, "1920x1080"))
            path = tr.save_pdf(doc, output_path(_one(filename_prefix, "pitch_deck")))
        except tr.TreatmentError as e:
            raise _err("Pitch Deck Builder", e) from e
        return (path, _to_tensor(doc.pages), len(doc.pages))


class StoryboardSheet:
    """Print-ready storyboard sheet with numbered panels and camera-move arrows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframes": ("IMAGE", {"tooltip": TIP["keyframes"]}),
                "shot_list_json": ("STRING", {"forceInput": True, "tooltip": TIP["shot_list_json"]}),
                "columns": ("INT", {"default": 4, "min": 1, "max": 8,
                    "tooltip": "Panels per row. Rows follow from the paper; extra shots go "
                               "onto further sheets."}),
                "paper": (list(tr.STORYBOARD_PAPERS), {"default": "A4 landscape",
                    "tooltip": "A4 landscape and A3 at 300 dpi; 16:9 at 3840 x 2160."}),
                "save_pdf": ("BOOLEAN", {"default": True,
                    "tooltip": "Also write a PDF to the output folder."}),
                "filename_prefix": ("STRING", {"default": "treatment/storyboard",
                                               "tooltip": TIP["filename_prefix"]}),
            },
            "optional": {
                "timing_json": ("STRING", {"forceInput": True, "tooltip": TIP["timing_json"]}),
                "brand": ("STRING", {"default": "", "tooltip": TIP["brand"]}),
                "date": ("STRING", {"default": "", "tooltip": TIP["date"]}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "pdf_path")
    OUTPUT_TOOLTIPS = (
        "The storyboard sheet(s) at print resolution.",
        "Path of the PDF, or empty when save_pdf is off.",
    )
    FUNCTION = "draw"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Storyboard sheet: numbered panels, camera arrows from the camera field "
                   "(push-in = frame lines, pan = arrow, tilt = vertical arrow), captions.")

    def draw(self, keyframes, shot_list_json, columns, paper, save_pdf, filename_prefix,
             timing_json=None, brand=None, date=None):
        try:
            P = tr.build_project(_frames(keyframes), _one(shot_list_json, ""),
                                 brand=_one(brand, ""), timing_json=_one(timing_json),
                                 date_text=_one(date, ""))
            doc = tr.render_storyboard(P, _one(paper, "A4 landscape"), int(_one(columns, 4)))
            path = ""
            if _one(save_pdf, True):
                path = tr.save_pdf(doc, output_path(_one(filename_prefix, "storyboard")))
        except tr.TreatmentError as e:
            raise _err("Storyboard Sheet", e) from e
        return (_to_tensor(doc.pages), path)


class PaletteFromShots:
    """Colour palette of the keyframes, with an accent and a readable text colour on it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframes": ("IMAGE", {"tooltip": TIP["keyframes"]}),
                "colors": ("INT", {"default": 6, "min": 1, "max": 12,
                    "tooltip": "Maximum number of swatches (near-identical colours are "
                               "merged, so you may get fewer)."}),
            },
            "optional": {
                "accent_hex": ("STRING", {"default": "",
                    "tooltip": "Force an accent colour. Empty = the most colourful swatch "
                               "that is present enough."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("swatches", "palette_json", "accent_hex", "text_on_accent_hex")
    OUTPUT_TOOLTIPS = (
        "Palette card image (1600 x 900).",
        "Swatches as JSON: hex, rgb, Lab, name and share of the picture.",
        "Accent colour (connect to Treatment Builder / Pitch Deck Builder accent_hex).",
        "Black or white text colour that passes WCAG AA on the accent.",
    )
    FUNCTION = "extract"
    CATEGORY = CATEGORY
    DESCRIPTION = "Extracts a named colour palette from the keyframes (k-means in Lab)."

    def extract(self, keyframes, colors, accent_hex=None):
        frames = _frames(keyframes)
        if not frames:
            raise RuntimeError("Palette From Shots: no keyframes connected")
        sw = pal.extract_palette(frames, int(_one(colors, 6)))
        acc = (_one(accent_hex, "") or "").strip()
        if acc:
            try:
                acc = pal.rgb_to_hex(pal.hex_to_rgb(acc))
            except ValueError as e:
                raise RuntimeError(f"Palette From Shots: {e}") from e
        summary = pal.palette_summary(sw, acc or None)
        card = tr.render_palette_card(sw, summary["accent_hex"], 1600)
        return (_to_tensor([card]), json.dumps(summary, indent=2), summary["accent_hex"],
                summary["text_on_accent_hex"])


TREATMENT_NODE_CLASS_MAPPINGS = {
    "TreatmentBuilder": TreatmentBuilder,
    "PitchDeckBuilder": PitchDeckBuilder,
    "StoryboardSheet": StoryboardSheet,
    "PaletteFromShots": PaletteFromShots,
}
TREATMENT_NODE_DISPLAY_NAME_MAPPINGS = {
    "TreatmentBuilder": "Treatment Builder",
    "PitchDeckBuilder": "Pitch Deck Builder",
    "StoryboardSheet": "Storyboard Sheet",
    "PaletteFromShots": "Palette From Shots",
}
