"""Treatment & Pitch Deck: layout and rendering (PIL + numpy + stdlib, no ComfyUI).

Turns a finished pre-production (shot list, keyframes, critic verdicts, timing, supers, VO,
brand info) into the documents an agency sends a client before the shoot:

* a director's treatment (portrait A4 / US Letter, rendered at 300 dpi),
* a 16:9 pitch deck (1920x1080 or 3840x2160),
* a print-ready storyboard sheet,
* a palette card,

and writes multi-page PDFs with Pillow's own PDF writer (no new dependencies).

Units and grid
--------------
Every layout is written in PostScript points (1 pt = 1/72 in) and converted to pixels once,
at draw time (``Canvas.u`` = pixels per point). A4 is 595 x 842 pt, Letter 612 x 792 pt, a
deck slide is 960 x 540 pt (13.33 x 7.5 in, the standard 16:9 slide), so one layout renders
at any resolution. The grid is stated in the constants below: 12 columns, fixed gutters and
margins per template, and every text baseline sits on a ``BASELINE`` grid measured from the
top margin; leadings are whole multiples of it.

Typography
----------
Display: Instrument Serif (regular + italic); text: Inter (variable, weight and optical size
axes), both SIL OFL, in ./fonts. Missing fonts fall back to DejaVu, then to PIL's default.
Text is wrapped without hyphenation (a word wider than the measure is broken between
letters, never with an inserted hyphen), the last line of a paragraph never holds a single
word when it can be avoided (widow control), display titles are balanced and shrink to fit
their box, and an ellipsis is used only when the smallest allowed size still does not fit.
"""

import datetime as _dt
import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:  # package import inside ComfyUI, plain import in tests / scripts
    from . import palette as pal
except ImportError:  # pragma: no cover - exercised by the test runner
    import palette as pal


class TreatmentError(ValueError):
    """A clear, user-facing problem with the inputs."""


# =========================================================================== constants / grid

PT_PER_INCH = 72.0
MM = PT_PER_INCH / 25.4                       # points per millimetre

TREATMENT_DPI = 300
PAGE_SIZES_PT = {                             # portrait, in points
    "A4": (210 * MM, 297 * MM),               # 595.3 x 841.9
    "Letter": (8.5 * 72, 11 * 72),            # 612 x 792
}
STORYBOARD_PAPERS = {                         # (width pt, height pt, dpi)
    "A4 landscape": (297 * MM, 210 * MM, 300),
    "A3": (420 * MM, 297 * MM, 300),          # landscape
    "16:9": (960.0, 540.0, 288),              # 3840 x 2160 px
}
DECK_SIZE_PT = (960.0, 540.0)                 # 13.33 x 7.5 in
DECK_RESOLUTIONS = {"1920x1080": 1920, "3840x2160": 3840}

GRID_COLUMNS = 12
BASELINE = 3.0             # pt: every baseline of the treatment sits on this grid
DECK_BASELINE = 4.0        # pt: same for slides (larger type, coarser grid)
PREVIEW_WIDTH = 1240       # px: treatment page previews returned to ComfyUI
JPEG_QUALITY = 90          # PDF page images (DCT); visually lossless for layouts

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_FILES = {
    "serif": "InstrumentSerif-Regular.ttf",
    "serif-italic": "InstrumentSerif-Italic.ttf",
    "sans": "Inter-Variable.ttf",
}
FALLBACK_FILES = {
    "serif": ("DejaVuSerif.ttf",),
    "serif-italic": ("DejaVuSerif-Italic.ttf", "DejaVuSerif.ttf"),
    "sans": ("DejaVuSans.ttf",),
    "sans-bold": ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"),
}
FALLBACK_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu",
                 "/usr/share/fonts/TTF", "/Library/Fonts", "C:\\Windows\\Fonts")
ELLIPSIS = "\u2026"


# =========================================================================== fonts

_FONT_CACHE = {}


def _find_font_file(key):
    path = os.path.join(FONT_DIR, FONT_FILES.get(key, ""))
    if key in FONT_FILES and os.path.isfile(path):
        return path, True
    for name in FALLBACK_FILES.get(key, ()):
        for d in FALLBACK_DIRS:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p, False
        try:                                   # PIL searches system font folders itself
            ImageFont.truetype(name, 10)
            return name, False
        except OSError:
            pass
    return None, False


def font_names():
    """Names of the faces actually in use (for the production notes page)."""
    serif, own_s = _find_font_file("serif")
    sans, own_t = _find_font_file("sans")
    return ("Instrument Serif" if own_s else ("DejaVu Serif" if serif else "PIL default"),
            "Inter" if own_t else ("DejaVu Sans" if sans else "PIL default"))


def get_font(family, px, weight=400, italic=False):
    """A FreeType font at `px` pixels. family: "serif" or "sans". Inter's variable axes are
    set (weight, optical size); fallbacks pick a bold file for weight >= 600."""
    px = max(4, int(round(px)))
    key = (family, px, int(weight), bool(italic))
    f = _FONT_CACHE.get(key)
    if f is not None:
        return f
    if family == "serif":
        path, own = _find_font_file("serif-italic" if italic else "serif")
    else:
        path, own = _find_font_file("sans")
        if path and not own and weight >= 600:
            path, _ = _find_font_file("sans-bold")
    if path:
        f = ImageFont.truetype(path, px)
        if own and family == "sans":
            try:
                # optical size axis is in points at 1x; large sizes get the tighter display cut
                f.set_variation_by_axes([float(min(32, max(14, px * 0.5))), float(weight)])
            except (OSError, ValueError, AttributeError):  # FreeType built without MM
                pass
    else:
        try:
            f = ImageFont.load_default(size=px)
        except TypeError:  # very old Pillow
            f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


@dataclass(frozen=True)
class TextStyle:
    family: str = "sans"          # "serif" | "sans"
    size: float = 10.0            # pt
    weight: int = 400
    italic: bool = False
    leading: float = 1.4          # line pitch as a multiple of size (snapped to the grid)
    tracking: float = 0.0         # em; > 0 for spaced capitals
    caps: bool = False
    color: tuple = (0, 0, 0)

    def but(self, **kw):
        d = dict(self.__dict__)
        d.update(kw)
        return TextStyle(**d)


# =========================================================================== text engine

class Measure:
    """Measures strings for a style at a given pixel scale (u px per pt); results in pt."""

    def __init__(self, style, u):
        self.style, self.u = style, u
        self.font = get_font(style.family, style.size * u, style.weight, style.italic)
        self.track = style.tracking * style.size

    def width(self, text):
        if not text:
            return 0.0
        w = self.font.getlength(text) / self.u
        if self.track:
            w += self.track * (len(text) - 1)
        return w

    def metrics(self):
        a, d = self.font.getmetrics()
        return a / self.u, d / self.u


def _split_long_word(word, width, max_w):
    """Break a word wider than the measure between letters (no hyphen is inserted)."""
    if width(word) <= max_w or len(word) <= 1:
        return [word]
    out, cur = [], ""
    for ch in word:
        if cur and width(cur + ch) > max_w:
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def wrap_paragraph(text, width, max_w, widow=True):
    """Greedy, hyphen-free line breaking of one paragraph. Returns a list of word lists.

    Widow control: when the last line holds a single word, words are pulled down from the
    line above (as long as that line keeps at least two words and the result fits)."""
    words = []
    for w in str(text).split():
        words.extend(_split_long_word(w, width, max_w))
    lines, cur = [], []
    for w in words:
        if cur and width(" ".join(cur + [w])) > max_w:
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    if widow and len(lines) >= 2 and len(lines[-1]) == 1:
        prev, last = lines[-2], lines[-1]
        if len(prev) >= 3:
            moved = [prev[-1]] + last
            if width(" ".join(moved)) <= max_w:
                lines[-2], lines[-1] = prev[:-1], moved
    return lines


def wrap_text(text, width, max_w, widow=True):
    """Wrap text (paragraphs separated by newlines) into lines (strings)."""
    out = []
    for para in str(text or "").replace("\r", "").split("\n"):
        if not para.strip():
            if out and out[-1] != "":
                out.append("")
            continue
        out.extend(" ".join(l) for l in wrap_paragraph(para, width, max_w, widow))
    while out and out[-1] == "":
        out.pop()
    return out


def balance_lines(text, width, max_w, widow=True):
    """Wrap with the narrowest measure that keeps the same number of lines, so display
    titles get an even rag instead of a long first line and a stub."""
    lines = wrap_text(text, width, max_w, widow)
    if len(lines) < 2:
        return lines
    lo, hi = max_w * 0.45, max_w
    for _ in range(14):
        mid = (lo + hi) / 2
        trial = wrap_text(text, width, mid, widow)
        if len(trial) <= len(lines) and max(width(l) for l in trial) <= mid + 1e-6:
            hi = mid
        else:
            lo = mid
    return wrap_text(text, width, hi, widow)


def ellipsize(line, width, max_w):
    """Trim a line word by word (then letter by letter) until line + ellipsis fits."""
    words = line.split()
    while words and width(" ".join(words) + ELLIPSIS) > max_w:
        words.pop()
    if words:
        return " ".join(words).rstrip(",;:.-\u2013\u2014") + ELLIPSIS
    s = line
    while s and width(s + ELLIPSIS) > max_w:
        s = s[:-1]
    return s + ELLIPSIS


def snap_up(v, unit):
    return math.ceil(v / unit - 1e-6) * unit


@dataclass
class TextBlock:
    lines: list
    style: TextStyle
    leading: float          # pt between baselines
    ascent: float
    descent: float
    widths: list
    truncated: bool = False

    @property
    def height(self):
        if not self.lines:
            return 0.0
        return self.ascent + self.leading * (len(self.lines) - 1) + self.descent

    @property
    def width(self):
        return max(self.widths) if self.widths else 0.0


def layout_text(text, style, max_w, u, max_h=None, max_lines=None, min_size=None,
                balance=False, widow=True, unit=BASELINE):
    """Lay text out in a box. Shrinks the size (down to min_size) until it fits max_h /
    max_lines; at the smallest size the last visible line is ellipsized (last resort)."""
    text = str(text or "").strip()
    if style.caps:
        text = text.upper()
    size = style.size
    min_size = style.size if min_size is None else min(min_size, style.size)
    while True:
        st = style.but(size=size)
        m = Measure(st, u)
        lines = (balance_lines if balance else wrap_text)(text, m.width, max_w, widow)
        lead = snap_up(size * st.leading, unit) if unit else size * st.leading
        asc, desc = m.metrics()
        blk = TextBlock(lines, st, lead, asc, desc, [m.width(l) for l in lines])
        fits = ((max_lines is None or len(lines) <= max_lines)
                and (max_h is None or blk.height <= max_h + 1e-6))
        if fits or size <= min_size + 1e-6:
            break
        size = max(min_size, size * 0.94)
    if not fits:
        keep = len(lines) if max_lines is None else min(len(lines), max_lines)
        if max_h is not None:
            while keep > 0 and asc + lead * (keep - 1) + desc > max_h + 1e-6:
                keep -= 1
        if keep < len(lines):
            lines = lines[:keep]
            if lines:
                lines[-1] = ellipsize(lines[-1], m.width, max_w)
            blk = TextBlock(lines, st, lead, asc, desc, [m.width(l) for l in lines], True)
    return blk


def fit_title(text, style, max_w, max_lines, u, min_size, max_h=None, unit=BASELINE):
    """Largest size (<= style.size) at which a display title fits in max_lines (balanced)."""
    return layout_text(text, style, max_w, u, max_h=max_h, max_lines=max_lines,
                       min_size=min_size, balance=True, unit=unit)


# =========================================================================== images

def to_pil(img):
    """numpy [H,W,3|4] float 0-1 / uint8, or PIL -> PIL RGB."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(pal.to_uint8_rgb(img), "RGB")


def saliency_map(img, side=96):
    """Cheap saliency on a small grey copy: local detail (gradient energy) plus brightness,
    blurred, times a soft centre prior. Returns a float [h,w] map."""
    a = pal.downsample(img, side) / 255.0
    g = a @ np.array([0.299, 0.587, 0.114])
    gx = np.abs(np.diff(g, axis=1, append=g[:, -1:]))
    gy = np.abs(np.diff(g, axis=0, append=g[-1:, :]))
    detail = gx + gy
    detail = detail / (detail.max() + 1e-6)
    bright = (g - g.min()) / (np.ptp(g) + 1e-6)
    s = 0.65 * detail + 0.35 * bright ** 2
    s = np.asarray(Image.fromarray((s * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(max(1.0, min(s.shape) / 14))), dtype=np.float64) / 255.0
    h, w = s.shape
    yy, xx = np.mgrid[0:h, 0:w]
    prior = np.exp(-(((xx / max(w - 1, 1)) - 0.5) ** 2 + ((yy / max(h - 1, 1)) - 0.5) ** 2) / 0.18)
    return s * (0.35 + 0.65 * prior)


def focal_point(img):
    """(fx, fy) in 0..1: the weighted centre of the most salient region."""
    s = saliency_map(img)
    h, w = s.shape
    thr = np.quantile(s, 0.9)
    m = np.where(s >= thr, s, 0.0) ** 2
    tot = m.sum()
    if tot <= 0 or not np.isfinite(tot):
        return 0.5, 0.5
    yy, xx = np.mgrid[0:h, 0:w]
    return (float((m * xx).sum() / tot + 0.5) / w, float((m * yy).sum() / tot + 0.5) / h)


def saliency_peaks(img, k=3):
    """Up to k salient points (fx, fy), spread apart (non-maximum suppression)."""
    s = saliency_map(img).copy()
    h, w = s.shape
    out = []
    r = max(2, int(min(h, w) * 0.22))
    for _ in range(k):
        y, x = np.unravel_index(int(np.argmax(s)), s.shape)
        if s[y, x] <= 0:
            break
        out.append(((x + 0.5) / w, (y + 0.5) / h))
        s[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = 0
    return out or [(0.5, 0.5)]


def crop_box(img_w, img_h, aspect, focal=(0.5, 0.5), zoom=1.0):
    """Largest (or 1/zoom of the largest) box of `aspect` (w/h) inside the image, centred
    on the focal point and clamped to the image. Returns integer (x0, y0, x1, y1)."""
    img_w, img_h = int(img_w), int(img_h)
    aspect = max(1e-3, float(aspect))
    if img_w / img_h > aspect:
        ch = float(img_h)
        cw = ch * aspect
    else:
        cw = float(img_w)
        ch = cw / aspect
    zoom = max(1.0, float(zoom))
    cw, ch = max(1.0, cw / zoom), max(1.0, ch / zoom)
    fx, fy = focal if focal else (0.5, 0.5)
    x0 = min(max(fx * img_w - cw / 2, 0.0), img_w - cw)
    y0 = min(max(fy * img_h - ch / 2, 0.0), img_h - ch)
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = min(img_w, x0 + int(round(cw))), min(img_h, y0 + int(round(ch)))
    return max(0, x0), max(0, y0), max(x0 + 1, x1), max(y0 + 1, y1)


def cover(img, w, h, focal=(0.5, 0.5), zoom=1.0):
    """Crop to the box's aspect (never stretch) around the focal point, then resize."""
    im = to_pil(img)
    w, h = max(1, int(w)), max(1, int(h))
    box = crop_box(im.width, im.height, w / h, focal, zoom)
    return im.resize((w, h), Image.LANCZOS, box=box, reducing_gap=3.0)


def fit_rect(aspect, x, y, w, h, align="center"):
    """Largest rect of `aspect` inside (x, y, w, h). align: center | top | left | topleft."""
    if w / h > aspect:
        nh, nw = h, h * aspect
    else:
        nw, nh = w, w / aspect
    nx = x if "left" in align else x + (w - nw) / 2
    ny = y if "top" in align else y + (h - nh) / 2
    return nx, ny, nw, nh


def placeholder_frame(w, h, number, colors):
    """Calm gradient card for a shot without a keyframe."""
    a, b = colors
    t = np.linspace(0, 1, int(h))[:, None, None]
    s = np.linspace(0, 1, int(w))[None, :, None]
    grad = np.array(a)[None, None] * (1 - 0.6 * t - 0.4 * s) + np.array(b)[None, None] * (0.6 * t + 0.4 * s)
    im = Image.fromarray(np.clip(grad, 0, 255).astype(np.uint8), "RGB")
    return im


def aa_layer(w, h, draw_fn, scale=4):
    """Anti-aliased RGBA layer: draw at `scale`x with draw_fn(draw, scale), downsample."""
    big = Image.new("RGBA", (max(1, int(w * scale)), max(1, int(h * scale))), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(big), scale)
    return big.resize((max(1, int(w)), max(1, int(h))), Image.LANCZOS)


# =========================================================================== data model

@dataclass
class Shot:
    number: int
    description: str = ""
    camera: str = ""
    duration: float = 2.0
    start: float = 0.0
    end: float = 2.0
    image: object = None          # PIL RGB or None
    focal: tuple = (0.5, 0.5)
    super_text: str = ""
    vo: str = ""
    verdict: dict = None          # {"score": 0-10 or None, "pass": bool|None, "issues": [...]}

    @property
    def headline(self):
        return first_clause(self.description) or f"Shot {self.number}"


@dataclass
class Project:
    brand: str
    title: str
    client_line: str
    director: str
    date: str
    logline: str
    tone_words: list
    approach: str
    style: str
    shots: list
    fps: float
    total_seconds: float
    palette: list
    accent: str                    # hex
    brand_rules: list
    pipeline: list                 # [(key, value)]
    logo: object = None            # PIL RGBA or None
    has_timing: bool = False
    has_verdicts: bool = False

    @property
    def hero(self):
        """Index of the hero keyframe: the best-scored shot, else the most detailed one."""
        scored = [(s.verdict or {}).get("score") for s in self.shots]
        if any(v is not None for v in scored):
            return max(range(len(self.shots)), key=lambda i: (scored[i] or -1, -i))
        return 0

    def frames(self):
        return [s.image for s in self.shots if s.image is not None]


DEFAULT_PIPELINE = "Keyframes: Qwen Image Edit + next-scene LoRA; QA: Shot Critic; Video: Wan"

DEFAULT_RULES = [
    "Clear space around the logo equals the height of its capital letters on every side.",
    "The logo appears on the end card only, never over action or faces.",
    "Supers are set in the display face, sentence case, two lines at most, inside the "
    "title-safe area.",
    "The accent colour is reserved for the call to action and the end card.",
    "Never recolour, stretch, outline or add effects to the logo.",
]

_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                 "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                 "seventeen", "eighteen", "nineteen", "twenty"]


def number_word(n):
    n = int(round(n))
    return _NUMBER_WORDS[n] if 0 <= n < len(_NUMBER_WORDS) else str(n)


def first_clause(text, max_words=9):
    parts = re.split(r"(?<=[.!?;:])\s|,\s|\s[-–—]\s", str(text or "").strip())
    s = parts[0].strip().rstrip(".!?;:,")
    if len(s.split()) < 3 and len(parts) > 1:      # "Overhead: the cup ..." -> both parts
        s = (s + ", " + parts[1].strip()).rstrip(".!?;:,")
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
    return s


def timecode(seconds, fps=24.0):
    """SMPTE-style non-drop timecode HH:MM:SS:FF (fps rounded to the nominal rate)."""
    nominal = max(1, int(round(float(fps or 24))))
    total = int(round(max(0.0, float(seconds)) * nominal))
    ff = total % nominal
    s = total // nominal
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}:{ff:02d}"


def short_duration(seconds):
    s = float(seconds)
    if abs(s - round(s)) < 0.05:
        return f"{int(round(s))} s"
    return f"{s:.1f} s"


# --------------------------------------------------------------------------- camera moves

CAMERA_MOVES = ("push-in", "pull-out", "pan-left", "pan-right", "tilt-up", "tilt-down",
                "track-left", "track-right", "crane-up", "crane-down", "orbit", "handheld",
                "static")

# storyboard arrow drawn for each move (None = no arrow)
CAMERA_ARROWS = {
    "push-in": "frames-in", "pull-out": "frames-out",
    "pan-left": "arrow-left", "pan-right": "arrow-right",
    "track-left": "arrow-left", "track-right": "arrow-right",
    "tilt-up": "arrow-up", "tilt-down": "arrow-down",
    "crane-up": "arrow-up", "crane-down": "arrow-down",
    "orbit": "arc", "handheld": None, "static": None,
}


def _direction(t, a, b, default):
    """'left'/'right' (or 'up'/'down') named in t; 'left to right' counts as right."""
    if re.search(rf"\b{a}\s+to\s+(the\s+)?{b}\b", t) or re.search(rf"\bto\s+(the\s+)?{b}\b", t):
        return b
    if re.search(rf"\b{b}\s+to\s+(the\s+)?{a}\b", t) or re.search(rf"\bto\s+(the\s+)?{a}\b", t):
        return a
    if re.search(rf"\b{a}(ward)?s?\b", t):
        return a
    if re.search(rf"\b{b}(ward)?s?\b", t):
        return b
    return default


def camera_move(text):
    """Canonical move for a free-text camera field ("slow push-in on the cup" -> push-in)."""
    t = str(text or "").lower().replace("_", " ")
    if re.search(r"push[\s-]*in|dolly[\s-]*in|zoom[\s-]*in|creep[\s-]*in|move[s]?\s+in\b|"
                 r"\bpush(es|ing)?\b", t):
        return "push-in"
    if re.search(r"pull[\s-]*(out|back)|dolly[\s-]*(out|back)|zoom[\s-]*out|\bpull(s|ing)?\b", t):
        return "pull-out"
    if re.search(r"\borbit|\barc(s|ing)?\b|\bcircl", t):
        return "orbit"
    if re.search(r"\bcrane|\bjib|\bpedestal|\bboom\b", t):
        return "crane-" + _direction(t, "down", "up", "up")
    if re.search(r"\btilt", t):
        return "tilt-" + _direction(t, "down", "up", "up")
    if re.search(r"\bpan(s|ning|ned)?\b|whip", t):
        return "pan-" + _direction(t, "left", "right", "right")
    if re.search(r"\btrack|\btruck|\bdolly|\btravel|\bslide|\blateral|\bsteadicam|follow", t):
        return "track-" + _direction(t, "left", "right", "right")
    if re.search(r"hand[\s-]*held|shoulder", t):
        return "handheld"
    return "static"


def camera_arrow(text):
    return CAMERA_ARROWS[camera_move(text)]


def camera_label(text):
    raw = str(text or "").strip()
    if raw:
        return raw[0].upper() + raw[1:]
    return "Static"


_MOVE_SENTENCE = {
    "static": "a locked-off camera that lets the light and the performance do the work",
    "push-in": "slow push-ins that draw us closer to the detail",
    "pull-out": "patient pull-backs that reveal where we are",
    "pan": "measured pans that follow the moment as it unfolds",
    "track": "gliding lateral moves that carry us through the space",
    "tilt": "vertical reveals that open each frame",
    "crane": "rising camera moves that lift the film",
    "orbit": "gentle arcs around the subject",
    "handheld": "an intimate handheld camera that breathes with the subject",
}


def approach_text(shots, tone_words, brand, total_seconds):
    """One paragraph of plain template sentences built from the shot list (no LLM)."""
    n = len(shots)
    # each clause becomes its own sentence, so capitals in the shot list stay correct
    clauses = [first_clause(s.description) for s in shots]
    clauses = [c[:1].upper() + c[1:] for c in clauses]
    parts = []
    if n == 1:
        parts.append(f"A single, held image: {clauses[0] or 'one frame'}.")
    else:
        parts.append(f"The film opens on its first image: {clauses[0] or 'a held frame'}.")
        mids = [c for c in clauses[1:-1] if c]
        if mids:
            shown = mids[:3]
            more = len(mids) - len(shown)
            word = "moment" if len(mids) == 1 else "moments"
            parts.append(f"From there it moves through {number_word(len(mids))} {word}.")
            parts.extend(f"{c}." for c in shown)
            parts.append(f"And {number_word(more)} more, each given room to breathe." if more
                         else ("It is" if len(mids) == 1 else "Each is") + " given room to breathe.")
    fam = [camera_move(s.camera).split("-")[0] for s in shots]
    fam = ["push-in" if f == "push" else "pull-out" if f == "pull" else f for f in fam]
    counts = {f: fam.count(f) for f in set(fam)}
    dom = max(sorted(counts), key=lambda f: counts[f])
    cam = _MOVE_SENTENCE.get(dom, _MOVE_SENTENCE["static"])
    if counts[dom] < max(2, math.ceil(n / 2)) and len(counts) > 1:
        cam = "a camera that changes gear with the story, from " + " to ".join(
            _MOVE_SENTENCE[f].split(" that ")[0] for f in sorted(counts, key=lambda f: -counts[f])[:2])
    tones = [t for t in tone_words if t]
    if tones:
        tl = tones[0] if len(tones) == 1 else ", ".join(tones[:-1]) + " and " + tones[-1]
        parts.append(f"The register is {tl[0].lower() + tl[1:]}, with {cam}.")
    else:
        parts.append(f"The camera language is simple: {cam}.")
    if n > 1:
        tail = f", so that {brand} is the last thing the audience holds on to" if brand else ""
        parts.append(f"The last image: {clauses[-1] or 'the final frame'}{tail}.")
    secs = int(round(total_seconds))
    parts.append(f"{number_word(n).capitalize()} {'shot' if n == 1 else 'shots'}, "
                 f"{number_word(secs) if secs <= 20 else secs} seconds, one idea.")
    return " ".join(parts)


# --------------------------------------------------------------------------- parsing

def _json(value, what):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except ValueError as e:
        raise TreatmentError(f"{what} is not valid JSON ({e})") from e


def parse_shot_list(shot_list_json):
    plan = _json(shot_list_json, "shot_list_json") or {}
    if isinstance(plan, list):
        plan = {"shots": plan}
    if not isinstance(plan, dict):
        raise TreatmentError("shot_list_json is not a Shot Planner list")
    shots = plan.get("shots") or []
    if not isinstance(shots, list):
        raise TreatmentError("shot_list_json: \"shots\" is not a list")
    return plan, [s if isinstance(s, dict) else {"description": str(s)} for s in shots]


def _num(v, default=None):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        if isinstance(v, str):
            m = re.search(r"\d+(\.\d+)?", v)
            if m:
                return float(m.group())
        return default


def parse_verdicts(verdicts_json):
    """Shot Critic / Shot Video QA verdicts -> {shot_number: {"score", "pass", "issues",
    "route"}}. Every field is optional; scores are normalised to 0-10."""
    v = _json(verdicts_json, "verdicts_json")
    if v is None:
        return {}
    if isinstance(v, dict):
        for key in ("verdicts", "shots", "results"):
            if isinstance(v.get(key), list):
                v = v[key]
                break
        else:
            if all(isinstance(x, dict) for x in v.values()):
                v = [dict(x, shot=x.get("shot", k)) for k, x in v.items()]
            else:
                v = [v]
    if not isinstance(v, list):
        raise TreatmentError("verdicts_json is not a list of per-shot verdicts")
    out = {}
    for i, item in enumerate(v):
        if not isinstance(item, dict):
            continue
        num = _num(item.get("shot", item.get("id")), i + 1)
        score = _num(item.get("score"))
        if score is not None:
            score = score * 10 if score <= 1.0 else score / 10 if score > 10 else score
            score = round(max(0.0, min(10.0, score)), 1)
        route = str(item.get("route") or "").strip().lower()
        ok = item.get("pass")
        if ok is None and route:
            ok = route in ("pass", "keep", "ok", "approve", "approved", "accept")
        issues = item.get("issues") or []
        if isinstance(issues, str):
            issues = [issues]
        issues = [str(x.get("text") or x.get("issue") or x) if isinstance(x, dict) else str(x)
                  for x in issues if x]
        out[int(num)] = {"score": score, "pass": None if ok is None else bool(ok),
                         "issues": issues, "route": route}
    return out


def parse_timing(timing_json):
    t = _json(timing_json, "timing_json")
    if t is None:
        return None
    if not isinstance(t, dict) or not isinstance(t.get("shots"), list):
        raise TreatmentError("timing_json is not Ad Timeline output (needs \"shots\")")
    return t


def parse_numbered_lines(text, n_shots, timing_shots=None, sequential=False):
    """'2 | text' lines -> {2: 'text'}. 'start-end | text' maps to the shot containing start.
    With sequential=True, plain lines are taken one per shot in order (VO scripts)."""
    out, plain = {}, []
    for raw in str(text or "").replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if "|" in line:
            key, val = (p.strip() for p in line.split("|", 1))
            m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)s?", key)
            if m and timing_shots:
                t0 = float(m.group(1))
                for s in timing_shots:
                    if s.start - 1e-6 <= t0 < s.end:
                        out[s.number] = (out.get(s.number, "") + " " + val).strip()
                        break
                continue
            k = _num(key)
            if k is not None and 1 <= int(k) <= max(1, n_shots):
                out[int(k)] = (out.get(int(k), "") + " " + val).strip()
        elif sequential:
            plain.append(line)
    if sequential and plain and not out:
        for i, line in enumerate(plain[:n_shots]):
            out[i + 1] = line
    return out


def parse_list_lines(text):
    """Bullet / numbered lines -> clean list of strings."""
    out = []
    for raw in str(text or "").replace("\r", "").split("\n"):
        s = re.sub(r"^\s*(?:[-*•–—]|\d+[.)])\s*", "", raw).strip()
        if s:
            out.append(s)
    return out


def parse_pipeline(text):
    """'Keyframes: X; QA: Y' (semicolons or newlines) -> [("Keyframes", "X"), ("QA", "Y")]."""
    out = []
    for part in re.split(r"[;\n]+", str(text or "")):
        part = part.strip().lstrip("-*• ").strip()
        if not part:
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            out.append((k.strip(), v.strip()))
        else:
            out.append(("", part))
    return out


def parse_tone_words(text):
    return [w.strip() for w in re.split(r"[,;/\n·]+", str(text or "")) if w.strip()]


def format_date(date_text=""):
    s = str(date_text or "").strip()
    if s:
        return s
    d = _dt.date.today()
    return f"{d.day} {d.strftime('%B')} {d.year}"


def logo_from_arrays(rgb, mask=None, invert_mask=False):
    """Logo IMAGE ([H,W,3] float) + optional MASK (ComfyUI: 1 = transparent) -> PIL RGBA,
    trimmed to its visible pixels."""
    rgb8 = pal.to_uint8_rgb(rgb)
    h, w = rgb8.shape[:2]
    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim == 3:
            m = m[0]
        if m.shape != (h, w):
            m = np.asarray(Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).resize(
                (w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        alpha = m if invert_mask else 1.0 - m
    else:
        alpha = np.ones((h, w), np.float32)
    a8 = (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(np.dstack([rgb8, a8]), "RGBA")
    bbox = im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    return im.crop(bbox) if bbox else im


def build_project(keyframes, shot_list_json="", brand="", client_line="", director="",
                  logline="", tone_words="", verdicts_json=None, timing_json=None,
                  supers_text="", voiceover_text="", brand_rules="", logo=None,
                  accent_hex="", pipeline_notes=DEFAULT_PIPELINE, date_text="",
                  palette_colors=6, title=""):
    """Normalise every input into a Project. keyframes: list of numpy/PIL images."""
    frames = [to_pil(k) for k in (keyframes or [])]
    plan, raw = parse_shot_list(shot_list_json)
    if not raw:
        raw = [{"description": ""} for _ in frames]
    if not raw:
        raise TreatmentError("nothing to lay out: no shots in shot_list_json and no keyframes")
    timing = parse_timing(timing_json)
    fps = float(_num((timing or {}).get("fps"), 24.0) or 24.0)
    tshots = {}
    for i, t in enumerate((timing or {}).get("shots") or []):
        if isinstance(t, dict):
            tshots[int(_num(t.get("shot"), i + 1))] = t
    shots, cursor = [], 0.0
    for i, s in enumerate(raw):
        num = i + 1
        dur = _num(s.get("duration"), None)
        t = tshots.get(num)
        if t is not None:
            start = _num(t.get("start_s"), cursor)
            end = _num(t.get("end_s"), None)
            d = _num(t.get("duration_s"), None)
            end = end if end is not None else start + (d if d else dur or 2.0)
        else:
            start, end = cursor, cursor + (dur if dur and dur > 0 else 2.0)
        end = max(end, start + 1.0 / fps)
        cursor = end
        img = frames[i] if i < len(frames) else None
        shots.append(Shot(number=num, description=str(s.get("description") or "").strip(),
                          camera=str(s.get("camera") or "").strip(), duration=end - start,
                          start=start, end=end, image=img,
                          focal=focal_point(img) if img is not None else (0.5, 0.5)))
    total = float(_num((timing or {}).get("total_seconds"), None) or shots[-1].end)
    total = max(total, shots[-1].end)
    for k, v in parse_numbered_lines(supers_text, len(shots), shots).items():
        shots[k - 1].super_text = v
    for k, v in parse_numbered_lines(voiceover_text, len(shots), shots, sequential=True).items():
        shots[k - 1].vo = v
    verdicts = parse_verdicts(verdicts_json)
    for s in shots:
        s.verdict = verdicts.get(s.number)
    swatches = pal.extract_palette(frames, palette_colors) if frames else [
        {"hex": "#8A8178", "rgb": [138, 129, 120], "lab": [54.6, 2.4, 5.9],
         "name": "warm grey", "share": 1.0}]
    accent = accent_hex.strip() if accent_hex and str(accent_hex).strip() else ""
    if accent:
        accent = pal.rgb_to_hex(pal.hex_to_rgb(accent))
    else:
        accent = pal.pick_accent(swatches)["hex"]
    brand = str(brand or "").strip()
    title = str(title or plan.get("title") or brand or "Untitled").strip()
    tones = parse_tone_words(tone_words)
    style = str(plan.get("style") or "").strip()
    logline = str(logline or "").strip()
    if not logline:
        logline = f"{title}. {style[0].upper() + style[1:]}." if style else (
            f"{title} — a {short_duration(total).replace(' s', '-second')} film"
            + (f" for {brand}." if brand else "."))
    return Project(
        brand=brand, title=title, client_line=str(client_line or "").strip(),
        director=str(director or "").strip(), date=format_date(date_text), logline=logline,
        tone_words=tones, approach=approach_text(shots, tones, brand, total), style=style,
        shots=shots, fps=fps, total_seconds=total, palette=swatches, accent=accent,
        brand_rules=parse_list_lines(brand_rules) or list(DEFAULT_RULES),
        pipeline=parse_pipeline(pipeline_notes if pipeline_notes is not None else DEFAULT_PIPELINE),
        logo=logo, has_timing=timing is not None, has_verdicts=bool(verdicts))


# =========================================================================== templates

def _rgb(h):
    return pal.hex_to_rgb(h)


@dataclass
class Theme:
    """A template: colour, type scale and layout choices (not just a colour swap)."""
    name: str
    bg: tuple
    fg: tuple
    muted: tuple
    faint: tuple            # hairlines
    card: tuple             # panels, placeholders
    display: str            # "serif" | "sans"
    display_weight: int
    cover: str              # bleed | framed | split
    shots: str              # side | alternate | stacked
    shots_per_page: int
    header: str             # chapter | numeral | index
    margins: tuple          # doc margins pt (left, right, top, bottom)
    gutter: float           # doc gutter pt
    deck_margin: float
    scale: dict             # doc type sizes pt
    deck_scale: dict        # slide type sizes pt
    mood: list              # doc mood-board tiles (col, row, cols, rows) on 12 x 12
    deck_mood: list         # slide mood-board tiles on 12 x 6
    label_tracking: float = 0.16
    tone_pills: bool = False


TEMPLATES = {
    "editorial dark": Theme(
        name="editorial dark", bg=_rgb("#13110F"), fg=_rgb("#EEE7DB"), muted=_rgb("#9C9388"),
        faint=_rgb("#3B3631"), card=_rgb("#1E1B18"), display="serif", display_weight=400,
        cover="bleed", shots="side", shots_per_page=3, header="chapter",
        margins=(46, 46, 50, 56), gutter=12, deck_margin=56,
        scale=dict(cover=66, cover_min=30, h1=42, h2=21, h3=14, body=9.3, small=7.4,
                   label=6.2, numeral=26, folio=5.8),
        deck_scale=dict(cover=78, cover_min=36, h1=46, h2=28, h3=18, body=13, small=10.5,
                        label=8.4, numeral=40, folio=8),
        mood=[(0, 0, 8, 7), (8, 0, 4, 4), (8, 4, 4, 3), (0, 7, 4, 5), (4, 7, 3, 5), (7, 7, 5, 5)],
        deck_mood=[(0, 0, 6, 6), (6, 0, 3, 3), (9, 0, 3, 3), (6, 3, 4, 3), (10, 3, 2, 3)]),
    "editorial light": Theme(
        name="editorial light", bg=_rgb("#F3EEE5"), fg=_rgb("#1C1A17"), muted=_rgb("#6E665C"),
        faint=_rgb("#D8CFC1"), card=_rgb("#E6DED1"), display="serif", display_weight=400,
        cover="framed", shots="alternate", shots_per_page=3, header="numeral",
        margins=(52, 52, 54, 60), gutter=14, deck_margin=60,
        scale=dict(cover=74, cover_min=32, h1=50, h2=23, h3=15, body=9.8, small=7.5,
                   label=6.2, numeral=78, folio=5.8),
        deck_scale=dict(cover=86, cover_min=40, h1=54, h2=30, h3=19, body=13.5, small=10.5,
                        label=8.4, numeral=96, folio=8),
        mood=[(0, 0, 12, 5), (0, 5, 5, 7), (5, 5, 7, 4), (5, 9, 3, 3), (8, 9, 4, 3)],
        deck_mood=[(0, 0, 4, 6), (4, 0, 8, 4), (4, 4, 3, 2), (7, 4, 5, 2)]),
    "agency minimal": Theme(
        name="agency minimal", bg=_rgb("#FFFFFF"), fg=_rgb("#111111"), muted=_rgb("#6B6B6B"),
        faint=_rgb("#E4E4E4"), card=_rgb("#F2F2F2"), display="sans", display_weight=600,
        cover="split", shots="stacked", shots_per_page=2, header="index",
        margins=(60, 60, 64, 66), gutter=16, deck_margin=72,
        scale=dict(cover=36, cover_min=20, h1=24, h2=15.5, h3=11.5, body=8.8, small=7,
                   label=6.2, numeral=11, folio=5.8),
        deck_scale=dict(cover=50, cover_min=26, h1=32, h2=22, h3=15, body=12.5, small=10,
                        label=8.4, numeral=14, folio=8),
        mood=[(0, 0, 7, 6), (7, 0, 5, 6), (0, 6, 4, 6), (4, 6, 8, 6)],
        deck_mood=[(0, 0, 5, 6), (5, 0, 7, 3), (5, 3, 3, 3), (8, 3, 4, 3)],
        label_tracking=0.09, tone_pills=True),
}
TEMPLATE_NAMES = list(TEMPLATES)


def get_theme(name):
    t = TEMPLATES.get(str(name or "").strip().lower())
    if t is None:
        raise TreatmentError(f"unknown template {name!r}; choose one of {', '.join(TEMPLATES)}")
    return t


class Styles:
    """The type hierarchy of one document, derived from a theme's scale."""

    def __init__(self, T, sc):
        disp = dict(family=T.display, weight=T.display_weight)
        serifish = T.display == "serif"
        self.cover = TextStyle(size=sc["cover"], leading=1.0 if serifish else 1.08,
                               color=T.fg, **disp)
        self.cover_min = sc["cover_min"]
        self.h1 = TextStyle(size=sc["h1"], leading=1.02 if serifish else 1.12, color=T.fg, **disp)
        self.h2 = TextStyle(family=T.display, size=sc["h2"], italic=serifish,
                            weight=T.display_weight if not serifish else 400,
                            leading=1.2 if serifish else 1.3, color=T.fg)
        self.h3 = TextStyle(size=sc["h3"], leading=1.2, color=T.fg, **disp)
        self.body = TextStyle(size=sc["body"], leading=1.5, color=T.fg)
        self.body_muted = self.body.but(color=T.muted)
        self.small = TextStyle(size=sc["small"], leading=1.45, color=T.muted)
        self.label = TextStyle(size=sc["label"], weight=600, caps=True,
                               tracking=T.label_tracking, leading=1.5, color=T.muted)
        self.numeral = TextStyle(family=T.display, size=sc["numeral"],
                                 weight=T.display_weight, italic=serifish, leading=1.0,
                                 color=T.fg)
        self.folio = TextStyle(size=sc["folio"], weight=500, caps=True, tracking=0.14,
                               color=T.muted, leading=1.4)
        self.mono = TextStyle(size=sc["small"], weight=500, leading=1.45, color=T.muted)


# =========================================================================== canvas

class Canvas:
    """A page in points. Every placed element is recorded as a box so tests can prove
    that nothing overlaps (layer "content") and that everything stays on the page."""

    def __init__(self, w_pt, h_pt, u, bg, unit=BASELINE):
        self.W, self.H, self.u, self.unit = float(w_pt), float(h_pt), float(u), float(unit)
        self.im = Image.new("RGB", (max(1, round(w_pt * u)), max(1, round(h_pt * u))), tuple(bg))
        self.draw = ImageDraw.Draw(self.im)
        self.bg = tuple(bg)
        self.boxes = []          # (layer, (x0, y0, x1, y1), tag)

    def px(self, v):
        return int(round(v * self.u))

    def record(self, x, y, w, h, tag, layer="content"):
        if w > 0 and h > 0:
            self.boxes.append((layer, (x, y, x + w, y + h), tag))

    # ---- shapes
    def rect(self, x, y, w, h, fill, tag=None, layer="content"):
        if w <= 0 or h <= 0:
            return
        self.draw.rectangle([self.px(x), self.px(y), self.px(x + w) - 1, self.px(y + h) - 1],
                            fill=tuple(fill))
        if tag:
            self.record(x, y, w, h, tag, layer)

    def hline(self, x0, x1, y, fill, weight=0.5):
        t = max(1, self.px(weight))
        y0 = self.px(y)
        self.draw.rectangle([self.px(x0), y0, self.px(x1) - 1, y0 + t - 1], fill=tuple(fill))

    def vline(self, x, y0, y1, fill, weight=0.5):
        t = max(1, self.px(weight))
        x0 = self.px(x)
        self.draw.rectangle([x0, self.px(y0), x0 + t - 1, self.px(y1) - 1], fill=tuple(fill))

    def paste_rgba(self, rgba, x, y):
        self.im.paste(rgba, (self.px(x), self.px(y)), rgba)

    def shape(self, x, y, w, h, fn, tag=None, layer="overlay"):
        """Anti-aliased vector drawing: fn(draw, s, wpx, hpx) on a supersampled layer."""
        wp, hp = max(1, self.px(x + w) - self.px(x)), max(1, self.px(y + h) - self.px(y))
        layer_im = aa_layer(wp, hp, lambda d, s: fn(d, s, wp * s, hp * s))
        self.im.paste(layer_im, (self.px(x), self.px(y)), layer_im)
        if tag:
            self.record(x, y, w, h, tag, layer)

    def pill(self, x, y, w, h, fill, outline=None, weight=0.6, tag=None, layer="content"):
        def fn(d, s, W, H):
            ow = max(1, int(round(weight * self.u * s))) if outline else 0
            d.rounded_rectangle([0, 0, W - 1, H - 1], radius=H / 2, fill=fill,
                                outline=outline, width=ow)
        self.shape(x, y, w, h, fn, tag, layer)

    def dot(self, cx, cy, r, fill):
        self.shape(cx - r, cy - r, 2 * r, 2 * r,
                   lambda d, s, W, H: d.ellipse([0, 0, W - 1, H - 1], fill=fill))

    def gradient(self, x, y, w, h, color, a0, a1, ease=True):
        """Vertical alpha ramp of a solid colour (a0 at top, a1 at bottom)."""
        wp, hp = self.px(x + w) - self.px(x), self.px(y + h) - self.px(y)
        if wp <= 0 or hp <= 0:
            return
        t = np.linspace(0, 1, hp)
        if ease:
            t = t * t * (3 - 2 * t)
        a = (a0 + (a1 - a0) * t) * 255
        alpha = np.repeat(np.clip(a, 0, 255).astype(np.uint8)[:, None], wp, axis=1)
        solid = Image.new("RGB", (wp, hp), tuple(color))
        self.im.paste(solid, (self.px(x), self.px(y)), Image.fromarray(alpha, "L"))

    # ---- images
    def image(self, img, x, y, w, h, focal=(0.5, 0.5), zoom=1.0, tag="image", layer="content",
              placeholder=None):
        x0, y0, x1, y1 = self.px(x), self.px(y), self.px(x + w), self.px(y + h)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return
        if img is None:
            a, b = placeholder or ((60, 56, 52), (110, 100, 92))
            tile = placeholder_frame(x1 - x0, y1 - y0, 0, (a, b))
        else:
            tile = cover(img, x1 - x0, y1 - y0, focal, zoom)
        self.im.paste(tile, (x0, y0))
        self.record(x, y, w, h, tag, layer)

    # ---- text
    def layout(self, text, style, w, max_h=None, max_lines=None, min_size=None, balance=False,
               widow=True):
        # one grid unit is reserved because the first baseline snaps down onto the grid
        mh = None if max_h is None else max(0.0, max_h - self.unit)
        return layout_text(text, style, max(1.0, w), self.u, max_h=mh, max_lines=max_lines,
                           min_size=min_size, balance=balance, widow=widow, unit=self.unit)

    def _baseline0(self, blk, y, anchor):
        if anchor == "baseline":                       # y is the exact first baseline
            return y
        if anchor == "bottom":
            last = math.floor((y - blk.descent) / self.unit + 1e-6) * self.unit
            return last - blk.leading * (len(blk.lines) - 1)
        return snap_up(y + blk.ascent, self.unit)

    def put(self, blk, x, y, w=None, align="left", anchor="top", tag="text", layer="content",
            color=None):
        """Draw a laid-out block with its top (or bottom) at y. Returns the box bottom."""
        if not blk.lines:
            return y
        st = blk.style
        font = get_font(st.family, st.size * self.u, st.weight, st.italic)
        fill = tuple(color or st.color)
        b0 = self._baseline0(blk, y, anchor)
        w = blk.width if w is None else w
        xs = []
        for i, (line, lw) in enumerate(zip(blk.lines, blk.widths)):
            lx = x + (w - lw) / 2 if align == "center" else x + w - lw if align == "right" else x
            xs.append(lx)
            by = self.px(b0 + i * blk.leading)
            if st.tracking:
                cx = lx * self.u
                tr = st.tracking * st.size * self.u
                for ch in line:
                    self.draw.text((cx, by), ch, font=font, fill=fill, anchor="ls")
                    cx += font.getlength(ch) + tr
            else:
                self.draw.text((self.px(lx), by), line, font=font, fill=fill, anchor="ls")
        top = b0 - blk.ascent
        bottom = b0 + blk.leading * (len(blk.lines) - 1) + blk.descent
        bx = min(xs)
        self.record(bx, top, max(xs[i] + blk.widths[i] for i in range(len(xs))) - bx,
                    bottom - top, tag, layer)
        return bottom

    def write(self, text, style, x, y, w, max_h=None, max_lines=None, min_size=None,
              balance=False, align="left", anchor="top", tag="text", layer="content",
              widow=True):
        blk = self.layout(text, style, w, max_h, max_lines, min_size, balance, widow)
        bottom = self.put(blk, x, y, w, align, anchor, tag, layer)
        return blk, bottom

    def block_height(self, blk):
        """Height the block takes from a grid-aligned top (includes the baseline snap)."""
        if not blk.lines:
            return 0.0
        return snap_up(blk.ascent, self.unit) + blk.leading * (len(blk.lines) - 1) + blk.descent


@dataclass
class Grid:
    W: float
    H: float
    ml: float
    mr: float
    mt: float
    mb: float
    gutter: float
    cols: int = GRID_COLUMNS

    @property
    def cw(self):
        return self.W - self.ml - self.mr

    @property
    def col_w(self):
        return (self.cw - self.gutter * (self.cols - 1)) / self.cols

    def x(self, col):
        return self.ml + col * (self.col_w + self.gutter)

    def span(self, col, n):
        """(x, width) of n columns starting at col."""
        return self.x(col), n * self.col_w + (n - 1) * self.gutter

    @property
    def bottom(self):
        return self.H - self.mb


@dataclass
class Ctx:
    P: Project
    T: Theme
    S: Styles
    G: Grid
    accent: tuple            # fills (bars, pills, swatch marks)
    accent_text: tuple       # accent reaching AA (large text) on the page background
    on_accent: tuple         # text on an accent fill
    kind: str = "doc"        # doc | deck
    page: int = 0
    total: int = 0


def make_ctx(P, T, kind, w, h):
    sc = T.scale if kind == "doc" else T.deck_scale
    if kind == "doc":
        ml, mr, mt, mb = T.margins
        G = Grid(w, h, ml, mr, mt, mb, T.gutter)
    else:
        m = T.deck_margin
        G = Grid(w, h, m, m, m * 0.8, m * 0.8, T.gutter * 1.5)
    acc = _rgb(P.accent)
    acc_text = _rgb(pal.ensure_contrast(acc, T.bg, pal.AA_LARGE))
    on_acc = _rgb(pal.pick_text_color(acc, (pal.rgb_to_hex(T.bg if sum(T.bg) > 380 else (255, 255, 255)),
                                             "#111111")))
    return Ctx(P, T, Styles(T, sc), G, acc, acc_text, on_acc, kind)


# =========================================================================== components

def wordmark(c, ctx, x, y, h, color=None, align="left", w=None, tag="brand", knockout=None,
             surface=None, layer="content"):
    """Logo (scaled to height h pt) or the brand name set as a tracked wordmark. With
    knockout=None the logo is recoloured (solid `color`, default the text colour) when its
    own colours would not read on `surface` (default the page background)."""
    P, S = ctx.P, ctx.S
    if P.logo is not None and knockout is None:
        surf = tuple(surface or ctx.T.bg)
        luma = _logo_luma(P.logo)
        knockout = abs(luma - pal.relative_luminance(surf) ** (1 / 2.2)) < 0.35
    if P.logo is not None:
        lg = P.logo
        lw = h * lg.width / max(1, lg.height)
        if w is not None and lw > w:
            lw, h = w, w * lg.height / max(1, lg.width)
        px_w, px_h = max(1, c.px(lw)), max(1, c.px(h))
        im = lg.resize((px_w, px_h), Image.LANCZOS)
        if knockout:
            col = tuple(color or ctx.T.fg)
            im = Image.merge("RGBA", (*Image.new("RGB", im.size, col).split(), im.getchannel("A")))
        lx = x + (w - lw) / 2 if (align == "center" and w) else x + w - lw if (align == "right" and w) else x
        c.im.paste(im, (c.px(lx), c.px(y)), im)
        c.record(lx, y, lw, h, tag, layer)
        return y + h
    name = P.brand or P.title
    st = S.label.but(size=h * 0.62, weight=650, tracking=0.32, color=tuple(color or ctx.T.fg),
                     family="sans")
    blk = c.layout(name, st, w if w else c.W, max_lines=1, min_size=h * 0.35)
    return c.put(blk, x, y, w, align=align, tag=tag, layer=layer)


def folio(c, ctx):
    """Running foot: brand / title left, page number right, on every inner page."""
    G, S = ctx.G, ctx.S
    y = c.H - G.mb * 0.55
    left = " · ".join(x for x in (ctx.P.brand, ctx.P.title) if x)
    num = f"{ctx.page:02d} / {ctx.total:02d}"
    st = S.folio
    nb = c.layout(num, st, G.cw * 0.3, max_lines=1)
    c.put(nb, G.ml + G.cw - nb.width, y, anchor="bottom", tag="folio")
    c.write(left, st, G.ml, y, G.cw - nb.width - G.gutter * 2, max_lines=1, anchor="bottom",
            tag="folio")


def section_header(c, ctx, number, title, kicker=""):
    """Section opener. Returns the y where the page content starts."""
    G, S, T = ctx.G, ctx.S, ctx.T
    deck = ctx.kind == "deck"
    y = G.mt
    if T.header == "chapter":                         # editorial dark
        kick = kicker or (ctx.P.brand or "Treatment")
        lab = S.label
        b1, _ = c.write(f"{number:02d}", lab.but(color=ctx.accent_text), G.ml, y, G.cw / 3,
                        max_lines=1, tag="hdr-num")
        c.write(kick, lab, G.ml + G.cw / 3, y, G.cw * 2 / 3, max_lines=1, align="right",
                tag="hdr-kicker")
        y += c.block_height(b1) + S.h1.size * (0.5 if not deck else 0.35)
        blk, bottom = c.write(title, S.h1, G.ml, y, G.cw, max_lines=2, min_size=S.h1.size * 0.6,
                              balance=True, tag="hdr-title")
        y = bottom + S.h1.size * 0.35
        c.hline(G.ml, G.ml + G.cw, y, T.faint)
        return y + S.body.size * (2.2 if not deck else 1.8)
    if T.header == "numeral":                         # editorial light
        nb = c.layout(f"{number:02d}", S.numeral.but(color=ctx.accent_text), G.cw / 4, max_lines=1)
        nbottom = c.put(nb, G.ml, y, tag="hdr-num")
        tx = G.ml + nb.width + G.gutter * 1.5
        tw = G.ml + G.cw - tx
        tb = c.layout(title, S.h1, tw, max_lines=2, min_size=S.h1.size * 0.55, balance=True)
        # title sits on the numeral's baseline
        bottom = c.put(tb, tx, nbottom - nb.descent + tb.descent, anchor="bottom", tag="hdr-title")
        y = max(nbottom, bottom) + S.body.size * 0.9
        c.hline(G.ml, G.ml + G.cw, y, T.fg, 0.7)
        if kicker:
            kb = c.layout(kicker, S.label, G.cw, max_lines=1)
            c.put(kb, G.ml, y + S.label.size * 0.8, tag="hdr-kicker")
            y += c.block_height(kb) + S.label.size * 0.8
        return y + S.body.size * (2.4 if not deck else 1.8)
    # index (agency minimal): small accent bar, "02  Mood board" in semibold sans
    c.rect(G.ml, y, S.h1.size * 0.9, max(1.5, S.h1.size * 0.09), ctx.accent, tag="hdr-bar")
    y += S.h1.size * 0.6
    nb = c.layout(f"{number:02d}", S.h1.but(color=T.muted, weight=400), G.cw / 6, max_lines=1)
    c.put(nb, G.ml, y, tag="hdr-num")
    tx = G.ml + max(nb.width + G.gutter, G.span(0, 1)[1] + G.gutter)
    blk, bottom = c.write(title, S.h1, tx, y, G.ml + G.cw - tx, max_lines=2,
                          min_size=S.h1.size * 0.6, tag="hdr-title")
    y = bottom
    if kicker:
        _, y = c.write(kicker, S.small, tx, y + S.small.size * 0.6, G.ml + G.cw - tx,
                       max_lines=2, tag="hdr-kicker")
    return y + S.body.size * (3.2 if not deck else 2.4)


def label_value(c, ctx, label, value, x, y, w, max_lines=2, value_style=None):
    """Small-caps label over a value. Returns the bottom."""
    S = ctx.S
    lb, b = c.write(label, S.label, x, y, w, max_lines=1, tag="lv-label")
    vs = value_style or S.body
    _, b2 = c.write(value or "—", vs, x, b + S.label.size * 0.45, w, max_lines=max_lines,
                    min_size=vs.size * 0.85, tag="lv-value")
    return b2


def score_badge(c, ctx, verdict, x, y, align="right", scale=1.0):
    """'QA 8.4' pill drawn over a frame's corner; colour dot shows pass / reroll."""
    if not verdict:
        return
    score, ok = verdict.get("score"), verdict.get("pass")
    if score is None and ok is None:
        return
    S = ctx.S
    txt = f"QA {score:.1f}" if score is not None else ("QA pass" if ok else "QA check")
    st = S.label.but(color=(250, 248, 244), size=S.label.size * scale, tracking=0.08)
    blk = c.layout(txt, st, 400, max_lines=1)
    h = st.size * 2.5
    dot_r = st.size * 0.36
    w = blk.width + h + dot_r * 2 + st.size * 0.6
    bx = x - w if align == "right" else x
    c.pill(bx, y, w, h, (12, 11, 10, 190), tag="qa-badge", layer="overlay")
    col = (122, 196, 132) if ok or (ok is None and (score or 0) >= 7) else (232, 163, 64)
    c.dot(bx + h * 0.5 + dot_r * 0.2, y + h / 2, dot_r, col + (255,))
    c.put(blk, bx + h * 0.5 + dot_r * 2 + st.size * 0.35,
          y + h / 2 + st.size * 0.36, anchor="baseline", tag="qa-text", layer="overlay")


def arrow_poly(kind, W, H, s=1.0):
    """Shapes for a storyboard camera arrow inside a W x H panel (pixel coords)."""
    m = min(W, H)
    t = m * 0.045                                    # shaft half thickness
    hw, hl = m * 0.11, m * 0.14                      # head half width, head length
    cx, cy = W / 2, H / 2
    if kind in ("arrow-left", "arrow-right"):
        L = W * 0.56
        x0, x1 = cx - L / 2, cx + L / 2
        pts = [(x0, cy - t), (x1 - hl, cy - t), (x1 - hl, cy - hw), (x1, cy),
               (x1 - hl, cy + hw), (x1 - hl, cy + t), (x0, cy + t)]
        if kind == "arrow-left":
            pts = [(W - px, py) for px, py in pts]
        return [("poly", pts)]
    if kind in ("arrow-up", "arrow-down"):
        L = H * 0.6
        y0, y1 = cy + L / 2, cy - L / 2
        pts = [(cx - t, y0), (cx - t, y1 + hl), (cx - hw, y1 + hl), (cx, y1),
               (cx + hw, y1 + hl), (cx + t, y1 + hl), (cx + t, y0)]
        if kind == "arrow-down":
            pts = [(px, H - py) for px, py in pts]
        return [("poly", pts)]
    if kind in ("frames-in", "frames-out"):
        out = []
        for f in (0.82, 0.6, 0.4):
            fw, fh = W * f, H * f
            out.append(("rect", (cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2)))
        # small corner arrows on the diagonal show the direction of the move
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            a = (cx + sx * W * 0.39, cy + sy * H * 0.39)
            b = (cx + sx * W * 0.25, cy + sy * H * 0.25)
            tip, tail = (b, a) if kind == "frames-in" else (a, b)
            out.append(("line", (tail, tip)))
            out.append(("head", (tail, tip)))
        return out
    if kind == "arc":
        return [("arc", (cx - W * 0.3, cy - H * 0.22, cx + W * 0.3, cy + H * 0.22))]
    return []


def draw_camera_arrow(c, kind, x, y, w, h, color=(255, 255, 255), outline=(0, 0, 0)):
    """Classic storyboard notation over a panel (anti-aliased, white with a dark edge)."""
    if not kind:
        return

    def fn(d, s, W, H):
        shapes = arrow_poly(kind, W, H)
        lw = max(2, int(min(W, H) * 0.012))
        col, ol = tuple(color) + (235,), tuple(outline) + (150,)
        for typ, data in shapes:
            if typ == "poly":
                d.polygon(data, fill=col, outline=ol, width=lw)
            elif typ == "rect":
                d.rectangle(data, outline=ol, width=lw * 3)
                d.rectangle(data, outline=col, width=lw)
            elif typ == "line":
                d.line(data, fill=ol, width=lw * 3)
                d.line(data, fill=col, width=lw * 2)
            elif typ == "head":
                (x0, y0), (x1, y1) = data
                ang = math.atan2(y1 - y0, x1 - x0)
                L = min(W, H) * 0.05
                p1 = (x1 - L * math.cos(ang - 0.5), y1 - L * math.sin(ang - 0.5))
                p2 = (x1 - L * math.cos(ang + 0.5), y1 - L * math.sin(ang + 0.5))
                d.polygon([(x1, y1), p1, p2], fill=col, outline=ol)
            elif typ == "arc":
                x0, y0, x1, y1 = data
                d.arc(data, 200, 340, fill=ol, width=lw * 4)
                d.arc(data, 200, 340, fill=col, width=lw * 2)
                ex = (x0 + x1) / 2 + (x1 - x0) / 2 * math.cos(math.radians(340))
                ey = (y0 + y1) / 2 + (y1 - y0) / 2 * math.sin(math.radians(340))
                L = min(W, H) * 0.07
                d.polygon([(ex + L * 0.9, ey + L * 0.1), (ex - L * 0.3, ey - L * 0.6),
                           (ex - L * 0.1, ey + L * 0.7)], fill=col, outline=ol)

    c.shape(x, y, w, h, fn, tag="camera-arrow", layer="overlay")


# =========================================================================== shared blocks

def gradient_h(c, x, y, w, h, color, a0, a1):
    """Horizontal alpha ramp (a0 at left, a1 at right)."""
    wp, hp = c.px(x + w) - c.px(x), c.px(y + h) - c.px(y)
    if wp <= 0 or hp <= 0:
        return
    t = np.linspace(0, 1, wp)
    t = t * t * (3 - 2 * t)
    a = np.clip((a0 + (a1 - a0) * t) * 255, 0, 255).astype(np.uint8)
    alpha = np.repeat(a[None, :], hp, axis=0)
    c.im.paste(Image.new("RGB", (wp, hp), tuple(color)), (c.px(x), c.px(y)),
               Image.fromarray(alpha, "L"))


def placeholder_colors(ctx):
    sw = ctx.P.palette
    a = tuple(sw[0]["rgb"]) if sw else ctx.T.card
    b = tuple(sw[1]["rgb"]) if len(sw) > 1 else ctx.T.muted
    return a, b


def put_shot_image(c, ctx, shot, x, y, w, h, zoom=1.0, focal=None, tag="frame", layer="content"):
    c.image(shot.image if shot else None, x, y, w, h, focal or (shot.focal if shot else (0.5, 0.5)),
            zoom, tag, layer, placeholder_colors(ctx))


def frame_aspect(P):
    im = next((s.image for s in P.shots if s.image is not None), None)
    if im is None:
        return 16 / 9
    return min(2.4, max(0.5, im.width / im.height))


def meta_items(P, deck=False):
    items = [("Director", P.director or "—"), ("Date", P.date)]
    n = len(P.shots)
    items.append(("Running time", f"{short_duration(P.total_seconds)} · {n} "
                                  f"{'shot' if n == 1 else 'shots'}"))
    return items


def meta_row_height(c, ctx, items, w, cols):
    cw = (w - ctx.G.gutter * (cols - 1)) / cols
    h = 0.0
    for lab, val in items:
        lb = c.layout(lab, ctx.S.label, cw, max_lines=1)
        vb = c.layout(val, ctx.S.body, cw, max_lines=2, min_size=ctx.S.body.size * 0.85)
        h = max(h, c.block_height(lb) + ctx.S.label.size * 0.45 + c.unit + c.block_height(vb))
    return h


def meta_row(c, ctx, items, x, y, w, cols, color=None):
    cw = (w - ctx.G.gutter * (cols - 1)) / cols
    bottom = y
    for i, (lab, val) in enumerate(items):
        vs = ctx.S.body if color is None else ctx.S.body.but(color=color)
        bottom = max(bottom, label_value(c, ctx, lab, val, x + i * (cw + ctx.G.gutter), y, cw,
                                         value_style=vs))
    return bottom


def stack_fields(c, ctx, fields, x, y, w, bottom, gap=None):
    """Label/value fields stacked top-down inside [y, bottom]. A field gets whatever space is
    left after reserving one line for every field below it, so nothing ever overflows; a
    field with no room at all is dropped. fields: [(label, text, style, max_lines)]."""
    S = ctx.S
    gap = S.body.size * 1.1 if gap is None else gap
    fields = [f for f in fields if f[1]]

    def min_h(f):
        lb = c.layout("X", S.label, w, max_lines=1) if f[0] else None
        vb = c.layout("X", f[2], w, max_lines=1)
        return (c.block_height(lb) + S.label.size * 0.45 + c.unit if lb else 0) \
            + c.block_height(vb) + gap

    # earlier fields win: keep fields in order while their one-line minimum still fits
    keep, used = [], 0.0
    for f in fields:
        m = min_h(f)
        if used + m - gap <= bottom - y + 1e-6:
            keep.append(f)
            used += m
    fields = keep
    mins = [min_h(f) for f in fields]
    for i, (lab, text, style, max_lines) in enumerate(fields):
        avail = bottom - y - sum(mins[i + 1:])
        if avail < mins[i] - gap:
            continue
        if lab:
            _, y = c.write(lab, S.label, x, y, w, max_lines=1, tag="field-label")
            y += S.label.size * 0.45
        _, y = c.write(text, style, x, y, w, max_h=bottom - y - sum(mins[i + 1:]),
                       max_lines=max_lines, min_size=style.size * 0.85, tag="field")
        y += gap
    return y


def write_columns(c, text, style, x, y, w, ncols, gutter, max_h, tag="columns"):
    """Set text in balanced columns (lines split evenly, so no column ends on a widow)."""
    cw = (w - gutter * (ncols - 1)) / ncols
    blk = c.layout(text, style, cw, min_size=style.size)
    size = style.size
    while True:
        lines = len(blk.lines)
        per = math.ceil(lines / ncols) if lines else 0
        fit = max(1, int((max_h - c.unit - blk.ascent - blk.descent) // blk.leading) + 1)
        if per <= fit or size <= style.size * 0.86:
            break
        size *= 0.95
        blk = c.layout(text, style.but(size=size), cw)
    lines = list(blk.lines)
    if per > fit:                                     # last resort: ellipsis
        lines = lines[:fit * ncols]
        m = Measure(blk.style, c.u)
        lines[-1] = ellipsize(lines[-1] + " " + ELLIPSIS, m.width, cw)
        per = fit
    bottom = y
    for i in range(ncols):
        part = lines[i * per:(i + 1) * per]
        if not part:
            break
        m = Measure(blk.style, c.u)
        sub = TextBlock(part, blk.style, blk.leading, blk.ascent, blk.descent,
                        [m.width(l) for l in part])
        bottom = max(bottom, c.put(sub, x + i * (cw + gutter), y, tag=tag))
    return bottom


def ruler_step(total):
    for step in (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300):
        if total / step <= 16:
            return step
    return 600


def timeline_lanes(c, ctx, x, y, w, lane_label_w, compact=False):
    """Timing ruler with three lanes (picture / supers / VO). Returns bottom y."""
    P, S, T = ctx.P, ctx.S, ctx.T
    total = max(P.total_seconds, 1e-3)
    lx, lw = x + lane_label_w, w - lane_label_w
    step = ruler_step(total)
    # tick labels
    tick_st = S.mono.but(size=S.mono.size * 0.95)
    lab_blk = c.layout("0 s", tick_st, 60, max_lines=1)
    base = y + c.block_height(lab_blk)
    t = 0.0
    last_right = -1e9
    while t <= total + 1e-6:
        tx = lx + lw * t / total
        txt = f"{t:g} s"
        blk = c.layout(txt, tick_st, 80, max_lines=1)
        bx = min(max(tx - blk.width / 2, lx), lx + lw - blk.width)
        if bx > last_right + 4:
            c.put(blk, bx, base, anchor="bottom", tag="tick")
            last_right = bx + blk.width
        c.vline(tx, base + 3, base + 8, T.muted, 0.5)
        t += step
    y = base + 10
    c.hline(lx, lx + lw, y, T.muted, 0.5)
    y += 6
    lanes = [("Picture", 24 if not compact else 20), ("Supers", 7), ("VO", 7)]
    pic_tones = [pal.mix(T.bg, T.fg, 0.16), pal.mix(T.bg, T.fg, 0.26)]
    lab_h = c.layout("X", S.label, 50, max_lines=1)
    lab_h = lab_h.ascent + lab_h.descent + 1
    for name, h in lanes:
        h = max(h, lab_h)                      # a lane is never shorter than its label
        lb = c.layout(name, S.label, lane_label_w - 6, max_lines=1)
        c.put(lb, x, y + h / 2 + S.label.size * 0.36, anchor="baseline", tag="lane-label")
        for i, s in enumerate(P.shots):
            sx = lx + lw * s.start / total
            sw = max(0.5, lw * (s.end - s.start) / total - 1.0)
            if name == "Picture":
                c.rect(sx, y, sw, h, pic_tones[i % 2])
                nb = c.layout(f"{s.number}", S.label.but(color=T.fg), sw, max_lines=1)
                if nb.width < sw - 4:
                    c.put(nb, sx + 3, y + h / 2 + S.label.size * 0.36, anchor="baseline",
                          tag="lane-num", layer="overlay")
            elif name == "Supers" and s.super_text:
                c.rect(sx, y + h / 2 - 3, sw, 6, ctx.accent)
            elif name == "VO" and s.vo:
                c.rect(sx, y + h / 2 - 3, sw, 6, T.muted)
        c.record(lx, y, lw, h, f"lane-{name}")
        y += h + 5
    return y


# =========================================================================== treatment pages

def doc_cover(c, ctx):
    P, T, S, G = ctx.P, ctx.T, ctx.S, ctx.G
    hero = P.shots[P.hero]
    items = meta_items(P)
    kicker = P.client_line or "Director's treatment"
    if T.cover == "bleed":
        # landscape keyframes bleed off the top 2/3 and dissolve into the page, so the
        # subject stays above the title instead of being cropped to a sliver
        ih = c.H * (0.68 if frame_aspect(P) > 1.0 else 1.0)
        put_shot_image(c, ctx, hero, 0, 0, c.W, ih, tag="hero", layer="bg")
        c.gradient(0, 0, c.W, c.H * 0.2, T.bg, 0.6, 0.0)
        c.gradient(0, ih * 0.55, c.W, ih * 0.45 + 1, T.bg, 0.0, 1.0)
        wordmark(c, ctx, G.ml, G.mt, 20, w=G.span(0, 6)[1])
        c.write("Director's treatment", S.label.but(color=T.fg), G.x(6), G.mt + 4,
                G.span(6, 6)[1], max_lines=1, align="right", tag="cover-kind")
        mh = meta_row_height(c, ctx, items, G.cw, 3)
        meta_y = G.bottom - mh
        meta_row(c, ctx, items, G.ml, meta_y, G.cw, 3)
        rule_y = meta_y - 16
        c.hline(G.ml, G.ml + G.cw, rule_y, pal.mix(T.bg, T.fg, 0.45))
        tw = G.span(0, 11)[1]
        kb = c.layout(kicker, S.label.but(color=ctx.accent_text), tw, max_lines=2)
        title_bottom = rule_y - 20
        room = title_bottom - c.H * 0.42 - c.block_height(kb) - 14
        tb = fit_title(P.title, S.cover, tw, 3, c.u, S.cover_min, room, c.unit)
        top = c.put(tb, G.ml, title_bottom, anchor="bottom", tag="title") - tb.height
        c.put(kb, G.ml, top - 14 - c.block_height(kb), tag="kicker")
        return
    if T.cover == "framed":
        wordmark(c, ctx, G.ml, G.mt, 18, w=G.span(0, 6)[1])
        c.write("Director's treatment", S.label, G.x(6), G.mt + 3, G.span(6, 6)[1],
                max_lines=1, align="right", tag="cover-kind")
        iy = G.mt + 36
        ih = c.H * 0.5
        put_shot_image(c, ctx, hero, G.ml, iy, G.cw, ih, tag="hero")
        mh = meta_row_height(c, ctx, items, G.cw, 3)
        meta_y = G.bottom - mh
        meta_row(c, ctx, items, G.ml, meta_y, G.cw, 3)
        rule_y = meta_y - 14
        c.hline(G.ml, G.ml + G.cw, rule_y, T.fg, 0.7)
        y = iy + ih + 22
        _, y = c.write(kicker, S.label.but(color=ctx.accent_text), G.ml, y, G.cw, max_lines=2,
                       tag="kicker")
        y += 10
        tb = fit_title(P.title, S.cover, G.cw, 3, c.u, S.cover_min, rule_y - 14 - y, c.unit)
        c.put(tb, G.ml, y, tag="title")
        return
    # split (agency minimal)
    sx = c.W * 0.44
    put_shot_image(c, ctx, hero, sx, 0, c.W - sx, c.H, tag="hero")
    lw = sx - G.ml - G.gutter * 2
    y = wordmark(c, ctx, G.ml, G.mt, 16, w=lw)
    c.rect(G.ml, y + 18, 22, 3, ctx.accent, tag="accent-bar")
    bottom = G.bottom
    for lab, val in reversed([("Client", kicker)] + items):
        lb = c.layout(lab, S.label, lw, max_lines=1)
        vb = c.layout(val, S.body, lw, max_lines=2, min_size=S.body.size * 0.85)
        h = c.block_height(lb) + S.label.size * 0.45 + c.unit + c.block_height(vb)
        label_value(c, ctx, lab, val, G.ml, bottom - h, lw)
        bottom -= h + 12
    room = bottom - 20 - (y + 50)
    tb = fit_title(P.title, S.cover, lw, 5, c.u, S.cover_min, room, c.unit)
    c.put(tb, G.ml, bottom - 20, anchor="bottom", tag="title")


def doc_idea(c, ctx):
    P, T, S, G = ctx.P, ctx.T, ctx.S, ctx.G
    y = section_header(c, ctx, 1, "The idea")
    bottom = G.bottom
    area = bottom - y
    editorial = T.display == "serif"
    qx, qw = (G.span(1, 10) if editorial else G.span(0, 9))
    if editorial:
        mark = c.layout("“", S.h2.but(size=S.h2.size * 2.6, italic=False,
                                           color=ctx.accent_text), G.col_w, max_lines=1)
        mark_bottom = c.put(mark, G.ml, y - S.h2.size * 0.2, tag="quote-mark")
    else:
        mark_bottom = y
    _, y = c.write(P.logline, S.h2, qx, y, qw, max_h=area * 0.3, min_size=S.h2.size * 0.72,
                   tag="logline")
    y = max(y, mark_bottom)
    y += S.body.size * 2.6
    if P.tone_words:
        _, y = c.write("Tone", S.label, G.ml, y, G.cw, max_lines=1, tag="tone-label")
        y += S.label.size * 0.9
        if T.tone_pills:
            st = S.body.but(weight=500)
            px_, py_ = G.ml, y
            ph = st.size * 2.4
            for wd in P.tone_words:
                blk = c.layout(wd, st, G.cw - ph, max_lines=1)
                pw = blk.width + ph
                if px_ + pw > G.ml + G.cw:
                    px_, py_ = G.ml, py_ + ph + 6
                    if py_ > y + (ph + 6) * 1.5:
                        break
                c.pill(px_, py_, pw, ph, None, outline=T.fg, tag="tone-pill")
                c.put(blk, px_ + ph / 2, py_ + ph / 2 + st.size * 0.36, anchor="baseline",
                      tag="tone-word", layer="overlay")
                px_ += pw + 6
            y = py_ + ph
        else:
            _, y = c.write("  ·  ".join(P.tone_words), S.h3.but(italic=True),
                           G.ml, y, G.cw, max_lines=2, min_size=S.h3.size * 0.8, tag="tone")
        y += S.body.size * 2.6
    _, y = c.write("Approach", S.label, G.ml, y, G.cw, max_lines=1, tag="approach-label")
    y += S.label.size * 0.9
    room = bottom - y
    if editorial:
        y = write_columns(c, P.approach, S.body, G.ml, y, G.cw, 2, G.gutter * 2, room * 0.62,
                          tag="approach")
    else:
        x, w = G.span(0, 8)
        _, y = c.write(P.approach, S.body, x, y, w, max_h=room * 0.62,
                       min_size=S.body.size * 0.88, tag="approach")
    y += S.body.size * 2.4
    if bottom - y >= 80:
        shot = P.shots[1] if len(P.shots) > 1 else P.shots[0]
        h = min(bottom - y, G.cw / 1.9)
        put_shot_image(c, ctx, shot, G.ml, bottom - h, G.cw, h, tag="idea-frame")


def mood_views(P, tiles):
    """(shot, focal, zoom) per tile: big tiles get whole frames, small tiles get details
    cropped around saliency peaks, spreading across the film."""
    n = len(P.shots)
    order = [P.hero] + [i for i in np.linspace(0, n - 1, num=min(n, 12)).round().astype(int)
                        if i != P.hero]
    order = list(dict.fromkeys(int(i) for i in order))
    areas = [t[2] * t[3] for t in tiles]
    rank = sorted(range(len(tiles)), key=lambda i: -areas[i])
    views = [None] * len(tiles)
    big = max(areas)
    k = 0
    peaks = {}
    for r, ti in enumerate(rank):
        shot = P.shots[order[k % len(order)]]
        k += 1
        if r == 0 or (areas[ti] >= big * 0.45 and k <= len(order)):
            views[ti] = (shot, shot.focal, 1.0)
        else:
            if shot.number not in peaks:
                peaks[shot.number] = saliency_peaks(shot.image, 3) if shot.image else [(0.5, 0.5)]
            pk = peaks[shot.number]
            views[ti] = (shot, pk[(r + k) % len(pk)], 2.3)
    return views


def mood_board(c, ctx, tiles, x, y, w, h, rows, gap):
    cols = 12
    mw = (w - gap * (cols - 1)) / cols
    mh = (h - gap * (rows - 1)) / rows
    for (c0, r0, cs, rs), (shot, focal, zoom) in zip(tiles, mood_views(ctx.P, tiles)):
        tx, ty = x + c0 * (mw + gap), y + r0 * (mh + gap)
        tw, th = cs * mw + (cs - 1) * gap, rs * mh + (rs - 1) * gap
        put_shot_image(c, ctx, shot, tx, ty, tw, th, zoom, focal, tag="mood-tile")


def doc_mood(c, ctx):
    P, T, S, G = ctx.P, ctx.T, ctx.S, ctx.G
    y = section_header(c, ctx, 2, "Mood")
    cap = (f"Keyframes and details from {len(P.shots)} "
           f"{'shot' if len(P.shots) == 1 else 'shots'}. Details are cropped around the most "
           f"salient region of each frame.")
    cb = c.layout(cap, S.small, G.span(0, 8)[1], max_lines=2)
    ch = c.block_height(cb)
    c.put(cb, G.ml, G.bottom - ch, tag="caption")
    mood_board(c, ctx, T.mood, G.ml, y, G.cw, G.bottom - ch - 14 - y, 12, G.gutter * 0.5)


def swatch_rows(c, ctx, x, y, w, bottom):
    P, S, T = ctx.P, ctx.S, ctx.T
    n = len(P.palette)
    row_h = min(78.0, (bottom - y) / max(n, 1))
    chip_w = w * 0.17
    for i, sw in enumerate(P.palette):
        ry = y + i * row_h
        rgb = tuple(sw["rgb"])
        c.rect(x, ry + 4, chip_w, row_h - 8, rgb, tag="chip")
        if pal.contrast_ratio(rgb, T.bg) < 1.3:
            c.hline(x, x + chip_w, ry + 4, T.faint)
            c.hline(x, x + chip_w, ry + row_h - 4.5, T.faint)
        tx = x + chip_w + 16
        name_w = w * 0.42
        nm = sw["name"][0].upper() + sw["name"][1:]
        nb = c.layout(nm, S.h3, name_w, max_lines=1, min_size=S.h3.size * 0.7)
        hb = c.layout(f"{sw['hex']}   RGB {rgb[0]} {rgb[1]} {rgb[2]}", S.mono, name_w, max_lines=1)
        is_acc = sw["hex"].upper() == P.accent.upper()
        blk_h = c.block_height(nb) + 4 + c.block_height(hb)
        top = ry + (row_h - blk_h) / 2
        b = c.put(nb, tx, top, tag="swatch-name")
        c.put(hb, tx, b + 4, tag="swatch-hex")
        sb = c.layout(f"{sw['share'] * 100:.0f}%", S.h3.but(size=S.h3.size * 1.25), w * 0.18,
                      max_lines=1)
        c.put(sb, x + w - sb.width, ry + row_h / 2 + sb.style.size * 0.34, anchor="baseline",
              tag="swatch-share")
        if is_acc:
            ab = c.layout("Accent", S.label.but(color=ctx.accent_text), w * 0.15, max_lines=1)
            c.put(ab, x + w * 0.66, ry + row_h / 2 + S.label.size * 0.36, anchor="baseline",
                  tag="accent-mark")
        if i < n - 1:
            c.hline(tx, x + w, ry + row_h, T.faint)


def swatch_cards(c, ctx, x, y, w, bottom, cols=3):
    P, S = ctx.P, ctx.S
    n = len(P.palette)
    rows = math.ceil(n / cols)
    gap = ctx.G.gutter
    cw = (w - gap * (cols - 1)) / cols
    cap_h = S.h3.size * 1.3 + S.mono.size * 1.6 * 2 + 10
    ch = min(cw * 0.9, (bottom - y - rows * (cap_h + gap)) / max(rows, 1))
    for i, sw in enumerate(P.palette):
        r, k = divmod(i, cols)
        cx, cy = x + k * (cw + gap), y + r * (ch + cap_h + gap)
        rgb = tuple(sw["rgb"])
        c.rect(cx, cy, cw, ch, rgb, tag="chip")
        on = _rgb(pal.pick_text_color(rgb, ("#FFFFFF", "#111111"), large=True))
        sb = c.layout(f"{sw['share'] * 100:.0f}%", S.h2.but(color=on, italic=False), cw - 16,
                      max_lines=1)
        c.put(sb, cx + 10, cy + ch - 10, anchor="bottom", tag="share", layer="overlay")
        if sw["hex"].upper() == P.accent.upper():
            ab = c.layout("Accent", S.label.but(color=on), cw - 16, max_lines=1)
            c.put(ab, cx + 10, cy + 10, tag="accent-mark", layer="overlay")
        nm = sw["name"][0].upper() + sw["name"][1:]
        _, b = c.write(nm, S.h3, cx, cy + ch + 8, cw, max_lines=1, min_size=S.h3.size * 0.7,
                       tag="swatch-name")
        c.write(f"{sw['hex']}  ·  RGB {rgb[0]} {rgb[1]} {rgb[2]}", S.mono, cx, b + 2, cw,
                max_lines=2, tag="swatch-hex")


def doc_palette(c, ctx):
    P, T, S, G = ctx.P, ctx.T, ctx.S, ctx.G
    y = section_header(c, ctx, 3, "Palette")
    cap = (f"Extracted from {len(P.frames())} keyframes: k-means in CIE Lab, near-identical "
           f"colours merged (ΔE < {pal.MERGE_DELTA_E:g}), sorted by share of the picture.")
    cb = c.layout(cap, S.small, G.span(0, 9)[1], max_lines=3)
    ch = c.block_height(cb)
    c.put(cb, G.ml, G.bottom - ch, tag="caption")
    bottom = G.bottom - ch - 20
    bar_h = min(96.0, (bottom - y) * 0.16)
    bx = G.ml
    for sw in P.palette:
        bw = G.cw * sw["share"]
        c.rect(bx, y, bw + 0.6, bar_h, tuple(sw["rgb"]))
        bx += bw
    c.record(G.ml, y, G.cw, bar_h, "share-bar")
    y += bar_h + S.body.size * 3
    if T.name == "editorial light":
        swatch_cards(c, ctx, G.ml, y, G.cw, bottom)
    else:
        swatch_rows(c, ctx, G.ml, y, G.cw, bottom)


def filmstrip(c, ctx, x, y, w, bottom, max_cols=8):
    """Grid of all shots with number, timecode and headline under each frame."""
    P, S, T = ctx.P, ctx.S, ctx.T
    n = len(P.shots)
    asp = frame_aspect(P)
    gap = ctx.G.gutter
    lab = c.layout("01", S.label, 100, max_lines=1)
    cap_one = c.layout("Xg", S.small, 100, max_lines=1)
    cap_h = 6 + c.block_height(lab) + 3 + c.block_height(cap_one) + cap_one.leading
    cols = 2 if n > 1 else 1
    if n == 1:
        cols = 1
    while True:
        tw = (w - gap * (cols - 1)) / cols
        th = tw / asp
        if n == 1:
            th = min(th, (bottom - y) - cap_h)
            tw = th * asp
        rows = math.ceil(n / cols)
        need = rows * (th + cap_h) + (rows - 1) * gap
        if need <= bottom - y or cols >= max_cols:
            break
        cols += 1
    if need > bottom - y:                        # very many shots: shrink frames to fit
        th = max(8.0, (bottom - y - (rows - 1) * gap) / rows - cap_h)
        tw = min(tw, th * asp)
    elif n > 1:                                  # spare room: taller crops (never past 4:3)
        th = min(tw / min(asp, 4 / 3), (bottom - y - (rows - 1) * gap) / rows - cap_h)
    for i, s in enumerate(P.shots):
        r, k = divmod(i, cols)
        tx, ty = x + k * (tw + gap), y + r * (th + cap_h + gap)
        put_shot_image(c, ctx, s, tx, ty, tw, th, tag="thumb")
        _, b = c.write(f"{s.number:02d}   {timecode(s.start, P.fps)}", S.label, tx, ty + th + 6,
                       tw, max_lines=1, tag="thumb-label")
        c.write(s.headline, S.small.but(color=T.fg), tx, b + 3, tw, max_lines=2,
                max_h=ty + th + cap_h - b - 3, tag="thumb-caption")
    return y + rows * (th + cap_h) + (rows - 1) * gap


def doc_story(c, ctx):
    P, S, G = ctx.P, ctx.S, ctx.G
    src = "Ad Timeline" if P.has_timing else "shot list durations"
    y = section_header(c, ctx, 4, "Story at a glance",
                       kicker=f"{short_duration(P.total_seconds)} at {P.fps:g} fps · "
                              f"timing from {src}")
    y = timeline_lanes(c, ctx, G.ml, y, G.cw, G.span(0, 2)[1])
    filmstrip(c, ctx, G.ml, y + S.body.size * 2.5, G.cw, G.bottom)


def shot_fields(ctx, s, compact=False):
    S = ctx.S
    vo_style = TextStyle(family="serif", italic=True, size=S.body.size * 1.22, leading=1.3,
                         color=ctx.T.fg)
    issues = ""
    v = s.verdict or {}
    if v.get("issues"):
        issues = "; ".join(v["issues"])
    return [
        ("Action", s.description or "\u2014", S.body, None),
        ("On screen", f"“{s.super_text}”" if s.super_text else "", S.body, 3),
        ("Voiceover", s.vo, vo_style, 4),
        ("Critic notes", issues, S.small, 3),
    ]


def shot_heading(c, ctx, s, x, y, w, camera=True):
    """Shot number with its timing (and camera) set beside it. Returns bottom."""
    P, S, T = ctx.P, ctx.S, ctx.T
    if T.header == "index":
        num = S.h2.but(color=T.fg)
    else:
        num = S.h1.but(size=S.h1.size * 0.85, italic=True, color=ctx.accent_text)
    nb = c.layout(f"{s.number:02d}", num, w, max_lines=1)
    meta = [f"{timecode(s.start, P.fps)} \u2013 {timecode(s.end, P.fps)}",
            short_duration(s.duration) + (f" \u00b7 {camera_label(s.camera)}" if camera else "")]
    mx = x + nb.width + S.body.size * 1.4
    mw = x + w - mx
    mono = S.mono.but(color=T.fg)
    probe = c.layout(meta[0], mono, mw, max_lines=1)
    if mw > 60 and probe.width <= mw and not probe.truncated:
        nbottom = c.put(nb, x, y, tag="shot-number")
        base = c._baseline0(nb, y, "top")
        mb = c.layout("\n".join(meta), mono, mw, max_lines=3)
        b0 = max(base - mb.leading * (len(mb.lines) - 1), snap_up(y + mb.ascent, c.unit))
        b = c.put(mb, mx, b0, anchor="baseline", tag="shot-time")
        return max(nbottom, b) + S.body.size * 1.3
    b = c.put(nb, x, y, tag="shot-number")
    _, b = c.write("\n".join(meta), mono, x, b + 4, w, max_lines=4, tag="shot-time")
    return b + S.body.size * 1.3


def shot_slot(c, ctx, s, x, y, w, h, layout, flip=False):
    P, S, G = ctx.P, ctx.S, ctx.G
    asp = frame_aspect(P)
    if layout == "stacked":
        fh = min(h * 0.6, w / asp)
        fw = min(w, fh * 2.39)          # widen to a scope crop rather than leave a gap
        put_shot_image(c, ctx, s, x, y, fw, fh)
        score_badge(c, ctx, s.verdict, x + fw - 8, y + 8)
        ty = y + fh + S.body.size * 1.6
        bottom = y + h
        a_x, a_w = x, (w - 2 * G.gutter) * 0.25
        b_x, b_w = a_x + a_w + G.gutter, (w - 2 * G.gutter) * 0.42
        c_x, c_w = b_x + b_w + G.gutter, (w - 2 * G.gutter) * 0.33
        shot_heading(c, ctx, s, a_x, ty, a_w)
        f = shot_fields(ctx, s)
        stack_fields(c, ctx, [f[0]], b_x, ty, b_w, bottom)
        stack_fields(c, ctx, f[1:], c_x, ty, c_w, bottom)
        return
    fcols = 7
    if flip:
        tx, tw = x, (w - G.gutter) * (12 - fcols) / 12 - G.gutter * 0.5
        fx, fw_max = x + tw + G.gutter * 1.5, w - tw - G.gutter * 1.5
    else:
        fx, fw_max = x, (w - G.gutter) * fcols / 12 + G.gutter * 0.5
        tx = fx + fw_max + G.gutter * 1.5
        tw = x + w - tx
    rx, ry, rw, rh = fit_rect(asp, fx, y, fw_max, h, align="top" if flip else "topleft")
    if flip:
        rx = fx + fw_max - rw
    put_shot_image(c, ctx, s, rx, ry, rw, rh)
    score_badge(c, ctx, s.verdict, rx + rw - 8, ry + 8)
    hb = shot_heading(c, ctx, s, tx, y, tw)
    stack_fields(c, ctx, shot_fields(ctx, s), tx, hb, tw, y + h)


def doc_shots(c, ctx, chunk, first, n_total):
    T, S, G = ctx.T, ctx.S, ctx.G
    a, b = chunk[0].number, chunk[-1].number
    rng = f"Shot {a} of {n_total}" if a == b else f"Shots {a}–{b} of {n_total}"
    y = section_header(c, ctx, 5, "Shot by shot", kicker=rng)
    k = T.shots_per_page
    gap = S.body.size * 3
    slot = (G.bottom - y - gap * (k - 1)) / k
    for i, s in enumerate(chunk):
        sy = y + i * (slot + gap)
        if i:
            c.hline(G.ml, G.ml + G.cw, sy - gap / 2, T.faint)
        flip = T.shots == "alternate" and (first + i) % 2 == 1
        shot_slot(c, ctx, s, G.ml, sy, G.cw, slot, T.shots, flip)


def _logo_luma(logo):
    a = np.asarray(logo.convert("RGBA"), dtype=np.float32)
    m = a[..., 3] > 32
    if not m.any():
        return 0.5
    return float((a[..., :3][m] @ np.array([0.2126, 0.7152, 0.0722])).mean() / 255.0)


def brand_panels(c, ctx, x, y, w, h):
    P, S, T, G = ctx.P, ctx.S, ctx.T, ctx.G
    pw = (w - G.gutter) / 2
    light, dark = (248, 246, 241), (22, 21, 20)
    luma = _logo_luma(P.logo) if P.logo is not None else 0.2
    lab_h = c.block_height(c.layout("X", S.label, pw, max_lines=1)) + 8
    ph = h - lab_h
    for i, (bgc, name) in enumerate(((light, "Positive · on light"),
                                     (dark, "Reversed · on dark"))):
        px_ = x + i * (pw + G.gutter)
        c.rect(px_, y, pw, ph, bgc, tag="brand-panel")
        if bgc == light and T.bg == (255, 255, 255) or pal.contrast_ratio(bgc, T.bg) < 1.2:
            c.hline(px_, px_ + pw, y, T.faint)
            c.hline(px_, px_ + pw, y + ph - 0.5, T.faint)
            c.vline(px_, y, y + ph, T.faint)
            c.vline(px_ + pw - 0.5, y, y + ph, T.faint)
        lh = ph * 0.22
        fg = (20, 19, 18) if bgc == light else (246, 244, 239)
        knock = P.logo is not None and ((bgc == dark and luma < 0.45) or (bgc == light and luma > 0.8))
        if P.logo is None:
            knock = False
        ly = y + (ph - lh) / 2
        if P.logo is not None:
            lw = min(pw * 0.6, lh * P.logo.width / max(1, P.logo.height))
            lh2 = lw * P.logo.height / max(1, P.logo.width)
            ly = y + (ph - lh2) / 2
            wordmark(c, ctx, px_, ly, lh2, color=fg, align="center", w=pw, knockout=knock,
                     tag="brand-logo", surface=bgc, layer="overlay")
            box = (px_ + (pw - lw) / 2, ly, lw, lh2)
        else:
            st = S.label.but(size=lh * 0.62, weight=650, tracking=0.32, color=fg)
            blk = c.layout(P.brand or P.title, st, pw * 0.8, max_lines=1, min_size=lh * 0.25)
            ly = y + ph / 2 + blk.ascent * 0.36
            c.put(blk, px_, ly, w=pw, align="center", anchor="baseline", tag="brand-logo",
                  layer="overlay")
            box = (px_ + (pw - blk.width) / 2, ly - blk.ascent * 0.72, blk.width, blk.ascent * 0.72)
        if i == 0:                                   # clear-space marks
            bx, by, bw, bh = box
            pad = max(4.0, bh * 0.5)
            col = ctx.accent
            for (x0, y0, x1, y1) in ((bx - pad, by - pad, bx + bw + pad, by - pad),
                                     (bx - pad, by + bh + pad, bx + bw + pad, by + bh + pad)):
                c.hline(x0, x1, y0, col, 0.5)
            c.vline(bx - pad, by - pad, by + bh + pad, col, 0.5)
            c.vline(bx + bw + pad, by - pad, by + bh + pad, col, 0.5)
        c.write(name, S.label, px_, y + ph + 8, pw, max_lines=1, tag="panel-label")
    return y + h


def doc_brand(c, ctx):
    P, T, S, G = ctx.P, ctx.T, ctx.S, ctx.G
    y = section_header(c, ctx, 6, "Brand")
    area = G.bottom - y
    rx_, rw_ = G.span(1, 10)
    rules_h = S.label.size * 3
    for rule in P.brand_rules:
        rules_h += c.block_height(c.layout(rule, S.body, rw_, max_lines=4)) + S.body.size * 1.1
    spec_need = S.h1.size * 1.2 + S.label.size * 2 + S.body.size * 2.2 + 40
    ph = max(area * 0.26, min(area * 0.44, area - rules_h - spec_need - S.body.size * 3))
    y = brand_panels(c, ctx, G.ml, y, G.cw, ph) + S.body.size * 3
    # type + colour spec, reserved at the foot of the page
    serif, sans = font_names()
    spec_h = S.h1.size * 1.2 + S.label.size * 2 + S.body.size * 2.2
    sy = G.bottom - spec_h
    c.hline(G.ml, G.ml + G.cw, sy - 12, T.faint)
    cols = [G.span(0, 4), G.span(4, 4), G.span(8, 4)]
    (x0, w0), (x1, w1), (x2, w2) = cols
    c.write("Display", S.label, x0, sy, w0, max_lines=1, tag="spec")
    c.write("Aa", TextStyle(family="serif", size=S.h1.size, color=T.fg, leading=1.0), x0,
            sy + S.label.size * 2, w0, max_lines=1, tag="spec-sample")
    c.write(serif, S.small, x0 + S.h1.size * 1.3, sy + S.label.size * 2 + S.h1.size * 0.35,
            w0 - S.h1.size * 1.3, max_lines=2, tag="spec-name")
    c.write("Text", S.label, x1, sy, w1, max_lines=1, tag="spec")
    c.write("Aa", TextStyle(family="sans", size=S.h1.size * 0.8, weight=500, color=T.fg,
                            leading=1.0), x1, sy + S.label.size * 2, w1, max_lines=1,
            tag="spec-sample")
    c.write(sans, S.small, x1 + S.h1.size * 1.3, sy + S.label.size * 2 + S.h1.size * 0.35,
            w1 - S.h1.size * 1.3, max_lines=2, tag="spec-name")
    c.write("Accent", S.label, x2, sy, w2, max_lines=1, tag="spec")
    chip = S.h1.size * 0.95
    c.rect(x2, sy + S.label.size * 2 + 2, chip, chip, ctx.accent, tag="spec-chip")
    c.write(f"{P.accent}\ntext on accent {pal.rgb_to_hex(ctx.on_accent)}", S.small,
            x2 + chip + 10, sy + S.label.size * 2, w2 - chip - 10, max_lines=3, tag="spec-name")
    # rules list
    _, y = c.write("Rules", S.label, G.ml, y, G.cw, max_lines=1, tag="rules-label")
    y += S.label.size
    limit = sy - 24
    nx, nw = G.span(0, 1)
    rx, rw = G.span(1, 10)
    for i, rule in enumerate(P.brand_rules):
        one = c.layout("X", S.body, rw, max_lines=1)
        if y + c.block_height(one) > limit:
            break
        c.write(f"{i + 1:02d}", S.mono.but(color=ctx.accent_text), nx, y, nw, max_lines=1,
                tag="rule-num")
        _, y = c.write(rule, S.body, rx, y, rw, max_h=limit - y, max_lines=4,
                       min_size=S.body.size * 0.9, tag="rule")
        y += S.body.size * 1.1


def qa_chart(c, ctx, x, y, w, bottom):
    """Per-shot score bars from the critic verdicts (or a compact shot table)."""
    P, S, T = ctx.P, ctx.S, ctx.T
    n = len(P.shots)
    scored = [s for s in P.shots if s.verdict and s.verdict.get("score") is not None]
    title = "Shot QA" if P.has_verdicts else "Shot list"
    _, y = c.write(title, S.label, x, y, w, max_lines=1, tag="qa-title")
    y += S.label.size
    foot_h = S.body.size * 3.6
    row_h = min(20.0, (bottom - y - foot_h) / max(n, 1))
    font_scale = min(1.0, row_h / 14.0)
    st = S.mono.but(size=max(4.0, S.mono.size * font_scale), color=T.fg)
    for i, s in enumerate(P.shots):
        ry = y + i * row_h
        base = ry + row_h / 2 + st.size * 0.36
        nb = c.layout(f"{s.number:02d}", st, 30, max_lines=1)
        c.put(nb, x, base, anchor="baseline", tag="qa-num")
        bx = x + 28
        if P.has_verdicts:
            bw = w - 28 - 44
            th = max(1.5, row_h * 0.28)
            c.rect(bx, ry + (row_h - th) / 2, bw, th, T.faint)
            v = s.verdict or {}
            sc = v.get("score")
            if sc is not None:
                ok = v.get("pass")
                col = ctx.accent if ok is not False else T.muted
                c.rect(bx, ry + (row_h - th) / 2, bw * sc / 10.0, th, col)
            txt = f"{sc:.1f}" if sc is not None else "—"
            if v.get("pass") is False:
                txt += " ↻"
            vb = c.layout(txt, st, 44, max_lines=1)
            c.put(vb, x + w - vb.width, base, anchor="baseline", tag="qa-score")
        else:
            row = f"{short_duration(s.duration):>6}   {camera_label(s.camera)}"
            c.write(row, st, bx, base, w - 28, max_lines=1, anchor="baseline", tag="qa-row")
    y += n * row_h + S.body.size
    if P.has_verdicts:
        mean = sum(s.verdict["score"] for s in scored) / len(scored) if scored else None
        passed = sum(1 for s in P.shots if s.verdict and s.verdict.get("pass"))
        judged = sum(1 for s in P.shots if s.verdict and s.verdict.get("pass") is not None)
        parts = []
        if mean is not None:
            parts.append(f"Mean score {mean:.1f} / 10")
        if judged:
            parts.append(f"{passed} of {judged} shots passed")
        summary = " · ".join(parts) or "Verdicts attached"
    else:
        summary = f"{n} shots · {short_duration(P.total_seconds)}"
    _, y = c.write(summary, S.small, x, y, w, max_h=bottom - y, max_lines=2, tag="qa-summary")
    return y


def doc_notes(c, ctx):
    P, S, G = ctx.P, ctx.S, ctx.G
    y = section_header(c, ctx, 7, "Production notes")
    lx, lw = G.span(0, 6)
    rx, rw = G.span(7, 5)
    serif, sans = font_names()
    rows = [("Format", P.client_line or "—"),
            ("Running time", f"{short_duration(P.total_seconds)} at {P.fps:g} fps, "
                             f"{len(P.shots)} {'shot' if len(P.shots) == 1 else 'shots'}")]
    rows += [(k or "Note", v) for k, v in P.pipeline]
    if P.style:
        rows.append(("Look", P.style))
    rows += [("Typefaces", f"{serif} (display), {sans} (text)"),
             ("Accent", f"{P.accent} · {pal.color_name(P.accent)}")]
    fields = [(k, v, S.body, 3) for k, v in rows]
    n = len(P.shots)
    table_min = S.label.size * 4 + min(n, 4) * 14
    yl = stack_fields(c, ctx, fields, lx, y, lw, G.bottom - table_min, gap=S.body.size * 1.3)
    qa_h = S.label.size * 2 + n * min(20.0, max(6.0, (G.bottom - y - table_min) * 0.6 / max(n, 1))) \
        + S.body.size * 5
    yr = qa_chart(c, ctx, rx, y, rw, min(G.bottom - table_min, y + qa_h))
    ty = max(yl, yr) + S.body.size * 2
    if G.bottom - ty >= table_min * 0.8:
        shot_table(c, ctx, G.ml, ty, G.cw, G.bottom)


def shot_table(c, ctx, x, y, w, bottom):
    """Production shot table: number, in/out timecodes, duration, camera, super, QA."""
    P, S, T = ctx.P, ctx.S, ctx.T
    _, y = c.write("Shot table", S.label, x, y, w, max_lines=1, tag="table-title")
    y += S.label.size * 0.8
    cols = [("#", 0.05), ("In", 0.15), ("Out", 0.15), ("Dur", 0.08), ("Camera", 0.27),
            ("On screen", 0.22), ("QA", 0.08)]
    xs, acc = [], x
    for _, f in cols:
        xs.append(acc)
        acc += w * f
    head = S.label.but(size=S.label.size * 0.95)
    hb = c.layout("X", head, w, max_lines=1)
    for (name, f), cx in zip(cols, xs):
        c.write(name, head, cx, y, w * f - 6, max_lines=1, tag="th")
    y += c.block_height(hb) + 4
    c.hline(x, x + w, y, T.fg, 0.5)
    y += 2
    n = len(P.shots)
    row_h = min(18.0, (bottom - y) / max(n, 1))
    st = S.mono.but(color=T.fg, size=min(S.mono.size, max(4.0, row_h * 0.52)))
    shown = n if row_h >= st.size * 1.25 else max(0, int((bottom - y) // (st.size * 1.3)) - 1)
    row_h = max(row_h, st.size * 1.3) if shown < n else row_h
    for i, s in enumerate(P.shots[:shown]):
        ry = y + i * row_h
        if i % 2 == 1:
            c.rect(x, ry, w, row_h, pal.mix(T.bg, T.fg, 0.04))
        v = s.verdict or {}
        sc = v.get("score")
        vals = [f"{s.number:02d}", timecode(s.start, P.fps), timecode(s.end, P.fps),
                short_duration(s.duration), camera_label(s.camera), s.super_text or "\u2014",
                f"{sc:.1f}" if sc is not None else "\u2014"]
        base = ry + row_h / 2 + st.size * 0.36
        for (name, f), cx, val in zip(cols, xs, vals):
            blk = c.layout(val, st, w * f - 6, max_lines=1)
            c.put(blk, cx, base, anchor="baseline", tag="td")
    y += shown * row_h
    if shown < n:
        c.write(f"+ {n - shown} more shots", S.small, x, y + 2, w, max_lines=1,
                max_h=bottom - y - 2, tag="table-more")
    return y


def doc_closing(c, ctx):
    P, T, S, G = ctx.P, ctx.T, ctx.S, ctx.G
    last = P.shots[-1]
    big = S.cover.but(size=min(S.cover.size, S.h1.size * 1.35))
    sign = " · ".join(x for x in (P.director and f"A film by {P.director}", P.date) if x)
    if T.cover == "bleed":
        ih = c.H * (0.62 if frame_aspect(P) > 1.0 else 1.0)
        put_shot_image(c, ctx, last, 0, c.H - ih, c.W, ih, tag="closing-frame", layer="bg")
        c.gradient(0, c.H - ih, c.W, ih * 0.5, T.bg, 1.0, 0.35)
        c.gradient(0, c.H - ih * 0.5, c.W, ih * 0.5, T.bg, 0.35, 0.85)
        cy = c.H * 0.42
        wordmark(c, ctx, G.ml, cy - 44, 22, align="center", w=G.cw)
        _, b = c.write("Thank you.", big, G.ml, cy, G.cw, max_lines=1, align="center",
                       tag="thanks")
        c.write(sign, S.body.but(color=T.fg), G.ml, b + 18, G.cw, max_lines=2, align="center",
                tag="sign")
        c.write(P.client_line, S.label.but(color=ctx.accent_text), G.ml, G.bottom - 12, G.cw,
                max_lines=1, align="center", anchor="bottom", tag="closing-client")
        return
    if T.cover == "framed":
        ih = c.H * 0.42
        put_shot_image(c, ctx, last, G.ml, G.mt, G.cw, ih, tag="closing-frame")
        y = G.mt + ih + 70
        y = wordmark(c, ctx, G.ml, y, 20, align="center", w=G.cw) + 26
        _, b = c.write("Thank you.", big, G.ml, y, G.cw, max_lines=1, align="center", tag="thanks")
        c.write(sign, S.body, G.ml, b + 18, G.cw, max_lines=2, align="center", tag="sign")
        c.write(P.client_line, S.label.but(color=ctx.accent_text), G.ml, G.bottom, G.cw,
                max_lines=1, align="center", anchor="bottom", tag="closing-client")
        return
    lx, lw = G.span(0, 7)
    y = wordmark(c, ctx, G.ml, G.mt, 16, w=lw)
    c.rect(G.ml, y + 18, 22, 3, ctx.accent, tag="accent-bar")
    ih = c.H * 0.34
    put_shot_image(c, ctx, last, G.x(5), G.bottom - ih, G.span(5, 7)[1], ih, tag="closing-frame")
    ty = c.H * 0.3
    _, b = c.write("Thank you.", big, lx, ty, lw, max_lines=1, tag="thanks")
    _, b = c.write(sign, S.body, lx, b + 14, lw, max_lines=2, tag="sign")
    c.write(P.client_line, S.label, lx, b + 10, lw, max_lines=2, tag="closing-client")


# =========================================================================== deck slides

def deck_folio(c, ctx):
    G, S = ctx.G, ctx.S
    y = c.H - G.mb * 0.5
    num = f"{ctx.page:02d}"
    nb = c.layout(num, S.folio, 80, max_lines=1)
    c.put(nb, G.ml + G.cw - nb.width, y, anchor="bottom", tag="folio")
    left = " · ".join(x for x in (ctx.P.brand, ctx.P.title) if x)
    c.write(left, S.folio, G.ml, y, G.cw * 0.7, max_lines=1, anchor="bottom", tag="folio")


def deck_cover(c, ctx):
    P, T, S, G = ctx.P, ctx.T, ctx.S, ctx.G
    hero = P.shots[P.hero]
    items = meta_items(P, deck=True)
    kicker = P.client_line or "Pitch"
    if T.cover == "bleed":
        ix = c.W * 0.34                  # hero sits right, dissolving into the page
        put_shot_image(c, ctx, hero, ix, 0, c.W - ix, c.H, tag="hero", layer="bg")
        gradient_h(c, ix - 1, 0, (c.W - ix) * 0.55, c.H, T.bg, 1.0, 0.0)
        c.gradient(0, c.H * 0.6, c.W, c.H * 0.4, T.bg, 0.0, 0.55)
        x, w = G.span(0, 7)
        wordmark(c, ctx, G.ml, G.mt, 24, w=w)
        mh = meta_row_height(c, ctx, items, w, 3)
        my = G.bottom - mh
        meta_row(c, ctx, items, x, my, w, 3)
        ry = my - 16
        c.hline(x, x + w, ry, pal.mix(T.bg, T.fg, 0.45))
        kb = c.layout(kicker, S.label.but(color=ctx.accent_text), w, max_lines=1)
        room = ry - 20 - (G.mt + 70) - c.block_height(kb) - 12
        tb = fit_title(P.title, S.cover, w, 3, c.u, S.cover_min, room, c.unit)
        top = c.put(tb, x, ry - 20, anchor="bottom", tag="title") - tb.height
        c.put(kb, x, top - 12 - c.block_height(kb), tag="kicker")
        return
    if T.cover == "framed":
        ih = c.H * 0.6
        put_shot_image(c, ctx, hero, 0, 0, c.W, ih, tag="hero", layer="bg")
        y = ih + 26
        x, w = G.span(0, 8)
        _, b = c.write(kicker, S.label.but(color=ctx.accent_text), x, y, w, max_lines=1,
                       tag="kicker")
        tb = fit_title(P.title, S.cover, w, 2, c.u, S.cover_min, G.bottom - b - 8, c.unit)
        c.put(tb, x, b + 8, tag="title")
        mx, mw = G.span(9, 3)
        yy = y
        for lab, val in items:
            yy = label_value(c, ctx, lab, val, mx, yy, mw, max_lines=1) + 8
        wordmark(c, ctx, G.ml, G.mt, 22, w=G.span(0, 5)[1], color=(255, 255, 255),
                 knockout=True)
        return
    sx = c.W * 0.46
    put_shot_image(c, ctx, hero, sx, 0, c.W - sx, c.H, tag="hero")
    lw = sx - G.ml - G.gutter * 2
    y = wordmark(c, ctx, G.ml, G.mt, 18, w=lw)
    c.rect(G.ml, y + 16, 26, 3.5, ctx.accent, tag="accent-bar")
    mh = meta_row_height(c, ctx, items, lw, 3)
    my = G.bottom - mh
    meta_row(c, ctx, items, G.ml, my, lw, 3)
    kb = c.layout(kicker, S.label, lw, max_lines=1)
    tb = fit_title(P.title, S.cover, lw, 4, c.u, S.cover_min,
                   my - 26 - (y + 50) - c.block_height(kb) - 10, c.unit)
    top = c.put(tb, G.ml, my - 26, anchor="bottom", tag="title") - tb.height
    c.put(kb, G.ml, top - 10 - c.block_height(kb), tag="kicker")


def deck_idea(c, ctx):
    P, T, S, G = ctx.P, ctx.T, ctx.S, ctx.G
    y = section_header(c, ctx, 1, "The idea")
    bottom = G.bottom
    lx, lw = G.span(0, 6)
    rx, rw = G.span(7, 5)
    c.write(P.logline, S.h2, lx, y, lw, max_h=bottom - y, min_size=S.h2.size * 0.6, tag="logline")
    ry = y
    if P.tone_words:
        _, ry = c.write("Tone", S.label, rx, ry, rw, max_lines=1, tag="tone-label")
        _, ry = c.write("  ·  ".join(P.tone_words),
                        S.h3.but(italic=T.display == "serif"), rx, ry + 6, rw,
                        max_lines=2, min_size=S.h3.size * 0.8, tag="tone")
        ry += S.body.size * 1.8
    _, ry = c.write("Approach", S.label, rx, ry, rw, max_lines=1, tag="approach-label")
    c.write(P.approach, S.body, rx, ry + 6, rw, max_h=bottom - ry - 6,
            min_size=S.body.size * 0.8, tag="approach")


def deck_mood(c, ctx):
    G = ctx.G
    y = section_header(c, ctx, 2, "Mood")
    mood_board(c, ctx, ctx.T.deck_mood, G.ml, y, G.cw, G.bottom - y, 6, G.gutter * 0.4)


def deck_palette(c, ctx):
    P, S, G = ctx.P, ctx.S, ctx.G
    y = section_header(c, ctx, 3, "Palette")
    n = len(P.palette)
    top = y
    h = c.H - top
    sw_w = c.W / n
    for i, sw in enumerate(P.palette):
        x = i * sw_w
        rgb = tuple(sw["rgb"])
        c.rect(x, top, sw_w + 0.5, h, rgb, tag=f"stripe-{i}", layer="bg")
        on = _rgb(pal.pick_text_color(rgb, ("#FFFFFF", "#111111")))
        tx, tw = x + 18, sw_w - 36
        sb = c.layout(f"{sw['share'] * 100:.0f}%", S.h1.but(color=on, italic=False), tw,
                      max_lines=1, min_size=S.h1.size * 0.4)
        nm = sw["name"][0].upper() + sw["name"][1:]
        nb = c.layout(nm, S.h3.but(color=on), tw, max_lines=2, min_size=S.h3.size * 0.6)
        hb = c.layout(sw["hex"], S.mono.but(color=on), tw, max_lines=1)
        yb = c.H - G.mb * 0.7
        yb = c.put(hb, tx, yb, anchor="bottom", tag="stripe-hex") - hb.height - 6
        yb = c.put(nb, tx, yb, anchor="bottom", tag="stripe-name") - nb.height - 10
        c.put(sb, tx, yb, anchor="bottom", tag="stripe-share")
        if sw["hex"].upper() == P.accent.upper():
            c.write("Accent", S.label.but(color=on), tx, top + 18, tw, max_lines=1,
                    tag="accent-mark")


def deck_shot(c, ctx, s):
    P, S, G = ctx.P, ctx.S, ctx.G
    asp = frame_aspect(P)
    n = len(P.shots)
    lab = c.layout(f"Shot {s.number} of {n}", S.label, G.cw / 2, max_lines=1)
    c.put(lab, G.ml, G.mt, tag="slide-label")
    y = G.mt + c.block_height(lab) + 14
    fx, fw = G.span(0, 9 if asp >= 1.2 else 7)
    rx, ry, rw, rh = fit_rect(asp, fx, y, fw, G.bottom - y, align="topleft")
    put_shot_image(c, ctx, s, rx, ry, rw, rh)
    score_badge(c, ctx, s.verdict, rx + rw - 12, ry + 12, scale=1.1)
    tx = rx + rw + G.gutter * 1.5
    tw = G.ml + G.cw - tx
    hb = shot_heading(c, ctx, s, tx, y, tw)
    stack_fields(c, ctx, shot_fields(ctx, s), tx, hb, tw, G.bottom)


def deck_timeline(c, ctx):
    P, S, G = ctx.P, ctx.S, ctx.G
    y = section_header(c, ctx, 4, "Timeline",
                       kicker=f"{short_duration(P.total_seconds)} · {len(P.shots)} shots")
    y = timeline_lanes(c, ctx, G.ml, y, G.cw, 70, compact=True) + 18
    n = len(P.shots)
    rows = 1 if n <= 10 else 2
    per = math.ceil(n / rows)
    gap = 4
    lab_h = c.block_height(c.layout("01", S.label, 50, max_lines=1)) + 6
    row_h = (G.bottom - y - (rows - 1) * 12) / rows - lab_h
    for r in range(rows):
        part = P.shots[r * per:(r + 1) * per]
        if not part:
            break
        tot = sum(s.duration for s in part)
        avail = G.cw - gap * (len(part) - 1)
        ws = [max(avail * 0.04, avail * s.duration / tot) for s in part]
        k = avail / sum(ws)
        ws = [w * k for w in ws]
        x = G.ml
        yy = y + r * (row_h + lab_h + 12)
        th = min(row_h, max(ws) / 0.9)
        for s, w in zip(part, ws):
            put_shot_image(c, ctx, s, x, yy, w, th, tag="strip")
            nb = c.layout(f"{s.number:02d}", S.label, w, max_lines=1)
            if nb.width <= w:
                c.put(nb, x, yy + th + 6, tag="strip-label")
            x += w + gap


def deck_end(c, ctx):
    P, T, S, G = ctx.P, ctx.T, ctx.S, ctx.G
    last = P.shots[-1]
    big = S.cover.but(size=min(S.cover.size, S.h1.size * 1.3))
    sign = " · ".join(x for x in (P.director and f"A film by {P.director}", P.date) if x)
    if T.cover in ("split", "framed"):
        sx = c.W * 0.5
        put_shot_image(c, ctx, last, sx, 0, c.W - sx, c.H, tag="closing-frame")
        lw = sx - G.ml - G.gutter * 2
        y = wordmark(c, ctx, G.ml, G.mt, 18, w=lw)
        c.rect(G.ml, y + 16, 26, 3.5, ctx.accent, tag="accent-bar")
        _, b = c.write("Thank you.", big, G.ml, c.H * 0.42, lw, max_lines=1, tag="thanks")
        _, b = c.write(sign, S.body, G.ml, b + 14, lw, max_lines=2, tag="sign")
        c.write(P.client_line, S.label, G.ml, b + 10, lw, max_lines=1, tag="closing-client")
        return
    put_shot_image(c, ctx, last, 0, 0, c.W, c.H, tag="closing-frame", layer="bg")
    c.gradient(0, 0, c.W, c.H, T.bg, 0.6, 0.9, ease=False)
    cy = c.H * 0.4
    wordmark(c, ctx, G.ml, cy - 54, 26, align="center", w=G.cw, color=T.fg)
    _, b = c.write("Thank you.", big, G.ml, cy, G.cw, max_lines=1, align="center", tag="thanks")
    c.write(sign, S.body.but(color=T.fg), G.ml, b + 16, G.cw, max_lines=1, align="center",
            tag="sign")
    c.write(P.client_line, S.label.but(color=ctx.accent_text), G.ml, G.bottom, G.cw,
            max_lines=1, align="center", anchor="bottom", tag="closing-client")


# =========================================================================== documents

@dataclass
class Document:
    pages: list                  # PIL RGB images
    names: list                  # page names
    boxes: list                  # per page: [(layer, (x0, y0, x1, y1), tag)] in pt
    size_pt: tuple
    dpi: float
    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: str = ""


def treatment_plan(P, T):
    k = T.shots_per_page
    chunks = [P.shots[i:i + k] for i in range(0, len(P.shots), k)]
    plan = [("cover", doc_cover, ()), ("idea", doc_idea, ()), ("mood", doc_mood, ()),
            ("palette", doc_palette, ()), ("story", doc_story, ())]
    for j, ch in enumerate(chunks):
        plan.append((f"shots-{j + 1}", doc_shots, (ch, j * k, len(P.shots))))
    plan += [("brand", doc_brand, ()), ("notes", doc_notes, ()), ("closing", doc_closing, ())]
    return plan


def treatment_page_count(n_shots, template="editorial dark"):
    return 8 + math.ceil(max(1, n_shots) / get_theme(template).shots_per_page)


def deck_plan(P):
    plan = [("cover", deck_cover, ()), ("idea", deck_idea, ()), ("mood", deck_mood, ()),
            ("palette", deck_palette, ())]
    plan += [(f"shot-{s.number}", deck_shot, (s,)) for s in P.shots]
    plan += [("timeline", deck_timeline, ()), ("end", deck_end, ())]
    return plan


def deck_slide_count(n_shots):
    return 6 + max(1, n_shots)


def _render(P, T, kind, w_pt, h_pt, u, plan, folio_fn, unit):
    pages, names, boxes = [], [], []
    for i, (name, fn, args) in enumerate(plan):
        ctx = make_ctx(P, T, kind, w_pt, h_pt)
        ctx.page, ctx.total = i + 1, len(plan)
        c = Canvas(w_pt, h_pt, u, T.bg, unit)
        fn(c, ctx, *args)
        if name not in ("cover", "closing", "end", "palette" if kind == "deck" else "") \
                and folio_fn:
            folio_fn(c, ctx)
        pages.append(c.im)
        names.append(name)
        boxes.append(c.boxes)
    return pages, names, boxes


def _meta(P, what):
    subject = f"{what} — {P.brand}" if P.brand else what
    kw = ", ".join([x for x in (P.brand, P.client_line) if x] + P.tone_words)
    return dict(title=P.title, author=P.director, subject=subject, keywords=kw)


def render_treatment(P, template="editorial dark", page_size="A4", dpi=TREATMENT_DPI):
    T = get_theme(template)
    if page_size not in PAGE_SIZES_PT:
        raise TreatmentError(f"page_size must be one of {', '.join(PAGE_SIZES_PT)}")
    w, h = PAGE_SIZES_PT[page_size]
    pages, names, boxes = _render(P, T, "doc", w, h, dpi / PT_PER_INCH, treatment_plan(P, T),
                                  folio, BASELINE)
    return Document(pages, names, boxes, (w, h), dpi, **_meta(P, "Director's treatment"))


def render_deck(P, template="editorial dark", resolution="1920x1080"):
    T = get_theme(template)
    width = DECK_RESOLUTIONS.get(resolution, None) if isinstance(resolution, str) else int(resolution)
    if not width:
        raise TreatmentError(f"resolution must be one of {', '.join(DECK_RESOLUTIONS)}")
    w, h = DECK_SIZE_PT
    u = width / w
    pages, names, boxes = _render(P, T, "deck", w, h, u, deck_plan(P), deck_folio, DECK_BASELINE)
    return Document(pages, names, boxes, (w, h), u * PT_PER_INCH, **_meta(P, "Pitch deck"))


# =========================================================================== storyboard

def storyboard_layout(n, columns, w, h, aspect, gap, cap_h):
    """Panels per sheet that fit at `columns` across: (panel_w, panel_h, rows, sheets)."""
    columns = max(1, min(int(columns), max(1, n)))
    pw = (w - gap * (columns - 1)) / columns
    ph = pw / aspect
    rows = max(1, int((h + gap) // (ph + cap_h + gap)))
    if rows == 1 and ph + cap_h > h:               # very wide grid on a short sheet
        ph = max(10.0, h - cap_h)
        pw = min(pw, ph * aspect)
    rows = min(rows, math.ceil(n / columns))
    sheets = math.ceil(n / (rows * columns))
    return pw, ph, rows, sheets


def render_storyboard(P, paper="A4 landscape", columns=4, dpi=None):
    if paper not in STORYBOARD_PAPERS:
        raise TreatmentError(f"paper must be one of {', '.join(STORYBOARD_PAPERS)}")
    W, H, pdpi = STORYBOARD_PAPERS[paper]
    dpi = dpi or pdpi
    u = dpi / PT_PER_INCH
    base = get_theme("editorial light")
    T = Theme(**{**base.__dict__, "bg": (251, 250, 247), "fg": (20, 20, 20),
                 "muted": (100, 98, 94), "faint": (214, 211, 205)})
    m = 30.0 if paper != "A3" else 40.0
    scale = W / 842.0
    sc = {k: v * (0.85 + 0.3 * scale) for k, v in base.scale.items()}
    S = Styles(T, sc)
    G = Grid(W, H, m, m, m, m, 12 * scale)
    acc = _rgb(P.accent)
    ctx = Ctx(P, T, S, G, acc, _rgb(pal.ensure_contrast(acc, T.bg, pal.AA_LARGE)),
              _rgb(pal.pick_text_color(acc)), "doc")
    asp = frame_aspect(P)
    gap = G.gutter
    probe = Canvas(10, 10, u, T.bg)
    lab_h = probe.block_height(probe.layout("01", S.label, 100, max_lines=1))
    sm = probe.layout("Xg", S.small, 100, max_lines=1)
    cap_h = 7 + lab_h + 3 + probe.block_height(sm) + sm.leading + 4
    kb_ = probe.layout("X", S.label, 100, max_lines=1)
    tb_ = probe.layout("Xg", S.h2.but(italic=False), 400, max_lines=1)
    hdr_h = probe.block_height(kb_) + 4 + probe.block_height(tb_) + 22
    pw, ph, rows, sheets = storyboard_layout(len(P.shots), columns, G.cw,
                                             G.bottom - G.mt - hdr_h, asp, gap, cap_h)
    per = rows * max(1, min(int(columns), len(P.shots)))
    cols = max(1, min(int(columns), len(P.shots)))
    pages, boxes = [], []
    for sh in range(sheets):
        c = Canvas(W, H, u, T.bg)
        ctx.page, ctx.total = sh + 1, sheets
        # header strip
        name = " · ".join(x for x in (P.brand, "Storyboard") if x)
        _, b = c.write(name, S.label, G.ml, G.mt, G.cw * 0.5, max_lines=1, tag="sb-kicker")
        right = f"{P.date}   ·   Sheet {sh + 1} / {sheets}"
        c.write(right, S.label, G.ml + G.cw * 0.5, G.mt, G.cw * 0.5, max_lines=1, align="right",
                tag="sb-sheet")
        _, b = c.write(P.title, S.h2.but(italic=False), G.ml, b + 4, G.cw, max_lines=1,
                       min_size=S.h2.size * 0.6, tag="sb-title")
        c.hline(G.ml, G.ml + G.cw, b + 10, T.fg, 0.6)
        grid_w = cols * pw + (cols - 1) * gap
        x0 = G.ml + (G.cw - grid_w) / 2
        used = min(rows, math.ceil((len(P.shots) - sh * per) / cols))
        grid_h = used * (ph + cap_h) + (used - 1) * gap
        room = G.bottom - (G.mt + hdr_h)
        y0 = G.mt + hdr_h + max(0.0, (room - grid_h) / 2) * (1.0 if sheets == 1 else 0.0)
        for j, s in enumerate(P.shots[sh * per:(sh + 1) * per]):
            r, k = divmod(j, cols)
            px_, py_ = x0 + k * (pw + gap), y0 + r * (ph + cap_h + gap)
            put_shot_image(c, ctx, s, px_, py_, pw, ph, tag="panel")
            for (a, b2, c2, d2) in ((px_, py_, px_ + pw, py_), (px_, py_ + ph - 0.6, px_ + pw, 0)):
                c.hline(a, c2, b2, T.fg, 0.6)
            c.vline(px_, py_, py_ + ph, T.fg, 0.6)
            c.vline(px_ + pw - 0.6, py_, py_ + ph, T.fg, 0.6)
            draw_camera_arrow(c, camera_arrow(s.camera), px_, py_, pw, ph)
            nb = c.layout(f"{s.number:02d}", S.label.but(color=(255, 255, 255), tracking=0.05),
                          60, max_lines=1)
            bw, bh = nb.width + S.label.size * 1.6, S.label.size * 2.4
            c.rect(px_, py_, bw, bh, T.fg, tag="panel-num-bg", layer="overlay")
            c.put(nb, px_ + S.label.size * 0.8, py_ + bh / 2 + S.label.size * 0.36,
                  anchor="baseline", tag="panel-num", layer="overlay")
            cap = f"{s.number:02d}  ·  {short_duration(s.duration)}  ·  {camera_label(s.camera)}"
            _, cb = c.write(cap, S.label.but(color=T.fg), px_, py_ + ph + 7, pw, max_lines=1,
                            tag="panel-label")
            c.write(s.description or s.headline, S.small.but(color=T.fg), px_, cb + 3, pw,
                    max_lines=2, max_h=py_ + ph + cap_h - cb - 3, tag="panel-caption")
        pages.append(c.im)
        boxes.append(c.boxes)
    return Document(pages, [f"sheet-{i + 1}" for i in range(len(pages))], boxes, (W, H), dpi,
                    **_meta(P, "Storyboard"))


# =========================================================================== palette card

def render_palette_card(swatches, accent_hex=None, width=1600):
    """Swatch image for the Palette From Shots node: share bar on top, then one column per
    colour with its name, hex and share in a colour that stays readable on it."""
    W, H = 960.0, 540.0
    u = width / W
    bg, muted = (243, 238, 229), (110, 102, 92)
    c = Canvas(W, H, u, bg, unit=DECK_BASELINE)
    acc = (accent_hex or (pal.pick_accent(swatches) or {"hex": "#888888"})["hex"]).upper()
    m = 36.0
    bar_h = 26.0
    x = m
    for sw in swatches:
        bw = (W - 2 * m) * sw["share"]
        c.rect(x, m, bw + 0.5, bar_h, tuple(sw["rgb"]))
        x += bw
    lab = TextStyle(size=8, weight=600, caps=True, tracking=0.16, color=muted)
    c.write("Palette · share of picture", lab, m, m + bar_h + 10, 400, max_lines=1)
    top = m + bar_h + 34
    n = max(1, len(swatches))
    gap = 8.0
    cw = (W - 2 * m - gap * (n - 1)) / n
    for i, sw in enumerate(swatches):
        cx = m + i * (cw + gap)
        rgb = tuple(sw["rgb"])
        c.rect(cx, top, cw, H - m - top, rgb)
        on = _rgb(pal.pick_text_color(rgb, ("#FFFFFF", "#111111")))
        tx, tw = cx + 12, cw - 24
        yb = H - m - 12
        hb = c.layout(f"{sw['hex']}   {sw['share'] * 100:.0f}%",
                      TextStyle(size=9.5, weight=500, color=on), tw, max_lines=2)
        yb = c.put(hb, tx, yb, anchor="bottom") - hb.height - 4
        nm = sw["name"][0].upper() + sw["name"][1:]
        nb = c.layout(nm, TextStyle(family="serif", size=22, color=on, leading=1.05), tw,
                      max_lines=3, min_size=11)
        c.put(nb, tx, yb, anchor="bottom")
        if sw["hex"].upper() == acc:
            c.write("Accent", lab.but(color=on), tx, top + 12, tw, max_lines=1)
    return c.im


# =========================================================================== output

def save_pdf(doc, path, quality=JPEG_QUALITY):
    """Multi-page PDF with Pillow's writer; page size follows the document's dpi, metadata
    (title, author, subject, keywords) is embedded."""
    if not doc.pages:
        raise TreatmentError("nothing to save: the document has no pages")
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    first, rest = doc.pages[0].convert("RGB"), [p.convert("RGB") for p in doc.pages[1:]]
    first.save(path, "PDF", save_all=True, append_images=rest, resolution=float(doc.dpi),
               quality=int(quality), title=doc.title or "Untitled", author=doc.author or "",
               subject=doc.subject or "", keywords=doc.keywords or "",
               creator="ComfyUI Shot Planner · Treatment & Pitch Deck", producer="Pillow")
    return path


def pdf_page_count(path):
    """Pages in a PDF, read back with Pillow's own PDF parser."""
    from PIL import PdfParser
    p = PdfParser.PdfParser(path, mode="rb")
    try:
        return len(p.pages)
    finally:
        p.close()


def pdf_info(path):
    from PIL import PdfParser
    p = PdfParser.PdfParser(path, mode="rb")
    try:
        out = {}
        for k, v in p.info.items():
            key = k.name.decode() if hasattr(k, "name") else str(k)
            if isinstance(v, bytes):
                v = v[2:].decode("utf-16-be") if v.startswith(b"\xfe\xff") else v.decode("latin-1")
            out[key] = v
        return out
    finally:
        p.close()


def previews(pages, width=PREVIEW_WIDTH):
    """Downscaled copies of pages (all the same size, ready for an IMAGE batch)."""
    out = []
    for p in pages:
        h = max(1, int(round(p.height * width / p.width)))
        out.append(p.resize((int(width), h), Image.LANCZOS, reducing_gap=3.0))
    return out
