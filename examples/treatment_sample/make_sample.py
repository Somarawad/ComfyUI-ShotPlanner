"""Render the sample treatment, deck, storyboard and palette card in this folder.

Every keyframe is procedurally generated (gradients and simple shapes, no stock imagery),
so the samples can be rebuilt anywhere:

    python examples/treatment_sample/make_sample.py
"""

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import treatment as tr  # noqa: E402

W, H = 1280, 720
RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------- painting helpers

def vgrad(stops, w=W, h=H):
    """Vertical gradient through (position, (r, g, b)) stops."""
    ys = np.linspace(0, 1, h)
    pos = [p for p, _ in stops]
    out = np.stack([np.interp(ys, pos, [c[i] for _, c in stops]) for i in range(3)], -1)
    return np.repeat(out[:, None, :], w, axis=1)


def radial(cx, cy, r, w=W, h=H):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.clip(1 - np.hypot(xx - cx, yy - cy) / r, 0, 1)


def blend(base, color, alpha):
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    return base * (1 - a) + np.array(color, dtype=np.float64) * a


def mask_from(draw_fn, blur=0.0, w=W, h=H):
    m = Image.new("L", (w, h), 0)
    draw_fn(ImageDraw.Draw(m))
    if blur:
        m = m.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(m, dtype=np.float64) / 255.0


def finish(img, grain=1.0, vignette=0.35):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    v = 1 - vignette * (((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2.2)
    img = img * v[..., None] + RNG.normal(0, grain, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- the six frames

def f1_rooftops():
    img = vgrad([(0, (28, 36, 64)), (0.45, (196, 120, 92)), (0.62, (240, 182, 120)), (1, (240, 190, 130))])
    img = blend(img, (255, 226, 170), radial(820, 470, 260) ** 2 * 0.9)
    img = blend(img, (255, 244, 214), radial(820, 470, 70) ** 0.6)
    rng = np.random.default_rng(3)

    def skyline(d):
        x = 0
        while x < W:
            bw = int(rng.integers(50, 140))
            top = int(rng.integers(430, 560))
            d.rectangle([x, top, x + bw, H], fill=255)
            if rng.random() < 0.4:
                d.polygon([(x, top), (x + bw // 2, top - 40), (x + bw, top)], fill=255)
            if rng.random() < 0.3:
                d.rectangle([x + bw // 3, top - 70, x + bw // 3 + 10, top], fill=255)
            x += bw
        d.polygon([(560, 470), (600, 300), (640, 470)], fill=255)      # a spire
        d.rectangle([596, 250, 604, 300], fill=255)
    img = blend(img, (44, 30, 26), mask_from(skyline, 1.2))
    return finish(img)


def f2_window():
    img = vgrad([(0, (120, 82, 58)), (1, (64, 42, 32))])
    shaft = mask_from(lambda d: d.polygon([(330, 0), (640, 0), (1160, H), (720, H)], fill=255), 40)
    img = blend(img, (250, 210, 150), shaft * 0.55)
    win = mask_from(lambda d: d.rectangle([150, 90, 520, 560], fill=255))
    img = blend(img, (255, 236, 196), win * 0.92)
    bars = mask_from(lambda d: [d.rectangle([150, 318, 520, 330], fill=255),
                                d.rectangle([330, 90, 342, 560], fill=255)])
    img = blend(img, (70, 46, 34), bars)
    table = mask_from(lambda d: d.rectangle([0, 600, W, H], fill=255))
    img = blend(img, (46, 30, 24), table)
    cup = mask_from(lambda d: [d.ellipse([820, 520, 980, 560], fill=255),
                               d.rectangle([840, 470, 960, 545], fill=255)], 1)
    img = blend(img, (238, 230, 214), cup)
    return finish(img)


def f3_steam():
    img = vgrad([(0, (22, 17, 14)), (1, (40, 28, 22))])
    img = blend(img, (120, 70, 40), radial(640, 520, 520) * 0.5)
    cup = mask_from(lambda d: [d.pieslice([470, 330, 810, 650], 0, 180, fill=255),
                               d.rectangle([470, 470, 810, 490], fill=255)], 1.5)
    img = blend(img, (236, 228, 214), cup)
    crema = mask_from(lambda d: d.ellipse([480, 455, 800, 505], fill=255), 2)
    img = blend(img, (176, 112, 62), crema)
    xs = np.arange(H)

    def steam(d):
        for k, x0 in enumerate((580, 640, 700)):
            pts = [(x0 + 26 * np.sin(y / 42.0 + k * 1.7), y) for y in range(450, 90, -6)]
            d.line(pts, fill=255, width=9)
    s = mask_from(steam, 9) * (np.linspace(0.05, 0.75, H)[::-1][:, None] ** 1.2)
    img = blend(img, (250, 242, 230), s * 0.8)
    del xs
    return finish(img, vignette=0.5)


def f4_overhead():
    img = vgrad([(0, (176, 150, 120)), (1, (150, 124, 98))])
    yy, xx = np.mgrid[0:H, 0:W]
    grain = (np.sin(xx / 9.0 + np.sin(yy / 40.0) * 3) * 0.5 + 0.5) * 14
    img = img - grain[..., None]
    sauc = mask_from(lambda d: d.ellipse([440, 160, 840, 560], fill=255), 2)
    img = blend(img, (243, 238, 228), sauc)
    img = blend(img, (0, 0, 0), mask_from(lambda d: d.ellipse([520, 250, 780, 510], fill=255), 18) * 0.25)
    cup = mask_from(lambda d: d.ellipse([510, 230, 770, 490], fill=255), 1.5)
    img = blend(img, (250, 247, 240), cup)
    coffee = mask_from(lambda d: d.ellipse([540, 260, 740, 460], fill=255), 1.5)
    img = blend(img, (70, 40, 24), coffee)
    img = blend(img, (196, 132, 74), radial(625, 345, 110) ** 1.5 * coffee * 0.9)
    spoon = mask_from(lambda d: [d.line([(860, 420), (1080, 600)], fill=255, width=14),
                                 d.ellipse([830, 390, 890, 440], fill=255)], 1)
    img = blend(img, (190, 190, 186), spoon)
    return finish(img)


def f5_tram():
    img = vgrad([(0, (160, 178, 186)), (0.5, (214, 196, 168)), (1, (92, 80, 70))])
    street = mask_from(lambda d: d.polygon([(0, H), (W, H), (W, 540), (0, 600)], fill=255))
    img = blend(img, (70, 62, 56), street)
    tram = mask_from(lambda d: d.rounded_rectangle([120, 300, 1180, 560], 40, fill=255), 1)
    img = blend(img, (218, 136, 46), tram)
    win = mask_from(lambda d: [d.rectangle([180 + i * 170, 330, 300 + i * 170, 430], fill=255)
                               for i in range(6)], 1)
    img = blend(img, (250, 228, 180), win * 0.85)
    stripe = mask_from(lambda d: d.rectangle([120, 470, 1180, 485], fill=255))
    img = blend(img, (250, 236, 210), stripe)
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    pil = pil.filter(ImageFilter.BoxBlur(0)).transform(pil.size, Image.AFFINE, (1, 0, 0, 0, 1, 0))
    arr = np.asarray(pil, dtype=np.float64)
    blurred = np.asarray(pil.filter(ImageFilter.GaussianBlur(0)), dtype=np.float64)
    k = 41
    pad = np.pad(blurred, ((0, 0), (k // 2, k // 2), (0, 0)), mode="edge")
    cs = np.cumsum(pad, axis=1)
    motion = (cs[:, k:] - cs[:, :-k]) / k
    motion = np.concatenate([motion, motion[:, -1:]], axis=1)[:, :W]
    band = np.zeros((H, 1))
    band[290:570] = 1
    img = arr * (1 - band[..., None] * 0.85) + motion * band[..., None] * 0.85
    wire = mask_from(lambda d: d.line([(0, 120), (W, 90)], fill=255, width=3))
    img = blend(img, (40, 36, 34), wire)
    return finish(img)


def f6_product():
    img = vgrad([(0, (36, 26, 21)), (1, (18, 14, 12))])
    img = blend(img, (176, 98, 52), radial(640, 300, 520) * 0.6)
    plinth = mask_from(lambda d: d.rectangle([0, 520, W, H], fill=255))
    img = blend(img, (26, 20, 17), plinth)
    cup = mask_from(lambda d: [d.pieslice([520, 330, 760, 560], 0, 180, fill=255),
                               d.rectangle([520, 435, 760, 448], fill=255),
                               d.ellipse([480, 505, 800, 535], fill=255)], 1.2)
    img = blend(img, (240, 233, 220), cup)
    handle = mask_from(lambda d: d.arc([735, 455, 805, 515], 270, 90, fill=255, width=12), 1)
    img = blend(img, (240, 233, 220), handle)
    band = mask_from(lambda d: d.rectangle([548, 470, 732, 486], fill=255))
    img = blend(img, (200, 116, 58), band)
    img = blend(img, (255, 230, 190), radial(560, 360, 90) ** 2 * 0.35)
    return finish(img, vignette=0.55)


FRAMES = [f1_rooftops, f2_window, f3_steam, f4_overhead, f5_tram, f6_product]

SHOT_LIST = {
    "title": "The First Light",
    "style": "Warm natural light, 35 mm, soft contrast, Milanese palette of terracotta and cream",
    "shots": [
        {"id": 1, "description": "Milan rooftops at dawn, the Duomo's spire catching the first sun",
         "camera": "slow pan left to right", "duration": 3.0},
        {"id": 2, "description": "A kitchen window throws a long shaft of light across the wall; a cup waits on the table",
         "camera": "slow push-in", "duration": 2.5},
        {"id": 3, "description": "Steam curls up from a fresh espresso, crema glowing in the dark",
         "camera": "static, macro", "duration": 2.5},
        {"id": 4, "description": "Overhead: the cup on its saucer, a spoon resting on the walnut table",
         "camera": "tilt down", "duration": 2.0},
        {"id": 5, "description": "An orange tram slides past the café window, the city waking up",
         "camera": "tracking right", "duration": 2.0},
        {"id": 6, "description": "The NOVARA cup alone on a dark plinth, warm light from the left",
         "camera": "slow push-in", "duration": 3.0},
    ],
}

TIMING = {"fps": 24, "total_seconds": 15.0, "shots": []}
_t = 0.0
for _i, _s in enumerate(SHOT_LIST["shots"]):
    TIMING["shots"].append({"shot": _i + 1, "start_s": _t, "end_s": _t + _s["duration"],
                            "duration_s": _s["duration"]})
    _t += _s["duration"]

VERDICTS = [
    {"shot": 1, "score": 8.6, "route": "pass", "pass": True, "issues": []},
    {"shot": 2, "score": 9.1, "route": "pass", "pass": True, "issues": []},
    {"shot": 3, "score": 7.4, "route": "pass", "pass": True, "issues": ["steam slightly too regular"]},
    {"shot": 4, "score": 6.2, "route": "detail_fix", "pass": False,
     "issues": ["spoon edge melts into the saucer", "grain direction inconsistent"]},
    {"shot": 5, "score": 8.0, "route": "pass", "pass": True, "issues": []},
    {"shot": 6, "score": 9.3, "route": "pass", "pass": True, "issues": []},
]

SUPERS = "1 | The first light in Milan.\n6 | NOVARA. Il primo caffè."
VO = ("Milan wakes slowly.\nLight finds the kitchen first.\nThen the coffee.\n"
      "A moment that is only yours.\nThe city can wait.\nNOVARA. The first light of the day.")
RULES = """- The NOVARA wordmark sits on the end card only, centred, never over the coffee.
- Clear space equals the height of the letter N on every side.
- Supers in sentence case, display serif, white or cream, never more than two lines.
- Terracotta is the accent: reserved for the call to action and the cup band.
- No people's faces: hands and objects carry the story."""


def make_logo(text="NOVARA"):
    """A simple wordmark logo with transparency (dark letters + MASK, 1 = transparent)."""
    from PIL import ImageFont
    font = ImageFont.truetype(os.path.join(ROOT, "fonts", "InstrumentSerif-Regular.ttf"), 160)
    w = int(font.getlength(text)) + 40
    im = Image.new("L", (w, 200), 0)
    ImageDraw.Draw(im).text((20, 10), text, font=font, fill=255)
    alpha = np.asarray(im, dtype=np.float32) / 255.0
    rgb = np.zeros(alpha.shape + (3,), np.float32) + np.array([0.12, 0.08, 0.06], np.float32)
    return rgb, 1.0 - alpha


def build(frames=None):
    frames = frames or [f() for f in FRAMES]
    logo_rgb, logo_mask = make_logo()
    return tr.build_project(
        frames, json.dumps(SHOT_LIST), brand="NOVARA",
        client_line="Spec commercial · 15 s · 16:9", director="Somar Awad",
        logline="Before Milan starts talking, there is a minute of light that belongs to "
                "nobody. NOVARA is the coffee you drink in it.",
        tone_words="warm, unhurried, Milanese morning, tactile",
        verdicts_json=json.dumps(VERDICTS), timing_json=json.dumps(TIMING),
        supers_text=SUPERS, voiceover_text=VO, brand_rules=RULES,
        logo=tr.logo_from_arrays(logo_rgb, logo_mask), date_text="2 November 2026")


def main():
    out = HERE
    frames = [f() for f in FRAMES]
    P = build(frames)
    doc = tr.render_treatment(P, "editorial dark", "A4")
    tr.save_pdf(doc, os.path.join(out, "treatment_editorial_dark_A4.pdf"), quality=84)
    deck = tr.render_deck(P, "editorial light", "1920x1080")
    tr.save_pdf(deck, os.path.join(out, "pitch_deck_editorial_light_1920.pdf"), quality=84)
    sb = tr.render_storyboard(P, "A4 landscape", 3)
    tr.save_pdf(sb, os.path.join(out, "storyboard_A4.pdf"), quality=84)

    def png(img, name, width):
        tr.previews([img], width)[0].save(os.path.join(out, name), optimize=True)

    idx = {n: i for i, n in enumerate(doc.names)}
    for name in ("cover", "idea", "mood", "palette", "story", "shots-1", "brand", "notes"):
        png(doc.pages[idx[name]], f"treatment_{name}.png", 760)
    for t in ("editorial light", "agency minimal"):
        d = tr.render_treatment(P, t, "A4", dpi=150)
        png(d.pages[0], f"treatment_cover_{t.replace(' ', '_')}.png", 760)
        png(d.pages[d.names.index("shots-1")], f"treatment_shots_{t.replace(' ', '_')}.png", 760)
    didx = {n: i for i, n in enumerate(deck.names)}
    for name in ("cover", "palette", "shot-2", "timeline"):
        png(deck.pages[didx[name]], f"deck_{name}.png", 1200)
    dark = tr.render_deck(P, "editorial dark", "1920x1080")
    png(dark.pages[0], "deck_cover_editorial_dark.png", 1200)
    png(sb.pages[0], "storyboard.png", 1400)
    card = tr.render_palette_card(P.palette, P.accent, 1600)
    png(card, "palette_card.png", 1200)
    print("pages:", len(doc.pages), "slides:", len(deck.pages), "sheets:", len(sb.pages))


if __name__ == "__main__":
    main()
