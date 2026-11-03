"""Colour palettes from keyframes, plus contrast helpers.

numpy + stdlib only (no ComfyUI, no torch) so it is unit-testable on its own.

Pipeline
--------
1. Every keyframe is downsampled (box filter) to about ``SAMPLE_SIDE`` pixels on its long
   side, so a 4K frame and a 512 px frame weigh the same.
2. Pixels are converted to CIE Lab (D65) and clustered with k-means (k-means++ seeding,
   fixed seed, so the same frames always give the same palette).
3. Clusters closer than ``MERGE_DELTA_E`` (CIE76 delta E) are merged, weighted by size:
   two slightly different browns become one brown.
4. Swatches are sorted by share of the picture and named from a curated list of
   evocative colour names ("espresso brown", "dusty rose", ...) by nearest Lab distance.

Contrast follows WCAG 2.x: relative luminance and contrast ratio; AA is 4.5:1 for body
text and 3:1 for large text.
"""

import math

import numpy as np

SAMPLE_SIDE = 96          # long side of each downsampled frame used for clustering
DEFAULT_COLORS = 6        # swatches returned by default
KMEANS_K = 10             # clusters before merging (more than DEFAULT_COLORS on purpose)
KMEANS_ITERS = 30
MERGE_DELTA_E = 12.0      # clusters closer than this (CIE76) are the same colour to the eye
MIN_SHARE = 0.004         # clusters smaller than 0.4 % of the picture are noise

AA_NORMAL = 4.5
AA_LARGE = 3.0

# --------------------------------------------------------------------------- hex / rgb


def hex_to_rgb(value, default=None):
    """'#A1B2C3', 'a1b2c3', '#abc' -> (161, 178, 195). Invalid -> default (or ValueError)."""
    s = str(value or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        if len(s) != 6:
            raise ValueError
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        if default is not None:
            return tuple(default)
        raise ValueError(f"{value!r} is not a hex colour like #C8743A") from None


def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*(int(max(0, min(255, round(c)))) for c in rgb[:3]))


# --------------------------------------------------------------------------- colour spaces

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def rgb_to_lab(rgb):
    """sRGB (0-255, [..., 3]) -> CIE Lab D65 [..., 3]."""
    lin = _srgb_to_linear(np.asarray(rgb, dtype=np.float64) / 255.0)
    xyz = lin @ _M_RGB2XYZ.T / _WHITE
    e, k = 216 / 24389, 24389 / 27
    f = np.where(xyz > e, np.cbrt(xyz), (k * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab):
    """CIE Lab [..., 3] -> sRGB 0-255 floats (clipped to gamut)."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    e, k = 216 / 24389, 24389 / 27
    def inv(f):
        return np.where(f ** 3 > e, f ** 3, (116 * f - 16) / k)
    xyz = np.stack([inv(fx), np.where(lab[..., 0] > k * e, fy ** 3, lab[..., 0] / k), inv(fz)],
                   axis=-1) * _WHITE
    return _linear_to_srgb(xyz @ _M_XYZ2RGB.T) * 255.0


def delta_e(lab1, lab2):
    """CIE76 colour difference (Euclidean distance in Lab)."""
    return np.linalg.norm(np.asarray(lab1, float) - np.asarray(lab2, float), axis=-1)


def chroma(lab):
    lab = np.asarray(lab, float)
    return np.hypot(lab[..., 1], lab[..., 2])


# --------------------------------------------------------------------------- WCAG contrast

def relative_luminance(rgb):
    lin = _srgb_to_linear(np.asarray(rgb[:3], dtype=np.float64) / 255.0)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def contrast_ratio(a, b):
    """WCAG contrast ratio of two colours (rgb tuples or hex strings), 1.0 - 21.0."""
    a = hex_to_rgb(a) if isinstance(a, str) else a
    b = hex_to_rgb(b) if isinstance(b, str) else b
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def meets_aa(fg, bg, large=False):
    return contrast_ratio(fg, bg) >= (AA_LARGE if large else AA_NORMAL) - 1e-9


def pick_text_color(bg, candidates=("#FFFFFF", "#111111"), large=False):
    """The first candidate that reaches WCAG AA on bg; if none does, the one with the
    highest contrast (pure black or white is added as a last resort). Returns hex."""
    bg = hex_to_rgb(bg) if isinstance(bg, str) else tuple(bg)
    for c in candidates:
        if meets_aa(hex_to_rgb(c), bg, large):
            return rgb_to_hex(hex_to_rgb(c))
    pool = [hex_to_rgb(c) for c in candidates] + [(255, 255, 255), (0, 0, 0)]
    return rgb_to_hex(max(pool, key=lambda c: contrast_ratio(c, bg)))


def ensure_contrast(fg, bg, min_ratio=AA_NORMAL):
    """Shift fg's Lab lightness (keeping hue and chroma) just enough to reach min_ratio on
    bg. Lighter on dark backgrounds, darker on light ones. Returns hex."""
    fg = hex_to_rgb(fg) if isinstance(fg, str) else tuple(int(round(c)) for c in fg)
    bg = hex_to_rgb(bg) if isinstance(bg, str) else tuple(bg)
    if contrast_ratio(fg, bg) >= min_ratio:
        return rgb_to_hex(fg)
    lab = rgb_to_lab(fg)
    toward_white = relative_luminance(bg) < 0.18
    lo, hi = (lab[0], 100.0) if toward_white else (0.0, lab[0])
    best = None
    for _ in range(30):
        mid = (lo + hi) / 2
        c = tuple(int(round(v)) for v in lab_to_rgb([mid, lab[1], lab[2]]))
        if contrast_ratio(c, bg) >= min_ratio:
            best = c
            if toward_white:
                hi = mid
            else:
                lo = mid
        elif toward_white:
            lo = mid
        else:
            hi = mid
    if best is None:
        return pick_text_color(bg, ())
    return rgb_to_hex(best)


def mix(a, b, t):
    """Linear blend of two rgb tuples (t=0 -> a, t=1 -> b)."""
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


# --------------------------------------------------------------------------- names

# Curated evocative names (hex anchors). Matching is nearest in Lab with lightness weighted
# a little less than hue, so "dusty rose" is found for muted pinks of any lightness nearby.
COLOR_NAMES = [
    ("espresso brown", "#3B2A20"), ("cocoa", "#5A3E2E"), ("chestnut", "#7A4A30"),
    ("walnut", "#5E4433"), ("mahogany", "#6B2F24"), ("oxblood", "#4E1A1A"),
    ("caramel", "#B5753C"), ("cinnamon", "#9C5A2E"), ("tobacco", "#7C5B3A"),
    ("camel", "#C19A6B"), ("latte", "#C7A88A"), ("mocha", "#8B6B57"),
    ("terracotta", "#C0623F"), ("burnt sienna", "#A9502E"), ("rust", "#9A4523"),
    ("copper", "#B8733D"), ("bronze", "#8C6A3A"), ("burnt orange", "#C9581F"),
    ("tangerine", "#EE8A2C"), ("apricot", "#F2B47E"), ("peach", "#F6C7A6"),
    ("amber", "#E39B2D"), ("honey", "#D8A24A"), ("saffron", "#EBB33A"),
    ("mustard", "#C99A2E"), ("ochre", "#B98B3A"), ("gold", "#C9A84C"),
    ("butter yellow", "#F2DE8C"), ("lemon", "#EEDD4A"), ("cream", "#F1E6CC"),
    ("ivory", "#F5F0E3"), ("bone", "#E3DACA"), ("linen", "#EDE3D2"),
    ("oat", "#D9CBB0"), ("sand", "#CDB894"), ("beige", "#D6C4A4"),
    ("taupe", "#8E7F70"), ("khaki", "#A89B70"), ("stone grey", "#9C978E"),
    ("warm grey", "#8A8178"), ("cool grey", "#8C9299"), ("pewter", "#6E7276"),
    ("charcoal", "#333436"), ("graphite", "#4A4B4D"), ("soot", "#1E1D1C"),
    ("ink black", "#101216"), ("jet black", "#0A0A0A"), ("smoke", "#B9B7B3"),
    ("ash", "#D2D0CC"), ("silver", "#C3C6CA"), ("pearl", "#ECEAE4"),
    ("chalk white", "#F7F6F2"), ("paper white", "#FBFAF7"),
    ("brick red", "#9E3B2C"), ("crimson", "#B3202E"), ("cherry", "#8F1D2C"),
    ("scarlet", "#D62D20"), ("vermilion", "#E04A23"), ("tomato", "#E2533D"),
    ("coral", "#EE7A63"), ("salmon", "#F09A84"), ("blush pink", "#F2C6C2"),
    ("dusty rose", "#C48B8B"), ("rosewood", "#7D3F3F"), ("raspberry", "#B3245A"),
    ("fuchsia", "#D23A9A"), ("magenta", "#B5307F"), ("orchid", "#C27EC0"),
    ("mauve", "#A7889A"), ("heather", "#9C8BA6"), ("lilac", "#C3B1D6"),
    ("lavender", "#B7AEE0"), ("violet", "#6E4AA8"), ("plum", "#5E2F52"),
    ("aubergine", "#3C1F3A"), ("indigo", "#2E2E6E"), ("navy", "#1D2A4A"),
    ("midnight blue", "#141C33"), ("cobalt", "#2449A8"), ("ultramarine", "#2D3FB3"),
    ("denim", "#3F5F8C"), ("steel blue", "#4F7196"), ("slate blue", "#5A6A86"),
    ("cornflower", "#6F8FD8"), ("sky blue", "#8EC2E8"), ("powder blue", "#BBD3E3"),
    ("ocean", "#1E5E7A"), ("petrol", "#1F4B55"), ("teal", "#2A7A78"),
    ("turquoise", "#3BB8B0"), ("aqua", "#7FD6D0"), ("seafoam", "#A8D8C2"),
    ("mint", "#B8E2C8"), ("jade", "#3F9A73"), ("emerald", "#1F7A4D"),
    ("bottle green", "#1E4A35"), ("forest green", "#2F4A2E"), ("pine", "#24402F"),
    ("moss green", "#68703A"), ("olive", "#6E6A35"), ("sage", "#9CAA8C"),
    ("eucalyptus", "#7F9C8A"), ("fern", "#5E8A4E"), ("avocado", "#7F8C3A"),
    ("pistachio", "#B5C98A"), ("lime", "#9CCB3B"), ("chartreuse", "#C3D63A"),
]
_NAME_LAB = rgb_to_lab(np.array([hex_to_rgb(h) for _, h in COLOR_NAMES], dtype=np.float64))


def color_name(rgb):
    """Nearest evocative name for an sRGB colour (tuple or hex)."""
    rgb = hex_to_rgb(rgb) if isinstance(rgb, str) else rgb
    lab = rgb_to_lab(np.asarray(rgb, dtype=np.float64))
    d = _NAME_LAB - lab
    dist = (0.8 * d[:, 0]) ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2
    return COLOR_NAMES[int(np.argmin(dist))][0]


# --------------------------------------------------------------------------- sampling

def to_uint8_rgb(img):
    """[H,W,3|4] float 0-1 / uint8 / PIL image -> uint8 [H,W,3]."""
    if hasattr(img, "convert") and hasattr(img, "size"):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(np.asarray(a, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return a


def downsample(img, side=SAMPLE_SIDE):
    """Box-filter downsample so the long side is at most `side` pixels. -> float [h,w,3]."""
    a = to_uint8_rgb(img).astype(np.float32)
    h, w = a.shape[:2]
    f = max(1, int(math.ceil(max(h, w) / float(side))))
    if f == 1:
        return a
    h2, w2 = h // f, w // f
    if h2 == 0 or w2 == 0:
        return a
    return a[:h2 * f, :w2 * f].reshape(h2, f, w2, f, 3).mean(axis=(1, 3))


def sample_pixels(images, side=SAMPLE_SIDE):
    """All downsampled pixels of all images as an [N,3] float array (0-255)."""
    parts = [downsample(im, side).reshape(-1, 3) for im in images]
    parts = [p for p in parts if len(p)]
    if not parts:
        raise ValueError("no pixels to sample (no keyframes?)")
    return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------- k-means

def kmeans(points, k, iters=KMEANS_ITERS, seed=0):
    """Plain k-means with k-means++ seeding. Returns (centers [k',D], labels [N]).
    k' <= k: seeding stops early when the points have fewer distinct values than k."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    rng = np.random.default_rng(seed)
    centers = [pts[rng.integers(n)]]
    d2 = np.sum((pts - centers[0]) ** 2, axis=1)
    while len(centers) < k:
        total = d2.sum()
        if total <= 1e-9:
            break
        idx = rng.choice(n, p=d2 / total)
        centers.append(pts[idx])
        d2 = np.minimum(d2, np.sum((pts - pts[idx]) ** 2, axis=1))
    centers = np.array(centers)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        dist = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = dist.argmin(1)
        if _ > 0 and np.array_equal(new, labels):
            break
        labels = new
        for j in range(len(centers)):
            m = labels == j
            if m.any():
                centers[j] = pts[m].mean(0)
    return centers, labels


def merge_clusters(labs, weights, threshold=MERGE_DELTA_E):
    """Merge clusters whose Lab centres are closer than `threshold`, closest pair first,
    weighting by size. Returns (labs [m,3], weights [m])."""
    labs = [np.asarray(l, dtype=np.float64) for l in labs]
    weights = [float(w) for w in weights]
    while len(labs) > 1:
        best = None
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                d = float(delta_e(labs[i], labs[j]))
                if d < threshold and (best is None or d < best[0]):
                    best = (d, i, j)
        if best is None:
            break
        _, i, j = best
        w = weights[i] + weights[j]
        labs[i] = (labs[i] * weights[i] + labs[j] * weights[j]) / max(w, 1e-12)
        weights[i] = w
        del labs[j], weights[j]
    return np.array(labs), np.array(weights)


def extract_palette(images, colors=DEFAULT_COLORS, k=KMEANS_K, merge_threshold=MERGE_DELTA_E,
                    seed=0):
    """Palette of up to `colors` swatches from a list of images.

    Returns a list of dicts sorted by share (largest first):
    {"hex", "rgb", "lab", "name", "share"}; shares sum to 1 over the returned swatches."""
    if not isinstance(images, (list, tuple)):
        images = [images]
    rgb = sample_pixels(images)
    lab = rgb_to_lab(rgb)
    centers, labels = kmeans(lab, max(1, int(k)), seed=seed)
    counts = np.bincount(labels, minlength=len(centers)).astype(np.float64)
    keep = counts > 0
    labs, weights = merge_clusters(centers[keep], counts[keep], merge_threshold)
    shares = weights / weights.sum()
    order = np.argsort(-shares)
    order = [i for i in order if shares[i] >= MIN_SHARE] or [int(order[0])]
    order = order[:max(1, int(colors))]
    total = sum(shares[i] for i in order)
    out = []
    for i in order:
        c = tuple(int(round(v)) for v in np.clip(lab_to_rgb(labs[i]), 0, 255))
        out.append({"hex": rgb_to_hex(c), "rgb": list(c),
                    "lab": [round(float(v), 2) for v in labs[i]],
                    "name": color_name(c), "share": round(float(shares[i] / total), 4)})
    _unique_names(out)
    return out


def _unique_names(swatches):
    """Two swatches with the same name get "deep" / "pale" so the list reads well."""
    seen = {}
    for s in swatches:
        seen.setdefault(s["name"], []).append(s)
    for name, group in seen.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda s: s["lab"][0])
        for i, s in enumerate(group):
            if i == 0:
                s["name"] = f"deep {name}"
            elif i == len(group) - 1:
                s["name"] = f"pale {name}"
            else:
                s["name"] = f"mid {name}"


def pick_accent(swatches):
    """The swatch that works best as an accent: colourful, not too dark or light, and
    present enough to feel like part of the film. Returns the swatch dict."""
    if not swatches:
        return None
    def score(s):
        L, a, b = s["lab"]
        c = math.hypot(a, b)
        light_penalty = 1.0 if 25 <= L <= 82 else 0.45
        return (c + 4) * light_penalty * (0.35 + math.sqrt(s["share"]))
    return max(swatches, key=score)


def palette_summary(swatches, accent_hex=None):
    """JSON-friendly summary used by the Palette From Shots node."""
    acc = accent_hex or (pick_accent(swatches) or {"hex": "#888888"})["hex"]
    return {"swatches": swatches, "accent_hex": acc,
            "text_on_accent_hex": pick_text_color(acc)}
