"""Ad Assembler core logic: timeline, transitions, supers, end card, audio mix, script.

Pure functions on numpy / torch / PIL only (no ComfyUI imports), so everything here can be
unit-tested on its own. Image tensors follow ComfyUI's IMAGE layout: float [B, H, W, 3] in 0-1.
Audio waveforms are float tensors [channels, samples] here; the nodes add/remove the batch dim.
"""

import json
import math
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:5"]
TRANSITIONS = ["cut", "crossfade", "dip-to-black", "whip-pan blur"]
SUPER_POSITIONS = ["lower third", "center", "top"]
SUPER_STYLES = ["white with soft shadow", "black box", "outline"]
DEFAULT_SHOT_SECONDS = 3.0


class AdAssemblyError(ValueError):
    pass


# --------------------------------------------------------------------------- durations & layout

def aspect_size(width, aspect_ratio):
    """Output (width, height) for a width and an aspect ratio like "9:16". Both even (for H.264)."""
    try:
        aw, ah = (float(x) for x in str(aspect_ratio).split(":"))
    except ValueError as e:
        raise AdAssemblyError(f"bad aspect ratio {aspect_ratio!r}") from e
    w = max(2, int(round(width / 2)) * 2)
    h = max(2, int(round(w * ah / aw / 2)) * 2)
    return w, h


def parse_durations(text):
    """"3, 2.5 4" -> [3.0, 2.5, 4.0]. Commas, semicolons or spaces separate values; an
    optional trailing "s" is allowed ("2.5s")."""
    out = []
    for tok in re.split(r"[,;\s]+", str(text or "").strip()):
        if not tok:
            continue
        tok = tok.lower().rstrip("s")
        try:
            v = float(tok)
        except ValueError as e:
            raise AdAssemblyError(f"durations: {tok!r} is not a number") from e
        if v <= 0:
            raise AdAssemblyError(f"durations: {v} must be > 0")
        out.append(v)
    return out


def durations_from_shot_list(shot_list_json):
    """Durations (seconds) of each shot in a Shot Planner shot_list_json string (or dict)."""
    if not shot_list_json:
        return []
    try:
        plan = json.loads(shot_list_json) if isinstance(shot_list_json, str) else shot_list_json
        shots = plan["shots"]
    except (ValueError, KeyError, TypeError) as e:
        raise AdAssemblyError("shot_list_json is not a Shot Planner list") from e
    out = []
    for s in shots:
        try:
            d = float(s.get("duration", DEFAULT_SHOT_SECONDS))
        except (TypeError, ValueError, AttributeError):
            d = DEFAULT_SHOT_SECONDS
        out.append(d if d > 0 else DEFAULT_SHOT_SECONDS)
    return out


def resolve_durations(n_shots, shot_list_json="", durations_text="", default=DEFAULT_SHOT_SECONDS):
    """One duration per shot. Priority: durations string, then shot_list_json, then default."""
    manual = parse_durations(durations_text)
    planned = durations_from_shot_list(shot_list_json)
    out = []
    for i in range(n_shots):
        if i < len(manual):
            out.append(manual[i])
        elif i < len(planned):
            out.append(planned[i])
        else:
            out.append(float(default))
    return out


def seconds_to_frames(seconds, fps):
    """Frame count for a duration (at least 1)."""
    return max(1, int(round(float(seconds) * float(fps))))


def plan_timeline(frame_counts, transition="cut", transition_frames=6):
    """Lay out shots on the timeline.

    Each shot "owns" frame_counts[i] frames between its cut points, so the total length is
    always sum(frame_counts), whatever the transition. Overlapping transitions (crossfade,
    whip-pan) are centred on the cut point: the outgoing shot runs on for the second half of
    the window and the incoming shot starts half a window early. dip-to-black does not
    overlap; it fades the outgoing shot to black before the cut and the new one up after it.

    Returns a dict with total_frames, the effective transition length T, and per shot:
    start/end (cut points, end exclusive) and render_start/render_end (frames where the shot
    is visible, including overlaps).
    """
    if transition not in TRANSITIONS:
        raise AdAssemblyError(f"unknown transition {transition!r}")
    counts = [int(c) for c in frame_counts]
    if not counts or min(counts) < 1:
        raise AdAssemblyError("need at least one shot with at least one frame")
    n = len(counts)
    T = 0 if transition == "cut" or n < 2 else max(0, min(int(transition_frames), min(counts)))
    lead = T // 2              # frames of the window before each cut point
    overlap = transition in ("crossfade", "whip-pan blur") and T > 0
    total = sum(counts)
    shots, cursor = [], 0
    for i, c in enumerate(counts):
        start, end = cursor, cursor + c
        r0 = start - lead if (overlap and i > 0) else start
        r1 = end - lead + T if (overlap and i < n - 1) else end
        shots.append({"start": start, "end": end, "render_start": r0, "render_end": r1})
        cursor = end
    return {"total_frames": total, "transition": transition, "transition_frames": T,
            "lead": lead, "shots": shots}


def timing_dict(layout, fps, width=None, height=None, kinds=None):
    """Seconds-based timing for downstream nodes (supers, audio)."""
    fps = float(fps)
    shots = []
    for i, s in enumerate(layout["shots"]):
        d = {"shot": i + 1, "start_frame": s["start"], "end_frame": s["end"],
             "start_s": round(s["start"] / fps, 4), "end_s": round(s["end"] / fps, 4),
             "duration_s": round((s["end"] - s["start"]) / fps, 4)}
        if kinds:
            d["source"] = kinds[i]
        shots.append(d)
    out = {"fps": fps, "total_frames": layout["total_frames"],
           "total_seconds": round(layout["total_frames"] / fps, 4),
           "transition": layout["transition"], "transition_frames": layout["transition_frames"],
           "shots": shots}
    if width:
        out["width"], out["height"] = int(width), int(height)
    return out


def load_timing(timing_json):
    """Parse timing_json (string or dict); None/'' -> None."""
    if not timing_json:
        return None
    try:
        t = json.loads(timing_json) if isinstance(timing_json, str) else timing_json
        t["shots"], t["fps"]
    except (ValueError, KeyError, TypeError) as e:
        raise AdAssemblyError("timing_json is not Ad Timeline output") from e
    return t


# --------------------------------------------------------------------------- image helpers

def center_crop_box(src_w, src_h, dst_w, dst_h):
    """(x0, y0, w, h) of the largest centred box in the source with the target aspect."""
    target = dst_w / dst_h
    if src_w / src_h > target:
        w, h = src_h * target, float(src_h)
    else:
        w, h = float(src_w), src_w / target
    return (src_w - w) / 2.0, (src_h - h) / 2.0, w, h


def center_crop_resize(images, width, height):
    """Center-crop a [B,H,W,3] batch to the target aspect, then resize to width x height."""
    b, h, w, c = images.shape
    x0, y0, cw, ch = center_crop_box(w, h, width, height)
    x0, y0 = int(round(x0)), int(round(y0))
    cw, ch = max(1, int(round(cw))), max(1, int(round(ch)))
    crop = images[:, y0:y0 + ch, x0:x0 + cw, :]
    if (cw, ch) == (width, height):
        return crop.contiguous()
    x = crop.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False,
                      antialias=(cw > width or ch > height))
    return x.permute(0, 2, 3, 1).clamp(0, 1).contiguous()


def clip_frame_index(i, n_src):
    """Source frame for output frame i of a clip: play through, then hold the last frame."""
    return min(max(int(i), 0), n_src - 1)


def ken_burns_scale(i, n, strength):
    """Zoom factor for frame i of n: linear push-in from 1.0 to 1 + strength."""
    if n <= 1:
        return 1.0
    return 1.0 + float(strength) * (i / (n - 1))


def zoom_frame(image, scale, width, height):
    """Centred zoom of one [H,W,3] image (already at the output aspect) by `scale` (>=1),
    sampled with sub-pixel precision so slow push-ins don't jitter."""
    x = image.permute(2, 0, 1).unsqueeze(0)
    s = 1.0 / float(scale)
    theta = torch.tensor([[[s, 0.0, 0.0], [0.0, s, 0.0]]], dtype=x.dtype, device=x.device)
    grid = F.affine_grid(theta, (1, 3, height, width), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0].permute(1, 2, 0).clamp(0, 1)


class ShotSource:
    """Produces output frames for one shot: a still (Ken Burns push-in) or a clip
    (trimmed, or held on the last frame), center-cropped to the output size."""

    def __init__(self, images, width, height, n_frames, ken_burns=0.0):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        self.images, self.width, self.height = images, int(width), int(height)
        self.n_frames, self.strength = int(n_frames), float(ken_burns)
        self.is_still = images.shape[0] == 1
        self._cache = (None, None)
        if self.is_still:
            # Pre-scale once to output size * max zoom so the per-frame zoom is a light resample.
            up = 1.0 + max(self.strength, 0.0)
            pw, ph = int(math.ceil(width * up)), int(math.ceil(height * up))
            self._still = center_crop_resize(images, pw, ph)[0]

    @property
    def kind(self):
        return "still" if self.is_still else "clip"

    def frame(self, i):
        if self.is_still:
            if self.strength <= 0:
                key = 0
                if self._cache[0] != key:
                    self._cache = (key, zoom_frame(self._still, 1.0, self.width, self.height))
                return self._cache[1]
            return zoom_frame(self._still, ken_burns_scale(i, self.n_frames, self.strength),
                              self.width, self.height)
        j = clip_frame_index(i, self.images.shape[0])
        if self._cache[0] != j:
            self._cache = (j, center_crop_resize(self.images[j:j + 1], self.width, self.height)[0])
        return self._cache[1]


# --------------------------------------------------------------------------- transitions

def crossfade(a, b, alpha):
    """Linear blend: alpha 0 -> a, 1 -> b."""
    return a * (1.0 - alpha) + b * alpha


def transition_progress(j, T):
    """Progress (0..1, exclusive) for frame j of a T-frame overlapping transition."""
    return (j + 1) / (T + 1)


def dip_gain(offset, T):
    """Brightness for a frame `offset` frames from a cut point (negative = before the cut)
    in a T-frame dip-to-black. The last frame before the cut is fully black."""
    if T <= 0:
        return 1.0
    before, after = T // 2, T - T // 2
    if -before <= offset < 0:
        return (-offset - 1) / before
    if 0 <= offset < after:
        return (offset + 1) / (after + 1)
    return 1.0


def horizontal_box_blur(image, radius):
    """Box blur along the width of one [H,W,3] image (motion blur for whip-pans)."""
    r = int(radius)
    if r < 1:
        return image
    x = image.permute(2, 0, 1).unsqueeze(0)             # [1,3,H,W]
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    c = torch.cumsum(x, dim=3)
    c = F.pad(c, (1, 0, 0, 0))
    out = (c[..., 2 * r + 1:] - c[..., :-2 * r - 1]) / (2 * r + 1)
    return out[0].permute(1, 2, 0)


def whip_pan(a, b, p, max_blur=0.12):
    """Whip-pan from a to b at progress p: both slide left, joined, with motion blur that
    peaks mid-move."""
    h, w, _ = a.shape
    shift = int(round(min(max(p, 0.0), 1.0) * w))
    strip = torch.cat([a, b], dim=1)[:, shift:shift + w, :]
    return horizontal_box_blur(strip, math.sin(math.pi * p) * max_blur * w).clamp(0, 1)


def assemble_timeline(shots, durations, fps, width, height, transition="cut",
                      transition_frames=6, ken_burns=0.05):
    """Render the full commercial. `shots` is a list of [B,H,W,3] tensors (B=1: still).
    Returns (frames [N,H,W,3], timing dict)."""
    if len(shots) != len(durations):
        raise AdAssemblyError("one duration per shot is required")
    counts = [seconds_to_frames(d, fps) for d in durations]
    layout = plan_timeline(counts, transition, transition_frames)
    T, lead = layout["transition_frames"], layout["lead"]
    sources = [ShotSource(img, width, height, s["render_end"] - s["render_start"], ken_burns)
               for img, s in zip(shots, layout["shots"])]
    out = torch.empty((layout["total_frames"], height, width, 3), dtype=torch.float32)
    n = len(sources)
    dip = transition == "dip-to-black" and T > 0
    # Pass 1: every shot draws the frames between its own cut points.
    for i, (src, s) in enumerate(zip(sources, layout["shots"])):
        for f in range(s["start"], s["end"]):
            fr = src.frame(f - s["render_start"])
            if dip:
                g = min(dip_gain(f - s["end"], T) if i < n - 1 else 1.0,
                        dip_gain(f - s["start"], T) if i > 0 else 1.0)
                fr = fr * g
            out[f] = fr
    # Pass 2: overlapping transitions overwrite the T-frame window centred on each cut.
    if transition in ("crossfade", "whip-pan blur") and T > 0:
        for i in range(n - 1):
            s, ns = layout["shots"][i], layout["shots"][i + 1]
            for j in range(T):
                f = s["end"] - lead + j
                a = sources[i].frame(f - s["render_start"])
                b = sources[i + 1].frame(f - ns["render_start"])
                p = transition_progress(j, T)
                out[f] = crossfade(a, b, p) if transition == "crossfade" else whip_pan(a, b, p)
    timing = timing_dict(layout, fps, width, height, [s.kind for s in sources])
    return out, timing


# --------------------------------------------------------------------------- text

_FONT_CANDIDATES = [
    "DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arialbd.ttf", "arial.ttf", "Arial Bold.ttf",
    "Arial.ttf", "Helvetica.ttc", "LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf", "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
]


def load_font(size, font_path=""):
    """A TrueType font at `size` px: font_path if given, else DejaVuSans / Arial, else PIL's
    built-in font (scalable on Pillow >= 10.1)."""
    size = max(1, int(size))
    for cand in ([font_path] if font_path else []) + _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, size)
        except (OSError, ValueError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text_width(font, text):
    box = font.getbbox(text)
    return box[2] - box[0]


def wrap_text(text, font, max_width):
    """Word-wrap so each line fits max_width px. Explicit newlines (or literal "\\n") are kept."""
    lines = []
    for para in str(text).replace("\\n", "\n").split("\n"):
        words, cur = para.split(), ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if cur and _text_width(font, trial) > max_width:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        lines.append(cur)
    return "\n".join(lines)


def parse_hex_color(value, default=(255, 255, 255)):
    """"#RRGGBB", "RRGGBB" or "#RGB" -> (r, g, b) ints."""
    s = str(value or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        if value:
            raise AdAssemblyError(f"bad hex colour {value!r}")
        return tuple(default)
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def render_text_layer(width, height, text, font_size=56, position="lower third",
                      style="white with soft shadow", font_path="", color=(255, 255, 255)):
    """RGBA uint8 numpy [H,W,4] with the styled text placed in the frame."""
    if position not in SUPER_POSITIONS:
        raise AdAssemblyError(f"unknown position {position!r}")
    if style not in SUPER_STYLES:
        raise AdAssemblyError(f"unknown style {style!r}")
    font = load_font(font_size, font_path)
    text = wrap_text(text, font, int(width * 0.84))
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    stroke = max(1, int(font_size / 14)) if style == "outline" else 0
    spacing = int(font_size * 0.25)
    l, t, r, b = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing,
                                         align="center", stroke_width=stroke)
    tw, th = r - l, b - t
    x = (width - tw) / 2 - l
    if position == "top":
        y = height * 0.08 - t
    elif position == "center":
        y = (height - th) / 2 - t
    else:  # lower third: bottom of the text block sits at 85% of the frame (title safe)
        y = height * 0.85 - th - t
    x, y = int(round(x)), int(round(y))
    fill = tuple(color) + (255,)
    if style == "black box":
        pad = int(font_size * 0.35)
        draw.rectangle((x + l - pad, y + t - pad, x + r + pad, y + b + pad), fill=(0, 0, 0, 160))
    elif style == "white with soft shadow":
        shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        off = max(1, int(font_size / 20))
        ImageDraw.Draw(shadow).multiline_text((x + off, y + off), text, font=font,
                                              fill=(0, 0, 0, 170), spacing=spacing, align="center")
        shadow = shadow.filter(ImageFilter.GaussianBlur(max(1, font_size / 12)))
        layer = Image.alpha_composite(layer, shadow)
        draw = ImageDraw.Draw(layer)
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="center",
                        stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    return np.array(layer)


def composite_layer(frame, layer, opacity=1.0):
    """Alpha-composite an RGBA uint8 [H,W,4] layer onto one [H,W,3] float frame (in place,
    only inside the layer's bounding box). Returns the frame."""
    alpha = layer[..., 3]
    ys, xs = np.nonzero(alpha)
    if len(ys) == 0 or opacity <= 0:
        return frame
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = torch.from_numpy(layer[y0:y1, x0:x1].astype(np.float32) / 255.0).to(frame.device)
    a = sub[..., 3:4] * float(opacity)
    region = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = region * (1 - a) + sub[..., :3] * a
    return frame


# --------------------------------------------------------------------------- supers

def parse_supers(text, timing=None, fps=24.0):
    """Parse supers lines into [{"start_s", "end_s", "text"}].

    Line formats: "2 | Text" (shot 2, needs timing) or "1.5-4 | Text" (seconds).
    Blank lines and lines starting with # are ignored. "\\n" in the text makes a new line."""
    out = []
    for n, raw in enumerate(str(text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise AdAssemblyError(f"supers line {n}: expected 'shot | text' or 'start-end | text'")
        when, body = (p.strip() for p in line.split("|", 1))
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*s?\s*-\s*(\d+(?:\.\d+)?)\s*s?", when)
        if m:
            start, end = float(m.group(1)), float(m.group(2))
            if end <= start:
                raise AdAssemblyError(f"supers line {n}: end must be after start")
        elif re.fullmatch(r"\d+", when):
            if not timing:
                raise AdAssemblyError(f"supers line {n}: shot numbers need timing_json")
            k = int(when)
            if not 1 <= k <= len(timing["shots"]):
                raise AdAssemblyError(f"supers line {n}: there is no shot {k}")
            sh = timing["shots"][k - 1]
            start, end = float(sh["start_s"]), float(sh["end_s"])
        else:
            raise AdAssemblyError(f"supers line {n}: can't read the time {when!r}")
        out.append({"start_s": start, "end_s": end, "text": body.replace("\\n", "\n")})
    return out


def super_frame_range(sup, fps, total_frames):
    """(first, last_exclusive) frames for a super, clipped to the clip."""
    a = max(0, int(round(sup["start_s"] * fps)))
    b = min(int(total_frames), int(round(sup["end_s"] * fps)))
    return a, b


def fade_alpha(f, start, end, fade_in, fade_out):
    """Opacity of frame f for an element visible in [start, end) with fades in frames."""
    if f < start or f >= end:
        return 0.0
    length = end - start
    fi, fo = min(int(fade_in), length // 2), min(int(fade_out), length // 2)
    a = 1.0
    if fi > 0:
        a = min(a, (f - start + 1) / fi)
    if fo > 0:
        a = min(a, (end - f) / fo)
    return max(0.0, min(1.0, a))


def apply_supers(frames, supers, fps, font_size=56, position="lower third",
                 style="white with soft shadow", fade_in=6, fade_out=6, font_path="",
                 color=(255, 255, 255)):
    """Draw parsed supers onto a copy of frames [N,H,W,3]."""
    out = frames.clone()
    n, h, w, _ = out.shape
    for sup in supers:
        a, b = super_frame_range(sup, fps, n)
        if b <= a:
            continue
        layer = render_text_layer(w, h, sup["text"], font_size, position, style, font_path, color)
        for f in range(a, b):
            composite_layer(out[f], layer, fade_alpha(f, a, b, fade_in, fade_out))
    return out


# --------------------------------------------------------------------------- end card

def blur_frame(frame, radius, darken=0.35):
    """Gaussian-blurred, darkened copy of one [H,W,3] frame (end-card background)."""
    img = Image.fromarray((frame.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(max(0.0, float(radius))))
    arr = np.asarray(img).astype(np.float32) / 255.0 * (1.0 - float(darken))
    return torch.from_numpy(arr)


def logo_rgba(logo, mask=None, invert_mask=False):
    """Logo IMAGE [H,W,3] (+ optional MASK [H,W]) -> RGBA uint8 PIL image. ComfyUI's Load
    Image gives MASK = 1 where transparent, so alpha = 1 - mask (invert_mask flips that)."""
    rgb = (logo.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    h, w = rgb.shape[:2]
    if mask is None:
        alpha = np.full((h, w), 255, np.uint8)
    else:
        m = mask.float()
        if m.ndim == 3:
            m = m[0]
        if tuple(m.shape) != (h, w):
            m = F.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        m = m.clamp(0, 1).cpu().numpy()
        alpha = ((m if invert_mask else 1.0 - m) * 255).round().astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def render_end_card(width, height, background=None, bg_color=(17, 17, 17), logo=None,
                    tagline="", cta="", logo_scale=0.3, text_color=(255, 255, 255),
                    font_path="", font_size=None):
    """One end-card frame [H,W,3]: background (tensor or solid colour), logo centred above
    the tagline and call-to-action."""
    if background is not None:
        canvas = Image.fromarray((background.clamp(0, 1).cpu().numpy() * 255).round()
                                 .astype(np.uint8)).convert("RGBA")
        if canvas.size != (width, height):
            canvas = canvas.resize((width, height), Image.LANCZOS)
    else:
        canvas = Image.new("RGBA", (width, height), tuple(bg_color) + (255,))
    base = font_size or max(12, int(min(width, height) * 0.06))
    f_tag, f_cta = load_font(base, font_path), load_font(int(base * 0.6), font_path)
    tag = wrap_text(tagline, f_tag, int(width * 0.84)) if tagline else ""
    cta_t = wrap_text(cta, f_cta, int(width * 0.84)) if cta else ""
    draw = ImageDraw.Draw(canvas)
    blocks = []  # (kind, w, h, payload)
    if logo is not None:
        lh = max(1, int(height * float(logo_scale)))
        lw = max(1, int(round(logo.width * lh / logo.height)))
        if lw > width * 0.84:
            lw = int(width * 0.84)
            lh = max(1, int(round(logo.height * lw / logo.width)))
        blocks.append(("logo", lw, lh, logo.resize((lw, lh), Image.LANCZOS)))
    for txt, font in ((tag, f_tag), (cta_t, f_cta)):
        if txt:
            l, t, r, b = draw.multiline_textbbox((0, 0), txt, font=font, align="center",
                                                 spacing=int(font.size * 0.25) if hasattr(font, "size") else 4)
            blocks.append(("text", r - l, b - t, (txt, font, l, t)))
    gap = int(height * 0.04)
    total_h = sum(b[2] for b in blocks) + gap * max(0, len(blocks) - 1)
    y = (height - total_h) // 2
    for kind, bw, bh, payload in blocks:
        x = (width - bw) // 2
        if kind == "logo":
            canvas.alpha_composite(payload, (x, y))
        else:
            txt, font, l, t = payload
            draw = ImageDraw.Draw(canvas)
            draw.multiline_text((x - l, y - t), txt, font=font, fill=tuple(text_color) + (255,),
                                align="center",
                                spacing=int(font.size * 0.25) if hasattr(font, "size") else 4)
        y += bh + gap
    arr = np.asarray(canvas.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def append_end_card(frames, card, duration_s, fps, fade_in_frames=12):
    """Append round(duration_s * fps) end-card frames; the first fade_in_frames cross-fade
    from the last frame of the spot into the card."""
    n_card = seconds_to_frames(duration_s, fps)
    last = frames[-1]
    tail = card.unsqueeze(0).repeat(n_card, 1, 1, 1)
    fi = min(int(fade_in_frames), n_card)
    for j in range(fi):
        tail[j] = crossfade(last, card, (j + 1) / (fi + 1))
    return torch.cat([frames, tail], dim=0)


# --------------------------------------------------------------------------- audio

def db_to_gain(db):
    return 10.0 ** (float(db) / 20.0)


def to_stereo(wave):
    """[C,S] -> [2,S]: mono is duplicated, >2 channels keep the first two."""
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    if wave.shape[0] == 1:
        return wave.repeat(2, 1)
    return wave[:2]


def resample(wave, sr_in, sr_out):
    """Resample [C,S] by band-limited (anti-aliased) linear interpolation."""
    sr_in, sr_out = int(sr_in), int(sr_out)
    if sr_in == sr_out or wave.shape[-1] == 0:
        return wave
    n_out = max(1, int(round(wave.shape[-1] * sr_out / sr_in)))
    x = wave.float()[None, :, None, :]                    # [1,C,1,S] so antialias works
    x = F.interpolate(x, size=(1, n_out), mode="bilinear", align_corners=False,
                      antialias=sr_out < sr_in)
    return x[0, :, 0, :]


def fit_length(wave, n, offset=0):
    """Place [C,S] at sample `offset` (may be negative) in a [C,n] buffer: pad or trim."""
    out = torch.zeros((wave.shape[0], int(n)), dtype=torch.float32)
    src0 = max(0, -int(offset))
    dst0 = max(0, int(offset))
    length = min(wave.shape[-1] - src0, int(n) - dst0)
    if length > 0:
        out[:, dst0:dst0 + length] = wave[:, src0:src0 + length]
    return out


def prepare_audio(audio, sample_rate):
    """ComfyUI AUDIO dict (or None) -> stereo float [2,S] at sample_rate (or None)."""
    if audio is None:
        return None
    w = audio["waveform"]
    if w.ndim == 3:
        w = w[0]
    return to_stereo(resample(w.float().cpu(), audio["sample_rate"], sample_rate))


def rms_envelope(wave, sample_rate, hop_ms=10.0, window_ms=30.0):
    """RMS (linear) of a [C,S] or [S] signal per hop. Returns (env [frames], hop_samples)."""
    mono = wave.float().mean(0) if wave.ndim == 2 else wave.float()
    hop = max(1, int(sample_rate * hop_ms / 1000))
    win = max(hop, int(sample_rate * window_ms / 1000))
    n_frames = int(math.ceil(mono.shape[-1] / hop)) if mono.shape[-1] else 0
    if n_frames == 0:
        return torch.zeros(0), hop
    sq = F.pad(mono ** 2, (win // 2, win + n_frames * hop))
    c = torch.cumsum(F.pad(sq, (1, 0)), 0)
    idx = torch.arange(n_frames) * hop
    env = ((c[idx + win] - c[idx]) / win).clamp_min(0).sqrt()
    return env, hop


def speech_activity(env, threshold_db=-40.0, relative_db=-30.0):
    """Boolean per envelope frame: louder than max(threshold_db, peak + relative_db) dBFS."""
    if env.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    db = 20 * torch.log10(env.clamp_min(1e-9))
    thr = max(float(threshold_db), float(db.max()) + float(relative_db))
    return db > thr


def smooth_gain(target, hop_s, attack_s=0.15, release_s=0.15):
    """One-pole smoothing of a per-frame gain curve: falls with the attack time constant,
    rises with the release time constant."""
    ka = math.exp(-hop_s / max(attack_s, 1e-6))
    kr = math.exp(-hop_s / max(release_s, 1e-6))
    vals = target.tolist()
    g = vals[0] if vals else 1.0
    res = []
    for t in vals:
        g = t + (g - t) * (ka if t < g else kr)
        res.append(g)
    return torch.tensor(res, dtype=target.dtype)


def ducking_gain(voice, sample_rate, n_samples, duck_db=-10.0, attack_s=0.15, release_s=0.15,
                 threshold_db=-40.0):
    """Per-sample gain [n_samples] for the music bed: duck_db while the voice is speaking.
    The detector looks ahead by the attack time so the dip is in place when speech starts."""
    if voice is None or duck_db >= 0:
        return torch.ones(n_samples)
    env, hop = rms_envelope(voice, sample_rate)
    active = speech_activity(env, threshold_db)
    if not bool(active.any()):
        return torch.ones(n_samples)
    hop_s = hop / sample_rate
    # hold short gaps between words (~250 ms) so the bed doesn't pump
    hold = max(1, int(round(0.25 / hop_s)))
    act = F.max_pool1d(active.float()[None, None], kernel_size=2 * hold + 1, stride=1,
                       padding=hold)[0, 0] > 0
    look = int(round(attack_s / hop_s))
    if look > 0:
        act = torch.cat([act[look:], act[-1:].repeat(look)]) | act
    target = torch.where(act, torch.tensor(db_to_gain(duck_db)), torch.tensor(1.0))
    g = smooth_gain(target, hop_s, attack_s, release_s)
    x = torch.arange(n_samples, dtype=torch.float32) / hop
    i0 = x.floor().long().clamp(0, g.numel() - 1)
    i1 = (i0 + 1).clamp(max=g.numel() - 1)
    frac = x - x.floor()
    return g[i0] * (1 - frac) + g[i1] * frac


def fade_out(wave, sample_rate, seconds):
    """Linear fade to silence over the last `seconds` (in place-safe copy)."""
    n = min(int(round(float(seconds) * sample_rate)), wave.shape[-1])
    if n <= 0:
        return wave
    out = wave.clone()
    out[..., -n:] *= torch.linspace(1.0, 0.0, n)
    return out


def soft_limit(wave, ceiling_db=-1.0, knee=0.8):
    """Soft-clip so |sample| never exceeds the ceiling: linear up to knee*ceiling, then a
    tanh curve that approaches the ceiling smoothly."""
    c = db_to_gain(ceiling_db)
    k = c * knee
    a = wave.abs()
    over = a > k
    if not bool(over.any()):
        return wave
    shaped = k + (c - k) * torch.tanh((a - k) / (c - k))
    return torch.where(over, torch.sign(wave) * shaped, wave)


def mix_ad_audio(voiceover=None, music=None, sfx=None, total_seconds=15.0, vo_start_s=0.0,
                 music_volume_db=-14.0, duck_db=-10.0, fade_out_s=1.0, sample_rate=48000,
                 vo_volume_db=0.0, sfx_volume_db=0.0, sfx_start_s=0.0, attack_ms=150.0,
                 release_ms=150.0):
    """Mix VO + music bed (ducked under the VO) + SFX into stereo [2, total*sr]."""
    sr = int(sample_rate)
    n = max(1, int(round(float(total_seconds) * sr)))
    vo = prepare_audio(voiceover, sr)
    mu = prepare_audio(music, sr)
    fx = prepare_audio(sfx, sr)
    mix = torch.zeros((2, n), dtype=torch.float32)
    vo_track = None
    if vo is not None:
        vo_track = fit_length(vo, n, int(round(float(vo_start_s) * sr))) * db_to_gain(vo_volume_db)
        mix += vo_track
    if mu is not None:
        bed = fit_length(mu, n) * db_to_gain(music_volume_db)
        bed *= ducking_gain(vo_track, sr, n, duck_db, attack_ms / 1000, release_ms / 1000)
        mix += bed
    if fx is not None:
        mix += fit_length(fx, n, int(round(float(sfx_start_s) * sr))) * db_to_gain(sfx_volume_db)
    mix = fade_out(mix, sr, fade_out_s)
    return soft_limit(mix, -1.0)


# --------------------------------------------------------------------------- script

_DANGLING = {"a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "with",
             "for", "from", "by", "as", "into", "onto", "while", "her", "his", "their", "its",
             "is", "are", "that", "who"}


def word_budget(seconds, words_per_second=2.5):
    """Max VO words for a shot (at least 1)."""
    return max(1, int(math.floor(float(seconds) * float(words_per_second))))


def fit_words(text, budget):
    """Trim text to at most `budget` words, drop dangling connectives, end with a period."""
    words = re.sub(r"\s+", " ", str(text)).strip().split(" ")
    words = [w for w in words if w][:max(1, budget)]
    while len(words) > 1 and words[-1].lower().strip(",.;:!?") in _DANGLING:
        words.pop()
    if not words:
        return ""
    line = " ".join(words).rstrip(",;:- ")
    line = line[0].upper() + line[1:]
    return line if line[-1] in ".!?" else line + "."


def _first_clause(text):
    return re.split(r"(?<=[.!?;])\s|,\s|\s[-\u2013\u2014]\s", str(text).strip(), maxsplit=1)[0]


def script_from_shots(shot_list_json, brand="", tagline="", words_per_second=2.5,
                      supers_mode="first and last"):
    """Offline template script: one VO line per shot that fits its duration, and supers in
    "shot | text" format. Returns (voiceover_text, supers_text)."""
    try:
        plan = json.loads(shot_list_json) if isinstance(shot_list_json, str) else shot_list_json
        shots = plan["shots"]
    except (ValueError, KeyError, TypeError) as e:
        raise AdAssemblyError("shot_list_json is not a Shot Planner list") from e
    if not shots:
        raise AdAssemblyError("the shot list has no shots")
    durations = durations_from_shot_list(plan)
    brand, tagline = str(brand).strip(), str(tagline).strip()
    n = len(shots)
    vo, supers = [], []
    for i, (s, d) in enumerate(zip(shots, durations)):
        budget = word_budget(d, words_per_second)
        last = i == n - 1
        if last and (brand or tagline):
            sign_off = " ".join(x.rstrip(".") + "." for x in (brand, tagline) if x)
            line = fit_words(sign_off, budget)
        else:
            line = fit_words(_first_clause(s.get("description", "")) or plan.get("title", ""), budget)
        vo.append(line)
        if supers_mode == "every shot" or i == 0 or last:
            if i == 0 and n > 1:
                sup = plan.get("title") or fit_words(line, 5).rstrip(".")
            elif last:
                sup = tagline or brand or fit_words(line, 5).rstrip(".")
            else:
                sup = fit_words(line, 5).rstrip(".")
            if sup:
                supers.append(f"{i + 1} | {sup}")
    return "\n".join(l for l in vo if l), "\n".join(supers)
