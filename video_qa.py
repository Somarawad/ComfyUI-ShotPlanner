"""Shot Video QA, Clip Reroll Merge and clip utilities: pure logic.

numpy + PIL + stdlib only (no ComfyUI, no torch). Clips are float arrays [N,H,W,3] in 0-1.
Offline metrics catch frozen clips, flicker, sudden jumps / morphs, drift away from the
approved keyframe and locally unstable ("melting") regions. An optional Gemini review looks
at a few sampled frames and suggests a fixed motion prompt.
"""

import base64
import io
import json
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from . import stage_utils as su
except ImportError:  # imported as a top-level module (tests)
    import stage_utils as su

StageError = su.StageError

ISSUE_TYPES = ["morph", "face", "hands", "logo", "physics", "camera", "other"]
DEFAULT_GEMINI_MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

THRESHOLDS = {
    "frozen": 0.002,      # mean abs frame-to-frame difference below this = frozen clip
    "flicker": 0.012,     # median |2nd difference| of mean luma above this = flicker
    "jump": 5.0,          # largest frame diff / median frame diff above this = jump / morph
    "jump_abs": 0.04,     # ...and the jump itself must be at least this big
    "drift": 0.62,        # keyframe vs last frame (0 = same, ~0.6+ = unrelated image); camera moves and walking reach ~0.5
    "melt": 4.0,          # hottest cell / median cell instability above this = melting region
    "melt_abs": 0.02,     # ...and the hottest cell must be at least this unstable
}
PENALTIES = {"frozen": 7, "flicker": 4, "jump": 4, "drift": 4, "melt": 4}

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# --------------------------------------------------------------------------- small helpers

def _clip(c):
    c = su.as_numpy(c)
    if c.ndim == 3:
        c = c[None]
    if c.ndim != 4 or c.shape[-1] < 3:
        raise StageError(f"expected an IMAGE batch [frames,H,W,3], got shape {tuple(c.shape)}")
    return c[..., :3]


def resize_frame(frame, w, h, resample=Image.BILINEAR):
    """Float [H,W,C] -> float [h,w,C] (PIL 'F' mode per channel, no 8-bit rounding)."""
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape[0] == h and frame.shape[1] == w:
        return frame.copy()
    chans = [np.asarray(Image.fromarray(np.ascontiguousarray(frame[..., c]), mode="F")
                        .resize((int(w), int(h)), resample))
             for c in range(frame.shape[-1])]
    return np.stack(chans, axis=-1).astype(np.float32)


def downscale(clip, max_side=64):
    """Small copy of a clip for metrics (keeps aspect ratio)."""
    clip = _clip(clip)
    _, h, w, _ = clip.shape
    s = min(1.0, max_side / max(h, w))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if (nw, nh) == (w, h):
        return clip.copy()
    return np.stack([resize_frame(f, nw, nh, Image.BOX) for f in clip])


def luma(clip):
    return np.tensordot(clip[..., :3], LUMA, axes=([-1], [0]))


def sample_indices(n, k):
    """k evenly spaced frame indices over n frames (first and last included, unique)."""
    n, k = int(n), max(1, int(k))
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


# --------------------------------------------------------------------------- offline metrics

def frame_diffs(small):
    """Mean abs difference between consecutive frames (len N-1)."""
    if small.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    return np.abs(np.diff(small, axis=0)).mean(axis=(1, 2, 3))


def flicker_score(small):
    """Median |second difference| of the mean luma: ~0 for steady shots, fades and pans,
    large when brightness pumps frame to frame. A single cut only affects 2 values, so the
    median ignores it."""
    m = luma(small).mean(axis=(1, 2))
    if m.shape[0] < 3:
        return 0.0
    return float(np.median(np.abs(np.diff(m, n=2))))


def jump_score(diffs):
    """(score, frame indices) for sudden jumps: biggest frame diff relative to the median.
    Index i means the change from frame i to i+1."""
    if diffs.size < 2:
        return 0.0, []
    med = float(np.median(diffs))
    score = float(diffs.max()) / (med + 0.005)
    thr = THRESHOLDS["jump"] * (med + 0.005)
    frames = [int(i) + 1 for i in np.where((diffs > thr) & (diffs > THRESHOLDS["jump_abs"]))[0]]
    return score, frames


def drift_score(keyframe, last_frame, size=32):
    """How far the last frame drifted from the approved keyframe, 0 (same) .. 1 (unrelated).

    Half colour (mean abs difference of 32x32 thumbnails, saturating at 0.25) and half
    structure ((1 - normalised cross-correlation of the luma thumbnails) / 2)."""
    a = resize_frame(np.asarray(keyframe, np.float32)[..., :3], size, size, Image.BOX)
    b = resize_frame(np.asarray(last_frame, np.float32)[..., :3], size, size, Image.BOX)
    colour = float(np.abs(a - b).mean())
    la, lb = luma(a[None])[0], luma(b[None])[0]
    la, lb = la - la.mean(), lb - lb.mean()
    den = float(np.sqrt((la ** 2).sum() * (lb ** 2).sum()))
    ncc = float((la * lb).sum() / den) if den > 1e-8 else (1.0 if colour < 0.02 else 0.0)
    structure = (1.0 - ncc) / 2.0
    return {"drift": round(0.5 * min(1.0, colour / 0.25) + 0.5 * structure, 4),
            "colour_diff": round(colour, 4), "structure_diff": round(structure, 4)}


def melt_heatmap(small, cells=8):
    """Per-cell temporal instability: mean |second temporal difference| of the luma inside
    each cell of a cells x cells grid. Smooth motion has a small 2nd difference; regions
    that boil, smear or morph light up. Returns [cells, cells] float."""
    y = luma(small)
    n, h, w = y.shape
    if n < 3:
        return np.zeros((cells, cells), dtype=np.float32)
    acc = np.abs(np.diff(y, n=2, axis=0)).mean(axis=0)          # [h,w]
    heat = np.zeros((cells, cells), dtype=np.float32)
    ys = np.linspace(0, h, cells + 1).astype(int)
    xs = np.linspace(0, w, cells + 1).astype(int)
    for i in range(cells):
        for j in range(cells):
            block = acc[ys[i]:max(ys[i] + 1, ys[i + 1]), xs[j]:max(xs[j] + 1, xs[j + 1])]
            heat[i, j] = float(block.mean()) if block.size else 0.0
    return heat


def melt_score(heat):
    med = float(np.median(heat))
    return float(heat.max()) / (med + 0.002), float(heat.max())


def heat_overlay(frame, heat, strength=0.6):
    """Red overlay of a heat map on a frame (float [H,W,3]); heat normalised to its max."""
    frame = np.asarray(frame, dtype=np.float32)[..., :3]
    h, w = frame.shape[:2]
    mx = float(heat.max())
    norm = (heat / mx) if mx > 1e-8 else heat * 0
    up = resize_frame(norm[..., None].astype(np.float32), w, h, Image.BILINEAR)[..., 0]
    a = np.clip(up, 0, 1)[..., None] * strength
    red = np.zeros_like(frame)
    red[..., 0] = 1.0
    grey = np.repeat(luma(frame[None])[0][..., None], 3, axis=-1) * 0.7
    return np.clip(grey * (1 - a) + red * a, 0, 1).astype(np.float32)


def clip_metrics(clip, keyframe=None, thresholds=None):
    """All offline metrics for one clip, plus pass/fail flags and the heat map."""
    t = dict(THRESHOLDS, **(thresholds or {}))
    clip = _clip(clip)
    small = downscale(clip, 64)
    diffs = frame_diffs(small)
    mean_diff = float(diffs.mean()) if diffs.size else 0.0
    j, jframes = jump_score(diffs)
    heat = melt_heatmap(small)
    m_ratio, m_max = melt_score(heat)
    out = {
        "frames": int(clip.shape[0]),
        "mean_motion": round(mean_diff, 5),
        "flicker": round(flicker_score(small), 5),
        "jump": round(j, 3),
        "jump_frames": jframes,
        "melt": round(m_ratio, 3),
        "melt_peak": round(m_max, 5),
    }
    flags = {
        "frozen": clip.shape[0] < 2 or mean_diff < t["frozen"],
        "flicker": out["flicker"] > t["flicker"],
        "jump": bool(jframes) and j > t["jump"],
        "melt": m_ratio > t["melt"] and m_max > t["melt_abs"],
        "drift": False,
    }
    if keyframe is not None:
        kf = _clip(keyframe)[0]
        d = drift_score(kf, clip[-1])
        out.update(d)
        flags["drift"] = d["drift"] > t["drift"]
    out["flags"] = {k: bool(v) for k, v in flags.items()}
    return out, heat


def offline_score(metrics):
    """10 minus penalties for each failed check; (score, reasons)."""
    f = metrics["flags"]
    score, reasons = 10.0, []
    if f["frozen"]:
        score -= PENALTIES["frozen"]
        reasons.append(f"frozen clip (mean motion {metrics['mean_motion']:.4f})")
    if f["flicker"]:
        score -= PENALTIES["flicker"]
        reasons.append(f"flicker (luma pumping {metrics['flicker']:.4f})")
    if f["jump"]:
        score -= PENALTIES["jump"]
        fr = ", ".join(str(x) for x in metrics["jump_frames"][:5])
        reasons.append(f"sudden jump / morph at frame {fr} ({metrics['jump']:.1f}x median motion)")
    if f["drift"]:
        score -= PENALTIES["drift"]
        reasons.append(f"drifted from the keyframe ({metrics['drift']:.2f})")
    if f["melt"]:
        score -= PENALTIES["melt"]
        reasons.append(f"unstable / melting region ({metrics['melt']:.1f}x the median cell)")
    return max(0.0, score), reasons


# --------------------------------------------------------------------------- LLM review

def frame_to_jpeg_b64(frame, max_side=512, quality=85):
    f = np.clip(np.asarray(frame, np.float32)[..., :3], 0, 1)
    img = Image.fromarray((f * 255 + 0.5).astype(np.uint8))
    s = min(1.0, max_side / max(img.size))
    if s < 1.0:
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def review_prompt(shot_number, n_frames, frame_numbers, checks="", brand_rules="",
                  motion_prompt="", metrics=None):
    lines = [
        f"You are a strict commercial video QA reviewer. These {n_frames} images are frames "
        f"{', '.join(str(i) for i in frame_numbers)} (0-based) sampled evenly from shot "
        f"{shot_number} of an AI-generated (image-to-video) commercial. The first image is the "
        "approved keyframe the clip was animated from when it is labelled KEYFRAME.",
        "Look for: morphing or melting objects, face or identity changes, broken hands or "
        "fingers, distorted or changing logos / text / product shape, impossible physics, "
        "and camera problems (jitter, sudden jumps, wrong direction).",
    ]
    if motion_prompt:
        lines.append(f"The motion prompt used was: {motion_prompt}")
    if checks and checks.strip():
        lines.append("Extra checks for this spot:\n" + checks.strip())
    if brand_rules and brand_rules.strip():
        lines.append("Brand rules (any violation is an issue of type logo or other):\n"
                     + brand_rules.strip())
    if metrics:
        lines.append("Automatic metrics (for context): " + json.dumps(
            {k: metrics[k] for k in ("mean_motion", "flicker", "jump", "drift", "melt")
             if k in metrics}))
    lines.append(
        "Reply with JSON only, exactly this shape: {\"score\": number 0-10 (10 = flawless, "
        "broadcast ready), \"issues\": [{\"type\": one of morph|face|hands|logo|physics|camera|"
        "other, \"frame\": frame number, \"note\": short description}], "
        "\"fixed_motion_prompt\": an improved image-to-video motion prompt that avoids the "
        "issues (keep the same action and camera move; empty string if no change is needed)}.")
    return "\n\n".join(lines)


def parse_review(text):
    """Validate a review reply -> {score, issues, fixed_motion_prompt}; StageError if bad."""
    data = su.extract_json_object(text)
    try:
        score = float(data["score"])
    except (KeyError, TypeError, ValueError) as e:
        raise StageError("review JSON has no numeric score") from e
    if not math.isfinite(score):
        raise StageError("review score is not a number")
    score = min(10.0, max(0.0, score))
    issues = []
    for it in data.get("issues") or []:
        if not isinstance(it, dict):
            it = {"note": str(it)}
        typ = str(it.get("type", "other")).strip().lower()
        if typ not in ISSUE_TYPES:
            typ = "other"
        try:
            frame = int(it.get("frame")) if it.get("frame") is not None else None
        except (TypeError, ValueError):
            frame = None
        issues.append({"type": typ, "frame": frame, "note": str(it.get("note", "")).strip()})
    fixed = data.get("fixed_motion_prompt") or ""
    return {"score": score, "issues": issues, "fixed_motion_prompt": str(fixed).strip()}


def _gemini_text(resp):
    try:
        parts = resp["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as e:
        reason = ""
        if isinstance(resp, dict):
            reason = (resp.get("promptFeedback") or {}).get("blockReason", "")
        raise StageError(f"Gemini returned no content{' (' + reason + ')' if reason else ''}") from e
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


def gemini_review(frames_b64, prompt, api_key, models=None, seed=0, timeout=120, post=None):
    """Ask Gemini for a JSON review. Each model gets one retry on HTTP 429/503 or on an
    unparseable reply; then the next model is tried. Returns (review, model_used, log)."""
    post = post or su.http_post_json
    models = [m for m in (models or DEFAULT_GEMINI_MODELS) if m]
    parts = [{"text": prompt}]
    for label, b64 in frames_b64:
        parts.append({"text": label})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2,
                                 "seed": int(seed) % 2147483647}}
    headers = {"x-goog-api-key": api_key}
    log = []
    for model in models:
        for attempt in range(2):
            try:
                resp = post(GEMINI_URL.format(model=model), body, headers, timeout)
                return parse_review(_gemini_text(resp)), model, log
            except su.HttpError as e:
                log.append(f"{model}: {e}")
                if e.status in (401, 403):
                    raise StageError(f"Gemini rejected the API key: {e}") from e
                if e.status in (429, 503) and attempt == 0:
                    su.sleep(2.0)
                    continue
                break           # other HTTP errors (404 model, 400, ...) -> next model
            except StageError as e:
                log.append(f"{model}: bad reply ({e})")
                if attempt == 0:
                    continue
                break
    raise StageError("Gemini review failed on every model: " + " | ".join(log))


# --------------------------------------------------------------------------- verdicts

def reroll_seed(seed, shot_number, attempt=0):
    """Deterministic new seed for a failed shot (differs per shot and per base seed)."""
    return random.Random(f"{int(seed)}:{int(shot_number)}:{int(attempt)}").randrange(1, 2 ** 31)


def combine_verdicts(per_shot, pass_score=7.0, seed=0, motion_prompts=None):
    """per_shot: list of dicts with metrics, offline_score, offline_reasons and optional
    llm (review dict). Final score = the lower of offline and LLM; reroll when below
    pass_score or when the clip is frozen. Returns the verdicts dict."""
    motion_prompts = list(motion_prompts or [])
    shots, rerolls = [], []
    for i, p in enumerate(per_shot):
        n = i + 1
        score = float(p["offline_score"])
        reasons = list(p.get("offline_reasons", []))
        llm = p.get("llm")
        issues, fixed = [], ""
        if llm:
            score = min(score, float(llm["score"]))
            issues = llm.get("issues", [])
            fixed = llm.get("fixed_motion_prompt", "")
            reasons += [f"{it['type']}"
                        + (f" @frame {it['frame']}" if it.get("frame") is not None else "")
                        + (f": {it['note']}" if it.get("note") else "") for it in issues]
            if float(llm["score"]) < pass_score:
                reasons.append(f"LLM score {float(llm['score']):g} < {pass_score:g}")
        hard = p["metrics"]["flags"].get("frozen", False)
        route = "reroll" if (score < float(pass_score) or hard) else "pass"
        original = motion_prompts[i] if i < len(motion_prompts) else ""
        v = {"shot": n, "route": route, "score": round(score, 2),
             "offline_score": round(float(p["offline_score"]), 2),
             "llm_score": round(float(llm["score"]), 2) if llm else None,
             "reasons": reasons, "issues": issues, "metrics": p["metrics"],
             "motion_prompt": original, "fixed_motion_prompt": fixed,
             "reroll_seed": reroll_seed(seed, n) if route == "reroll" else None}
        if llm and p.get("model"):
            v["model"] = p["model"]
        shots.append(v)
        if route == "reroll":
            rerolls.append(n)
    return {"pass_score": float(pass_score), "seed": int(seed), "shots": shots,
            "reroll_shots": rerolls}


def reroll_lists(verdicts):
    """(reroll_shots string, prompts list, seeds list) in shot order."""
    shots = [v for v in verdicts["shots"] if v["route"] == "reroll"]
    prompts = [v.get("fixed_motion_prompt") or v.get("motion_prompt") or "" for v in shots]
    seeds = [int(v["reroll_seed"]) for v in shots]
    return ", ".join(str(v["shot"]) for v in shots), prompts, seeds


def report_text(verdicts):
    lines = [f"Shot Video QA: {len(verdicts['shots'])} shots, pass score "
             f"{verdicts['pass_score']:g}, reroll: {', '.join(map(str, verdicts['reroll_shots'])) or 'none'}"]
    for v in verdicts["shots"]:
        m = v["metrics"]
        llm = f", LLM {v['llm_score']:g}" if v.get("llm_score") is not None else ""
        lines.append(f"Shot {v['shot']}: {v['route'].upper()} score {v['score']:g} "
                     f"(offline {v['offline_score']:g}{llm}) | motion {m['mean_motion']:.4f} "
                     f"flicker {m['flicker']:.4f} jump {m['jump']:.1f} melt {m['melt']:.1f}"
                     + (f" drift {m['drift']:.2f}" if "drift" in m else ""))
        for r in v["reasons"]:
            lines.append(f"    - {r}")
        if v["route"] == "reroll":
            lines.append(f"    reroll seed {v['reroll_seed']}"
                         + (f"; fixed prompt: {v['fixed_motion_prompt']}" if v["fixed_motion_prompt"] else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------- contact sheet

def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def contact_sheet(clips, verdicts, heats, frames_per_clip=6, thumb_h=128):
    """One row per shot: a label cell (shot, PASS/REROLL, score), the sampled frames and the
    melting-region heat overlay. Returns float [1,H,W,3]."""
    rows = []
    label_w = int(thumb_h * 1.4)
    font_big, font_small = _font(max(10, thumb_h // 6)), _font(max(8, thumb_h // 10))
    widths = []
    for clip, v, heat in zip(clips, verdicts["shots"], heats):
        clip = _clip(clip)
        n, h, w, _ = clip.shape
        tw = max(1, int(round(w * thumb_h / h)))
        idx = sample_indices(n, frames_per_clip)
        thumbs = [resize_frame(clip[i], tw, thumb_h, Image.BILINEAR) for i in idx]
        mid = clip[n // 2]
        thumbs.append(heat_overlay(resize_frame(mid, tw, thumb_h), heat))
        strip = np.concatenate(thumbs, axis=1)
        ok = v["route"] == "pass"
        lab = Image.new("RGB", (label_w, thumb_h), (24, 110, 48) if ok else (160, 30, 30))
        d = ImageDraw.Draw(lab)
        d.text((8, 6), f"Shot {v['shot']}", fill=(255, 255, 255), font=font_small)
        d.text((8, thumb_h // 3), "PASS" if ok else "REROLL", fill=(255, 255, 255), font=font_big)
        d.text((8, thumb_h * 2 // 3), f"score {v['score']:g}", fill=(255, 255, 255), font=font_small)
        row = np.concatenate([np.asarray(lab, np.float32) / 255.0, strip], axis=1)
        rows.append(row)
        widths.append(row.shape[1])
    if not rows:
        raise StageError("no clips to draw")
    W = max(widths)
    gap = np.full((4, W, 3), 0.08, np.float32)
    padded = []
    for r in rows:
        if r.shape[1] < W:
            r = np.concatenate([r, np.full((r.shape[0], W - r.shape[1], 3), 0.08, np.float32)], 1)
        padded += [r, gap]
    return np.concatenate(padded[:-1], axis=0)[None]


# --------------------------------------------------------------------------- run QA

def run_video_qa(clips, keyframes=None, checks="", pass_score=7.0, use_llm=False,
                 model="", frames_per_clip=6, seed=0, brand_rules="", api_key="",
                 motion_prompts=None, thresholds=None, post=None):
    """Full QA pass. Returns dict(verdicts, report, contact_sheet, reroll_shots,
    reroll_prompts, reroll_seeds)."""
    clips = [_clip(c) for c in clips]
    if not clips:
        raise StageError("no clips connected")
    keyframes = list(keyframes or [])
    if len(keyframes) == 1 and len(clips) > 1 and _clip(keyframes[0]).shape[0] == len(clips):
        kb = _clip(keyframes[0])
        keyframes = [kb[i:i + 1] for i in range(kb.shape[0])]
    if keyframes and len(keyframes) != len(clips):
        raise StageError(f"{len(clips)} clips but {len(keyframes)} keyframes: connect one "
                         "approved still per shot, in the same order")
    key = su.resolve_key("gemini", api_key) if use_llm else ""
    models = [model.strip()] if model and model.strip() else DEFAULT_GEMINI_MODELS
    motion_prompts = list(motion_prompts or [])
    per_shot, heats = [], []
    for i, clip in enumerate(clips):
        kf = keyframes[i] if keyframes else None
        metrics, heat = clip_metrics(clip, kf, thresholds)
        score, reasons = offline_score(metrics)
        p = {"metrics": metrics, "offline_score": score, "offline_reasons": reasons}
        if use_llm:
            idx = sample_indices(clip.shape[0], frames_per_clip)
            imgs = []
            if kf is not None:
                imgs.append(("KEYFRAME (approved still):", frame_to_jpeg_b64(_clip(kf)[0])))
            imgs += [(f"Frame {j}:", frame_to_jpeg_b64(clip[j])) for j in idx]
            prompt = review_prompt(i + 1, len(imgs), idx, checks, brand_rules,
                                   motion_prompts[i] if i < len(motion_prompts) else "", metrics)
            try:
                review, used, _ = gemini_review(imgs, prompt, key, models, seed + i, post=post)
            except StageError as e:
                raise StageError(f"shot {i + 1}: {e}") from e
            p["llm"], p["model"] = review, used
        per_shot.append(p)
        heats.append(heat)
    verdicts = combine_verdicts(per_shot, pass_score, seed, motion_prompts)
    shots_s, prompts, seeds = reroll_lists(verdicts)
    return {"verdicts": verdicts, "report": report_text(verdicts),
            "contact_sheet": contact_sheet(clips, verdicts, heats, frames_per_clip),
            "reroll_shots": shots_s, "reroll_prompts": prompts, "reroll_seeds": seeds}


# --------------------------------------------------------------------------- reroll merge

def _is_missing(x):
    if x is None:
        return True
    if isinstance(x, (list, tuple)):
        return len(x) == 0 or all(v is None for v in x)
    return False


def load_verdicts(verdicts_json):
    v = su.load_json(verdicts_json, "verdicts_json")
    if not isinstance(v, dict) or not isinstance(v.get("shots"), list):
        raise StageError("verdicts_json is not Shot Video QA output")
    return v


def reroll_numbers(verdicts):
    return [int(s["shot"]) for s in verdicts["shots"] if s.get("route") == "reroll"]


def needs_rerolled(verdicts_json, rerolled):
    """True when some shot failed and the rerolled clips have not been evaluated yet."""
    return bool(reroll_numbers(load_verdicts(verdicts_json))) and _is_missing(rerolled)


def merge_clips(clips, verdicts_json, rerolled=None):
    """Replace failed clips with the rerolled ones (given in shot order); returns
    (clips list, report)."""
    v = load_verdicts(verdicts_json)
    clips = list(clips)
    if len(v["shots"]) != len(clips):
        raise StageError(f"verdicts_json covers {len(v['shots'])} shots but {len(clips)} "
                         "clips are connected")
    fails = reroll_numbers(v)
    if not fails:
        return clips, f"All {len(clips)} shots passed; nothing replaced."
    if _is_missing(rerolled):
        raise StageError(f"shots {', '.join(map(str, fails))} need a reroll but no rerolled "
                         "clips are connected")
    rer = [r for r in rerolled if r is not None]
    if len(rer) != len(fails):
        raise StageError(f"{len(fails)} shots were marked for reroll ({', '.join(map(str, fails))}) "
                         f"but {len(rer)} rerolled clips arrived; connect exactly one per failed "
                         "shot, in shot order")
    out = list(clips)
    for n, clip in zip(fails, rer):
        out[n - 1] = clip
    return out, f"Replaced shots {', '.join(map(str, fails))}; kept {len(clips) - len(fails)}."


# --------------------------------------------------------------------------- clip utilities

def split_batch(n_frames, counts):
    """[(start, end)] ranges for splitting a batch of n_frames by counts (must add up)."""
    counts = [int(c) for c in counts]
    if not counts or min(counts) < 1:
        raise StageError("frame counts must be at least 1 each")
    total = sum(counts)
    if total != n_frames:
        raise StageError(f"frame counts add up to {total} but the batch has {n_frames} frames")
    out, cur = [], 0
    for c in counts:
        out.append((cur, cur + c))
        cur += c
    return out


def equal_counts(n_frames, n_clips):
    n_clips = int(n_clips)
    if n_clips < 1 or n_clips > n_frames:
        raise StageError(f"cannot split {n_frames} frames into {n_clips} clips")
    base, extra = divmod(n_frames, n_clips)
    return [base + (1 if i < extra else 0) for i in range(n_clips)]


def resample_count(n, src_fps, dst_fps):
    """Frames after changing frame rate with the same duration: round(n * dst / src)."""
    if src_fps <= 0 or dst_fps <= 0:
        raise StageError("fps must be > 0")
    return max(1, int(math.floor(n * float(dst_fps) / float(src_fps) + 0.5)))


def resample_plan(n, src_fps, dst_fps):
    """For each output frame: (index a, index b, weight of b) to blend."""
    m = resample_count(n, src_fps, dst_fps)
    out = []
    for j in range(m):
        p = min(j * float(src_fps) / float(dst_fps), n - 1)
        a = int(math.floor(p))
        b = min(a + 1, n - 1)
        out.append((a, b, float(p - a)))
    return out


def resample_fps(clip, src_fps, dst_fps, method="blend"):
    """Change a clip's frame rate keeping its duration (16 -> 24 fps for Wan clips).
    blend = linear blend of the two neighbouring frames; nearest = duplicate/drop."""
    clip = _clip(clip)
    plan = resample_plan(clip.shape[0], src_fps, dst_fps)
    frames = []
    for a, b, w in plan:
        if method == "nearest":
            frames.append(clip[b] if w >= 0.5 else clip[a])
        else:
            frames.append(clip[a] * (1 - w) + clip[b] * w if w > 1e-6 else clip[a])
    return np.stack(frames).astype(np.float32)
