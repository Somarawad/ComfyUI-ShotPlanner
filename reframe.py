"""Ad Reframe and Ad Cutdown: social versions of a finished spot. numpy + PIL only.

Reframe: a crop window of the target aspect ratio follows the subject (mask centre of mass,
or a centre-weighted edge-energy saliency fallback), smoothed over time, never leaving the
frame. Cutdown: picks and trims shots to a target length using Ad Timeline's timing_json;
end card frames after the last shot are always kept.
"""

import numpy as np
from PIL import Image

try:
    from . import stage_utils as su
except ImportError:  # imported as a top-level module (tests)
    import stage_utils as su

StageError = su.StageError

REFRAME_RATIOS = ["9:16", "1:1", "4:5", "16:9"]


def parse_ratio(text):
    try:
        a, b = (float(x) for x in str(text).split(":"))
    except ValueError as e:
        raise StageError(f"bad aspect ratio {text!r}") from e
    if a <= 0 or b <= 0:
        raise StageError(f"bad aspect ratio {text!r}")
    return a / b


def crop_size(src_w, src_h, aspect_ratio):
    """Largest (w, h) of the target ratio inside the source frame."""
    r = parse_ratio(aspect_ratio)
    if src_w / src_h > r:
        h = src_h
        w = min(src_w, max(1, int(round(h * r))))
    else:
        w = src_w
        h = min(src_h, max(1, int(round(w / r))))
    return w, h


def output_size(output_width, aspect_ratio):
    r = parse_ratio(aspect_ratio)
    w = max(2, int(round(output_width / 2)) * 2)
    return w, max(2, int(round(w / r / 2)) * 2)


def mask_centers(masks, frame_w, frame_h):
    """Centre of mass (x, y in pixels of the frame) per mask frame; None where empty."""
    m = su.as_numpy(masks)
    if m.ndim == 2:
        m = m[None]
    out = []
    for f in m:
        tot = float(f.sum())
        if tot <= 1e-6:
            out.append(None)
            continue
        h, w = f.shape
        ys, xs = np.arange(h) + 0.5, np.arange(w) + 0.5
        cx = float((f.sum(axis=0) * xs).sum() / tot) / w * frame_w
        cy = float((f.sum(axis=1) * ys).sum() / tot) / h * frame_h
        out.append((cx, cy))
    return out


def saliency_center(frame, max_side=96, sigma=0.35):
    """Centre-weighted edge energy centroid of one frame [H,W,3] -> (x, y) pixels."""
    f = np.asarray(frame, np.float32)
    H, W = f.shape[:2]
    s = min(1.0, max_side / max(H, W))
    w, h = max(2, int(W * s)), max(2, int(H * s))
    g = f[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    g = np.asarray(Image.fromarray(g, mode="F").resize((w, h), Image.BOX))
    e = np.zeros_like(g)
    e[:, 1:] += np.abs(np.diff(g, axis=1))
    e[1:, :] += np.abs(np.diff(g, axis=0))
    yy, xx = np.mgrid[0:h, 0:w]
    wgt = np.exp(-(((xx + 0.5) / w - 0.5) ** 2 + ((yy + 0.5) / h - 0.5) ** 2) / (2 * sigma ** 2))
    e = e * wgt
    tot = float(e.sum())
    if tot <= 1e-8:
        return W / 2.0, H / 2.0
    return (float((e * (xx + 0.5)).sum() / tot) / w * W,
            float((e * (yy + 0.5)).sum() / tot) / h * H)


def fill_missing(centers, default):
    """Replace None with the nearest known centre (or the default if none known)."""
    known = [i for i, c in enumerate(centers) if c is not None]
    if not known:
        return [default] * len(centers)
    out = []
    for i, c in enumerate(centers):
        if c is None:
            j = min(known, key=lambda k: abs(k - i))
            c = centers[j]
        out.append(c)
    return out


def smooth_path(values, smoothing):
    """Zero-phase exponential smoothing. smoothing 0 = raw, 1 = fixed at the mean."""
    v = np.asarray(values, np.float64)
    s = float(np.clip(smoothing, 0.0, 1.0))
    if v.size < 2 or s <= 0:
        return v.copy()
    if s >= 1.0:
        return np.full_like(v, v.mean())
    a = (1.0 - s) ** 2            # EMA weight of the new sample
    fwd = np.empty_like(v)
    fwd[0] = v[0]
    for i in range(1, v.size):
        fwd[i] = a * v[i] + (1 - a) * fwd[i - 1]
    out = np.empty_like(v)
    out[-1] = fwd[-1]
    for i in range(v.size - 2, -1, -1):
        out[i] = a * fwd[i] + (1 - a) * out[i + 1]
    return out


def crop_boxes(centers, frame_w, frame_h, aspect_ratio, smoothing=0.5):
    """[(x0, y0, w, h)] per frame: window of the target ratio centred on the smoothed
    subject path, clamped so it never leaves the frame."""
    cw, ch = crop_size(frame_w, frame_h, aspect_ratio)
    cx = smooth_path([c[0] for c in centers], smoothing)
    cy = smooth_path([c[1] for c in centers], smoothing)
    boxes = []
    for x, y in zip(cx, cy):
        x0 = int(round(min(max(x - cw / 2.0, 0), frame_w - cw)))
        y0 = int(round(min(max(y - ch / 2.0, 0), frame_h - ch)))
        boxes.append((x0, y0, cw, ch))
    return boxes


def resize_float(img, w, h):
    if img.shape[1] == w and img.shape[0] == h:
        return img.copy()
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(img[..., c]), mode="F")
                                .resize((w, h), Image.BICUBIC)) for c in range(img.shape[-1])],
                    axis=-1)


def plan_reframe(n_frames, frame_w, frame_h, aspect_ratio="9:16", subject_mask=None,
                 smoothing=0.6, frame_at=None):
    """Crop boxes for every frame. frame_at(i) -> float [H,W,3] is only called (on up to
    ~48 frames) for the saliency fallback when there is no mask."""
    N, W, H = int(n_frames), int(frame_w), int(frame_h)
    if N < 1:
        raise StageError("no frames")
    if subject_mask is not None:
        cents = mask_centers(subject_mask, W, H)
        if len(cents) == 1 and N > 1:
            cents = cents * N
        if len(cents) != N:
            raise StageError(f"subject_mask has {len(cents)} frames but there are {N} frames")
        cents = fill_missing(cents, (W / 2.0, H / 2.0))
    else:
        if frame_at is None:
            raise StageError("no subject_mask and no frames for the saliency fallback")
        step = max(1, N // 48)     # saliency on a subset of frames, interpolated
        idx = list(range(0, N, step))
        if idx[-1] != N - 1:
            idx.append(N - 1)
        sc = [saliency_center(frame_at(i)) for i in idx]
        xs = np.interp(np.arange(N), idx, [c[0] for c in sc])
        ys = np.interp(np.arange(N), idx, [c[1] for c in sc])
        cents = list(zip(xs, ys))
    return crop_boxes(cents, W, H, aspect_ratio, smoothing)


def reframe(frames, aspect_ratio="9:16", output_width=1080, subject_mask=None, smoothing=0.6):
    """frames [N,H,W,3] -> (reframed [N,h,w,3], boxes). (The ComfyUI node uses
    plan_reframe and resizes on torch, which is much faster at 1080p.)"""
    f = su.as_numpy(frames)
    if f.ndim != 4:
        raise StageError("frames must be an IMAGE batch [N,H,W,3]")
    N, H, W, _ = f.shape
    boxes = plan_reframe(N, W, H, aspect_ratio, subject_mask, smoothing, lambda i: f[i])
    ow, oh = output_size(output_width, aspect_ratio)
    out = np.empty((N, oh, ow, 3), np.float32)
    for i, (x0, y0, cw, ch) in enumerate(boxes):
        out[i] = np.clip(resize_float(f[i, y0:y0 + ch, x0:x0 + cw, :3], ow, oh), 0, 1)
    return out, boxes


# --------------------------------------------------------------------------- cutdown

def _scores(verdicts_json):
    v = su.load_json(verdicts_json, "verdicts_json")
    if not v:
        return {}
    try:
        return {int(s["shot"]): float(s.get("score", 0)) for s in v["shots"]}
    except (KeyError, TypeError, ValueError) as e:
        raise StageError("verdicts_json is not Shot Video QA output") from e


def choose_shots(n_shots, keep="auto", max_shots=None, scores=None):
    """Shot numbers (1-based, sorted) to keep. auto = first + last + the best-scored middle
    shots (ties: earlier first) up to max_shots; with room for one shot, the last."""
    if str(keep).strip().lower() != "auto":
        nums = su.parse_int_list(keep, "keep")
        bad = [k for k in nums if k < 1 or k > n_shots]
        if bad:
            raise StageError(f"keep: shot {bad[0]} does not exist (the spot has {n_shots} shots)")
        return sorted(set(nums))
    max_shots = n_shots if max_shots is None else max(1, int(max_shots))
    if n_shots == 1 or max_shots == 1:
        return [n_shots]             # room for one shot: the last (product / pack) shot
    chosen = [1, n_shots]
    scores = scores or {}
    middle = sorted(range(2, n_shots), key=lambda k: (-scores.get(k, 0.0), k))
    for k in middle:
        if len(chosen) >= max_shots:
            break
        chosen.append(k)
    return sorted(chosen)


def allocate_frames(lengths, budget):
    """Frames per kept shot: the full shot if everything fits, else proportional to the
    original lengths (each at least 1, never more than the shot has)."""
    lengths = [int(l) for l in lengths]
    if sum(lengths) <= budget:
        return lengths
    if budget < len(lengths):
        raise StageError(f"{len(lengths)} shots cannot fit in {budget} frames")
    alloc = [max(1, min(l, int(budget * l / sum(lengths)))) for l in lengths]
    i = 0
    while sum(alloc) < budget:          # hand out the rounding remainder
        j = i % len(alloc)
        if alloc[j] < lengths[j]:
            alloc[j] += 1
        i += 1
    while sum(alloc) > budget:
        j = int(np.argmax(alloc))
        alloc[j] -= 1
    return alloc


def plan_cutdown(total_frames, timing_json, target_seconds=6.0, keep="auto",
                 verdicts_json=None, min_shot_s=1.5):
    """-> dict(ranges [(src_start, src_end)], timing (new), report). The first kept shot
    keeps its head, the last its tail (the product / pack shot), others their middle."""
    t = su.load_timing(timing_json)
    if t is None:
        raise StageError("timing_json is required (connect Ad Timeline's timing_json)")
    fps = float(t["fps"])
    shots = t["shots"]
    if not shots:
        raise StageError("timing_json has no shots")
    last_end = int(shots[-1]["end_frame"])
    if last_end > total_frames:
        raise StageError(f"timing_json ends at frame {last_end} but only {total_frames} frames "
                         "are connected")
    card = total_frames - last_end
    target = int(round(float(target_seconds) * fps))
    budget = target - card
    if budget < 1:
        raise StageError(f"the end card alone is {card / fps:.2f} s, longer than the "
                         f"{float(target_seconds):g} s target")
    max_shots = max(1, int(budget // max(1, round(float(min_shot_s) * fps))))
    nums = choose_shots(len(shots), keep, max_shots, _scores(verdicts_json))
    lengths = [int(shots[k - 1]["end_frame"]) - int(shots[k - 1]["start_frame"]) for k in nums]
    alloc = allocate_frames(lengths, budget)
    ranges, new_shots, cursor = [], [], 0
    for pos, (k, n_keep, n_full) in enumerate(zip(nums, alloc, lengths)):
        s0 = int(shots[k - 1]["start_frame"])
        if pos == 0 and len(nums) > 1:
            a = s0
        elif pos == len(nums) - 1:
            a = s0 + n_full - n_keep
        else:
            a = s0 + (n_full - n_keep) // 2
        ranges.append((a, a + n_keep))
        new_shots.append({"shot": pos + 1, "source_shot": k, "start_frame": cursor,
                          "end_frame": cursor + n_keep, "start_s": round(cursor / fps, 4),
                          "end_s": round((cursor + n_keep) / fps, 4),
                          "duration_s": round(n_keep / fps, 4)})
        cursor += n_keep
    if card:
        ranges.append((last_end, total_frames))
    total = cursor + card
    timing = {"fps": fps, "total_frames": cursor, "total_seconds": round(cursor / fps, 4),
              "transition": "cut", "transition_frames": 0, "shots": new_shots,
              "end_card_frames": card, "with_end_card_frames": total,
              "with_end_card_seconds": round(total / fps, 4)}
    for k in ("width", "height"):
        if k in t:
            timing[k] = t[k]
    msg = (f"Cutdown: kept shots {', '.join(map(str, nums))} of {len(shots)} "
           f"({cursor / fps:.2f} s) + end card {card / fps:.2f} s = {total / fps:.2f} s "
           f"(target {float(target_seconds):g} s)")
    if total < target:
        msg += "; shorter than the target because the kept shots are shorter"
    return {"ranges": ranges, "timing": timing, "report": msg, "total_frames": total}
