"""Animatic tools: turn one keyframe per shot into a timed animatic with camera motion,
or stitch per-shot video clips into one sequence. Pure torch, no extra models."""

import json
import re

import torch
import torch.nn.functional as F

# camera keyword -> (zoom_start, zoom_end, pan_x_start, pan_x_end, pan_y_start, pan_y_end)
# pan values are fractions of the free space left after zooming (-1 = left/top, 1 = right/bottom)
MOTIONS = [
    (r"pull[- ]?back|pull[- ]?out|zoom[- ]?out|dolly[- ]?out|reveal", (1.0, 0.0, 0, 0, 0, 0)),
    (r"push[- ]?in|zoom[- ]?in|dolly[- ]?in|creep", (0.0, 1.0, 0, 0, 0, 0)),
    (r"pan[- ]?left|move[s]? left", (0.6, 0.6, 1, -1, 0, 0)),
    (r"pan[- ]?right|move[s]? right|tracking|track(s)? |follow", (0.6, 0.6, -1, 1, 0, 0)),
    (r"tilt[- ]?up|crane[- ]?up|rising|rise", (0.6, 0.6, 0, 0, 1, -1)),
    (r"tilt[- ]?down|crane[- ]?down|descend", (0.6, 0.6, 0, 0, -1, 1)),
    (r"orbit|rotat|arc", (0.4, 0.7, -1, 1, 0, 0)),
    (r"static|locked|still|tripod", (0.1, 0.25, 0, 0, 0, 0)),
]
DEFAULT_MOTION = (0.0, 0.6, 0, 0, 0, 0)


def motion_for(camera):
    text = (camera or "").lower()
    for pattern, motion in MOTIONS:
        if re.search(pattern, text):
            return motion
    return DEFAULT_MOTION


def _even(x):
    return max(2, int(round(x / 2)) * 2)


def _ease(t):
    return t * t * (3 - 2 * t)  # smoothstep


def ken_burns(image, frames, out_w, out_h, motion, strength):
    """image: [H, W, C] float tensor. Returns [frames, out_h, out_w, C]."""
    h, w, c = image.shape
    src = image.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    target_ar = out_w / out_h
    # largest crop with the output aspect ratio that fits the source
    base_h = min(h, w / target_ar)
    base_w = base_h * target_ar
    z0, z1, px0, px1, py0, py1 = motion
    out = torch.empty((frames, out_h, out_w, c), dtype=image.dtype)
    for f in range(frames):
        t = _ease(f / max(frames - 1, 1))
        zoom = 1.0 + strength * (z0 + (z1 - z0) * t)
        cw, ch = base_w / zoom, base_h / zoom
        free_x, free_y = (w - cw) / 2, (h - ch) / 2
        cx = w / 2 + free_x * (px0 + (px1 - px0) * t)
        cy = h / 2 + free_y * (py0 + (py1 - py0) * t)
        x0 = min(max(cx - cw / 2, 0), w - cw)
        y0 = min(max(cy - ch / 2, 0), h - ch)
        # affine grid sample for sub-pixel smooth motion
        theta = torch.tensor([[[cw / w, 0, (x0 + cw / 2) / w * 2 - 1],
                               [0, ch / h, (y0 + ch / 2) / h * 2 - 1]]], dtype=src.dtype)
        grid = F.affine_grid(theta, (1, c, out_h, out_w), align_corners=False)
        frame = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)
        out[f] = frame[0].permute(1, 2, 0)
    return out


def resize_frames(frames, out_w, out_h):
    """frames [N,H,W,C] -> cover-crop to out size."""
    n, h, w, c = frames.shape
    if (w, h) == (out_w, out_h):
        return frames
    scale = max(out_w / w, out_h / h)
    nw, nh = max(out_w, round(w * scale)), max(out_h, round(h * scale))
    x = F.interpolate(frames.permute(0, 3, 1, 2), size=(nh, nw), mode="bilinear", align_corners=False)
    x0, y0 = (nw - out_w) // 2, (nh - out_h) // 2
    return x[:, :, y0:y0 + out_h, x0:x0 + out_w].permute(0, 2, 3, 1)


def join_with_crossfade(clips, fade):
    """clips: list of [N,H,W,C] (same size). Overlaps `fade` frames between clips."""
    out = clips[0]
    for nxt in clips[1:]:
        k = min(fade, len(out) - 1, len(nxt) - 1)
        if k <= 0:
            out = torch.cat([out, nxt])
            continue
        w = torch.linspace(0, 1, k + 2, dtype=out.dtype)[1:-1].view(k, 1, 1, 1)
        blend = out[-k:] * (1 - w) + nxt[:k] * w
        out = torch.cat([out[:-k], blend, nxt[k:]])
    return out


def shots_from_json(shot_list_json):
    if not shot_list_json or not shot_list_json.strip():
        return []
    try:
        return json.loads(shot_list_json).get("shots", [])
    except (ValueError, AttributeError):
        raise RuntimeError("shot_list_json is not a Shot Planner list.")


def _first(v, default=None):
    if isinstance(v, list):
        return v[0] if v else default
    return v


class ShotAnimatic:
    """One keyframe per shot -> a timed animatic with camera moves read from the shot list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframes": ("IMAGE", {"tooltip": "One image per shot (connect the batch output of your sampler chain)."}),
                "fps": ("INT", {"default": 12, "min": 6, "max": 30,
                                "tooltip": "Frame rate of the animatic. 12 is plenty for a preview."}),
                "width": ("INT", {"default": 960, "min": 256, "max": 1920, "step": 16,
                                  "tooltip": "Output width in pixels; height follows the aspect ratio."}),
                "aspect_ratio": (["from keyframe", "16:9", "9:16", "1:1", "4:5", "2.39:1"],
                                 {"default": "from keyframe",
                                  "tooltip": "Frame of the animatic. Keyframes are cropped to it."}),
                "motion_strength": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01,
                                              "tooltip": "How far the camera moves (0 = still frames)."}),
                "crossfade_frames": ("INT", {"default": 4, "min": 0, "max": 24,
                                             "tooltip": "Frames each cut dissolves over (0 = hard cuts)."}),
                "default_seconds": ("INT", {"default": 3, "min": 1, "max": 20,
                                            "tooltip": "Used when no shot list is connected."}),
            },
            "optional": {
                "shot_list_json": ("STRING", {"forceInput": True,
                                              "tooltip": "From Shot Planner: sets each shot's duration and camera move."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "fps", "timing")
    OUTPUT_TOOLTIPS = ("All animatic frames as one batch (connect CreateVideo).", "Number of frames.",
                       "Frame rate (connect CreateVideo fps).", "Length and camera move of every shot.")
    FUNCTION = "build"
    CATEGORY = "Shot Planner"
    DESCRIPTION = "Builds a timed animatic from keyframes, with push-ins, pans and tilts taken from each shot's camera note."

    def build(self, keyframes, fps, width, aspect_ratio, motion_strength, crossfade_frames,
              default_seconds, shot_list_json=None):
        fps, width = _first(fps), _first(width)
        aspect_ratio, strength = _first(aspect_ratio), _first(motion_strength)
        fade, default_seconds = _first(crossfade_frames), _first(default_seconds)
        shots = shots_from_json(_first(shot_list_json, ""))
        images = [img for batch in keyframes for img in batch]  # flatten list-of-batches
        if not images:
            raise RuntimeError("Shot Animatic: no keyframes received.")
        h0, w0 = images[0].shape[0], images[0].shape[1]
        if aspect_ratio == "from keyframe":
            ar = w0 / h0
        else:
            a, b = aspect_ratio.split(":")
            ar = float(a) / float(b)
        out_w, out_h = _even(width), _even(width / ar)
        clips, timing = [], []
        for i, img in enumerate(images):
            shot = shots[i] if i < len(shots) else {}
            secs = float(shot.get("duration", default_seconds) or default_seconds)
            n = max(2, int(round(secs * fps)))
            motion = motion_for(shot.get("camera", ""))
            clips.append(ken_burns(img.float().cpu(), n, out_w, out_h, motion, strength))
            timing.append(f"Shot {i + 1}: {secs:g}s, {shot.get('camera', 'default push-in') or 'default push-in'}")
        frames = join_with_crossfade(clips, fade)
        return (frames, int(frames.shape[0]), float(fps), "\n".join(timing))


class ClipStitcher:
    """Join per-shot video clips (e.g. one image-to-video run per shot) into one sequence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clips": ("IMAGE", {"tooltip": "Frames from your video model. Runs once per shot when fed by Shot Planner lists."}),
                "fps": ("INT", {"default": 16, "min": 6, "max": 60,
                                "tooltip": "Frame rate of the clips (Wan = 16)."}),
                "crossfade_frames": ("INT", {"default": 4, "min": 0, "max": 24,
                                             "tooltip": "Frames each cut dissolves over (0 = hard cuts)."}),
                "trim_to_shot_duration": ("BOOLEAN", {"default": True,
                    "tooltip": "Cut each clip to its shot's duration from the shot list. Shots without "
                               "a duration keep their full clip."}),
            },
            "optional": {"shot_list_json": ("STRING", {"forceInput": True,
                "tooltip": "Shot Planner plan, for the per-shot durations used by trimming."})},
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "frame_count", "fps")
    OUTPUT_TOOLTIPS = ("All clips joined as one batch.", "Number of frames.", "Frame rate (connect CreateVideo fps).")
    FUNCTION = "stitch"
    CATEGORY = "Shot Planner"
    DESCRIPTION = "Concatenates one clip per shot, resized to match, trimmed to each shot's duration, with crossfades."

    def stitch(self, clips, fps, crossfade_frames, trim_to_shot_duration, shot_list_json=None):
        fps, fade, trim = _first(fps), _first(crossfade_frames), _first(trim_to_shot_duration)
        shots = shots_from_json(_first(shot_list_json, ""))
        if not clips:
            raise RuntimeError("Clip Stitcher: no clips received.")
        out_h, out_w = clips[0].shape[1], clips[0].shape[2]
        prepared = []
        for i, clip in enumerate(clips):
            c = resize_frames(clip.float().cpu(), out_w, out_h)
            if trim and i < len(shots):
                secs = float(shots[i].get("duration", 0) or 0)
                n = max(2, int(round(secs * fps))) if secs > 0 else 0  # no duration: keep the clip
                if n and len(c) > n:
                    c = c[:n]
            prepared.append(c)
        frames = join_with_crossfade(prepared, fade)
        return (frames, int(frames.shape[0]), float(fps))


ANIMATIC_NODE_CLASS_MAPPINGS = {"ShotAnimatic": ShotAnimatic, "ClipStitcher": ClipStitcher}
ANIMATIC_NODE_DISPLAY_NAME_MAPPINGS = {"ShotAnimatic": "Shot Animatic (Keyframes to Video)",
                                       "ClipStitcher": "Clip Stitcher"}
