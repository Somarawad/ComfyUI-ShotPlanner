"""Shot Motion Prompts: per-shot image-to-video prompts and frame counts per engine.

Pure Python (no ComfyUI). Frame counts are snapped to each engine's latent grid:
Wan 4k+1, MiniMax H3 17k+5, LTX 8k+1. API engines (Kling, Seedance, Hailuo) take a
duration in seconds from a fixed set, so they get the nearest allowed duration.
"""

import math
import re

try:
    from . import stage_utils as su
except ImportError:  # imported as a top-level module (tests)
    import stage_utils as su

StageError = su.StageError

# kind "grid": frames = step*k + offset (k >= 1), clamped to max_frames.
# kind "seconds": allowed durations in seconds.
ENGINES = {
    "wan2.1": {"fps": 16, "kind": "grid", "step": 4, "offset": 1, "max_frames": 81},
    "wan2.2": {"fps": 16, "kind": "grid", "step": 4, "offset": 1, "max_frames": 121},
    "minimax-h3": {"fps": 24, "kind": "grid", "step": 17, "offset": 5, "max_frames": 243},
    "ltx": {"fps": 24, "kind": "grid", "step": 8, "offset": 1, "max_frames": 257},
    "kling": {"fps": 24, "kind": "seconds", "allowed": [5, 10]},
    "seedance": {"fps": 24, "kind": "seconds", "allowed": list(range(4, 13))},
    "hailuo": {"fps": 24, "kind": "seconds", "allowed": [6, 10]},
}
ENGINE_NAMES = list(ENGINES)
MOTION_STRENGTHS = ["subtle", "natural", "dynamic"]
DEFAULT_SECONDS = 3.0

SPEED = {"subtle": "very slow", "natural": "slow", "dynamic": "fast"}
SUBJECT_MOTION = {
    "subtle": "subtle, minimal motion, mostly still, gentle natural movement",
    "natural": "natural, fluid motion at a realistic pace",
    "dynamic": "energetic, dynamic motion with clear momentum",
}

# (regex on the lower-cased camera text, camera phrase). First match wins, so specific
# moves come before generic words. {speed} is filled from motion_strength.
CAMERA_VOCAB = [
    (r"push[\s-]?in|dolly[\s-]?in|move[s]? in\b|creep[s]? in", "{speed} dolly push-in"),
    (r"pull[\s-]?(back|out)|dolly[\s-]?(out|back)", "{speed} dolly pull-back"),
    (r"zoom[\s-]?in", "{speed} zoom in"),
    (r"zoom[\s-]?out", "{speed} zoom out"),
    (r"whip", "fast whip pan"),
    (r"pan(s|ning)?\s+left", "{speed} pan left"),
    (r"pan(s|ning)?\s+right", "{speed} pan right"),
    (r"tilt(s|ing)?\s+up", "{speed} tilt up"),
    (r"tilt(s|ing)?\s+down", "{speed} tilt down"),
    (r"top[\s-]?down|overhead|bird", "{speed} overhead rotation"),
    (r"orbit|arc(s|ing)?\b|360|rotat", "{speed} orbit around the subject"),
    (r"crane|jib|boom", "{speed} crane up revealing the scene"),
    (r"drone|aerial|fly[\s-]?over", "{speed} aerial drone glide forward"),
    (r"track|follow|steadicam|gimbal", "{speed} tracking shot following the subject"),
    (r"truck(s|ing)?\s+left", "{speed} truck left"),
    (r"truck(s|ing)?\s+right|truck", "{speed} lateral truck"),
    (r"\bpan", "{speed} lateral pan"),
    (r"\btilt", "{speed} tilt"),
    (r"rack[\s-]?focus|focus pull", "locked-off camera with a rack focus"),
    (r"handheld|hand-held|shaky", "subtle handheld drift"),
    (r"static|locked|tripod|still camera|fixed", "locked-off static camera"),
]
# framing words carried over as-is (no motion of their own)
FRAMING = [
    (r"extreme close[\s-]?up|\becu\b", "extreme close-up"),
    (r"macro", "macro close-up"),
    (r"close[\s-]?up|\bcu\b", "close-up"),
    (r"medium|\bmid\b|\bms\b", "medium shot"),
    (r"over[\s-]?the[\s-]?shoulder|\bots\b", "over-the-shoulder shot"),
    (r"establishing|extreme wide|wide|\bws\b|long shot", "wide shot"),
    (r"low[\s-]?angle", "low angle"),
    (r"high[\s-]?angle", "high angle"),
]
# when the camera field only gives framing, pick a move that suits the strength
DEFAULT_MOVE = {"subtle": "almost static camera with a gentle drift",
                "natural": "slow dolly push-in",
                "dynamic": "fast tracking shot following the subject"}

NEGATIVE_BASE = ("static frame, frozen, still image, morphing, warping, melting, distorted "
                 "faces, deformed hands, extra fingers, extra limbs, flicker, jitter, "
                 "sudden cut, scene change, blurry, low quality, compression artifacts, "
                 "text, subtitles, watermark, logo distortion")
NEGATIVE_EXTRA = {
    "wan2.1": ", overexposed, oversaturated, grey cast, worst quality, jpeg artifacts",
    "wan2.2": ", overexposed, oversaturated, grey cast, worst quality, jpeg artifacts",
    "ltx": ", worst quality, inconsistent motion, jittery",
    "minimax-h3": ", inconsistent identity",
}


def camera_move(camera, strength="natural"):
    """Map a shot's free-text camera field to a camera-motion phrase for video models.

    "push-in" -> "slow dolly push-in"; "close-up, static" -> "close-up, locked-off static
    camera"; framing-only text gets a default move for the strength."""
    if strength not in SPEED:
        raise StageError(f"unknown motion_strength {strength!r}")
    text = str(camera or "").lower()
    framing = next((f for pat, f in FRAMING if re.search(pat, text)), "")
    move = ""
    for pat, phrase in CAMERA_VOCAB:
        if re.search(pat, text):
            move = phrase.format(speed=SPEED[strength])
            break
    if not move:
        move = DEFAULT_MOVE[strength]
    return f"{framing}, {move}" if framing else move


def _subject_action(shot):
    desc = str(shot.get("description") or "").strip()
    if not desc:
        desc = str(shot.get("video_prompt") or shot.get("image_prompt") or "").strip()
    desc = re.sub(r"\s+", " ", desc).rstrip(" .")
    return desc


def motion_prompt(shot, strength="natural", style_suffix=""):
    """One image-to-video prompt: subject action + motion pacing + camera move + style."""
    action = _subject_action(shot) or "The scene comes to life"
    parts = [action[0].upper() + action[1:] + ".",
             SUBJECT_MOTION[strength][0].upper() + SUBJECT_MOTION[strength][1:] + ".",
             "Camera: " + camera_move(shot.get("camera", ""), strength) + "."]
    if style_suffix and style_suffix.strip():
        parts.append(style_suffix.strip().rstrip(".") + ".")
    return " ".join(parts)


def snap_frames(seconds, engine, fps=None):
    """(frame_count, seconds) for an engine. Grid engines snap to the nearest valid count
    (ties go to the longer one) within [first grid point, max_frames]; seconds engines pick
    the nearest allowed duration (ties go to the longer one)."""
    if engine not in ENGINES:
        raise StageError(f"unknown engine {engine!r} (choose from {', '.join(ENGINE_NAMES)})")
    spec = ENGINES[engine]
    fps = float(fps or spec["fps"])
    seconds = float(seconds)
    if seconds <= 0:
        raise StageError(f"duration {seconds:g} s must be > 0")
    if spec["kind"] == "seconds":
        allowed = spec["allowed"]
        best = min(allowed, key=lambda a: (abs(a - seconds), -a))
        return int(round(best * fps)), float(best)
    step, off = spec["step"], spec["offset"]
    target = seconds * fps
    k_max = max(1, (spec["max_frames"] - off) // step)
    k = math.floor((target - off) / step + 0.5)
    k = min(max(1, k), k_max)
    frames = step * k + off
    return int(frames), round(frames / fps, 4)


def shot_durations(plan, durations_text=""):
    """Seconds per shot: the durations string overrides the plan for the shots it covers."""
    manual = su.parse_number_list(durations_text, "durations")
    out = []
    for i, s in enumerate(plan["shots"]):
        if i < len(manual):
            out.append(manual[i])
            continue
        try:
            d = float(s.get("duration", DEFAULT_SECONDS))
        except (TypeError, ValueError, AttributeError):
            d = DEFAULT_SECONDS
        out.append(d if d > 0 else DEFAULT_SECONDS)
    return out


def negative_prompt(engine, strength="natural"):
    neg = NEGATIVE_BASE + NEGATIVE_EXTRA.get(engine, "")
    if strength == "dynamic":
        neg = neg.replace("static frame, ", "")
    return neg


def build_motion_prompts(shot_list_json, engine="wan2.1", strength="natural", fps_override=0,
                         style_suffix="", durations_text=""):
    """-> dict(prompts, frame_counts, seconds, negative_prompt, fps)."""
    plan = su.load_shot_list(shot_list_json)
    if plan is None or not plan["shots"]:
        raise StageError("shot_list_json is empty: connect Shot Planner's shot_list_json")
    if engine not in ENGINES:
        raise StageError(f"unknown engine {engine!r}")
    if strength not in MOTION_STRENGTHS:
        raise StageError(f"unknown motion_strength {strength!r}")
    fps = int(fps_override) if fps_override and int(fps_override) > 0 else ENGINES[engine]["fps"]
    shots = [s if isinstance(s, dict) else {"description": str(s)} for s in plan["shots"]]
    secs_in = shot_durations({"shots": shots}, durations_text)
    prompts, frames, secs = [], [], []
    for shot, d in zip(shots, secs_in):
        prompts.append(motion_prompt(shot, strength, style_suffix))
        f, s = snap_frames(d, engine, fps)
        frames.append(f)
        secs.append(s)
    return {"prompts": prompts, "frame_counts": frames, "seconds": secs,
            "negative_prompt": negative_prompt(engine, strength), "fps": fps}
