"""Core logic for ComfyUI Shot Planner: turns a brief or script into a shot list
with ready-to-use image and video prompts. No ComfyUI imports here, so it can be
tested on its own."""

import json
import os
import re
import urllib.error
import urllib.request

PROVIDERS = ["offline", "anthropic", "openai", "gemini"]

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}

ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:5", "2.39:1"]

CAMERA_CYCLE = [
    "wide establishing shot, slow push-in",
    "medium shot, eye level, gentle handheld",
    "close-up, shallow depth of field, static",
    "over-the-shoulder shot, slight rack focus",
    "tracking shot following the subject",
    "low-angle shot, slow tilt up",
    "top-down overhead shot, slow rotation",
    "extreme close-up on a detail, macro lens",
]

SYSTEM_PROMPT = """You are a film director and storyboard artist who writes prompts for AI image and video models.
Turn the user's brief into a shot list. Reply with JSON only, no commentary, using exactly this shape:
{"title": str, "style": str, "shots": [{"id": int, "description": str, "camera": str, "duration": int,
"image_prompt": str, "video_prompt": str}]}
Rules:
- image_prompt describes ONE still frame: subject, setting, composition, lighting, lens, the global style. No camera motion.
- video_prompt describes what MOVES in that frame over the shot: subject action, camera move, pacing. Keep it to 1-3 sentences.
- Keep characters, wardrobe and locations consistent across shots by repeating their key visual details.
- Do not put text, logos or captions in the frame unless the brief asks for them."""


class ShotPlannerError(Exception):
    pass


def _user_prompt(brief, num_shots, style, aspect_ratio, shot_seconds, extra):
    parts = [
        f"Brief:\n{brief.strip()}",
        f"Number of shots: {num_shots}",
        f"Visual style: {style.strip() or 'cinematic, photoreal'}",
        f"Aspect ratio: {aspect_ratio}",
        f"Target duration per shot: about {shot_seconds} seconds",
    ]
    if extra and extra.strip():
        parts.append(f"Extra instructions:\n{extra.strip()}")
    return "\n\n".join(parts)


def _post_json(url, payload, headers, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        raise ShotPlannerError(f"API error {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise ShotPlannerError(f"Could not reach the API: {e.reason}") from e


def _call_anthropic(model, key, system, user):
    r = _post_json("https://api.anthropic.com/v1/messages",
                   {"model": model, "max_tokens": 4096, "system": system,
                    "messages": [{"role": "user", "content": user}]},
                   {"x-api-key": key, "anthropic-version": "2023-06-01"})
    return "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")


def _call_openai(model, key, system, user):
    r = _post_json("https://api.openai.com/v1/chat/completions",
                   {"model": model, "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}]},
                   {"Authorization": f"Bearer {key}"})
    return r["choices"][0]["message"]["content"]


def _call_gemini(model, key, system, user):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = _post_json(url,
                   {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"responseMimeType": "application/json"}},
                   {"x-goog-api-key": key})
    return "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])


CALLERS = {"anthropic": _call_anthropic, "openai": _call_openai, "gemini": _call_gemini}


def extract_json(text):
    """Pull the first JSON object out of a model reply (handles code fences and chatter)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ShotPlannerError("The model did not return JSON.")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ShotPlannerError("The model's JSON was cut off.")


def normalize(plan, num_shots, style, shot_seconds):
    """Make sure every shot has every field, ids are 1..n, and durations are ints."""
    shots = plan.get("shots") or []
    if not isinstance(shots, list) or not shots:
        raise ShotPlannerError("No shots in the plan.")
    out = []
    for i, s in enumerate(shots[:num_shots], 1):
        s = s if isinstance(s, dict) else {"description": str(s)}
        desc = str(s.get("description", "")).strip()
        try:
            dur = int(round(float(s.get("duration", shot_seconds))))
        except (TypeError, ValueError):
            dur = shot_seconds
        out.append({
            "id": i,
            "description": desc,
            "camera": str(s.get("camera", "")).strip(),
            "duration": max(1, dur),
            "image_prompt": str(s.get("image_prompt") or desc).strip(),
            "video_prompt": str(s.get("video_prompt") or desc).strip(),
        })
    return {"title": str(plan.get("title", "Untitled")).strip() or "Untitled",
            "style": str(plan.get("style", style)).strip() or style,
            "shots": out}


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def plan_offline(brief, num_shots, style, aspect_ratio, shot_seconds):
    """Deterministic planner with no API: splits the brief into beats and builds prompts
    from templates. Useful for testing a workflow or working without a key."""
    beats = _sentences(brief) or ["An empty scene."]
    groups = [[] for _ in range(num_shots)]
    if len(beats) >= num_shots:
        for i, b in enumerate(beats):
            groups[i * num_shots // len(beats)].append(b)
    else:
        for i in range(num_shots):
            groups[i].append(beats[i * len(beats) // num_shots])
    style = style.strip() or "cinematic, photoreal"
    shots = []
    for i, g in enumerate(groups):
        desc = " ".join(g)
        cam = CAMERA_CYCLE[i % len(CAMERA_CYCLE)]
        shots.append({
            "id": i + 1,
            "description": desc,
            "camera": cam,
            "duration": shot_seconds,
            "image_prompt": f"{desc} {cam.split(',')[0]}. {style}, {aspect_ratio} frame, detailed, consistent characters and setting.",
            "video_prompt": f"{desc} Camera: {cam}. Smooth, natural motion over {shot_seconds} seconds.",
        })
    title = (beats[0][:60].rstrip(".!?") if beats else "Untitled")
    return {"title": title, "style": style, "shots": shots}


def plan_shots(brief, provider="offline", model="", api_key="", num_shots=6,
               style="cinematic, photoreal", aspect_ratio="16:9", shot_seconds=5,
               extra_instructions=""):
    if not brief or not brief.strip():
        raise ShotPlannerError("The brief is empty.")
    if provider not in PROVIDERS:
        raise ShotPlannerError(f"Unknown provider '{provider}'.")
    if provider == "offline":
        plan = plan_offline(brief, num_shots, style, aspect_ratio, shot_seconds)
    else:
        key = (api_key or "").strip() or os.environ.get(ENV_KEYS[provider], "")
        if not key:
            raise ShotPlannerError(
                f"No API key for {provider}. Set the {ENV_KEYS[provider]} environment variable "
                f"or paste a key into the api_key field.")
        user = _user_prompt(brief, num_shots, style, aspect_ratio, shot_seconds, extra_instructions)
        text = CALLERS[provider]((model or "").strip() or DEFAULT_MODELS[provider], key, SYSTEM_PROMPT, user)
        plan = extract_json(text)
    return normalize(plan, num_shots, style, shot_seconds)


def storyboard_text(plan):
    lines = [f"# {plan['title']}", f"Style: {plan['style']}", ""]
    for s in plan["shots"]:
        lines += [f"## Shot {s['id']} ({s['duration']}s) - {s['camera']}",
                  s["description"],
                  f"IMAGE: {s['image_prompt']}",
                  f"VIDEO: {s['video_prompt']}", ""]
    return "\n".join(lines).strip()
