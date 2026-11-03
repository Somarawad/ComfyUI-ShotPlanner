"""Core logic for Shot Critic: a vision LLM reviews generated shots against the shot list
and your rules, scores them and rewrites the prompt of every shot that fails.
No ComfyUI imports here, so it can be tested on its own."""

import base64
from concurrent.futures import ThreadPoolExecutor
import re
import time

try:
    from .shot_planner import ENV_KEYS, ShotPlannerError, _post_json, env_key, extract_json
except ImportError:  # imported on its own (tests)
    from shot_planner import ENV_KEYS, ShotPlannerError, _post_json, env_key, extract_json

CRITIC_PROVIDERS = ["gemini", "anthropic", "openai"]

CRITIC_MODELS = {
    # first model is tried first; the next ones are fallbacks if it is busy or unavailable
    "gemini": ["gemini-flash-lite-latest", "gemini-flash-latest"],
    "anthropic": ["claude-sonnet-4-5"],
    "openai": ["gpt-4o-mini"],
}

DEFAULT_CHECKS = """Anatomy: hands, fingers, limbs, eyes, faces (no extra or fused fingers, no warped faces)
Text and logos: spelled exactly as required, clean shapes, no gibberish lettering
Matches the shot: subject, action, framing and camera described in the shot's prompt
Continuity: same characters, wardrobe, product and location look across shots
Artifacts: melted objects, duplicated people, broken perspective, obvious AI smears"""

SYSTEM_PROMPT = """You are a strict art director and QC supervisor reviewing AI-generated frames for a commercial production.
For every shot you get the image and, when available, the prompt it was generated from.
First, for every shot write "seen": one or two factual sentences on what is actually visible (who is in it, what
each person wears, what they are doing, the setting, any text). Then judge the shot by comparing "seen" with the
shot's prompt, the checks and the brand rules, and compare shots with each other for continuity.
Score each shot from 0 to 10 (10 = ready for the client, 7 = acceptable, 4 or less = must be redone).
List concrete, visible issues only; do not invent problems. Tag every issue with how it should be fixed:
- "detail": a small local area that can be repaired in place without changing the rest of the shot
  (misspelled or garbled text and logos, eyes, hands and fingers, a small object).
  Give "target": one or two plain words naming that area so a segmentation model can find it
  (e.g. "logo", "text", "hand", "cup").
- "consistency": the shot is fine, but a recurring character or product looks different from how it looks
  in another shot or reference (different face, hair, clothing, colours, product design).
  Give "target": a plain noun a segmentation model can find for that subject, chosen so it picks the right one
  if several people are visible (e.g. "woman", "man", "cup"), and "reference": the shot that shows the correct
  look, as "shot N" (or "reference N" for a reference image). Only reference a shot whose look is correct.
- "reroll": the shot itself is wrong and must be generated again (wrong action, location, framing,
  missing subject or product, composition).
For every shot write fixed_prompt: a rewrite of its prompt that keeps the intent and fixes the reroll issues
(describe what should be visible, e.g. "relaxed hand with five fingers"; avoid negations).
For shots with consistency issues also write edit_instruction for an image editor that receives a close-up crop of
that subject from this shot as Picture 1 and a crop of the same subject from the reference as Picture 2. Name exactly
what must change and say everything else stays, e.g. "Make the woman in Picture 1 the same person as the woman in
Picture 2: same face and long light-brown loose hair. Change only her face and hair; keep her pose, clothing, the cup,
the background and the lighting of Picture 1 unchanged."
Reply with JSON only, exactly this shape:
{"shots": [{"shot": int, "seen": str, "score": int, "issues": [{"issue": str, "fix": "detail" or "consistency" or "reroll",
"target": str, "reference": str}], "fixed_prompt": str, "edit_instruction": str}], "continuity_notes": str}"""


def _b64(png_bytes):
    return base64.b64encode(png_bytes).decode("ascii")


def build_user_text(shots, checks, brand_rules, n_refs):
    parts = ["Checks:\n" + (checks.strip() or DEFAULT_CHECKS)]
    if brand_rules and brand_rules.strip():
        parts.append("Brand rules (must be respected):\n" + brand_rules.strip())
    if n_refs:
        parts.append(f"{n_refs} reference image(s) follow first, labeled 'Reference N'. "
                     "They show how the product, logo or character must look.")
    lines = []
    for i, s in enumerate(shots, 1):
        p = (s.get("image_prompt") or "").strip()
        lines.append(f"Shot {i}: " + (f"prompt: {p}" if p else "(no prompt given)"))
    parts.append("Shots to review (images follow in this order, labeled 'Shot N'):\n" + "\n".join(lines))
    return "\n\n".join(parts)


TIMEOUT = 150  # seconds per attempt; a hung model falls through to the next one sooner


def _gemini(model, key, system, text, images):
    parts = [{"text": text}]
    for label, png in images:
        parts.append({"text": label})
        parts.append({"inline_data": {"mime_type": "image/png", "data": _b64(png)}})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": "low"}}}
    try:
        r = _post_json(url, body, {"x-goog-api-key": key}, timeout=TIMEOUT, retries=1)
    except ShotPlannerError as e:
        if getattr(e, "status", None) != 400:
            raise
        body["generationConfig"].pop("thinkingConfig")  # model without thinking levels
        r = _post_json(url, body, {"x-goog-api-key": key}, timeout=TIMEOUT, retries=1)
    parts = r["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _anthropic(model, key, system, text, images):
    content = [{"type": "text", "text": text}]
    for label, png in images:
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(png)}})
    r = _post_json("https://api.anthropic.com/v1/messages",
                   {"model": model, "max_tokens": 4096, "system": system,
                    "messages": [{"role": "user", "content": content}]},
                   {"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=TIMEOUT, retries=1)
    return "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")


def _openai(model, key, system, text, images):
    content = [{"type": "text", "text": text}]
    for label, png in images:
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(png)}})
    r = _post_json("https://api.openai.com/v1/chat/completions",
                   {"model": model, "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": content}]},
                   {"Authorization": f"Bearer {key}"}, timeout=TIMEOUT, retries=1)
    return r["choices"][0]["message"]["content"]


CALLERS = {"gemini": _gemini, "anthropic": _anthropic, "openai": _openai}


FIXES = ("detail", "consistency", "reroll")


def _ref(value):
    """'shot 3' -> ('shot', 3); 'reference 1' -> ('reference', 1); anything else -> None."""
    m = re.search(r"(shot|ref\w*)\s*#?\s*(\d+)", str(value or ""), re.I)
    if not m:
        return None
    return ("shot" if m.group(1).lower() == "shot" else "reference", int(m.group(2)))


def _issue(x):
    if isinstance(x, dict):
        text = str(x.get("issue") or x.get("text") or "").strip()
        f = str(x.get("fix", "")).lower()
        fix = next((k for k in FIXES if f.startswith(k[:6])), "reroll")
        target = str(x.get("target") or "").strip().lower() if fix != "reroll" else ""
        ref = _ref(x.get("reference")) if fix == "consistency" else None
    else:
        text, fix, target, ref = str(x).strip(), "reroll", "", None
    if not text:
        return None
    return {"issue": text, "fix": fix, "target": target, "reference": ref}


def _auto_instruction(issues):
    targets = ", ".join(sorted({d["target"] for d in issues if d["target"]})) or "subject"
    return (f"Make the {targets} in Picture 1 look exactly like the {targets} in Picture 2: same face, hair, clothing, "
            f"colours and design. Keep the pose, framing, background and lighting of Picture 1 unchanged.")


def normalize_verdicts(raw, shots, pass_score):
    """One verdict per shot, with clean types and a route (most serious first):
    reroll      - the shot itself is wrong: generate it again from fixed_prompt
    consistency - a character or product doesn't match another shot: edit it with that shot as reference
    detail      - only local problems (text, logos, hands): repair in place, e.g. with Detail Fixer
    pass        - good as it is"""
    by_num = {}
    for v in raw.get("shots") or []:
        if not isinstance(v, dict):
            continue
        try:
            n = int(v.get("shot"))
        except (TypeError, ValueError):
            continue
        by_num[n] = v
    out = []
    for i, s in enumerate(shots, 1):
        v = by_num.get(i, {})
        try:
            score = max(0, min(10, int(round(float(v.get("score", 0))))))
        except (TypeError, ValueError):
            score = 0
        issues = [d for d in (_issue(x) for x in (v.get("issues") or [])) if d]
        original = (s.get("image_prompt") or "").strip()
        fixed = str(v.get("fixed_prompt") or "").strip() or original
        if not v:
            issues = [{"issue": "The critic returned no verdict for this shot.", "fix": "reroll", "target": "", "reference": None}]
        by_fix = {k: [d for d in issues if d["fix"] == k] for k in FIXES}
        # a consistency fix needs a usable reference: another shot or a reference image
        cons = [d for d in by_fix["consistency"] if d["reference"] and d["reference"] != ("shot", i)]
        if not v or by_fix["reroll"] or (score < pass_score and not (cons or by_fix["detail"])):
            route = "reroll"
        elif cons:
            route = "consistency"
        elif by_fix["detail"]:
            route = "detail"
        else:
            route = "pass"
        ref = cons[0]["reference"] if cons else None
        cons_target = next((d["target"] for d in cons if d["target"]), "person") if cons else ""
        instruction = str(v.get("edit_instruction") or "").strip() or (_auto_instruction(cons) if cons else "")
        out.append({"shot": i, "seen": str(v.get("seen") or "").strip(), "score": score, "route": route,
                    "pass": route != "reroll",
                    "issues": issues, "detail_targets": sorted({d["target"] for d in by_fix["detail"] if d["target"]}),
                    "reference": list(ref) if ref else None, "edit_instruction": instruction,
                    "consistency_target": cons_target,
                    "prompt": original, "fixed_prompt": fixed})
    return {"shots": out, "continuity_notes": str(raw.get("continuity_notes") or "").strip(),
            "pass_score": pass_score}


SEVERITY = {"pass": 0, "detail": 1, "consistency": 2, "reroll": 3}


def _prior_votes(prior, i):
    """Votes an earlier review cast for shot i, unless that shot was sent to be re-rendered
    (then it is a different image now and the old votes don't apply)."""
    if not prior or i >= len(prior.get("shots", [])):
        return []
    old = prior["shots"][i]
    if old.get("route") == "reroll":
        return []
    routes = old.get("votes") or [old.get("route", "pass")] * int(prior.get("reviews", 1) or 1)
    return [dict(old, route=r) for r in routes]


def combine_reviews(reviews, prior=None):
    """Majority vote over several independent reviews of the same shots (plus, optionally, the votes of an
    earlier review for shots that haven't changed since). A shot gets the most serious route that at least
    half of the votes reached; its issues and instructions come from a vote that chose that route.
    Fixes random misses of fast models."""
    if len(reviews) == 1 and not prior:
        return reviews[0]
    out = []
    for i in range(len(reviews[0]["shots"])):
        fresh = [r["shots"][i] for r in reviews]
        votes = fresh + _prior_votes(prior, i)
        need = (len(votes) + 1) // 2
        route = "pass"
        for cand in ("reroll", "consistency", "detail"):
            if sum(1 for v in votes if SEVERITY[v["route"]] >= SEVERITY[cand]) >= need:
                route = cand
                break
        pick = next((v for v in fresh if v["route"] == route), None) or \
            next((v for v in votes if v["route"] == route), None) or \
            min(votes, key=lambda v: abs(SEVERITY[v["route"]] - SEVERITY[route]))
        scores = sorted(v["score"] for v in votes)
        merged = dict(pick, route=route)
        merged["score"] = scores[len(scores) // 2]
        merged["pass"] = route != "reroll"
        merged["votes"] = [v["route"] for v in votes]
        out.append(merged)
    first = reviews[0]
    combined = {"shots": out, "continuity_notes": first.get("continuity_notes", ""),
                "pass_score": first["pass_score"], "reviews": len(reviews)}
    for k in ("model",):
        if k in first:
            combined[k] = first[k]
    combined["seconds"] = max(r.get("seconds", 0) for r in reviews)
    return combined


def _one_review(images_png, shots, checks, brand_rules, refs, provider, model, key, pass_score):
    labeled = [(f"Reference {i}:", p) for i, p in enumerate(refs, 1)]
    labeled += [(f"Shot {i}:", p) for i, p in enumerate(images_png, 1)]
    text = build_user_text(shots, checks, brand_rules, len(refs))
    models = [model.strip()] if model and model.strip() else CRITIC_MODELS[provider]
    attempts = [m for m in models for _ in range(2)]  # each model gets a second try (e.g. malformed JSON)
    last = None
    for m in attempts:
        try:
            t0 = time.time()
            reply = CALLERS[provider](m, key, SYSTEM_PROMPT, text, labeled)
            try:
                parsed = extract_json(reply)
            except (ValueError, KeyError, IndexError) as e:
                raise ShotPlannerError(f"The critic's reply was not valid JSON ({e}).") from e
            v = normalize_verdicts(parsed, shots, pass_score)
            v["model"], v["seconds"] = m, round(time.time() - t0, 1)
            if last is not None:
                v["fallback_reason"] = str(last)[:200]
            return v
        except ShotPlannerError as e:  # busy / unavailable model: try the next one
            last = e
            if getattr(e, "status", None) not in (404, 429, 500, 502, 503, 529, None):
                raise
            if getattr(e, "status", None) == 404:  # model doesn't exist: skip its second try
                continue
    raise last


def critique(images_png, shots=None, checks="", brand_rules="", references_png=None,
             provider="gemini", model="", api_key="", pass_score=7, reviews=1, prior=None):
    """images_png: list of PNG bytes (one per shot). shots: list of dicts with image_prompt.
    reviews > 1 runs that many independent reviews in parallel and takes a majority vote."""
    if not images_png:
        raise ShotPlannerError("Shot Critic got no images.")
    if provider not in CRITIC_PROVIDERS:
        raise ShotPlannerError(f"Unknown provider '{provider}'.")
    shots = list(shots or [])
    shots = (shots + [{}] * len(images_png))[:len(images_png)]
    key = (api_key or "").strip() or env_key(provider)
    if not key:
        raise ShotPlannerError(f"No API key for {provider}. Set {ENV_KEYS[provider]} or paste a key.")
    refs = list(references_png or [])
    args = (images_png, shots, checks, brand_rules, refs, provider, model, key, pass_score)
    n = max(1, int(reviews or 1))
    if n == 1 and not prior:
        return _one_review(*args)
    with ThreadPoolExecutor(max_workers=max(1, n)) as pool:
        futures = [pool.submit(_one_review, *args) for _ in range(n)]
        results, errors = [], []
        for f in futures:
            try:
                results.append(f.result())
            except ShotPlannerError as e:
                errors.append(e)
    if not results:
        raise errors[0]
    return combine_reviews(results, prior)


def report_text(verdicts):
    shots = verdicts["shots"]
    counts = {r: sum(1 for v in shots if v["route"] == r) for r in ("pass", "detail", "consistency", "reroll")}
    lines = [f"Shot Critic: {counts['pass']} pass, {counts['detail']} need a detail fix, "
             f"{counts['consistency']} need a consistency fix, "
             f"{counts['reroll']} need a re-render (threshold {verdicts['pass_score']}/10)"]
    if verdicts.get("model"):
        n = verdicts.get("reviews", 1)
        lines.append(f"Reviewed by {verdicts['model']} in {verdicts.get('seconds', '?')}s"
                     + (f", majority of {n} reviews" if n > 1 else "")
                     + (" (fallback: first model failed or timed out)" if verdicts.get("fallback_reason") else ""))
    tags = {"pass": "PASS", "detail": "DETAIL FIX", "consistency": "CONSISTENCY", "reroll": "RE-RENDER"}
    for v in shots:
        votes = f"  (votes: {', '.join(v['votes'])})" if v.get("votes") else ""
        lines.append(f"\nShot {v['shot']}  {tags[v['route']]}  {v['score']}/10{votes}")
        if v.get("seen"):
            lines.append(f"  Seen: {v['seen']}")
        for d in v["issues"]:
            where = f" -> {d['target']}" if d["target"] else ""
            if d.get("reference"):
                where += f" (like {d['reference'][0]} {d['reference'][1]})"
            lines.append(f"  - [{d['fix']}{where}] {d['issue']}")
        if v["route"] == "reroll":
            lines.append(f"  New prompt: {v['fixed_prompt']}")
        if v["route"] == "consistency":
            kind, n = v["reference"]
            lines.append(f"  Edit with {kind} {n} as reference: {v['edit_instruction']}")
    if verdicts.get("continuity_notes"):
        lines.append("\nContinuity: " + verdicts["continuity_notes"])
    return "\n".join(lines)
