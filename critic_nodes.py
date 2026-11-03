import io
import json

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .shot_critic import CRITIC_PROVIDERS, DEFAULT_CHECKS, critique, report_text
from .shot_planner import ShotPlannerError


def _first(v, default=None):
    if isinstance(v, list):
        return v[0] if v else default
    return v


def _to_pil(img):
    return Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))


def _png(pil, max_side=1024):
    pil = pil.copy()
    pil.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _clip_strip(frames, n=3):
    """A video clip (batch of frames) -> one image with start / middle / end frames side by side."""
    idx = sorted({0, len(frames) // 2, len(frames) - 1})[:n]
    pils = [_to_pil(frames[i]) for i in idx]
    h = 512
    pils = [p.resize((max(1, int(p.width * h / p.height)), h)) for p in pils]
    strip = Image.new("RGB", (sum(p.width for p in pils) + 8 * (len(pils) - 1), h), "white")
    x = 0
    for p in pils:
        strip.paste(p, (x, 0))
        x += p.width + 8
    return strip


def _font(size):
    for f in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


ROUTE_STYLE = {"pass": ((46, 160, 67), "PASS"), "detail": ((214, 140, 18), "DETAIL FIX"),
               "consistency": ((124, 77, 196), "CONSISTENCY"), "reroll": ((218, 54, 51), "RE-RENDER")}


def contact_sheet(pils, verdicts, cell_w=384):
    n = len(pils)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    ar = pils[0].height / pils[0].width
    cell_h = int(cell_w * ar)
    bar = 40
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + bar)), (24, 24, 24))
    f = _font(20)
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(pils):
        v = verdicts["shots"][i]
        t = p.copy()
        t.thumbnail((cell_w, cell_h))
        x, y = (i % cols) * cell_w, (i // cols) * (cell_h + bar)
        sheet.paste(t, (x + (cell_w - t.width) // 2, y + bar + (cell_h - t.height) // 2))
        color, label = ROUTE_STYLE.get(v.get("route", "pass" if v["pass"] else "reroll"))
        d.rectangle([x, y, x + cell_w - 1, y + bar - 2], fill=color)
        d.text((x + 10, y + 9), f"Shot {v['shot']}  {label}  {v['score']}/10", fill="white", font=f)
    return torch.from_numpy(np.array(sheet).astype(np.float32) / 255.0).unsqueeze(0)


class ShotCritic:
    """A vision model reviews your shots like an art director: scores each one, lists the issues,
    and rewrites the prompt of every shot that fails."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "One image per shot (the list from your sampler chain), or video clips."}),
                "provider": (CRITIC_PROVIDERS, {"default": "gemini",
                    "tooltip": "Vision LLM that reviews the shots: gemini (GEMINI_API_KEY / GOOGLE_API_KEY), "
                               "anthropic (ANTHROPIC_API_KEY) or openai (OPENAI_API_KEY)."}),
                "model": ("STRING", {"default": "", "tooltip": "Leave empty for the default (Gemini: Flash Lite, falls back to Flash if unavailable)."}),
                "pass_score": ("INT", {"default": 7, "min": 1, "max": 10,
                                       "tooltip": "Shots scoring below this fail and get a rewritten prompt."}),
                "input_type": (["images (one per shot)", "video clips (one batch per shot)"],
                               {"default": "images (one per shot)",
                                "tooltip": "For clips, the start, middle and end frames are reviewed."}),
                "checks": ("STRING", {"multiline": True, "default": DEFAULT_CHECKS,
                    "tooltip": "What the critic checks on every shot, one topic per line."}),
                "brand_rules": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "Anything that must be true, e.g. 'The cup logo reads NOVARA in white capitals'."}),
                "candidates": ("INT", {"default": 3, "min": 1, "max": 8,
                                       "tooltip": "How many new versions to render for each shot that needs a re-render. "
                                                  "Review them with a second Shot Critic and Shot Reroll Merge keeps the best."}),
                "reviews": ("INT", {"default": 3, "min": 1, "max": 5,
                                    "tooltip": "Independent reviews run in parallel; each shot gets the majority verdict. "
                                               "Makes fast models (e.g. Gemini Flash Lite) much more reliable."}),
                "seed": ("INT", {"default": 1000, "min": 0, "max": 2**31 - 1,
                                 "tooltip": "Base seed for the re-renders (each candidate gets its own)."}),
            },
            "optional": {
                "shot_list_json": ("STRING", {"forceInput": True, "tooltip": "From Shot Planner: gives the critic each shot's intent."}),
                "prompts": ("STRING", {"forceInput": True, "tooltip": "Or connect the list of prompts used for the images."}),
                "references": ("IMAGE", {"tooltip": "Optional: product, logo or character references the shots must match."}),
                "prior_verdicts_json": ("STRING", {"forceInput": True,
                                                   "tooltip": "Optional: verdicts of an earlier Shot Critic on the same shots. Its votes "
                                                              "count again for shots that weren't re-rendered since, so one flaky "
                                                              "review can't drop a real problem."}),
                "api_key": ("STRING", {"default": "", "tooltip": "Prefer the environment variable (e.g. GEMINI_API_KEY)."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "INT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("report", "contact_sheet", "verdicts_json", "reroll_prompts", "reroll_seeds",
                    "rerender_shots", "detail_shots", "detail_targets", "consistency_shots")
    OUTPUT_IS_LIST = (False, False, False, True, True, False, False, False, False)
    OUTPUT_TOOLTIPS = (
        "Readable review: score, issues (tagged detail or reroll) and new prompt per shot.",
        "All shots labeled PASS / DETAIL FIX / RE-RENDER.",
        "Machine-readable verdicts (connect to Shot Reroll Merge).",
        "Rewritten prompts for the shots that need a re-render, repeated once per candidate. Feed to a sampler.",
        "One seed per re-render prompt (connect to the sampler's seed) so candidates differ.",
        "Numbers of the shots that need a re-render, e.g. '2, 5'.",
        "Numbers of the shots that only need a local fix (logo, text, face, hands).",
        "What the detail fixes should look for, e.g. 'logo, hand' (use as SAM 3 prompts).",
        "Numbers of the shots where a character or product doesn't match another shot.",
    )
    FUNCTION = "review"
    CATEGORY = "Shot Planner"
    DESCRIPTION = "Reviews generated shots against the shot list and your rules, and rewrites prompts for the ones that fail."

    def review(self, images, provider, model, pass_score, input_type, checks, brand_rules, candidates=1, reviews=1, seed=1000,
               shot_list_json=None, prompts=None, references=None, prior_verdicts_json=None, api_key=None):
        provider, model, pass_score = _first(provider), _first(model, ""), _first(pass_score)
        candidates, seed = max(1, int(_first(candidates, 1))), int(_first(seed, 1000))
        reviews = max(1, int(_first(reviews, 1)))
        input_type, checks, brand_rules = _first(input_type), _first(checks, ""), _first(brand_rules, "")
        api_key = _first(api_key, "") or ""
        if "clips" in input_type:
            pils = [_clip_strip(batch) for batch in images]
        else:
            pils = [_to_pil(img) for batch in images for img in batch]
        shots = []
        sl = _first(shot_list_json, "")
        if sl:
            try:
                shots = json.loads(sl).get("shots", [])
            except (ValueError, AttributeError):
                raise RuntimeError("Shot Critic: shot_list_json is not a Shot Planner list.")
        if prompts:
            shots = [{"image_prompt": p} for p in prompts]
        refs = [_png(_to_pil(img)) for batch in (references or []) for img in batch]
        try:
            verdicts = critique([_png(p) for p in pils], shots, checks, brand_rules, refs,
                                provider, model, api_key, pass_score, reviews,
                                json.loads(_first(prior_verdicts_json, "") or "null") or None)
        except ShotPlannerError as e:
            raise RuntimeError(f"Shot Critic: {e}") from e
        verdicts["candidates"] = candidates
        rer = [v for v in verdicts["shots"] if v["route"] == "reroll"]
        det = [v for v in verdicts["shots"] if v["route"] == "detail"]
        prompts_out = [v["fixed_prompt"] for v in rer for _ in range(candidates)] or [""]
        seeds_out = [seed + v["shot"] * 100 + c for v in rer for c in range(candidates)] or [seed]
        targets = sorted({t for v in verdicts["shots"] for t in v["detail_targets"]})
        return (report_text(verdicts), contact_sheet(pils, verdicts), json.dumps(verdicts, indent=2),
                prompts_out, seeds_out, ", ".join(str(v["shot"]) for v in rer),
                ", ".join(str(v["shot"]) for v in det), ", ".join(targets),
                ", ".join(str(v["shot"]) for v in verdicts["shots"] if v["route"] == "consistency"))


class ShotRerollMerge:
    """Put re-rendered shots back in place of the ones that needed a re-render, keeping the best
    candidate when a second Shot Critic reviewed them. If nothing needed a re-render, the
    re-render branch does not run at all."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The original shots (same list the critic reviewed)."}),
                "verdicts_json": ("STRING", {"forceInput": True,
                                             "tooltip": "Shot Critic verdicts_json: says which shots to replace."}),
                "route": (["reroll", "consistency", "detail"], {"default": "reroll",
                          "tooltip": "Which shots to replace: re-rendered ones, ones edited for consistency, or ones given a detail fix."}),
            },
            "optional": {
                "rerolled": ("IMAGE", {"lazy": True,
                                       "tooltip": "Images rendered from reroll_prompts, in the same order."}),
                "candidate_verdicts_json": ("STRING", {"forceInput": True, "lazy": True,
                                                       "tooltip": "Optional: verdicts_json of a second Shot Critic run on the "
                                                                  "re-renders. The highest-scoring candidate of each shot is kept."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "summary")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_TOOLTIPS = ("All shots in order, with the chosen route's shots replaced.",
                       "Which shots were replaced, and by which candidate.")
    FUNCTION = "merge"
    CATEGORY = "Shot Planner"
    DESCRIPTION = "Replaces shots that needed a re-render with the best re-render. Skips the branch when nothing failed."

    @staticmethod
    def _plan(verdicts_json, route="reroll"):
        v = json.loads(_first(verdicts_json, "{}") or "{}")
        shots = [s["shot"] for s in v.get("shots", []) if s.get("route", "pass" if s.get("pass") else "reroll") == route]
        return shots, (int(v.get("candidates", 1) or 1) if route == "reroll" else 1)

    @staticmethod
    def _missing(x):
        return x is None or (isinstance(x, (list, tuple)) and any(i is None for i in x))

    def check_lazy_status(self, images, verdicts_json, route="reroll", rerolled=None, candidate_verdicts_json="__unset__"):
        shots, _ = self._plan(verdicts_json, _first(route, "reroll"))
        if not shots:
            return []
        need = []
        if self._missing(rerolled):
            need.append("rerolled")
        if candidate_verdicts_json != "__unset__" and self._missing(candidate_verdicts_json):
            need.append("candidate_verdicts_json")
        return need

    def merge(self, images, verdicts_json, route="reroll", rerolled=None, candidate_verdicts_json="__unset__"):
        flat = [img.unsqueeze(0) for batch in images for img in batch]
        route = _first(route, "reroll")
        shots, n = self._plan(verdicts_json, route)
        if not shots:
            return (flat, f"No shots needed a {route} fix.")
        new = [img.unsqueeze(0) for batch in (rerolled or []) for img in batch]
        if len(new) < len(shots) * n:
            raise RuntimeError(f"Shot Reroll Merge: expected {len(shots) * n} re-renders "
                               f"({len(shots)} shots x {n} candidates), got {len(new)}.")
        scores = None
        if candidate_verdicts_json not in ("__unset__", None):
            cv = json.loads(_first(candidate_verdicts_json, "{}") or "{}")
            scores = [s.get("score", 0) for s in cv.get("shots", [])]
        notes = []
        for k, shot in enumerate(shots):
            group = list(range(k * n, k * n + n))
            best = max(group, key=lambda j: scores[j] if scores and j < len(scores) else 0) if scores else group[0]
            img = new[best]
            i = shot - 1
            if 0 <= i < len(flat):
                if img.shape[1:3] != flat[i].shape[1:3]:
                    img = torch.nn.functional.interpolate(img.permute(0, 3, 1, 2), size=flat[i].shape[1:3],
                                                          mode="bicubic", align_corners=False).permute(0, 2, 3, 1).clamp(0, 1)
                flat[i] = img
            pick = f"candidate {best - k * n + 1}/{n}" + (f" ({scores[best]}/10)" if scores and best < len(scores) else "")
            notes.append(f"Shot {shot}: {pick}")
        return (flat, ("Re-rendered: " if route == "reroll" else "Consistency edits: ") + "; ".join(notes))


class ShotConsistencyPrep:
    """For every shot whose character or product doesn't match the rest, pairs the shot with the shot
    (or reference image) that shows the correct look, plus the critic's edit instruction.
    Feed the three lists to an image-edit model (e.g. Qwen Image Edit: Picture 1 = shot, Picture 2 = reference),
    then put the results back with Shot Reroll Merge (route = consistency)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The shots the critic reviewed."}),
                "verdicts_json": ("STRING", {"forceInput": True,
                                             "tooltip": "Shot Critic verdicts_json (consistency route and reference)."}),
            },
            "optional": {
                "references": ("IMAGE", {"tooltip": "The same reference images given to the critic, if any."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("shots", "references", "instructions", "targets", "summary")
    OUTPUT_IS_LIST = (True, True, True, True, False)
    OUTPUT_TOOLTIPS = (
        "The shots that need a consistency fix.",
        "For each of them, the shot or reference image that shows the correct look.",
        "The critic's edit instruction for each (Picture 1 = subject in the shot, Picture 2 = subject in the reference).",
        "What to find in both images, e.g. 'woman' (use as SAM 3 prompt to crop the subject for a close-up edit).",
        "Which shot is fixed from which reference.",
    )
    FUNCTION = "prep"
    CATEGORY = "Shot Planner"
    DESCRIPTION = ("Pairs every shot whose character or product drifted with the shot or reference that "
                   "shows the correct look, plus the critic's edit instruction, for an image-edit model.")

    def prep(self, images, verdicts_json, references=None):
        flat = [img.unsqueeze(0) for batch in images for img in batch]
        refs = [img.unsqueeze(0) for batch in (references or []) for img in batch]
        v = json.loads(_first(verdicts_json, "{}") or "{}")
        shots, refimgs, instr, targets, notes = [], [], [], [], []
        for s in v.get("shots", []):
            if s.get("route") != "consistency" or not s.get("reference"):
                continue
            kind, n = s["reference"]
            pool = flat if kind == "shot" else refs
            if not (1 <= n <= len(pool)) or not (1 <= s["shot"] <= len(flat)):
                continue
            shots.append(flat[s["shot"] - 1])
            refimgs.append(pool[n - 1])
            instr.append(s.get("edit_instruction") or "Make the subject in Picture 1 match Picture 2.")
            targets.append(s.get("consistency_target") or "person")
            notes.append(f"Shot {s['shot']} <- {kind} {n} ({targets[-1]})")
        if not shots:  # placeholder so the graph stays valid; the merge won't request it
            return ([flat[0]], [flat[0]], [""], ["person"], "No consistency fixes needed.")
        return (shots, refimgs, instr, targets, "; ".join(notes))


SEVERITY = {"pass": 0, "detail": 1, "consistency": 2, "reroll": 3}


class ShotKeepBetter:
    """'Never make it worse': after a fix, keep each shot's fixed version only if the critic scores it at
    least as well as before. Connect the shots and verdicts from before the fix, and the fixed shots
    plus a Shot Critic review of them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "before": ("IMAGE", {"tooltip": "The shots before the fix (the list the first Shot Critic reviewed)."}),
            "before_verdicts_json": ("STRING", {"forceInput": True,
                                                "tooltip": "verdicts_json of the Shot Critic that reviewed 'before'."}),
            "after": ("IMAGE", {"tooltip": "The same shots after the fix (e.g. Shot Reroll Merge output)."}),
            "after_verdicts_json": ("STRING", {"forceInput": True,
                                               "tooltip": "verdicts_json of a second Shot Critic run on 'after'."}),
        }}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "verdicts_json", "summary")
    OUTPUT_IS_LIST = (True, False, False)
    OUTPUT_TOOLTIPS = ("Per shot, the fixed version if it scored at least as well, else the original.",
                       "Verdicts matching the images that were kept.",
                       "Which fixes were kept or rejected, with the scores.")
    FUNCTION = "choose"
    CATEGORY = "Shot Planner"
    DESCRIPTION = ("Never make it worse: keeps each fixed shot only if the critic scores it at least as well "
                   "as the original (and no worse route).")

    def choose(self, before, before_verdicts_json, after, after_verdicts_json):
        b = [img.unsqueeze(0) for batch in before for img in batch]
        a = [img.unsqueeze(0) for batch in after for img in batch]
        vb = json.loads(_first(before_verdicts_json, "{}") or "{}")
        va = json.loads(_first(after_verdicts_json, "{}") or "{}")
        sb, sa = vb.get("shots", []), va.get("shots", [])
        out, shots, notes = [], [], []
        for i in range(len(b)):
            x, y = (sb[i] if i < len(sb) else {}), (sa[i] if i < len(sa) else {})
            same = i >= len(a) or (a[i].shape == b[i].shape and torch.equal(a[i], b[i]))
            better = (not same and y and y.get("score", 0) >= x.get("score", 0)
                      and SEVERITY.get(y.get("route"), 3) <= SEVERITY.get(x.get("route"), 3))
            if better:
                out.append(a[i])
                shots.append(y)
                notes.append(f"Shot {i + 1}: fix kept ({x.get('score', '?')} -> {y.get('score', '?')})")
            else:
                out.append(b[i])
                shots.append(x or y)
                if not same:
                    notes.append(f"Shot {i + 1}: fix rejected ({x.get('score', '?')} -> {y.get('score', '?')}), original kept")
        merged = dict(vb, shots=shots)
        return (out, json.dumps(merged, indent=2), "; ".join(notes) or "Nothing was changed.")


class ShotContinuityPrompts:
    """Consistency first: shot 1 is generated from text, every later shot is generated from shot 1
    with an image-edit model (e.g. Qwen Image Edit), so characters, wardrobe, product and location stay the same."""

    DEFAULT_NOTE = ("Keep the same people (faces, hair, clothing), the same product and the same location and "
                    "photographic style as Picture 1; this is the next shot of the same film.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shot_list_json": ("STRING", {"forceInput": True, "tooltip": "Shot Planner shot_list_json."}),
            "continuity_note": ("STRING", {"multiline": True, "default": cls.DEFAULT_NOTE,
                                           "tooltip": "Added to every later shot's prompt. Name the fixed details, e.g. "
                                                      "'the barista always wears a black top and black apron'."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("first_prompt", "next_prompts", "next_count")
    OUTPUT_IS_LIST = (False, True, False)
    OUTPUT_TOOLTIPS = ("Image prompt of shot 1, for the text-to-image model.",
                       "Prompts of shots 2..N with the continuity note added, for the image-edit model "
                       "(Picture 1 = shot 1).",
                       "Number of later shots.")
    FUNCTION = "split"
    CATEGORY = "Shot Planner"
    DESCRIPTION = ("Splits a shot list for consistency-first generation: shot 1 from text, every later shot "
                   "edited from shot 1 so people, product and place stay the same.")

    def split(self, shot_list_json, continuity_note):
        shots = json.loads(shot_list_json or "{}").get("shots", [])
        if not shots:
            raise RuntimeError("Shot Continuity Prompts: the shot list is empty.")
        first = shots[0].get("image_prompt", "")
        rest = [f"{s.get('image_prompt', '')} {continuity_note}".strip() for s in shots[1:]]
        return (first, rest or [first], len(rest))


class ShotJoin:
    """Shot 1 plus the list of later shots -> one list of shots (for Shot Critic, Animatic...)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "first": ("IMAGE", {"tooltip": "Shot 1 (only the first image is used)."}),
            "rest": ("IMAGE", {"tooltip": "Shots 2..N, as a list or one batch. Resized to shot 1's size if needed."}),
        }}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("All shots as a list, one image per shot.",)
    FUNCTION = "join"
    CATEGORY = "Shot Planner"
    DESCRIPTION = "Joins shot 1 and the later shots into one list of shots, all at shot 1's size."

    def join(self, first, rest):
        f = [img.unsqueeze(0) for batch in first for img in batch][:1]
        r = [img.unsqueeze(0) for batch in rest for img in batch]
        out = f + r
        h, w = out[0].shape[1:3]
        fixed = []
        for img in out:
            if img.shape[1:3] != (h, w):
                img = torch.nn.functional.interpolate(img.permute(0, 3, 1, 2), size=(h, w), mode="bicubic",
                                                      align_corners=False).permute(0, 2, 3, 1).clamp(0, 1)
            fixed.append(img)
        return (fixed,)


class ShotSelect:
    """The shots the critic sent to one route (e.g. detail), so a fix only touches those shots.
    Put them back with Shot Reroll Merge set to the same route."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The shots (or clips) that were reviewed."}),
            "verdicts_json": ("STRING", {"forceInput": True,
                                         "tooltip": "Shot Critic or Shot Video QA verdicts_json."}),
            "route": (["detail", "consistency", "reroll"], {"default": "detail",
                      "tooltip": "Which shots to pick: those the reviewer sent to this route."}),
        }}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("shots", "targets", "summary")
    OUTPUT_IS_LIST = (True, True, False)
    OUTPUT_TOOLTIPS = ("The picked shots, in shot order. When none match, shot 1 is passed as a placeholder "
                       "(Shot Reroll Merge then skips the branch, so it never runs).",
                       "Per picked shot, what the detail fix should look for, e.g. 'logo, text' (SAM 3 prompt).",
                       "Which shots were picked.")
    FUNCTION = "select"
    CATEGORY = "Shot Planner"
    DESCRIPTION = ("Picks the shots a reviewer sent to one route, so a fix only touches those; put them back "
                   "with Shot Reroll Merge set to the same route.")

    def select(self, images, verdicts_json, route):
        flat = [img.unsqueeze(0) for batch in images for img in batch]
        route = _first(route, "detail")
        v = json.loads(_first(verdicts_json, "{}") or "{}")
        picked = [s for s in v.get("shots", []) if s.get("route") == route and 1 <= s["shot"] <= len(flat)]
        if not picked:
            return ([flat[0]], ["logo"], f"No shots need a {route} fix.")
        return ([flat[s["shot"] - 1] for s in picked],
                [", ".join(s.get("detail_targets") or []) or "logo" for s in picked],
                "Shots " + ", ".join(str(s["shot"]) for s in picked))


CRITIC_NODE_CLASS_MAPPINGS = {"ShotCritic": ShotCritic, "ShotRerollMerge": ShotRerollMerge,
                              "ShotConsistencyPrep": ShotConsistencyPrep, "ShotKeepBetter": ShotKeepBetter,
                              "ShotContinuityPrompts": ShotContinuityPrompts, "ShotJoin": ShotJoin,
                              "ShotSelect": ShotSelect}
CRITIC_NODE_DISPLAY_NAME_MAPPINGS = {"ShotCritic": "Shot Critic", "ShotRerollMerge": "Shot Reroll Merge",
                                     "ShotConsistencyPrep": "Shot Consistency Prep",
                                     "ShotKeepBetter": "Shot Keep Better (never make it worse)",
                                     "ShotContinuityPrompts": "Shot Continuity Prompts", "ShotJoin": "Shot Join",
                                     "ShotSelect": "Shot Select (by route)"}
