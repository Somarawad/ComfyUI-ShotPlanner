"""ComfyUI nodes for video QA, reroll merging and clip utilities (Shot Planner/Video).

Logic lives in video_qa.py; these classes convert torch IMAGE tensors to numpy and back.
"""

import json

import torch

from . import stage_utils as su
from . import video_qa as vq

CATEGORY = "Shot Planner/Video"


def _one(v, default=None):
    if isinstance(v, list):
        return v[0] if v else default
    return v


def _np(t):
    return t.detach().float().cpu().numpy()


def _t(a):
    return torch.from_numpy(a).float()


class ShotVideoQA:
    """Checks every generated clip (offline metrics + optional Gemini review) and routes
    each shot to pass or reroll."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clips": ("IMAGE", {"tooltip": "One clip (batch of frames) per shot, in shot "
                                               "order: the list output of your image-to-video node."}),
                "keyframes": ("IMAGE", {"tooltip": "The approved stills, one per shot (list, or one "
                                                   "batch with one image per shot). Used for drift."}),
                "checks": ("STRING", {"multiline": True,
                    "default": "Product shape and label stay identical.\nNo extra fingers.\nFaces keep their identity.",
                    "tooltip": "Extra things the LLM reviewer must check, one per line. Ignored "
                               "when use_llm is off."}),
                "pass_score": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Shots scoring below this (the lower of the offline and LLM "
                               "scores) are routed to reroll. Frozen clips always reroll."}),
                "use_llm": ("BOOLEAN", {"default": False,
                    "tooltip": "Also send sampled frames to Gemini for a visual review (needs "
                               "GEMINI_API_KEY or GOOGLE_API_KEY). Off = offline metrics only."}),
                "provider": (["gemini"], {"default": "gemini",
                    "tooltip": "LLM provider for the visual review."}),
                "model": ("STRING", {"default": "",
                    "tooltip": "Empty = gemini-flash-lite-latest, falling back to "
                               "gemini-flash-latest when it is busy or replies badly."}),
                "frames_per_clip": ("INT", {"default": 6, "min": 2, "max": 16,
                    "tooltip": "Frames sampled evenly from each clip for the LLM and the "
                               "contact sheet."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Base seed: the LLM call uses it, and every failed shot gets a "
                               "new reroll seed derived from it and the shot number."}),
            },
            "optional": {
                "brand_rules": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Brand rules for the reviewer, e.g. \"logo never distorted; "
                               "bottle always upright\"."}),
                "motion_prompts": ("STRING", {"forceInput": True,
                    "tooltip": "Shot Motion Prompts output. Given to the reviewer and used as "
                               "the reroll prompt when it suggests no fix."}),
                "api_key": ("STRING", {"default": "",
                    "tooltip": "Optional Gemini key. Prefer the environment variable: a key "
                               "typed here is saved inside the workflow."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("report", "contact_sheet", "verdicts_json", "reroll_shots",
                    "reroll_prompts", "reroll_seeds")
    OUTPUT_IS_LIST = (False, False, False, False, True, True)
    OUTPUT_TOOLTIPS = (
        "Readable report: score, metrics and reasons per shot.",
        "One row per shot: sampled frames, a melting-region heat overlay and a PASS/REROLL label.",
        "Per-shot route (pass / reroll), score, reasons, metrics and reroll seed. Feed Clip "
        "Reroll Merge.",
        "Failed shot numbers, e.g. \"2, 5\" (empty when all passed).",
        "One motion prompt per failed shot (list): the reviewer's fix, or the original.",
        "One new seed per failed shot (list).",
    )
    FUNCTION = "check"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Detects frozen clips, flicker, jumps/morphs, drift from the keyframe and "
                   "melting regions, optionally asks Gemini, and routes shots to pass or reroll.")

    def check(self, clips, keyframes, checks, pass_score, use_llm, provider, model,
              frames_per_clip, seed, brand_rules=None, motion_prompts=None, api_key=None):
        try:
            r = vq.run_video_qa(
                [_np(c) for c in clips], [_np(k) for k in keyframes], _one(checks, ""),
                float(_one(pass_score, 7.0)), bool(_one(use_llm, False)), _one(model, ""),
                int(_one(frames_per_clip, 6)), int(_one(seed, 0)), _one(brand_rules, "") or "",
                _one(api_key, "") or "", list(motion_prompts or []))
        except su.StageError as e:
            raise RuntimeError(f"Shot Video QA: {e}") from e
        return (r["report"], _t(r["contact_sheet"]), json.dumps(r["verdicts"], indent=2),
                r["reroll_shots"], r["reroll_prompts"], r["reroll_seeds"])


class ClipRerollMerge:
    """Puts rerolled clips back in place of the failed ones."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clips": ("IMAGE", {"tooltip": "The original clips, one per shot (same list "
                                               "that went into Shot Video QA)."}),
                "verdicts_json": ("STRING", {"forceInput": True,
                    "tooltip": "Shot Video QA verdicts: says which shots failed."}),
            },
            "optional": {
                "rerolled": ("IMAGE", {"lazy": True,
                    "tooltip": "Re-generated clips for the failed shots only, in shot order "
                               "(drive your video node with reroll_prompts / reroll_seeds). "
                               "Lazy: not computed at all when every shot passed."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("clips", "report")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_TOOLTIPS = ("All clips in shot order, failed ones replaced. Feed Clip Resample FPS "
                       "or Ad Timeline.", "Which shots were replaced.")
    FUNCTION = "merge"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Replaces the clips Shot Video QA marked for reroll and passes the rest "
                   "through; the reroll branch only runs when something failed.")

    def check_lazy_status(self, clips=None, verdicts_json=None, rerolled=None, **_):
        try:
            if vq.needs_rerolled(_one(verdicts_json), rerolled):
                return ["rerolled"]
        except su.StageError as e:
            raise RuntimeError(f"Clip Reroll Merge: {e}") from e
        return []

    def merge(self, clips, verdicts_json, rerolled=None):
        try:
            out, report = vq.merge_clips(list(clips), _one(verdicts_json), rerolled)
        except su.StageError as e:
            raise RuntimeError(f"Clip Reroll Merge: {e}") from e
        return (out, report)


class ClipsFromBatch:
    """Splits one long batch of frames into per-shot clips."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "One batch holding all shots back to back (if "
                                                "several batches arrive they are joined first)."}),
                "frame_counts_text": ("STRING", {"default": "",
                    "tooltip": "Frames per clip, e.g. \"81, 49, 81\". Must add up to the batch "
                               "length. Empty = use frame_counts, else split equally."}),
                "num_clips": ("INT", {"default": 1, "min": 1, "max": 256,
                    "tooltip": "Used only when no frame counts are given: split into this many "
                               "equal clips."}),
            },
            "optional": {
                "frame_counts": ("INT", {"forceInput": True,
                    "tooltip": "Shot Motion Prompts frame_counts (list)."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("clips",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("One clip per shot (list).",)
    FUNCTION = "split"
    CATEGORY = CATEGORY
    DESCRIPTION = "Splits one batch into N clips by frame counts."

    def split(self, frames, frame_counts_text, num_clips, frame_counts=None):
        batch = torch.cat([f for f in frames], dim=0) if len(frames) > 1 else frames[0]
        n = batch.shape[0]
        try:
            text = _one(frame_counts_text, "") or ""
            if text.strip():
                counts = su.parse_int_list(text, "frame_counts_text")
            elif frame_counts:
                counts = [int(c) for c in frame_counts]
            else:
                counts = vq.equal_counts(n, int(_one(num_clips, 1)))
            ranges = vq.split_batch(n, counts)
        except su.StageError as e:
            raise RuntimeError(f"Clips From Batch: {e}") from e
        return ([batch[a:b] for a, b in ranges],)


class ClipLastFrame:
    """Last frame of each clip (to start the next shot from where this one ended)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clips": ("IMAGE", {"tooltip": "Clips, one per shot (list)."})}}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("last_frames",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("The last frame of every clip (list of single images), e.g. as the "
                       "start image of the next shot.",)
    FUNCTION = "last"
    CATEGORY = CATEGORY
    DESCRIPTION = "Outputs the last frame of each clip, for chaining shots."

    def last(self, clips):
        return ([c[-1:].clone() for c in clips],)


class ClipResampleFPS:
    """Changes clip frame rate, keeping duration (frame blending)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clips": ("IMAGE", {"tooltip": "Clips, one per shot (list)."}),
                "source_fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 240.0,
                    "tooltip": "Frame rate the clips were generated at (Wan = 16)."}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0,
                    "tooltip": "Frame rate of the timeline (Ad Timeline fps)."}),
                "method": (["blend", "nearest"], {"default": "blend",
                    "tooltip": "blend = mix neighbouring frames (smooth); nearest = repeat/drop "
                               "frames (sharp, slightly steppy)."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("clips", "fps")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_TOOLTIPS = ("Resampled clips (list), round(frames x target / source) frames each.",
                       "The target fps.")
    FUNCTION = "resample"
    CATEGORY = CATEGORY
    DESCRIPTION = "Converts clips between frame rates (e.g. 16 fps Wan clips for a 24 fps timeline)."

    def resample(self, clips, source_fps, target_fps, method):
        src, dst, how = float(_one(source_fps)), float(_one(target_fps)), _one(method, "blend")
        out = []
        try:
            for c in clips:
                plan = vq.resample_plan(c.shape[0], src, dst)
                a = torch.tensor([p[0] for p in plan], dtype=torch.long)
                b = torch.tensor([p[1] for p in plan], dtype=torch.long)
                w = torch.tensor([p[2] for p in plan], dtype=c.dtype, device=c.device).view(-1, 1, 1, 1)
                if how == "nearest":
                    out.append(torch.where(w >= 0.5, c[b], c[a]))
                else:
                    out.append(c[a] * (1 - w) + c[b] * w)
        except su.StageError as e:
            raise RuntimeError(f"Clip Resample FPS: {e}") from e
        return (out, dst)


VIDEO_QA_NODE_CLASS_MAPPINGS = {
    "ShotVideoQA": ShotVideoQA,
    "ClipRerollMerge": ClipRerollMerge,
    "ClipsFromBatch": ClipsFromBatch,
    "ClipLastFrame": ClipLastFrame,
    "ClipResampleFPS": ClipResampleFPS,
}
VIDEO_QA_NODE_DISPLAY_NAME_MAPPINGS = {
    "ShotVideoQA": "Shot Video QA",
    "ClipRerollMerge": "Clip Reroll Merge",
    "ClipsFromBatch": "Clips From Batch",
    "ClipLastFrame": "Clip Last Frame",
    "ClipResampleFPS": "Clip Resample FPS",
}
