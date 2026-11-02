"""ComfyUI nodes for social versions (Shot Planner/Ad). Logic lives in reframe.py."""

import json

import torch
import torch.nn.functional as F

from . import reframe as rf
from . import stage_utils as su

CATEGORY = "Shot Planner/Ad"


class AdReframe:
    """Subject-following crop to 9:16, 1:1 or 4:5."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "The finished spot (or any clip)."}),
                "aspect_ratio": (rf.REFRAME_RATIOS, {"default": "9:16",
                    "tooltip": "Target aspect ratio. The crop is the largest window of this "
                               "ratio that fits in the frame."}),
                "output_width": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 2,
                    "tooltip": "Output width in pixels (1080 -> 1080x1920 for 9:16)."}),
                "smoothing": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How calmly the crop follows the subject: 0 = follows every "
                               "frame, 1 = one fixed crop at the average position."}),
            },
            "optional": {
                "subject_mask": ("MASK", {"tooltip": "Subject mask per frame (e.g. SAM video "
                    "tracking). The crop centres on its centre of mass. Without it, a "
                    "centre-weighted edge-energy saliency guess is used."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    OUTPUT_TOOLTIPS = ("Reframed frames.",)
    FUNCTION = "reframe"
    CATEGORY = CATEGORY
    DESCRIPTION = "Crops a spot to a social aspect ratio, following the subject smoothly."

    def reframe(self, frames, aspect_ratio, output_width, smoothing, subject_mask=None):
        n, h, w, _ = frames.shape
        try:
            boxes = rf.plan_reframe(n, w, h, aspect_ratio, subject_mask, smoothing,
                                    lambda i: frames[i].float().cpu().numpy())
            ow, oh = rf.output_size(output_width, aspect_ratio)
        except su.StageError as e:
            raise RuntimeError(f"Ad Reframe: {e}") from e
        out = torch.empty((n, oh, ow, 3), dtype=torch.float32)
        for i, (x0, y0, cw, ch) in enumerate(boxes):
            crop = frames[i:i + 1, y0:y0 + ch, x0:x0 + cw, :3].float().permute(0, 3, 1, 2)
            if (cw, ch) != (ow, oh):
                crop = F.interpolate(crop, size=(oh, ow), mode="bicubic", align_corners=False,
                                     antialias=cw > ow)
            out[i] = crop[0].permute(1, 2, 0).clamp(0, 1).cpu()
        return (out,)


class AdCutdown:
    """Shorter version (e.g. 6 s or 15 s) from the full spot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "The full spot including the end card (Ad End "
                                                "Card or Ad Supers output)."}),
                "timing_json": ("STRING", {"forceInput": True,
                    "tooltip": "Ad Timeline timing of the full spot. Frames after its last shot "
                               "are the end card and are always kept."}),
                "target_seconds": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Length of the cutdown including the end card (6 = bumper, 15)."}),
                "keep": ("STRING", {"default": "auto",
                    "tooltip": "Shot numbers to keep, e.g. \"1, 4, 6\", or auto = first shot, "
                               "best-scored middle shots, last shot."}),
            },
            "optional": {
                "verdicts_json": ("STRING", {"forceInput": True,
                    "tooltip": "Shot Video QA verdicts: their scores rank the middle shots in "
                               "auto mode."}),
                "min_shot_s": ("FLOAT", {"default": 1.5, "min": 0.2, "max": 10.0, "step": 0.1,
                    "tooltip": "auto keeps only as many shots as fit at this length each."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "timing_json", "frame_count", "total_seconds", "report")
    OUTPUT_TOOLTIPS = (
        "Cutdown frames (hard cuts), end card included.",
        "Timing of the kept shots (for Ad Supers / Ad Voiceover on the cutdown).",
        "Number of frames.",
        "Length in seconds including the end card: connect to Ad Audio Mix total_seconds.",
        "Which shots were kept and how long it runs.",
    )
    FUNCTION = "cut"
    CATEGORY = CATEGORY
    DESCRIPTION = "Makes a 6 s / 15 s cutdown: picks and trims shots and keeps the end card."

    def cut(self, frames, timing_json, target_seconds, keep, verdicts_json=None, min_shot_s=1.5):
        try:
            p = rf.plan_cutdown(frames.shape[0], timing_json, target_seconds, keep,
                                verdicts_json, min_shot_s)
        except su.StageError as e:
            raise RuntimeError(f"Ad Cutdown: {e}") from e
        out = torch.cat([frames[a:b] for a, b in p["ranges"]], dim=0)
        fps = float(p["timing"]["fps"])
        return (out, json.dumps(p["timing"], indent=2), out.shape[0],
                round(out.shape[0] / fps, 4), p["report"])


REFRAME_NODE_CLASS_MAPPINGS = {"AdReframe": AdReframe, "AdCutdown": AdCutdown}
REFRAME_NODE_DISPLAY_NAME_MAPPINGS = {"AdReframe": "Ad Reframe", "AdCutdown": "Ad Cutdown"}
