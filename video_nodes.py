"""ComfyUI node for the video stage prompts (category: Shot Planner/Video).

Logic lives in motion_prompts.py; this file only wraps inputs and outputs.
"""

from . import motion_prompts as mp

CATEGORY = "Shot Planner/Video"


class ShotMotionPrompts:
    """Shot list -> one image-to-video motion prompt and frame count per shot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_list_json": ("STRING", {"forceInput": True,
                    "tooltip": "Shot Planner plan (or Shot List Editor). One motion prompt is "
                               "written per shot from its description and camera field."}),
                "engine": (mp.ENGINE_NAMES, {"default": "wan2.1",
                    "tooltip": "Image-to-video engine. Sets the fps and the frame grid: Wan "
                               "4k+1 at 16 fps, MiniMax H3 17k+5 at 24 fps, LTX 8k+1 at 24 fps. "
                               "API engines get the nearest allowed duration: Kling 5/10 s, "
                               "Seedance 4-12 s, Hailuo 6/10 s."}),
                "motion_strength": (mp.MOTION_STRENGTHS, {"default": "natural",
                    "tooltip": "subtle = slow, mostly still (product shots, faces); natural = "
                               "realistic pace; dynamic = energetic action and faster camera."}),
                "fps_override": ("INT", {"default": 0, "min": 0, "max": 120,
                    "tooltip": "0 = the engine's native fps. Set it if your workflow renders "
                               "at another rate; frame counts are snapped to the grid at this fps."}),
            },
            "optional": {
                "style_suffix": ("STRING", {"default": "",
                    "tooltip": "Appended to every prompt, e.g. \"cinematic, 35mm film grain, "
                               "soft natural light\"."}),
                "durations": ("STRING", {"default": "",
                    "tooltip": "Optional seconds per shot, e.g. \"3, 2.5, 4\" (or Ad Beat "
                               "Sync's durations). Overrides the shot list for the shots it covers."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING", "INT")
    RETURN_NAMES = ("motion_prompts", "frame_counts", "seconds", "negative_prompt", "fps")
    OUTPUT_IS_LIST = (True, True, True, False, False)
    OUTPUT_TOOLTIPS = (
        "One motion prompt per shot (list): subject action, pacing and camera move.",
        "Frames per shot (list), snapped to the engine's grid; connect to the video node's "
        "length / num_frames.",
        "Seconds per shot (list) after snapping; for API engines this is the duration to request.",
        "Negative prompt for the engine (connect once; it is the same for every shot).",
        "Frame rate of the engine (or fps_override).",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Writes per-shot image-to-video prompts (subject action + camera vocabulary) "
                   "and frame counts that suit the chosen engine.")

    def build(self, shot_list_json, engine, motion_strength, fps_override, style_suffix="",
              durations=""):
        try:
            r = mp.build_motion_prompts(shot_list_json, engine, motion_strength, fps_override,
                                        style_suffix or "", durations or "")
        except mp.StageError as e:
            raise RuntimeError(f"Shot Motion Prompts: {e}") from e
        return (r["prompts"], r["frame_counts"], r["seconds"], r["negative_prompt"], r["fps"])


VIDEO_NODE_CLASS_MAPPINGS = {"ShotMotionPrompts": ShotMotionPrompts}
VIDEO_NODE_DISPLAY_NAME_MAPPINGS = {"ShotMotionPrompts": "Shot Motion Prompts"}
