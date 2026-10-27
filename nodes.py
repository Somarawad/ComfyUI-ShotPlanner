import json

from .shot_planner import ASPECT_RATIOS, PROVIDERS, ShotPlannerError, plan_shots, storyboard_text


class ShotPlanner:
    """Brief or script in, shot list out: one image prompt and one video prompt per shot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "brief": ("STRING", {"multiline": True, "default":
                    "A tired office worker finds a tiny dragon living in the office coffee machine. "
                    "It refuses to make coffee until she brings it a croissant."}),
                "provider": (PROVIDERS, {"default": "offline",
                    "tooltip": "offline needs no key (template-based). Others call the LLM API."}),
                "model": ("STRING", {"default": "", "tooltip": "Leave empty for the provider's default model."}),
                "num_shots": ("INT", {"default": 6, "min": 1, "max": 30}),
                "style": ("STRING", {"default": "cinematic, photoreal, natural light"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "shot_seconds": ("INT", {"default": 5, "min": 1, "max": 20}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "",
                    "tooltip": "Optional. Prefer the environment variable so the key is not saved in the workflow."}),
                "extra_instructions": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("shot_list_json", "shot_count", "image_prompts", "video_prompts", "storyboard")
    OUTPUT_IS_LIST = (False, False, True, True, False)
    OUTPUT_TOOLTIPS = (
        "Full plan as JSON. Connect to Shot Selector to pick one shot.",
        "Number of shots in the plan.",
        "List of image prompts. Downstream nodes run once per shot (batch).",
        "List of video prompts, one per shot.",
        "Readable storyboard text.",
    )
    FUNCTION = "plan"
    CATEGORY = "Shot Planner"
    DESCRIPTION = "Turns a brief or script into a shot list with image and video prompts."

    def plan(self, brief, provider, model, num_shots, style, aspect_ratio, shot_seconds,
             api_key="", extra_instructions=""):
        try:
            plan = plan_shots(brief, provider, model, api_key, num_shots, style,
                              aspect_ratio, shot_seconds, extra_instructions)
        except ShotPlannerError as e:
            raise RuntimeError(f"Shot Planner: {e}") from e
        shots = plan["shots"]
        return (json.dumps(plan, indent=2, ensure_ascii=False), len(shots),
                [s["image_prompt"] for s in shots], [s["video_prompt"] for s in shots],
                storyboard_text(plan))


class ShotSelector:
    """Pick one shot from a Shot Planner list by number."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shot_list_json": ("STRING", {"forceInput": True}),
            "shot_number": ("INT", {"default": 1, "min": 1, "max": 30,
                                    "tooltip": "1 = first shot. Clamped to the last shot."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("image_prompt", "video_prompt", "duration", "camera", "description")
    FUNCTION = "select"
    CATEGORY = "Shot Planner"
    DESCRIPTION = "Outputs the prompts and duration for one shot."

    def select(self, shot_list_json, shot_number):
        try:
            shots = json.loads(shot_list_json)["shots"]
        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError("Shot Selector: input is not a Shot Planner list.") from e
        if not shots:
            raise RuntimeError("Shot Selector: the list has no shots.")
        s = shots[min(max(shot_number, 1), len(shots)) - 1]
        return (s["image_prompt"], s["video_prompt"], int(s["duration"]), s["camera"], s["description"])


class ShotListEditor:
    """Paste or edit a shot list JSON by hand (e.g. after tweaking the planner's output)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"shot_list_json": ("STRING", {"multiline": True, "default":
            '{"title": "My film", "style": "cinematic", "shots": [{"id": 1, "description": "", '
            '"camera": "", "duration": 5, "image_prompt": "", "video_prompt": ""}]}'})}}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("shot_list_json", "shot_count")
    FUNCTION = "edit"
    CATEGORY = "Shot Planner"
    DESCRIPTION = "Validates a hand-edited shot list so it can feed Shot Selector."

    def edit(self, shot_list_json):
        try:
            plan = json.loads(shot_list_json)
            n = len(plan["shots"])
        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"Shot List Editor: invalid JSON ({e}).") from e
        return (json.dumps(plan, indent=2, ensure_ascii=False), n)


NODE_CLASS_MAPPINGS = {
    "ShotPlanner": ShotPlanner,
    "ShotSelector": ShotSelector,
    "ShotListEditor": ShotListEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShotPlanner": "Shot Planner (Brief to Shots)",
    "ShotSelector": "Shot Selector",
    "ShotListEditor": "Shot List Editor",
}
