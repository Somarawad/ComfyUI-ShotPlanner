"""Shot Video Engine: one switch in front of the image-to-video stage.

Every engine branch (local Wan, local MiniMax H3, an API node such as Kling, Seedance or
Hailuo) is wired into this node; only the branch picked in ``engine`` is evaluated, the
others are never run (lazy inputs). The output is always a list of IMAGE clips, one per
shot, ready for Shot Video QA or Ad Timeline.
"""

ENGINES = ["local: Wan", "local: MiniMax H3", "api (VIDEO input)"]
_INPUT_FOR = {"local: Wan": "wan_clips", "local: MiniMax H3": "h3_clips", "api (VIDEO input)": "api_videos"}
DEFAULT_FPS = {"local: Wan": 16.0, "local: MiniMax H3": 24.0, "api (VIDEO input)": 24.0}


def _first(v, default=None):
    if isinstance(v, (list, tuple)):
        return v[0] if v else default
    return default if v is None else v


def _missing(x):
    return x is None or (isinstance(x, (list, tuple)) and (len(x) == 0 or all(i is None for i in x)))


def needed_input(engine):
    """Name of the lazy input the chosen engine reads."""
    if engine not in _INPUT_FOR:
        raise ValueError(f"Unknown engine '{engine}'. Choose one of: {', '.join(ENGINES)}")
    return _INPUT_FOR[engine]


def video_to_frames(video):
    """A ComfyUI VIDEO object (API nodes, LoadVideo) -> (IMAGE batch, fps)."""
    comp = video.get_components()
    fps = float(comp.frame_rate) if getattr(comp, "frame_rate", None) else 24.0
    return comp.images, fps


def collect_clips(engine, wan_clips=None, h3_clips=None, api_videos=None):
    """Return (list of IMAGE clips, fps) from the branch of the chosen engine."""
    name = needed_input(engine)
    src = {"wan_clips": wan_clips, "h3_clips": h3_clips, "api_videos": api_videos}[name]
    if _missing(src):
        raise ValueError(f"Engine '{engine}' is selected, but nothing is connected to '{name}'. "
                         f"Wire that engine's output into '{name}', or pick another engine.")
    items = [x for x in (src if isinstance(src, (list, tuple)) else [src]) if x is not None]
    if name == "api_videos":
        clips, rates = [], []
        for v in items:
            frames, fps = video_to_frames(v)
            clips.append(frames)
            rates.append(fps)
        return clips, (rates[0] if rates else DEFAULT_FPS[engine])
    return list(items), DEFAULT_FPS[engine]


class ShotVideoEngine:
    """Pick which image-to-video engine renders the shots. Only the chosen branch runs."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": (ENGINES, {"default": ENGINES[0],
                                     "tooltip": "Which image-to-video branch to run. The other branches are "
                                                "skipped, so they cost nothing."}),
            },
            "optional": {
                "wan_clips": ("IMAGE", {"lazy": True,
                                        "tooltip": "Clips from the local Wan branch (VAE Decode), one per shot."}),
                "h3_clips": ("IMAGE", {"lazy": True,
                                       "tooltip": "Clips from the local MiniMax H3 branch, one per shot."}),
                "api_videos": ("VIDEO", {"lazy": True,
                                         "tooltip": "VIDEO outputs of an API node (Kling, Seedance, Hailuo, Veo...), "
                                                    "one per shot."}),
                "fps_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 1.0,
                                           "tooltip": "0 = the engine's usual frame rate (Wan 16, H3 24, API from "
                                                      "the file). Set it if your branch renders at another rate."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("clips", "source_fps", "summary")
    OUTPUT_IS_LIST = (True, False, False)
    OUTPUT_TOOLTIPS = ("One clip (IMAGE batch) per shot.",
                       "Frame rate the clips were rendered at; use it to convert to the timeline's fps.",
                       "Which engine ran and the clip lengths.")
    FUNCTION = "run"
    CATEGORY = "Shot Planner/Video"

    def check_lazy_status(self, engine, wan_clips=None, h3_clips=None, api_videos=None, fps_override=None):
        name = needed_input(_first(engine, ENGINES[0]))
        val = {"wan_clips": wan_clips, "h3_clips": h3_clips, "api_videos": api_videos}[name]
        return [name] if _missing(val) else []

    def run(self, engine, wan_clips=None, h3_clips=None, api_videos=None, fps_override=None):
        eng = _first(engine, ENGINES[0])
        clips, fps = collect_clips(eng, wan_clips, h3_clips, api_videos)
        over = float(_first(fps_override, 0.0) or 0.0)
        if over > 0:
            fps = over
        lengths = ", ".join(str(int(c.shape[0]) if hasattr(c, "shape") else len(c)) for c in clips)
        summary = f"{eng}: {len(clips)} clips at {fps:g} fps (frames: {lengths})"
        return (clips, fps, summary)


VIDEO_ENGINE_NODE_CLASS_MAPPINGS = {"ShotVideoEngine": ShotVideoEngine}
VIDEO_ENGINE_NODE_DISPLAY_NAME_MAPPINGS = {"ShotVideoEngine": "Shot Video Engine (switch)"}
