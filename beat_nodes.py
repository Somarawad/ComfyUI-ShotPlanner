"""ComfyUI node for beat-synced cutting (Shot Planner/Audio). Logic lives in beat_sync.py."""

import json

from . import beat_sync as bs
from . import stage_utils as su

CATEGORY = "Shot Planner/Audio"


class AdBeatSync:
    """Music -> tempo, beats and shot durations whose cuts land on the beat."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "music": ("AUDIO", {"tooltip": "The music track (Load Audio)."}),
                "shot_count": ("INT", {"default": 6, "min": 1, "max": 100,
                    "tooltip": "Number of shots to cut."}),
                "target_total_s": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 600.0, "step": 0.5,
                    "tooltip": "Wanted length of the shots (without the end card). The result "
                               "ends on the beat grid closest to this."}),
                "min_shot_s": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 30.0, "step": 0.1,
                    "tooltip": "No shot shorter than this."}),
                "beats_per_cut": (bs.BEATS_PER_CUT, {"default": "auto",
                    "tooltip": "Cuts fall every N beats from the first downbeat. auto picks the "
                               "coarsest grid that still lets shot lengths vary."}),
            },
            "optional": {
                "shot_list_json": ("STRING", {"forceInput": True,
                    "tooltip": "Shot Planner plan: its durations are the preferred lengths "
                               "(scaled to target_total_s). Without it all shots prefer equal length."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0,
                    "tooltip": "Timeline fps: durations are snapped to whole frames so cuts stay "
                               "on the beat after Ad Timeline rounds them."}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT", "AUDIO", "FLOAT")
    RETURN_NAMES = ("durations", "bpm", "beat_times_json", "music_offset_s", "music_aligned",
                    "total_seconds")
    OUTPUT_TOOLTIPS = (
        "Seconds per shot, e.g. \"3.25, 2.5, ...\". Connect to Ad Timeline durations (and Shot "
        "Motion Prompts durations).",
        "Detected tempo.",
        "Beat times (original and aligned to the first downbeat), cut times and settings.",
        "Seconds to trim from the start of the music so the first downbeat lands on frame 0.",
        "The music already trimmed by music_offset_s: feed Ad Audio Mix music.",
        "Sum of the durations.",
    )
    FUNCTION = "sync"
    CATEGORY = CATEGORY
    DESCRIPTION = "Detects tempo and beats and picks shot durations that cut on the beat."

    def sync(self, music, shot_count, target_total_s, min_shot_s, beats_per_cut,
             shot_list_json=None, fps=24.0):
        try:
            wave, sr = su.audio_to_numpy(music)
            r = bs.beat_sync(wave, sr, shot_count, target_total_s, min_shot_s, beats_per_cut,
                             shot_list_json, fps)
        except su.StageError as e:
            raise RuntimeError(f"Ad Beat Sync: {e}") from e
        off = int(round(r["music_offset_s"] * sr))
        w = music["waveform"][..., off:]
        info = {k: r[k] for k in ("bpm", "period_s", "music_offset_s", "beats_per_cut",
                                  "cut_times", "beat_times", "aligned_beat_times")}
        return (r["durations"], float(r["bpm"]), json.dumps(info, indent=2),
                float(r["music_offset_s"]), {"waveform": w.contiguous(), "sample_rate": sr},
                float(r["total_seconds"]))


BEAT_NODE_CLASS_MAPPINGS = {"AdBeatSync": AdBeatSync}
BEAT_NODE_DISPLAY_NAME_MAPPINGS = {"AdBeatSync": "Ad Beat Sync"}
