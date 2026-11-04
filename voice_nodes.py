"""ComfyUI nodes for voiceover (Shot Planner/Audio). Logic lives in voice.py."""

import json

import torch

from . import stage_utils as su
from . import voice as vo

CATEGORY = "Shot Planner/Audio"


def _audio(wave, sr):
    return {"waveform": torch.from_numpy(wave).float().unsqueeze(0), "sample_rate": int(sr)}


class AdVoiceover:
    """Text-to-speech voiceover, optionally placed line by line on the edit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True,
                    "default": "1 | Every morning starts somewhere.\n3 | Novara. Made for the way you move.",
                    "tooltip": "Voiceover script. Either \"shot_number | line\" per line, one "
                               "line per shot (Ad Script From Shots' voiceover_text), or plain "
                               "text read in one go. Lines starting with # are ignored."}),
                "provider": (vo.PROVIDERS, {"default": "gemini",
                    "tooltip": "gemini (GEMINI_API_KEY / GOOGLE_API_KEY), openai "
                               "(OPENAI_API_KEY) or elevenlabs (ELEVENLABS_API_KEY)."}),
                "voice": ("STRING", {"default": "",
                    "tooltip": "Empty = Kore (gemini), coral (openai), Rachel "
                               "21m00Tcm4TlvDq8ikWAM (elevenlabs; use a voice ID)."}),
                "model": ("STRING", {"default": "",
                    "tooltip": "Empty = gemini-2.5-flash-preview-tts, gpt-4o-mini-tts or "
                               "eleven_multilingual_v2."}),
                "style_prompt": ("STRING", {"default": "warm, unhurried, confident",
                    "tooltip": "How to read it, e.g. \"warm, unhurried Italian-accented "
                               "English\". Sent as the reading instruction to Gemini and OpenAI "
                               "(gpt-4o-mini-tts); ElevenLabs ignores it (pick the voice instead)."}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Speaking rate. Native for OpenAI (0.25-4) and ElevenLabs "
                               "(0.7-1.2); Gemini gets it as a pacing instruction."}),
            },
            "optional": {
                "timing_json": ("STRING", {"forceInput": True,
                    "tooltip": "Ad Timeline timing. With per-shot lines, each line is "
                               "synthesised separately and starts at its shot's start + "
                               "line_offset_s, so the VO follows the edit."}),
                "text_mode": (vo.TEXT_MODES, {"default": "auto",
                    "tooltip": "auto: numbered if every line has \"N |\", one line per shot "
                               "if the line count matches timing_json, else plain text."}),
                "line_offset_s": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Delay of each line after its shot starts."}),
                "trim_silence": ("BOOLEAN", {"default": True,
                    "tooltip": "Cut the silence TTS adds before and after each line, so lines "
                               "start exactly where they are placed."}),
                "api_key": ("STRING", {"default": "",
                    "tooltip": "Optional key. Prefer the environment variable: a key typed "
                               "here is saved inside the workflow."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("voiceover", "vo_timing_json")
    OUTPUT_TOOLTIPS = (
        "48 kHz stereo voiceover. When placed on the timeline, set Ad Audio Mix vo_start_s to 0.",
        "Start/end of every line (and its shot), for Ad Voice Fit.",
    )
    FUNCTION = "speak"
    CATEGORY = CATEGORY
    DESCRIPTION = "Text-to-speech voiceover (Gemini, OpenAI or ElevenLabs), placed per shot."

    def speak(self, text, provider, voice, model, style_prompt, speed, timing_json=None,
              text_mode="auto", line_offset_s=0.3, trim_silence=True, api_key=""):
        try:
            wave, timing = vo.build_voiceover(text, provider, voice, model, style_prompt, speed,
                                              timing_json, api_key or "", text_mode,
                                              line_offset_s, trim_silence)
        except su.StageError as e:
            raise RuntimeError(f"Ad Voiceover: {e}") from e
        return (_audio(wave, vo.OUTPUT_SR), json.dumps(timing, indent=2))


class AdVoiceFit:
    """Speeds up VO lines that overrun their shot (up to max_stretch)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voiceover": ("AUDIO", {"tooltip": "Ad Voiceover output."}),
                "vo_timing_json": ("STRING", {"forceInput": True,
                    "tooltip": "Ad Voiceover vo_timing_json (where each line sits)."}),
                "timing_json": ("STRING", {"forceInput": True,
                    "tooltip": "Ad Timeline timing (where each shot ends)."}),
                "max_stretch": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Most a line may be sped up: 0.12 = up to 12% faster (pitch is "
                               "kept, WSOLA). Lines that still overrun are reported."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("voiceover", "report", "vo_timing_json")
    OUTPUT_TOOLTIPS = ("Voiceover with overrunning lines tightened. Feed Ad Audio Mix.",
                       "Per line: fits / sped up by N% / OVERRUN by N s.",
                       "Updated line timing.")
    FUNCTION = "fit"
    CATEGORY = CATEGORY
    DESCRIPTION = "Time-stretches voiceover lines that run past the end of their shot."

    def fit(self, voiceover, vo_timing_json, timing_json, max_stretch):
        try:
            wave, sr = su.audio_to_numpy(voiceover)
            out, report, vt = vo.fit_voiceover(wave, sr, vo_timing_json, timing_json, max_stretch)
        except su.StageError as e:
            raise RuntimeError(f"Ad Voice Fit: {e}") from e
        return (_audio(out, sr), report, json.dumps(vt, indent=2))


VOICE_NODE_CLASS_MAPPINGS = {"AdVoiceover": AdVoiceover, "AdVoiceFit": AdVoiceFit}
VOICE_NODE_DISPLAY_NAME_MAPPINGS = {"AdVoiceover": "Ad Voiceover (TTS)", "AdVoiceFit": "Ad Voice Fit"}
