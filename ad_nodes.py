"""ComfyUI nodes for the Ad Assembler stage (category: Shot Planner/Ad).

All the logic lives in ad_assembly.py; these classes only unwrap ComfyUI inputs and wrap
outputs. Frames from Ad Timeline / Ad End Card plus the Ad Audio Mix output go into the core
CreateVideo node, then SaveVideo writes the MP4.
"""

import json

from . import ad_assembly as aa

CATEGORY = "Shot Planner/Ad"


def _one(v, default=None):
    """Unwrap a scalar that arrives as a list on INPUT_IS_LIST nodes."""
    if isinstance(v, list):
        return v[0] if v else default
    return v


def _err(node, e):
    return RuntimeError(f"{node}: {e}")


class AdTimeline:
    """Shots (stills or clips) -> one timed sequence with transitions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shots": ("IMAGE", {"tooltip":
                    "One image or one clip (a batch of frames) per shot, in order. Connect the "
                    "list output of your image or image-to-video node. A single image becomes a "
                    "still with a gentle push-in."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120,
                    "tooltip": "Frame rate of the finished spot."}),
                "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 2,
                    "tooltip": "Output width in pixels. Height follows the aspect ratio "
                               "(use 1080 for a 1080x1920 9:16 spot)."}),
                "aspect_ratio": (aa.ASPECT_RATIOS, {"default": "16:9",
                    "tooltip": "Every shot is center-cropped to this aspect ratio."}),
                "transition": (aa.TRANSITIONS, {"default": "cut",
                    "tooltip": "How shots join. crossfade and whip-pan blur overlap the shots "
                               "around each cut; dip-to-black fades out and back in. The total "
                               "length stays the sum of the shot durations."}),
                "transition_frames": ("INT", {"default": 6, "min": 0, "max": 120,
                    "tooltip": "Length of each transition in frames (ignored for cut). "
                               "Clamped to the shortest shot."}),
                "ken_burns": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Push-in strength for stills: 0.06 = zoom from 100% to 106% over "
                               "the shot. 0 = static still."}),
            },
            "optional": {
                "shot_list_json": ("STRING", {"forceInput": True,
                    "tooltip": "Shot Planner plan. Each shot's duration (seconds) is used. "
                               "Shots without a duration get 3 s."}),
                "durations": ("STRING", {"default": "",
                    "tooltip": "Optional seconds per shot, e.g. \"3, 2.5, 4\". Overrides the "
                               "shot list for the shots it covers."}),
                "split_single_batch": ("BOOLEAN", {"default": False,
                    "tooltip": "If only one IMAGE batch is connected, treat each image in it "
                               "as its own still shot instead of as one clip."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING", "FLOAT")
    RETURN_NAMES = ("frames", "frame_count", "fps", "timing_json", "total_seconds")
    OUTPUT_TOOLTIPS = (
        "All frames of the spot as one batch. Feed Ad Supers, Ad End Card or CreateVideo.",
        "Number of frames.",
        "Frame rate (for CreateVideo).",
        "Start/end of every shot in frames and seconds, for Ad Supers and the audio.",
        "Length in seconds (connect to Ad Audio Mix total_seconds if there is no end card).",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Times each shot, conforms it to the aspect ratio (stills get a push-in, "
                   "clips are trimmed or held) and joins them with transitions.")

    def build(self, shots, fps, width, aspect_ratio, transition, transition_frames, ken_burns,
              shot_list_json=None, durations=None, split_single_batch=None):
        fps, width = _one(fps), _one(width)
        aspect_ratio, transition = _one(aspect_ratio), _one(transition)
        transition_frames, ken_burns = _one(transition_frames), _one(ken_burns)
        shot_list_json, durations = _one(shot_list_json, ""), _one(durations, "")
        split = bool(_one(split_single_batch, False))
        shots = list(shots)
        if split and len(shots) == 1 and shots[0].shape[0] > 1:
            shots = [shots[0][i:i + 1] for i in range(shots[0].shape[0])]
        try:
            w, h = aa.aspect_size(width, aspect_ratio)
            secs = aa.resolve_durations(len(shots), shot_list_json, durations)
            frames, timing = aa.assemble_timeline([s.float().cpu() for s in shots], secs, fps,
                                                  w, h, transition, transition_frames, ken_burns)
        except aa.AdAssemblyError as e:
            raise _err("Ad Timeline", e) from e
        return (frames, frames.shape[0], float(fps), json.dumps(timing, indent=2),
                float(timing["total_seconds"]))


class AdSupers:
    """Text overlays (supers) on top of the frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Frames from Ad Timeline."}),
                "supers": ("STRING", {"multiline": True, "default": "1 | Meet the new Novara\n4 | Available now",
                    "tooltip": "One super per line: \"shot_number | text\" (needs timing_json) or "
                               "\"start_s-end_s | text\" (e.g. \"2.5-5 | Text\"). Use \\n for a "
                               "line break; lines starting with # are ignored."}),
                "font_size": ("INT", {"default": 56, "min": 8, "max": 400,
                    "tooltip": "Text height in pixels. Long lines wrap to 84% of the width."}),
                "position": (aa.SUPER_POSITIONS, {"default": "lower third",
                    "tooltip": "lower third = above the bottom title-safe margin."}),
                "style": (aa.SUPER_STYLES, {"default": "white with soft shadow",
                    "tooltip": "White text with a soft drop shadow, white on a translucent "
                               "black box, or white with a black outline."}),
                "fade_in_frames": ("INT", {"default": 6, "min": 0, "max": 120,
                    "tooltip": "Frames to fade each super in."}),
                "fade_out_frames": ("INT", {"default": 6, "min": 0, "max": 120,
                    "tooltip": "Frames to fade each super out."}),
            },
            "optional": {
                "timing_json": ("STRING", {"forceInput": True,
                    "tooltip": "From Ad Timeline. Needed for shot numbers; also supplies fps."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0,
                    "tooltip": "Only used when timing_json is not connected."}),
                "text_color": ("STRING", {"default": "#FFFFFF",
                    "tooltip": "Text colour as hex, e.g. #FFFFFF."}),
                "font_path": ("STRING", {"default": "",
                    "tooltip": "Optional .ttf/.otf path for your brand font. Empty = DejaVu Sans "
                               "/ Arial / built-in fallback."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    OUTPUT_TOOLTIPS = ("Frames with the supers drawn on.",)
    FUNCTION = "draw"
    CATEGORY = CATEGORY
    DESCRIPTION = "Draws timed text supers (titles, claims, legal lines) with fades."

    def draw(self, frames, supers, font_size, position, style, fade_in_frames, fade_out_frames,
             timing_json=None, fps=24.0, text_color="#FFFFFF", font_path=""):
        try:
            timing = aa.load_timing(timing_json)
            rate = float(timing["fps"]) if timing else float(fps)
            parsed = aa.parse_supers(supers, timing, rate)
            out = aa.apply_supers(frames.float().cpu(), parsed, rate, font_size, position, style,
                                  fade_in_frames, fade_out_frames, font_path,
                                  aa.parse_hex_color(text_color))
        except aa.AdAssemblyError as e:
            raise _err("Ad Supers", e) from e
        return (out,)


class AdEndCard:
    """Brand end card (logo, tagline, call-to-action) appended to the frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "The spot so far (Ad Timeline or Ad Supers). "
                                                "The card uses the same size."}),
                "tagline": ("STRING", {"default": "Made for the way you move.",
                    "tooltip": "Main line under the logo. Empty = none."}),
                "cta": ("STRING", {"default": "novara.it",
                    "tooltip": "Call to action / URL, drawn smaller under the tagline."}),
                "background": (["blurred last frame", "solid color"], {"default": "blurred last frame",
                    "tooltip": "Blurred and darkened last frame of the spot, or a solid colour."}),
                "bg_color": ("STRING", {"default": "#111111",
                    "tooltip": "Background colour (hex) for solid color mode."}),
                "duration_s": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 30.0, "step": 0.1,
                    "tooltip": "How long the end card stays on screen, in seconds."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0,
                    "tooltip": "Frame rate of the spot (connect Ad Timeline fps)."}),
                "logo_scale": ("FLOAT", {"default": 0.3, "min": 0.02, "max": 0.9, "step": 0.01,
                    "tooltip": "Logo height as a fraction of the frame height."}),
                "fade_in_frames": ("INT", {"default": 12, "min": 0, "max": 240,
                    "tooltip": "Frames to cross-fade from the last shot into the card "
                               "(part of the card's duration)."}),
            },
            "optional": {
                "logo": ("IMAGE", {"tooltip": "Brand logo (first image of the batch is used)."}),
                "logo_mask": ("MASK", {"tooltip": "Transparency for the logo, e.g. the MASK from "
                                                  "Load Image (1 = transparent)."}),
                "invert_mask": ("BOOLEAN", {"default": False,
                    "tooltip": "Turn on if your mask is 1 where the logo is visible."}),
                "text_color": ("STRING", {"default": "#FFFFFF",
                    "tooltip": "Tagline / CTA colour as hex."}),
                "blur_radius": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 200.0,
                    "tooltip": "Blur radius in pixels for the blurred last frame background."}),
                "font_path": ("STRING", {"default": "",
                    "tooltip": "Optional .ttf/.otf path for your brand font."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "frame_count", "total_seconds")
    OUTPUT_TOOLTIPS = (
        "Spot plus end card. Feed CreateVideo.",
        "Total number of frames.",
        "Total length in seconds: connect to Ad Audio Mix total_seconds.",
    )
    FUNCTION = "append"
    CATEGORY = CATEGORY
    DESCRIPTION = "Renders a brand end card (logo, tagline, CTA) and appends it to the spot."

    def append(self, frames, tagline, cta, background, bg_color, duration_s, fps, logo_scale,
               fade_in_frames, logo=None, logo_mask=None, invert_mask=False,
               text_color="#FFFFFF", blur_radius=24.0, font_path=""):
        frames = frames.float().cpu()
        _, h, w, _ = frames.shape
        try:
            bg = aa.blur_frame(frames[-1], blur_radius) if background == "blurred last frame" else None
            logo_img = None
            if logo is not None:
                m = logo_mask.cpu() if logo_mask is not None else None
                logo_img = aa.logo_rgba(logo[0].float().cpu(), m, invert_mask)
            card = aa.render_end_card(w, h, bg, aa.parse_hex_color(bg_color, (17, 17, 17)),
                                      logo_img, tagline, cta, logo_scale,
                                      aa.parse_hex_color(text_color), font_path)
            out = aa.append_end_card(frames, card, duration_s, fps, fade_in_frames)
        except aa.AdAssemblyError as e:
            raise _err("Ad End Card", e) from e
        return (out, out.shape[0], round(out.shape[0] / float(fps), 4))


class AdAudioMix:
    """Voiceover + ducked music bed + SFX, conformed to the spot length."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_seconds": ("FLOAT", {"default": 15.0, "min": 0.1, "max": 600.0, "step": 0.01,
                    "tooltip": "Length of the mix. Connect total_seconds from Ad End Card (or "
                               "Ad Timeline) so audio and picture match exactly."}),
                "vo_start_s": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 600.0, "step": 0.05,
                    "tooltip": "When the voiceover starts, in seconds."}),
                "music_volume_db": ("FLOAT", {"default": -14.0, "min": -60.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Music bed level in dB (before ducking)."}),
                "duck_db": ("FLOAT", {"default": -10.0, "min": -40.0, "max": 0.0, "step": 0.5,
                    "tooltip": "Extra music reduction while the voiceover is speaking "
                               "(detected from its RMS level). 0 = no ducking."}),
                "fade_out_s": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Fade the whole mix out over the last N seconds."}),
                "sample_rate": ("INT", {"default": 48000, "min": 8000, "max": 192000,
                    "tooltip": "Output sample rate. Inputs are resampled; output is stereo."}),
            },
            "optional": {
                "voiceover": ("AUDIO", {"tooltip": "Voiceover (e.g. from a TTS node or Load Audio)."}),
                "music": ("AUDIO", {"tooltip": "Music bed. Starts at 0 s; padded with silence "
                                               "or trimmed to the spot length."}),
                "sfx": ("AUDIO", {"tooltip": "Sound effects / ambience track."}),
                "vo_volume_db": ("FLOAT", {"default": 0.0, "min": -40.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Voiceover gain in dB."}),
                "sfx_volume_db": ("FLOAT", {"default": 0.0, "min": -40.0, "max": 12.0, "step": 0.5,
                    "tooltip": "SFX gain in dB."}),
                "sfx_start_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.05,
                    "tooltip": "When the SFX track starts, in seconds."}),
                "attack_ms": ("FLOAT", {"default": 150.0, "min": 5.0, "max": 2000.0,
                    "tooltip": "How fast the music dips when speech starts (it also starts this "
                               "early, so the dip is in place for the first word)."}),
                "release_ms": ("FLOAT", {"default": 150.0, "min": 5.0, "max": 5000.0,
                    "tooltip": "How fast the music comes back after speech."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_TOOLTIPS = ("Stereo mix, exactly total_seconds long, peaks soft-limited to -1 dBFS. "
                       "Connect to CreateVideo audio.",)
    FUNCTION = "mix"
    CATEGORY = CATEGORY
    DESCRIPTION = "Mixes voiceover, music (ducked under the voice) and SFX to the spot length."

    def mix(self, total_seconds, vo_start_s, music_volume_db, duck_db, fade_out_s, sample_rate,
            voiceover=None, music=None, sfx=None, vo_volume_db=0.0, sfx_volume_db=0.0,
            sfx_start_s=0.0, attack_ms=150.0, release_ms=150.0):
        wave = aa.mix_ad_audio(voiceover, music, sfx, total_seconds, vo_start_s,
                               music_volume_db, duck_db, fade_out_s, sample_rate, vo_volume_db,
                               sfx_volume_db, sfx_start_s, attack_ms, release_ms)
        return ({"waveform": wave.unsqueeze(0), "sample_rate": int(sample_rate)},)


class AdScriptFromShots:
    """Offline voiceover script + supers list from a Shot Planner plan."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_list_json": ("STRING", {"forceInput": True,
                    "tooltip": "Shot Planner plan; each shot's duration sets its word budget."}),
                "brand": ("STRING", {"default": "Novara",
                    "tooltip": "Brand name, said on the last shot."}),
                "tagline": ("STRING", {"default": "Made for the way you move.",
                    "tooltip": "Tagline, said on the last shot and used as its super."}),
                "words_per_second": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Speaking rate used to fit each line to its shot (2.5 = relaxed "
                               "commercial read)."}),
                "supers_mode": (["first and last", "every shot"], {"default": "first and last",
                    "tooltip": "Which shots get a super in supers_text."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("voiceover_text", "supers_text")
    OUTPUT_TOOLTIPS = (
        "One VO line per shot, each short enough for its shot. Feed a TTS node.",
        "Supers in \"shot_number | text\" format for Ad Supers.",
    )
    FUNCTION = "write"
    CATEGORY = CATEGORY
    DESCRIPTION = "Template (offline) voiceover and supers that fit the shot timing."

    def write(self, shot_list_json, brand, tagline, words_per_second, supers_mode):
        try:
            return aa.script_from_shots(shot_list_json, brand, tagline, words_per_second, supers_mode)
        except aa.AdAssemblyError as e:
            raise _err("Ad Script From Shots", e) from e


AD_NODE_CLASS_MAPPINGS = {
    "AdTimeline": AdTimeline,
    "AdSupers": AdSupers,
    "AdEndCard": AdEndCard,
    "AdAudioMix": AdAudioMix,
    "AdScriptFromShots": AdScriptFromShots,
}

AD_NODE_DISPLAY_NAME_MAPPINGS = {
    "AdTimeline": "Ad Timeline",
    "AdSupers": "Ad Supers",
    "AdEndCard": "Ad End Card",
    "AdAudioMix": "Ad Audio Mix",
    "AdScriptFromShots": "Ad Script From Shots",
}
