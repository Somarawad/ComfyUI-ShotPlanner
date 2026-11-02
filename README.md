# ComfyUI Shot Planner

Turn a brief or a script into a **shot list** inside ComfyUI, with a ready-to-use **image prompt** and **video prompt** for every shot, then generate the whole sequence in one run.

Built from the way I make AI films and ads (*To Nora*, *Bloom*): plan the shots first, make one keyframe per shot, then animate each keyframe with an image-to-video model (Seedance, Kling, MiniMax H3, Wan...).

![Three keyframes generated from one brief](docs/demo_keyframes.jpg)
*One brief, one queue: shots 1-3 rendered with JuggernautXL using `02_shot_planner_image_batch.json`.*

```
Brief  ─►  Shot Planner  ─►  image_prompts (list) ─► your image model ─► keyframes
                         ├─►  video_prompts (list) ─► your video model ─► clips
                         ├─►  storyboard (text)
                         └─►  shot_list_json ─► Shot Selector ─► prompts for one shot
```

## Nodes

| Node | What it does |
|---|---|
| **Shot Planner (Brief to Shots)** | Brief in, shot list out. Choose the number of shots, style, aspect ratio and seconds per shot. Uses Claude, GPT or Gemini to write the shots, or `offline` mode (no key, template-based). |
| **Shot Selector** | Picks one shot by number and outputs its image prompt, video prompt, duration, camera and description. |
| **Shot List Editor** | Paste or hand-edit a shot list JSON (for example after tweaking the planner's output) and feed it to Shot Selector. |

`image_prompts` and `video_prompts` are **list outputs**: any node connected to them runs once per shot, so a single queue renders every keyframe.

## Install

**ComfyUI Manager:** *Install via Git URL* and paste this repo's URL.

**Manual:**
```
cd ComfyUI/custom_nodes
git clone https://github.com/Somarawad/ComfyUI-ShotPlanner
```
Restart ComfyUI. No extra Python packages needed (standard library only).

## API keys

`offline` works with no key. For LLM-written shots, set an environment variable before starting ComfyUI:

| Provider | Variable | Default model |
|---|---|---|
| anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-5 |
| openai | `OPENAI_API_KEY` | gpt-4o-mini |
| gemini | `GEMINI_API_KEY` | gemini-flash-lite-latest (free tier at aistudio.google.com) |

Windows portable: add `set ANTHROPIC_API_KEY=sk-...` to the top of `run_nvidia_gpu.bat`.
You can also paste a key into the node's `api_key` field, **but it is then saved inside the workflow file**, so don't share that workflow.
Any model name the provider supports can be typed into the `model` field. Busy-server errors (429/503) are retried automatically.

## Example workflows

Drag a file from `examples/` onto the ComfyUI canvas:

- `01_shot_planner_storyboard.json`: brief → storyboard + one selected shot's prompts. Runs with no models and no key.
- `02_shot_planner_image_batch.json`: brief → one SDXL keyframe per shot. Pick your own checkpoint; swap the sampler for Flux, Krea, Qwen or an API image node.

To animate: connect `video_prompts` (or Shot Selector's `video_prompt`) and each keyframe to your image-to-video node.

## Tips

- Describe characters' look and wardrobe in the brief; the planner repeats those details in every shot for consistency.
- Use `extra_instructions` for things like "no dialogue", "end on the product", "every shot handheld".
- Change the brief or the seed of your sampler to get new variations; the plan itself is cached until an input changes.

## Tests

```
python tests/test_shot_planner.py
```
Covers offline planning, JSON extraction from chatty replies, and the request format for all three providers (mocked).

## Ad Assembler

Nodes under **Shot Planner/Ad** that turn finished shots (keyframes or short clips) into a finished 15–30 s commercial: timing, transitions, supers, brand end card and a ducked voiceover/music mix. The core `CreateVideo` and `SaveVideo` nodes write the MP4.

```
Shot Planner ─► keyframes ─► (image-to-video per shot) ─► Ad Timeline ─► Ad Supers ─► Ad End Card ─► CreateVideo ─► SaveVideo
     │                                                                                                  ▲ audio
     └─► Ad Script From Shots ─► voiceover_text ─► your TTS node ─► Ad Audio Mix (+ music, sfx) ──────────┘
                             └─► supers_text ─► Ad Supers
```

| Node | What it does | Outputs |
|---|---|---|
| **Ad Timeline** | Takes a list of shots (one still or one clip per shot). Durations come from `shot_list_json` (or a `durations` string like `3, 2.5, 4`; default 3 s). Stills get a gentle Ken Burns push-in; clips are trimmed, or held on their last frame, to fit. Everything is center-cropped to 16:9, 9:16, 1:1 or 4:5. Transitions: cut, crossfade, dip-to-black, whip-pan blur. Transitions are centred on the cut, so the total length is always the sum of the durations. | `frames`, `frame_count`, `fps`, `timing_json`, `total_seconds` |
| **Ad Supers** | Text overlays, one per line: `2 \| Text` (during shot 2) or `1.5-4 \| Text` (seconds). Lower third / center / top; white with soft shadow, black box or outline; fade in/out; optional brand font. | `frames` |
| **Ad End Card** | Logo (with optional mask for transparency), tagline and CTA on a solid colour or the blurred last frame, cross-faded in and appended for N seconds. | `frames`, `frame_count`, `total_seconds` |
| **Ad Audio Mix** | Voiceover, music and SFX, resampled to stereo at 48 kHz, padded/trimmed to `total_seconds`. The music sits at `music_volume_db` (-14) and ducks a further `duck_db` (-10) while the voice speaks (RMS detection, ~150 ms attack/release with look-ahead). Fade out at the end, peaks soft-limited to -1 dBFS. | `audio` |
| **Ad Script From Shots** | Offline template: one voiceover line per shot that fits its duration (~2.5 words/s), brand + tagline on the last shot, and a supers list in `shot \| text` format. | `voiceover_text`, `supers_text` |

Tips:
- Connect **Ad End Card → total_seconds** to **Ad Audio Mix → total_seconds**, and **Ad Timeline → fps** to the end card and to `CreateVideo`, so picture and sound are exactly the same length.
- With `INPUT_IS_LIST`, Ad Timeline runs once for all shots. If you have one batch of keyframes instead of a list, turn on `split_single_batch`.
- `width` is the output width: use 1920 for 16:9 and 1080 for 9:16 / 1:1 / 4:5.
- A full 1080p spot is held in memory as float frames (~25 MB per frame, ~18 GB for 30 s at 24 fps). Draft at a lower width (e.g. 960) if RAM is tight.

Tests (numpy, torch and Pillow only; no GPU, no ComfyUI):
```
cd tests
python -m unittest
```

## License

MIT
