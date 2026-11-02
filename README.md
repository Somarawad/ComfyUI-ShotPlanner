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

## Video stage

Nodes under **Shot Planner/Video** for the step between approved keyframes and the edit: write motion prompts and frame counts that suit your image-to-video engine, check the clips it produces, re-roll only the bad ones and conform frame rates. The image-to-video node itself is your choice (local Wan, local MiniMax H3, LTX, or an API node such as Kling, Seedance or Hailuo).

```
keyframes ─► Shot Motion Prompts ─► motion_prompts / frame_counts ─► (image-to-video engine) ─► clips
                                                                                                │
             ┌─────────────────────── reroll_prompts / reroll_seeds ◄── Shot Video QA ◄──────────┤
             ▼                                                              │ verdicts_json     │
   (same engine, failed shots only) ─► rerolled ─► Clip Reroll Merge ◄──────┴───────────────────┘
                                                        │
                                                        ▼
                                   Clip Resample FPS (16 → 24) ─► Ad Timeline ─► ...
```

| Node | What it does | Outputs |
|---|---|---|
| **Shot Motion Prompts** | One image-to-video prompt per shot: the shot's action, a pacing line for `motion_strength` (subtle / natural / dynamic) and the `camera` field mapped to a camera vocabulary ("push-in" → "slow dolly push-in", "pan left", "orbit", "crane", "handheld"...). Frame counts snap to the engine's grid: Wan 2.1/2.2 4k+1 at 16 fps, MiniMax H3 17k+5 at 24 fps, LTX 8k+1 at 24 fps. API engines get the nearest allowed duration: Kling 5/10 s, Seedance 4–12 s, Hailuo 6/10 s. | `motion_prompts` (list), `frame_counts` (list), `seconds` (list), `negative_prompt`, `fps` |
| **Shot Video QA** | Offline metrics per clip (numpy): frozen clip, flicker (luma pumping), sudden jumps / morphs (frame-difference spikes against the clip's median), drift of the last frame from the approved keyframe, and a melting-region heat map (per-cell temporal instability). Optionally sends `frames_per_clip` sampled frames to Gemini (`gemini-flash-lite-latest`, falling back to `gemini-flash-latest`) for a JSON review with issues and a fixed motion prompt. Score = the lower of offline and LLM; below `pass_score` (or frozen) → reroll with a new seed. | `report`, `contact_sheet`, `verdicts_json`, `reroll_shots`, `reroll_prompts` (list), `reroll_seeds` (list) |
| **Clip Reroll Merge** | Swaps the rerolled clips into the failed shots' places. Its `rerolled` input is lazy: when every shot passed, the reroll branch is never executed. | `clips` (list), `report` |
| **Clips From Batch** | Splits one long batch into per-shot clips by frame counts (text, the `frame_counts` list, or N equal parts). | `clips` (list) |
| **Clip Last Frame** | Last frame of every clip, to start the next shot where the previous one ended. | `last_frames` (list) |
| **Clip Resample FPS** | Changes frame rate and keeps the duration (frame blending or nearest), e.g. 16 fps Wan clips into a 24 fps timeline: 81 frames → 122. | `clips` (list), `fps` |

Tips:
- To reroll, drive a second copy of your video node with `reroll_prompts` and `reroll_seeds` (and the matching keyframes) and connect its output to Clip Reroll Merge `rerolled`; the lists contain only the failed shots, in order.
- The offline checks are deliberately simple; use the `contact_sheet` (red heat = unstable regions) to judge borderline shots, and turn on `use_llm` for faces, hands and logos. The LLM review needs `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).

## Voiceover, beat sync and social cuts

| Node | Category | What it does | Outputs |
|---|---|---|---|
| **Ad Voiceover (TTS)** | Audio | Text-to-speech with Gemini (`gemini-2.5-flash-preview-tts`, voice Kore), OpenAI (`gpt-4o-mini-tts`, voice coral) or ElevenLabs (`eleven_multilingual_v2`). `style_prompt` steers the read ("warm, unhurried Italian-accented English"). Text can be `shot_number \| line` lines, one line per shot (Ad Script From Shots) or plain text. With `timing_json`, each line is synthesised separately and placed at its shot's start + 0.3 s. Decoded with the standard library (no ffmpeg); output 48 kHz stereo. | `voiceover`, `vo_timing_json` |
| **Ad Voice Fit** | Audio | Lines that run past the end of their shot are sped up (WSOLA, pitch kept) by at most `max_stretch` (12%); anything still too long is reported. | `voiceover`, `report`, `vo_timing_json` |
| **Ad Beat Sync** | Audio | Tempo and beats from the music (onset envelope + autocorrelation, numpy only), then shot durations whose cuts land on the beat, as close as possible to the shot list's durations and never shorter than `min_shot_s`. `music_offset_s` / `music_aligned` put the first downbeat on frame 0. | `durations`, `bpm`, `beat_times_json`, `music_offset_s`, `music_aligned`, `total_seconds` |
| **Ad Reframe** | Ad | 9:16, 1:1 or 4:5 versions: the crop follows the subject's centre of mass from a `subject_mask` (e.g. SAM video tracking), or a centre-weighted edge-energy guess without one, smoothed over time and never outside the frame. | `frames` |
| **Ad Cutdown** | Ad | 6 s / 15 s cutdowns from the full spot: `keep` = shot numbers or `auto` (first shot, best-scored middle shots from `verdicts_json`, last shot), trimmed to fit. The end card is always kept. | `frames`, `timing_json`, `frame_count`, `total_seconds`, `report` |

The full chain:

```
keyframes ─► Shot Motion Prompts ─► (image-to-video engine) ─► Shot Video QA ─► Clip Reroll Merge ─► Clip Resample FPS
   ─► Ad Timeline ─► Ad Supers ─► Ad End Card ─► CreateVideo ─► SaveVideo
                                                    ▲ audio
   Ad Voiceover ─► Ad Voice Fit ─► Ad Audio Mix ────┘
                                       ▲ music_aligned
   music ─► Ad Beat Sync ── durations ─► Ad Timeline (and Shot Motion Prompts)

   Ad End Card frames ─► Ad Reframe (9:16) / Ad Cutdown (6 s) ─► CreateVideo ─► SaveVideo
```

Tips:
- Run Ad Beat Sync first and connect its `durations` to both Shot Motion Prompts and Ad Timeline, so clips are generated at the right length and every cut lands on a beat. Feed `music_aligned` (not the original music) to Ad Audio Mix.
- When Ad Voiceover places lines on the timeline, set Ad Audio Mix `vo_start_s` to 0.
- For a 9:16 cutdown, run Ad Cutdown first, then Ad Reframe, and connect Ad Cutdown `total_seconds` to Ad Audio Mix.
- Keys: `GEMINI_API_KEY` / `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`. A key typed into an `api_key` field is saved inside the workflow file.

Every input, output and tooltip of these nodes is listed in [docs/NODES.md](docs/NODES.md).

Tests (numpy, torch and Pillow only; all HTTP is mocked, no GPU, no network):
```
cd tests
python -m unittest
```

## License

MIT
