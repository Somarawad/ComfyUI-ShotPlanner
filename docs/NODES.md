# Node reference: video stage, voiceover, beat sync, social cuts

Generated from the node definitions (`INPUT_TYPES`, `RETURN_NAMES`, `OUTPUT_TOOLTIPS`), so the tooltip text below is exactly what ComfyUI shows. For the Shot Planner and Ad Assembler nodes see the README.

## Video stage

### Shot Motion Prompts

`ShotMotionPrompts` · category **Shot Planner/Video**

Writes per-shot image-to-video prompts (subject action + camera vocabulary) and frame counts that suit the chosen engine.

**Inputs**

| Input | Type | Default | Tooltip |
|---|---|---|---|
| `shot_list_json` | STRING |  | Shot Planner plan (or Shot List Editor). One motion prompt is written per shot from its description and camera field. |
| `engine` | choice: `wan2.1` / `wan2.2` / `minimax-h3` / `ltx` / `kling` / `seedance` / `hailuo` | `wan2.1` | Image-to-video engine. Sets the fps and the frame grid: Wan 4k+1 at 16 fps, MiniMax H3 17k+5 at 24 fps, LTX 8k+1 at 24 fps. API engines get the nearest allowed duration: Kling 5/10 s, Seedance 4-12 s, Hailuo 6/10 s. |
| `motion_strength` | choice: `subtle` / `natural` / `dynamic` | `natural` | subtle = slow, mostly still (product shots, faces); natural = realistic pace; dynamic = energetic action and faster camera. |
| `fps_override` | INT | `0` | 0 = the engine's native fps. Set it if your workflow renders at another rate; frame counts are snapped to the grid at this fps. |
| `style_suffix` *(optional)* | STRING | `""` | Appended to every prompt, e.g. "cinematic, 35mm film grain, soft natural light". |
| `durations` *(optional)* | STRING | `""` | Optional seconds per shot, e.g. "3, 2.5, 4" (or Ad Beat Sync's durations). Overrides the shot list for the shots it covers. |

**Outputs**

| Output | Type | Tooltip |
|---|---|---|
| `motion_prompts` | STRING (list) | One motion prompt per shot (list): subject action, pacing and camera move. |
| `frame_counts` | INT (list) | Frames per shot (list), snapped to the engine's grid; connect to the video node's length / num_frames. |
| `seconds` | FLOAT (list) | Seconds per shot (list) after snapping; for API engines this is the duration to request. |
| `negative_prompt` | STRING | Negative prompt for the engine (connect once; it is the same for every shot). |
| `fps` | INT | Frame rate of the engine (or fps_override). |

### Shot Video QA

`ShotVideoQA` · category **Shot Planner/Video** · `INPUT_IS_LIST` (runs once for all shots)

Detects frozen clips, flicker, jumps/morphs, drift from the keyframe and melting regions, optionally asks Gemini, and routes shots to pass or reroll.

**Inputs**

| Input | Type | Default | Tooltip |
|---|---|---|---|
| `clips` | IMAGE |  | One clip (batch of frames) per shot, in shot order: the list output of your image-to-video node. |
| `keyframes` | IMAGE |  | The approved stills, one per shot (list, or one batch with one image per shot). Used for drift. |
| `checks` | STRING | `Product shape and label stay identical.\nNo extra fingers.\nFaces keep their identity.` | Extra things the LLM reviewer must check, one per line. Ignored when use_llm is off. |
| `pass_score` | FLOAT | `7.0` | Shots scoring below this (the lower of the offline and LLM scores) are routed to reroll. Frozen clips always reroll. |
| `use_llm` | BOOLEAN | `False` | Also send sampled frames to Gemini for a visual review (needs GEMINI_API_KEY or GOOGLE_API_KEY). Off = offline metrics only. |
| `provider` | choice: `gemini` | `gemini` | LLM provider for the visual review. |
| `model` | STRING | `""` | Empty = gemini-flash-lite-latest, falling back to gemini-flash-latest when it is busy or replies badly. |
| `frames_per_clip` | INT | `6` | Frames sampled evenly from each clip for the LLM and the contact sheet. |
| `seed` | INT | `0` | Base seed: the LLM call uses it, and every failed shot gets a new reroll seed derived from it and the shot number. |
| `brand_rules` *(optional)* | STRING | `""` | Brand rules for the reviewer, e.g. "logo never distorted; bottle always upright". |
| `motion_prompts` *(optional)* | STRING |  | Shot Motion Prompts output. Given to the reviewer and used as the reroll prompt when it suggests no fix. |
| `api_key` *(optional)* | STRING | `""` | Optional Gemini key. Prefer the environment variable: a key typed here is saved inside the workflow. |

**Outputs**

| Output | Type | Tooltip |
|---|---|---|
| `report` | STRING | Readable report: score, metrics and reasons per shot. |
| `contact_sheet` | IMAGE | One row per shot: sampled frames, a melting-region heat overlay and a PASS/REROLL label. |
| `verdicts_json` | STRING | Per-shot route (pass / reroll), score, reasons, metrics and reroll seed. Feed Clip Reroll Merge. |
| `reroll_shots` | STRING | Failed shot numbers, e.g. "2, 5" (empty when all passed). |
| `reroll_prompts` | STRING (list) | One motion prompt per failed shot (list): the reviewer's fix, or the original. |
| `reroll_seeds` | INT (list) | One new seed per failed shot (list). |

### Clip Reroll Merge

`ClipRerollMerge` · category **Shot Planner/Video** · `INPUT_IS_LIST` (runs once for all shots)

Replaces the clips Shot Video QA marked for reroll and passes the rest through; the reroll branch only runs when something failed.

**Inputs**

| Input | Type | Default | Tooltip |
|---|---|---|---|
| `clips` | IMAGE |  | The original clips, one per shot (same list that went into Shot Video QA). |
| `verdicts_json` | STRING |  | Shot Video QA verdicts: says which shots failed. |
| `rerolled` *(optional)* *(lazy)* | IMAGE |  | Re-generated clips for the failed shots only, in shot order (drive your video node with reroll_prompts / reroll_seeds). Lazy: not computed at all when every shot passed. |

**Outputs**

| Output | Type | Tooltip |
|---|---|---|
| `clips` | IMAGE (list) | All clips in shot order, failed ones replaced. Feed Clip Resample FPS or Ad Timeline. |
| `report` | STRING | Which shots were replaced. |

### Clips From Batch

`ClipsFromBatch` · category **Shot Planner/Video** · `INPUT_IS_LIST` (runs once for all shots)

Splits one batch into N clips by frame counts.

**Inputs**

| Input | Type | Default | Tooltip |
|---|---|---|---|
| `frames` | IMAGE |  | One batch holding all shots back to back (if several batches arrive they are joined first). |
| `frame_counts_text` | STRING | `""` | Frames per clip, e.g. "81, 49, 81". Must add up to the batch length. Empty = use frame_counts, else split equally. |
| `num_clips` | INT | `1` | Used only when no frame counts are given: split into this many equal clips. |
| `frame_counts` *(optional)* | INT |  | Shot Motion Prompts frame_counts (list). |

**Outputs**

| Output | Type | Tooltip |
|---|---|---|
| `clips` | IMAGE (list) | One clip per shot (list). |

### Clip Last Frame

`ClipLastFrame` · category **Shot Planner/Video** · `INPUT_IS_LIST` (runs once for all shots)

Outputs the last frame of each clip, for chaining shots.

**Inputs**

| Input | Type | Default | Tooltip |
|---|---|---|---|
| `clips` | IMAGE |  | Clips, one per shot (list). |

**Outputs**

| Output | Type | Tooltip |
|---|---|---|
| `last_frames` | IMAGE (list) | The last frame of every clip (list of single images), e.g. as the start image of the next shot. |

### Clip Resample FPS

`ClipResampleFPS` · category **Shot Planner/Video** · `INPUT_IS_LIST` (runs once for all shots)

Converts clips between frame rates (e.g. 16 fps Wan clips for a 24 fps timeline).

**Inputs**

| Input | Type | Default | Tooltip |
|---|---|---|---|
| `clips` | IMAGE |  | Clips, one per shot (list). |
| `source_fps` | FLOAT | `16.0` | Frame rate the clips were generated at (Wan = 16). |
| `target_fps` | FLOAT | `24.0` | Frame rate of the timeline (Ad Timeline fps). |
| `method` | choice: `blend` / `nearest` | `blend` | blend = mix neighbouring frames (smooth); nearest = repeat/drop frames (sharp, slightly steppy). |

**Outputs**

| Output | Type | Tooltip |
|---|---|---|
| `clips` | IMAGE (list) | Resampled clips (list), round(frames x target / source) frames each. |
| `fps` | FLOAT | The target fps. |

## Voiceover

### Ad Voiceover (TTS)

`AdVoiceover` · category **Shot Planner/Audio**

Text-to-speech voiceover (Gemini, OpenAI or ElevenLabs), placed per shot.

**Inputs**

| Input | Type | Default | Tooltip |
|---|---|---|---|
| `text` | STRING | `1 \| Every morning starts somewhere.\n3 \| Novara. Made for the way you move.` | Voiceover script. Either "shot_number \| line" per line, one line per shot (Ad Script From Shots' voiceover_text), or plain text read in one go. Lines starting with # are ignored. |
| `provider` | choice: `gemini` / `openai` / `elevenlabs` | `gemini` | gemini (GEMINI_API_KEY / GOOGLE_API_KEY), openai (OPENAI_API_KEY) or elevenlabs (ELEVENLABS_API_KEY). |
| `voice` | STRING | `""` | Empty = Kore (gemini), coral (openai), Rachel 21m00Tcm4TlvDq8ikWAM (elevenlabs; use a voice ID). |
| `model` | STRING | `""` | Empty = gemini-2.5-flash-preview-tts, gpt-4o-mini-tts or eleven_multilingual_v2. |
| `style_prompt` | STRING | `warm, unhurried, confident` | How to read it, e.g. "warm, unhurried Italian-accented English". Sent as the reading instruction to Gemini and OpenAI (gpt-4o-mini-tts); ElevenLabs ignores it (pick the voice instead). |
| `speed` | FLOAT | `1.0` | Speaking rate. Native for OpenAI (0.25-4) and ElevenLabs (0.7-1.2); Gemini gets it as a pacing instruction. |
| `timing_json` *(optional)* | STRING |  | Ad Timeline timing. With per-shot lines, each line is synthesised separately and starts at its shot's start + line_offset_s, so the VO follows the edit. |
| `text_mode` *(optional)* | choice: `auto` / `numbered lines` / `one line per shot` / `plain text` | `auto` | auto: numbered if every line has "N \|", one line per shot if the line count matches timing_json, else plain text. |
| `line_offset_s` *(optional)* | FLOAT | `0.3` | Delay of each line after its shot starts. |
| `trim_silence` *(optional)* | BOOLEAN | `True` | Cut the silence TTS adds before and after each line, so lines start exactly where they are placed. |
| `api_key` *(optional)* | STRING | `""` | Optional key. Prefer the environment variable: a key typed here is saved inside the workflow. |

**Outputs**

| Output | Type | Tooltip |
|---|---|---|
| `voiceover` | AUDIO | 48 kHz stereo voiceover. When placed on the timeline, set Ad Audio Mix vo_start_s to 0. |
| `vo_timing_json` | STRING | Start/end of every line (and its shot), for Ad Voice Fit. |

### Ad Voice Fit

`AdVoiceFit` · category **Shot Planner/Audio**

Time-stretches voiceover lines that run past the end of their shot.

**Inputs**

| Input | Type | Default | Tooltip |
|---|---|---|---|
| `voiceover` | AUDIO |  | Ad Voiceover output. |
| `vo_timing_json` | STRING |  | Ad Voiceover vo_timing_json (where each line sits). |
| `timing_json` | STRING |  | Ad Timeline timing (where each shot ends). |
| `max_stretch` | FLOAT | `0.12` | Most a line may be sped up: 0.12 = up to 12% faster (pitch is kept, WSOLA). Lines that still overrun are reported. |

**Outputs**

| Output | Type | Tooltip |
|---|---|---|
| `voiceover` | AUDIO | Voiceover with overrunning lines tightened. Feed Ad Audio Mix. |
| `report` | STRING | Per line: fits / sped up by N% / OVERRUN by N s. |
| `vo_timing_json` | STRING | Updated line timing. |

## Beat sync

### Ad Beat Sync

`AdBeatSync` · category **Shot Planner/Audio**

Detects tempo and beats and picks shot durations that cut on the beat.

**Inputs**

| Input | Type | Default | Tooltip |
|---|---|---|---|
| `music` | AUDIO |  | The music track (Load Audio). |
| `shot_count` | INT | `6` | Number of shots to cut. |
| `target_total_s` | FLOAT | `15.0` | Wanted length of the shots (without the end card). The result ends on the beat grid closest to this. |
| `min_shot_s` | FLOAT | `1.5` | No shot shorter than this. |
| `beats_per_cut` | choice: `auto` / `1` / `2` / `4` / `8` | `auto` | Cuts fall every N beats from the first downbeat. auto picks the coarsest grid that still lets shot lengths vary. |
| `shot_list_json` *(optional)* | STRING |  | Shot Planner plan: its durations are the preferred lengths (scaled to target_total_s). Without it all shots prefer equal length. |
| `fps` *(optional)* | FLOAT | `24.0` | Timeline fps: durations are snapped to whole frames so cuts stay on the beat after Ad Timeline rounds them. |

**Outputs**

| Output | Type | Tooltip |
|---|---|---|
| `durations` | STRING | Seconds per shot, e.g. "3.25, 2.5, ...". Connect to Ad Timeline durations (and Shot Motion Prompts durations). |
| `bpm` | FLOAT | Detected tempo. |
| `beat_times_json` | STRING | Beat times (original and aligned to the first downbeat), cut times and settings. |
| `music_offset_s` | FLOAT | Seconds to trim from the start of the music so the first downbeat lands on frame 0. |
| `music_aligned` | AUDIO | The music already trimmed by music_offset_s: feed Ad Audio Mix music. |
| `total_seconds` | FLOAT | Sum of the durations. |

## Social versions

### Ad Reframe

`AdReframe` · category **Shot Planner/Ad**

Crops a spot to a social aspect ratio, following the subject smoothly.

**Inputs**

| Input | Type | Default | Tooltip |
|---|---|---|---|
| `frames` | IMAGE |  | The finished spot (or any clip). |
| `aspect_ratio` | choice: `9:16` / `1:1` / `4:5` / `16:9` | `9:16` | Target aspect ratio. The crop is the largest window of this ratio that fits in the frame. |
| `output_width` | INT | `1080` | Output width in pixels (1080 -> 1080x1920 for 9:16). |
| `smoothing` | FLOAT | `0.6` | How calmly the crop follows the subject: 0 = follows every frame, 1 = one fixed crop at the average position. |
| `subject_mask` *(optional)* | MASK |  | Subject mask per frame (e.g. SAM video tracking). The crop centres on its centre of mass. Without it, a centre-weighted edge-energy saliency guess is used. |

**Outputs**

| Output | Type | Tooltip |
|---|---|---|
| `frames` | IMAGE | Reframed frames. |

### Ad Cutdown

`AdCutdown` · category **Shot Planner/Ad**

Makes a 6 s / 15 s cutdown: picks and trims shots and keeps the end card.

**Inputs**

| Input | Type | Default | Tooltip |
|---|---|---|---|
| `frames` | IMAGE |  | The full spot including the end card (Ad End Card or Ad Supers output). |
| `timing_json` | STRING |  | Ad Timeline timing of the full spot. Frames after its last shot are the end card and are always kept. |
| `target_seconds` | FLOAT | `6.0` | Length of the cutdown including the end card (6 = bumper, 15). |
| `keep` | STRING | `auto` | Shot numbers to keep, e.g. "1, 4, 6", or auto = first shot, best-scored middle shots, last shot. |
| `verdicts_json` *(optional)* | STRING |  | Shot Video QA verdicts: their scores rank the middle shots in auto mode. |
| `min_shot_s` *(optional)* | FLOAT | `1.5` | auto keeps only as many shots as fit at this length each. |

**Outputs**

| Output | Type | Tooltip |
|---|---|---|
| `frames` | IMAGE | Cutdown frames (hard cuts), end card included. |
| `timing_json` | STRING | Timing of the kept shots (for Ad Supers / Ad Voiceover on the cutdown). |
| `frame_count` | INT | Number of frames. |
| `total_seconds` | FLOAT | Length in seconds including the end card: connect to Ad Audio Mix total_seconds. |
| `report` | STRING | Which shots were kept and how long it runs. |
