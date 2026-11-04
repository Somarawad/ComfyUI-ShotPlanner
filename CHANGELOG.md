# Changelog

All notable changes to this project. Versions follow [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-11-03

From shot list to finished spot: 30 new nodes in six stages, example workflows of the NOVARA spec ad,
and the pack is ready for the Comfy Registry.

### Added

- **Animatic** (`Shot Planner`): Shot Animatic (Keyframes to Video), timed per shot, with the camera
  move of each shot as a Ken Burns move; Clip Stitcher.
- **Shot Critic** (`Shot Planner`): a vision LLM (Gemini, Claude or GPT) reviews the shots against the
  shot list, checks and brand rules. Three parallel reviews with a majority vote. Every shot is routed
  to pass, detail, consistency or reroll, with a contact sheet. Gemini Flash Lite falls back to Flash
  when busy.
- **Fix helpers**: Shot Select (by route), Shot Reroll Merge (lazy; keeps the best-scored candidate),
  Shot Consistency Prep, Shot Keep Better (never make it worse), Shot Continuity Prompts, Shot Join.
- **Ad Assembler** (`Shot Planner/Ad`): Ad Timeline (transitions, push-in on stills, any aspect ratio),
  Ad Supers, Ad End Card, Ad Audio Mix (ducked music, soft limiter), Ad Script From Shots.
- **Video stage** (`Shot Planner/Video`): Shot Video Engine (switch between Wan, MiniMax H3 and API
  nodes; only the chosen branch runs), Shot Motion Prompts, Shot Video QA (frozen, flicker, jump,
  drift and melting checks, optional Gemini review), Clip Reroll Merge (lazy), Clips From Batch,
  Clip Last Frame, Clip Resample FPS.
- **Audio** (`Shot Planner/Audio`): Ad Voiceover (Gemini, OpenAI, ElevenLabs TTS placed per shot),
  Ad Voice Fit, Ad Beat Sync.
- **Social versions**: Ad Reframe (subject-following 9:16 / 1:1 / 4:5), Ad Cutdown (6 s / 15 s).
- **Treatment & Pitch Deck** (`Shot Planner/Treatment`): Treatment Builder, Pitch Deck Builder,
  Storyboard Sheet, Palette From Shots. Print-ready PDFs in three templates, with bundled OFL fonts.
- **Example workflows** in `examples/workflows/`, each stage grouped with a Note listing the models:
  brief to animatic, NOVARA keyframes, NOVARA stills spot, NOVARA video stage, treatment & pitch deck.
  The API-format prompts are in `examples/api/`.
- **Docs**: a new README, `docs/GUIDE.md`, `docs/NODES.md` (generated from the code by
  `tools/gen_nodes_md.py`) and `docs/PUBLISHING.md`.
- **Tooling**: `tools/comfy_smoke_test.py` (loads the pack in a real ComfyUI and validates the
  example prompts), `tools/api_to_workflow.py` and `tools/check_workflows.js` (build the UI workflows,
  check them in the frontend, take screenshots).
- **CI**: unit tests on Python 3.10, 3.11 and 3.12 with CPU torch. A manual-only publish workflow for
  the Comfy Registry.
- Every input of every node has a tooltip. Every node has output tooltips and a description.

### Fixed

- Shot Video QA lost its `brand_rules` and `api_key` values every time a saved workflow was reopened
  (ComfyUI frontend 1.57).
- Clip Stitcher cut a clip to 2 frames when its shot had no duration; it now keeps the whole clip.
- Shot Planner and Shot Critic ignored `GOOGLE_API_KEY`, which the other Gemini nodes accept.

### Changed

- `pyproject.toml` is complete for the Comfy Registry (publisher `somarawad`, icon, URLs).
- Treatment Builder / Pitch Deck Builder: `director` now defaults to empty.

## [1.0.0] - 2026-10-27

First release.

- Shot Planner (Brief to Shots): a brief or script becomes a shot list with one image prompt and one
  video prompt per shot (Claude, GPT, Gemini or offline). List outputs, so one queue renders every
  shot.
- Shot Selector and Shot List Editor.
- Example workflows: storyboard (no models) and an SDXL image batch.

[1.1.0]: https://github.com/Somarawad/ComfyUI-ShotPlanner/compare/516c4e6...release-1.1
[1.0.0]: https://github.com/Somarawad/ComfyUI-ShotPlanner/tree/516c4e6
