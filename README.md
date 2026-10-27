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
git clone https://github.com/<your-username>/ComfyUI-ShotPlanner
```
Restart ComfyUI. No extra Python packages needed (standard library only).

## API keys

`offline` works with no key. For LLM-written shots, set an environment variable before starting ComfyUI:

| Provider | Variable | Default model |
|---|---|---|
| anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-5 |
| openai | `OPENAI_API_KEY` | gpt-4o-mini |
| gemini | `GEMINI_API_KEY` | gemini-2.5-flash |

Windows portable: add `set ANTHROPIC_API_KEY=sk-...` to the top of `run_nvidia_gpu.bat`.
You can also paste a key into the node's `api_key` field, **but it is then saved inside the workflow file**, so don't share that workflow.
Any model name the provider supports can be typed into the `model` field.

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

## License

MIT
