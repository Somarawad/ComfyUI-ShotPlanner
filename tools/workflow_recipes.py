"""How tools/api_to_workflow.py turns each prompt in examples/api/ into a UI workflow in
examples/workflows/: which nodes form which stage (group), the Note shown on each stage, and
file names to make portable (Somar's Windows sub-folders and output prefixes become plain names).
"""

# --------------------------------------------------------------------------- shared text

KEYS = ("API keys: set them as environment variables before starting ComfyUI, never in the "
        "workflow. GEMINI_API_KEY (or GOOGLE_API_KEY), ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "ELEVENLABS_API_KEY.")

QWEN_MODELS = (
    "Qwen Image Edit 2509 (fits a 10 GB GPU):\n"
    "- models/unet/Qwen-Image-Edit-2509-Q3_K_S.gguf  (QuantStack/Qwen-Image-Edit-2509-GGUF)\n"
    "- models/loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors  (lightx2v/Qwen-Image-Lightning)\n"
    "- models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors  (Comfy-Org/Qwen-Image_ComfyUI)\n"
    "- models/vae/qwen_image_vae.safetensors  (Comfy-Org/Qwen-Image_ComfyUI)\n"
    "Needs the ComfyUI-GGUF pack (city96) for 'Unet Loader (GGUF)'."
)
DETAIL_MODELS = (
    "Detail fix:\n"
    "- models/checkpoints/sam3.1_multiplex_fp16.safetensors  (SAM 3, loaded with Load Checkpoint; "
    "SAM3 Detect is a core node)\n"
    "- models/upscale_models/4x-UltraSharp.pth\n"
    "- input/novara_logo.png and input/novara_logo_white.png: your logo, dark and white versions\n"
    "Needs the Detail Fixer pack (Detail Detect / Crop / Prepare Logo / Paste Back)."
)

RENAME = {
    # Windows sub-folders on the machine these prompts ran on -> plain file names
    "gguf\\Qwen-Image-Edit-2509-Q3_K_S.gguf": "Qwen-Image-Edit-2509-Q3_K_S.gguf",
    "qwen-image-edit-lightning\\Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors":
        "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
    "qwen\\qwen_2.5_vl_7b_fp8_scaled.safetensors": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "Wan21_CausVid_14B_T2V_lora_rank32281%29.safetensors": "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
    # output prefixes: one NOVARA/ folder, and each workflow loads what the previous one saved
    "Pipeline8/generated": "NOVARA/keyframes",
    "Pipeline8/animatic": "NOVARA/animatic",
    "Pipeline8/sheet1": "NOVARA/critic_sheet_1",
    "Pipeline8/sheet2": "NOVARA/critic_sheet_2",
    "Pipeline8/final": "NOVARA/keyframes_fixed",
    "SpecAd/sheet1": "NOVARA/critic_sheet",
    "SpecAd/final": "NOVARA/final",
    "SpecAd/novara_spot": "NOVARA/spot_stills",
    "SpecAd/novara_spot_v3b": "NOVARA/spot_video",
    "SpecAd/video_qa_v3b": "NOVARA/video_qa",
    **{f"Pipeline8/generated_0000{i}_.png [output]": f"NOVARA/keyframes_0000{i}_.png [output]" for i in range(1, 5)},
    **{f"SpecAd/final_v3_0000{i}_.png [output]": f"NOVARA/final_0000{i}_.png [output]" for i in range(1, 5)},
}

# --------------------------------------------------------------------------- shared stages

def detail_fix_stage(extra_nodes=()):
    return {
        "title": "3 · Detail fix on flagged shots", "color": "purple",
        "nodes": ["sel", "up", "up2", "d1", "d2", "det", "crop", "er", "gd", "q", "qn", "enc", "gm", "nm",
                  "qk", "qd", "pb", "mgD", "p2", *extra_nodes],
        "note": ("Shot Select picks only the shots the critic routed to 'detail' (a wrong logo, garbled "
                 "text). SAM 3 finds the logo / text, Detail Fixer crops it, and Qwen Image Edit repaints "
                 "it from your logo reference in 4 steps (Lightning LoRA). The crop is pasted back with a "
                 "feathered mask, and Shot Reroll Merge (route = detail) puts the fixed shots in place.\n"
                 "When no shot needs a detail fix, this whole group is skipped (lazy input): it costs nothing."),
    }


def critic_stage(nodes, title="2 · Shot Critic"):
    return {
        "title": title, "color": "red", "nodes": nodes,
        "note": ("Shot Critic sends every shot to a vision LLM (Gemini Flash Lite by default, 3 independent "
                 "reviews, majority vote) with the shot list and your brand rules. Each shot gets a score and "
                 "a route: pass, detail (local fix), consistency (edit against another shot) or reroll. The "
                 "contact sheet shows the verdicts.\n" + KEYS + "\n"
                 "Slow or 503 errors: Gemini is busy; the critic falls back from Flash Lite to Flash by itself."),
    }


def keep_better_stage(nodes):
    return {
        "title": "4 · Review again, keep only improvements", "color": "teal", "nodes": nodes,
        "note": ("A second Shot Critic reviews the fixed shots. Shot Keep Better keeps a fix only if it "
                 "scores at least as well as the original (never make it worse), and saves the approved "
                 "keyframes."),
    }


# --------------------------------------------------------------------------- recipes

RECIPES = [
    {
        "api": "brief_to_animatic_api.json", "out": "01_brief_to_animatic.json",
        "stages": [
            {"title": "1 · Brief to shot list", "color": "yellow", "nodes": ["sp", "ps"],
             "note": ("Write the brief. Shot Planner returns a shot list (JSON) plus one image prompt and one "
                      "video prompt per shot. provider = offline needs no key (template-based); gemini, "
                      "anthropic or openai write better shots.\n" + KEYS)},
            {"title": "2 · Keyframes (one image per shot)", "color": "green",
             "nodes": ["ck", "pos", "neg", "lat", "ks", "vd", "s0"],
             "note": ("image_prompts is a list, so CLIP Text Encode, KSampler and VAE Decode run once per shot.\n"
                      "Model: models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors (Juggernaut XL, "
                      "RunDiffusion). Any SDXL checkpoint works; keep 1344x768 for 16:9.")},
            {"title": "3 · Animatic", "color": "blue", "nodes": ["an", "pt", "cv", "sv"],
             "note": ("Shot Animatic times every keyframe from the shot list and adds the camera move named in "
                      "each shot (push-in, pan, tilt...) as a smooth Ken Burns move, with crossfades. No model "
                      "needed. Save Video writes an MP4 to output/ShotPlanner/.")},
        ],
    },
    {
        "api": "novara_keyframes_api.json", "out": "02_novara_keyframes.json", "rename": RENAME,
        "third_party": ["LoaderGGUF (ComfyUI-GGUF)", "Detail Fixer"],
        "stages": [
            {"title": "0 · Models and references", "color": "black",
             "nodes": ["ck", "neg", "upm", "sam", "t1", "t2", "refT", "refR", "gg", "lo", "ms", "cn", "nl", "clip",
                       "vae"],
             "note": ("Download (folders are inside your ComfyUI folder):\n"
                      "- models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors  (Juggernaut XL)\n"
                      "- models/loras/next-scene_lora-v2-3000.safetensors  (lovis93/next-scene-qwen-image-lora-2509)\n"
                      + QWEN_MODELS + "\n" + DETAIL_MODELS)},
            {"title": "1 · Shot list and consistent keyframes", "color": "green",
             "nodes": ["sp", "p5", "sel1", "pos", "lat", "ks", "vd", "sel2", "ca2", "cb2", "eq2", "ek2", "ev2",
                       "sel3", "ca3", "cb3", "eq3", "ek3", "ev3", "sel4", "ca4", "cb4", "eq4", "ek4", "ev4",
                       "elat", "bb1", "bb2", "jn", "s0"],
             "note": ("Shot Planner (Gemini) writes 4 shots from the brief. Shot 1 is rendered with Juggernaut XL. "
                      "Shots 2-4 are each edited from the previous shot with Qwen Image Edit + the next-scene "
                      "LoRA ('Next Scene: ...'), so the barista, the customer and the NOVARA cup stay the same. "
                      "Shot Join makes one list of 4 keyframes.")},
            critic_stage(["cr1", "p1", "s1"]),
            detail_fix_stage(),
            keep_better_stage(["cr2", "p3", "s2", "kb", "p4", "s3"]),
            {"title": "5 · Animatic", "color": "blue", "nodes": ["an", "cv", "sv"],
             "note": "The approved keyframes become a timed animatic with camera moves from the shot list (MP4)."},
        ],
    },
    {
        "api": "novara_stills_spot_api.json", "out": "03_novara_stills_spot.json", "rename": RENAME,
        "third_party": ["LoaderGGUF (ComfyUI-GGUF)", "Detail Fixer"],
        "stages": [
            {"title": "0 · Models and references", "color": "black",
             "nodes": ["upm", "sam", "t1", "t2", "refT", "refR", "lw", "gg", "lo", "ms", "cn", "clip", "vae"],
             "note": ("Download (folders are inside your ComfyUI folder):\n" + QWEN_MODELS + "\n" + DETAIL_MODELS
                      + "\n- input/novara_logo_white_transparent.png: logo with transparency for the end card")},
            {"title": "1 · Keyframes", "color": "green", "nodes": ["i1", "i2", "i3", "i4", "b1", "b2", "b3"],
             "note": ("The four keyframes saved by 02_novara_keyframes (output/NOVARA/keyframes_*.png). Pick your "
                      "own stills here: one per shot, in shot order.")},
            critic_stage(["cr1", "p1", "s1"]),
            detail_fix_stage(),
            keep_better_stage(["cr2", "p4", "kb", "p5", "s3"]),
            {"title": "5 · Ad assembly (MP4)", "color": "blue", "nodes": ["tl", "su", "ec", "cv", "sv"],
             "note": ("Ad Timeline times the stills (durations in seconds per shot) with a gentle push-in and "
                      "crossfades. Ad Supers draws the on-screen lines ('shot | text'). Ad End Card adds the logo, "
                      "tagline and URL over the blurred last frame. Create Video + Save Video write the MP4. "
                      "Add a voiceover and music with Ad Voiceover and Ad Audio Mix (see docs/NODES.md).")},
        ],
    },
    {
        "api": "novara_video_stage_api.json", "out": "04_novara_video_stage.json", "rename": RENAME,
        "third_party": ["CR Prompt List (Comfyroll)"],
        "stages": [
            {"title": "0 · Wan 2.1 models", "color": "black",
             "nodes": ["um", "lo", "ms", "cl", "va", "cvl", "neg", "lw"],
             "note": ("Wan 2.1 image-to-video on a 10 GB GPU (folders inside your ComfyUI folder):\n"
                      "- models/diffusion_models/Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors  (Kijai/WanVideo_comfy; "
                      "any Wan 2.1 I2V 14B fp8 works)\n"
                      "- models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors  (CausVid, Kijai/WanVideo_comfy): "
                      "6 steps at cfg 1\n"
                      "- models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors  (Comfy-Org/Wan_2.1_ComfyUI_repackaged)\n"
                      "- models/clip_vision/clip_vision_h.safetensors  (Comfy-Org/Wan_2.1_ComfyUI_repackaged)\n"
                      "- models/vae/Wan2_1_VAE_fp32.safetensors  (Kijai/WanVideo_comfy; wan_2.1_vae also works)\n"
                      "Low VRAM: start ComfyUI with --lowvram, keep 832x480 and 65 frames.")},
            {"title": "1 · Approved keyframes and motion prompts", "color": "green",
             "nodes": ["i1", "i2", "i3", "i4", "b1", "b2", "ball", "jn", "sc", "pl"],
             "note": ("The approved stills from 03_novara_stills_spot (output/NOVARA/final_*.png), scaled to "
                      "832x480. CR Prompt List gives one motion prompt per shot (or use Shot Motion Prompts "
                      "from the shot list).")},
            {"title": "2 · Shot Video Engine: Wan branch", "color": "pale_blue",
             "nodes": ["pos", "cve", "wi", "ks", "de", "eng", "pe"],
             "note": ("Wan 2.1 I2V renders one clip per shot. Shot Video Engine is the switch in front of the "
                      "video stage: wire other branches (MiniMax H3, or an API node such as Kling / Seedance / "
                      "Hailuo) into its other inputs and pick one; only the chosen branch runs.")},
            {"title": "3 · Shot Video QA and reroll", "color": "red",
             "nodes": ["qa", "qs", "pq", "rs", "rsc", "rpos", "rcve", "rwi", "rks", "rde", "mg", "pm"],
             "note": ("Shot Video QA checks every clip offline (frozen, flicker, jumps, drift from the keyframe, "
                      "melting regions) and with Gemini (use_llm). Failed shots get a new prompt and seed; the "
                      "reroll branch renders only those, and Clip Reroll Merge puts them back. When every clip "
                      "passes, the reroll branch never runs.\n" + KEYS)},
            {"title": "4 · Resample and assemble (MP4)", "color": "blue",
             "nodes": ["fps", "tl", "ec", "su", "cv", "sv"],
             "note": ("Clip Resample FPS turns the 16 fps Wan clips into 24 fps for the timeline. Ad Timeline "
                      "joins the clips with crossfades, Ad End Card and Ad Supers finish the spot, Save Video "
                      "writes the MP4.")},
        ],
    },
    {
        "api": "treatment_pitch_deck_api.json", "out": "05_treatment_pitch_deck.json",
        "stages": [
            {"title": "1 · Keyframes, shot list, logo", "color": "green",
             "nodes": ["i1", "i2", "i3", "i4", "b1", "b2", "jn", "lw", "sl"],
             "note": ("Approved keyframes (one per shot, here the stills from 03_novara_stills_spot), the shot "
                      "list (from Shot Planner or pasted into Shot List Editor) and a logo with transparency. "
                      "No models or API keys needed: everything below runs on the CPU.")},
            {"title": "2 · Palette and script", "color": "yellow", "nodes": ["pal", "pv4", "scr"],
             "note": ("Palette From Shots extracts named colours and an accent from the keyframes. Ad Script From "
                      "Shots writes voiceover lines and supers that fit each shot's duration.")},
            {"title": "3 · Treatment, pitch deck, storyboard (PDF)", "color": "brown",
             "nodes": ["tb", "pv1", "pa1", "pd", "pv2", "pa2", "sb", "pv3"],
             "note": ("Treatment Builder lays out a director's treatment (A4 / Letter, 300 dpi), Pitch Deck Builder "
                      "a 16:9 deck, Storyboard Sheet a print-ready board with camera arrows. PDFs are written to "
                      "output/treatment/. Templates: editorial dark, editorial light, agency minimal.")},
        ],
    },
]
