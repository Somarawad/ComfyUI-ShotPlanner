"""Generate docs/NODES.md from the node classes, so the reference never drifts from the code.

Usage (from the repository root; needs torch, numpy and Pillow, not ComfyUI):

    python tools/gen_nodes_md.py           # rewrite docs/NODES.md
    python tools/gen_nodes_md.py --check   # exit 1 if docs/NODES.md is out of date (used by CI)

Everything comes from the classes: INPUT_TYPES (types, defaults, ranges, tooltips),
RETURN_TYPES / RETURN_NAMES / OUTPUT_IS_LIST / OUTPUT_TOOLTIPS, DESCRIPTION and CATEGORY.
The pack is loaded as a package exactly like ComfyUI loads it, so the node list is the one
registered in __init__.py.
"""

import argparse
import importlib.util
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "NODES.md")
PKG = "comfyui_shotplanner_docs"

# Section per module, in pipeline order: (module, title, one-line intro)
SECTIONS = [
    ("nodes", "Planning", "Brief or script in, shot list and prompts out."),
    ("animatic", "Animatic", "Keyframes or clips to a timed preview, no extra models."),
    ("critic_nodes", "Shot Critic and fixes",
     "A vision LLM reviews the shots like an art director; the helper nodes route each shot "
     "to a re-render, a consistency edit or a local detail fix and keep only improvements."),
    ("video_engine", "Video engine", "One switch in front of the image-to-video stage."),
    ("video_nodes", "Video prompts", "Per-shot motion prompts and frame counts for the chosen engine."),
    ("video_qa_nodes", "Video QA and clip utilities",
     "Offline (plus optional Gemini) clip checks, reroll merging and clip helpers."),
    ("ad_nodes", "Ad assembly", "Timeline, supers, end card, audio mix and an offline script."),
    ("voice_nodes", "Voiceover", "Text-to-speech placed on the edit, and fitting it to the shots."),
    ("beat_nodes", "Beat sync", "Shot durations that cut on the music's beat."),
    ("reframe_nodes", "Social versions", "Subject-following reframes and short cutdowns."),
    ("treatment_nodes", "Treatment and pitch deck",
     "Print-ready director's treatment, pitch deck, storyboard sheet and palette."),
]


def load_pack():
    """Import the pack as a package (relative imports and __init__ registration run as in ComfyUI)."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    spec = importlib.util.spec_from_file_location(PKG, os.path.join(ROOT, "__init__.py"),
                                                  submodule_search_locations=[ROOT])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = pkg
    spec.loader.exec_module(pkg)
    return pkg


def _cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _fmt_default(v):
    if isinstance(v, bool):
        return "`true`" if v else "`false`"
    if isinstance(v, str):
        if v == "":
            return '`""`'
        short = v if len(v) <= 60 else v[:57] + "..."
        return "`" + short.replace("\n", "\\n").replace("|", "\\|").replace("`", "'") + "`"
    return f"`{v}`"


def _type_cell(spec):
    kind = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(kind, (list, tuple)):
        t = "choice: " + " / ".join(f"`{_cell(o)}`" for o in kind)
    else:
        t = str(kind)
        lo, hi = opts.get("min"), opts.get("max")
        if lo is not None or hi is not None:
            t += f" ({'' if lo is None else lo}-{'' if hi is None else hi})"
    return t


def _flags(opts, optional):
    f = []
    if optional:
        f.append("optional")
    if opts.get("lazy"):
        f.append("lazy")
    if opts.get("forceInput"):
        f.append("input only")
    return f" *({', '.join(f)})*" if f else ""


def node_markdown(key, cls, display):
    lines = [f"### {display}", "", f"`{key}` · category **{cls.CATEGORY}**", ""]
    desc = getattr(cls, "DESCRIPTION", "") or inspect.getdoc(cls) or ""
    if desc:
        lines += [" ".join(desc.split()), ""]
    it = cls.INPUT_TYPES()
    if getattr(cls, "INPUT_IS_LIST", False):
        lines += ["Receives whole lists (one item per shot) and runs once.", ""]
    lines += ["**Inputs**", "", "| Input | Type | Default | Tooltip |", "|---|---|---|---|"]
    for section in ("required", "optional"):
        for name, spec in (it.get(section) or {}).items():
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            default = _fmt_default(opts["default"]) if "default" in opts else ""
            lines.append(f"| `{name}`{_flags(opts, section == 'optional')} | {_cell(_type_cell(spec))} "
                         f"| {default} | {_cell(opts.get('tooltip', ''))} |")
    lines += ["", "**Outputs**", "", "| Output | Type | Tooltip |", "|---|---|---|"]
    names = getattr(cls, "RETURN_NAMES", None) or cls.RETURN_TYPES
    is_list = getattr(cls, "OUTPUT_IS_LIST", None) or (False,) * len(cls.RETURN_TYPES)
    tips = getattr(cls, "OUTPUT_TOOLTIPS", None) or ("",) * len(cls.RETURN_TYPES)
    for n, t, li, tip in zip(names, cls.RETURN_TYPES, is_list, tips):
        lines.append(f"| `{n}` | {t}{' (list)' if li else ''} | {_cell(tip)} |")
    return "\n".join(lines) + "\n"


def build():
    pkg = load_pack()
    mappings, names = pkg.NODE_CLASS_MAPPINGS, pkg.NODE_DISPLAY_NAME_MAPPINGS
    by_module = {}
    for key, cls in mappings.items():
        mod = cls.__module__.rsplit(".", 1)[-1]
        by_module.setdefault(mod, []).append((key, cls))
    known = [m for m, _, _ in SECTIONS]
    unknown = sorted(set(by_module) - set(known))
    if unknown:
        raise SystemExit(f"gen_nodes_md: add a section for module(s) {unknown} in SECTIONS")
    head = [
        "# Node reference",
        "",
        "<!-- Generated by tools/gen_nodes_md.py from the node classes. Do not edit by hand: "
        "change the tooltips in the code and run `python tools/gen_nodes_md.py`. -->",
        "",
        f"All {len(mappings)} nodes of the pack, in pipeline order. Tooltips are exactly what "
        "ComfyUI shows when you hover an input. *lazy* inputs are only evaluated when the node "
        "needs them, so the branch feeding them costs nothing otherwise; *input only* inputs have "
        "no widget and must be connected.",
        "",
        "## Contents",
        "",
    ]
    body = []
    for mod, title, intro in SECTIONS:
        nodes = by_module.get(mod, [])
        if not nodes:
            continue
        anchor = title.lower().replace(" ", "-")
        head.append(f"- [{title}](#{anchor}): " + ", ".join(names.get(k, k) for k, _ in nodes))
        body += [f"## {title}", "", intro, ""]
        for key, cls in nodes:
            body.append(node_markdown(key, cls, names.get(key, key)))
    return "\n".join(head) + "\n\n" + "\n".join(body).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if docs/NODES.md is out of date")
    args = ap.parse_args()
    text = build()
    if args.check:
        current = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if current != text:
            print("docs/NODES.md is out of date: run python tools/gen_nodes_md.py", file=sys.stderr)
            sys.exit(1)
        print("docs/NODES.md is up to date.")
        return
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"Wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
