"""Convert ComfyUI API-format prompts into UI-format workflows (what the canvas saves and loads).

Nodes are laid out left to right by dependency depth, inside one group per stage, with a Note at
the top of every group explaining the stage. The node definitions (input order, widgets, output
types) come from a running ComfyUI's /object_info, so widget values land in the right slots.

Usage (from the repository root):

    # all example workflows (recipes below) -> examples/workflows/
    python tools/api_to_workflow.py --object-info http://127.0.0.1:8188 --all

    # any API prompt, one group per dependency depth band, no notes
    python tools/api_to_workflow.py --object-info http://127.0.0.1:8188 in_api.json out.json

--object-info also takes a saved /object_info JSON file. --sizes takes a JSON file of measured
node sizes {"<workflow file>": {"<node id>": [w, h]}} written by tools/check_workflows.js
--measure, so the layout uses the sizes the frontend really draws.

Nodes from packs that are not installed where /object_info came from (GGUF loader, CR Prompt
List, Detail Fixer...) are still written: their slots are inferred from the links and their
widget values keep the order of the API prompt, so they load fine once the pack is installed.
"""

import argparse
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(ROOT, "examples", "api")
OUT_DIR = os.path.join(ROOT, "examples", "workflows")

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
GAP_X, GAP_Y, PAD = 60, 36, 30
STAGE_GAP = 140
COLORS = {  # LiteGraph group palette
    "red": "#b06634", "brown": "#7a5a3a", "green": "#3f789e", "blue": "#335", "pale_blue": "#3f5159",
    "cyan": "#2a6b6b", "purple": "#6a4a8a", "yellow": "#b58b2a", "black": "#222",
    "teal": "#2f6f5f", "olive": "#5a6b2f",
}
NOTE_COLOR, NOTE_BG = "#432", "#653"


# --------------------------------------------------------------------------- object_info

def load_object_info(src):
    if src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src.rstrip("/") + "/object_info", timeout=120) as r:
            return json.loads(r.read())
    with open(src, encoding="utf-8") as f:
        return json.load(f)


def _spec_parts(spec):
    """object_info input spec -> (type, options, combo values or None)."""
    t = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(t, list):
        return "COMBO", opts, t
    if t == "COMBO":
        return "COMBO", opts, opts.get("options", [])
    return t, opts, None


def is_widget(spec):
    t, opts, _ = _spec_parts(spec)
    return t in WIDGET_TYPES and not opts.get("forceInput")


def default_value(spec):
    t, opts, combo = _spec_parts(spec)
    if "default" in opts:
        return opts["default"]
    if combo:
        return combo[0]
    return {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(t, None)


def ordered_inputs(info):
    """[(name, spec, optional)] in the order the frontend creates them."""
    out = []
    order = info.get("input_order") or {}
    for section in ("required", "optional"):
        specs = (info.get("input") or {}).get(section) or {}
        names = order.get(section) or list(specs)
        out += [(n, specs[n], section == "optional") for n in names if n in specs]
    return out


# --------------------------------------------------------------------------- graph helpers

def is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def depths(prompt):
    memo = {}

    def d(n, seen=()):
        if n in memo:
            return memo[n]
        if n in seen:
            return 0
        ups = [v[0] for v in prompt[n]["inputs"].values() if is_link(v) and v[0] in prompt]
        memo[n] = 0 if not ups else 1 + max(d(u, seen + (n,)) for u in ups)
        return memo[n]

    for n in prompt:
        d(n)
    return memo


def topo_order(prompt):
    dep = depths(prompt)
    return sorted(prompt, key=lambda n: (dep[n], list(prompt).index(n)))


# --------------------------------------------------------------------------- node building

def estimate_size(node):
    widgets = [w for w in node.get("_widget_specs", [])]
    multiline = sum(1 for _, s in widgets if _spec_parts(s)[1].get("multiline"))
    rows = max(len([i for i in node["inputs"] if "widget" not in i]), len(node["outputs"]))
    h = 30 + rows * 22 + (len(widgets) - multiline) * 26 + multiline * 110 + 12
    w = 420 if multiline else 330
    ct = node["type"]
    if ct in ("LoadImage",):
        h += 260
    if ct in ("SaveImage", "PreviewImage", "SaveVideo", "PreviewAny", "Note"):
        h = max(h, 260)
    if ct == "PreviewAny":
        w = 380
    return [w, h]


class Builder:
    def __init__(self, object_info, prompt, rename=None):
        self.oi, self.prompt = object_info, prompt
        self.rename = rename or {}
        self.ids = {k: i + 1 for i, k in enumerate(prompt)}  # API id -> UI int id
        self.links, self.nodes = [], {}
        self.missing = sorted({v["class_type"] for v in prompt.values() if v["class_type"] not in object_info})

    def out_types(self, api_id):
        ct = self.prompt[api_id]["class_type"]
        if ct in self.oi:
            info = self.oi[ct]
            return list(info.get("output", [])), list(info.get("output_name", [])), \
                list(info.get("output_is_list", [False] * len(info.get("output", []))))
        # unknown node: as many outputs as the highest slot used, typed by their consumers
        used = {}
        for n, node in self.prompt.items():
            cinfo = self.oi.get(node["class_type"])
            for name, v in node["inputs"].items():
                if is_link(v) and v[0] == api_id:
                    t = "*"
                    if cinfo:
                        spec = dict((x[0], x[1]) for x in ordered_inputs(cinfo)).get(name)
                        if spec is not None:
                            t = _spec_parts(spec)[0]
                    used.setdefault(v[1], t)
        n = max(used) + 1 if used else 1
        types = [used.get(i, "*") for i in range(n)]
        return types, [t.lower() if t != "*" else f"output_{i}" for i, t in enumerate(types)], [False] * n

    def build_node(self, api_id):
        node = self.prompt[api_id]
        ct = node["class_type"]
        inputs = {k: self.rename.get(v, v) if isinstance(v, str) else v for k, v in node["inputs"].items()}
        uid = self.ids[api_id]
        slots, widgets, widget_specs, converted = [], [], [], []
        if ct in self.oi:
            for name, spec, optional in ordered_inputs(self.oi[ct]):
                t, opts, _ = _spec_parts(spec)
                v = inputs.get(name)
                if is_widget(spec):
                    widget_specs.append((name, spec))
                    widgets.append(default_value(spec) if (v is None or is_link(v)) else v)
                    if name in ("seed", "noise_seed") or opts.get("control_after_generate"):
                        if t == "INT":
                            widgets.append("fixed")
                    if opts.get("image_upload"):
                        widgets.append("image")
                    if is_link(v):
                        converted.append({"name": name, "type": t, "widget": {"name": name},
                                          "link": None, "_src": v})
                else:
                    slots.append({"name": name, "type": t if t != "COMBO" else "COMBO", "link": None,
                                  **({"shape": 7} if optional else {}), "_src": v if is_link(v) else None})
        else:  # node pack not installed here: keep the prompt's input order
            for name, v in inputs.items():
                if is_link(v):
                    st = self.out_types(v[0])[0]
                    slots.append({"name": name, "type": st[v[1]] if v[1] < len(st) else "*",
                                  "link": None, "_src": v})
                else:
                    widgets.append(v)
                    widget_specs.append((name, ["STRING", {}]))
        all_slots = slots + converted
        for i, s in enumerate(all_slots):
            src = s.pop("_src")
            if src is not None:
                lid = len(self.links) + 1
                st = self.out_types(src[0])[0]
                ltype = st[src[1]] if src[1] < len(st) else s["type"]
                self.links.append([lid, self.ids[src[0]], src[1], uid, i, ltype])
                s["link"] = lid
        types, names, is_list = self.out_types(api_id)
        outputs = []
        for i, (t, n) in enumerate(zip(types, names)):
            o = {"name": n, "type": t, "links": [], "slot_index": i}
            if i < len(is_list) and is_list[i]:
                o["shape"] = 6
            outputs.append(o)
        title = (node.get("_meta") or {}).get("title")
        ui = {"id": uid, "type": ct, "pos": [0, 0], "size": [330, 100], "flags": {}, "order": 0,
              "mode": 0, "inputs": all_slots, "outputs": outputs,
              "properties": {"Node name for S&R": ct}, "widgets_values": widgets}
        if title and title != ct:
            ui["title"] = title
        ui["_widget_specs"] = widget_specs
        self.nodes[api_id] = ui
        return ui

    def build(self):
        for i, api_id in enumerate(topo_order(self.prompt)):
            self.build_node(api_id)["order"] = i
        for lid, src, sslot, _, _, _ in self.links:
            for n in self.nodes.values():
                if n["id"] == src:
                    n["outputs"][sslot]["links"].append(lid)
        return self


# --------------------------------------------------------------------------- layout

MAX_COL_H = 1500   # a depth column taller than this wraps into a second column
ASPECT = 16 / 9    # stage groups are packed into rows to give roughly this canvas shape


def _stage_local(st, nodes, prompt, dep, sizes, note_id):
    """Lay out one stage at (0, 0): note on top, then one column per dependency depth.
    Returns (positions {api id: [x, y]}, note dict, width, height)."""
    members = [m for m in st["nodes"] if m in nodes]
    cols = {}
    if members:
        base = min(dep[m] for m in members)
        for m in members:
            cols.setdefault(dep[m] - base, []).append(m)
    # split tall columns
    columns = []
    for c in sorted(cols):
        cur, h = [], 0
        for m in sorted(cols[c], key=lambda m: list(prompt).index(m)):
            nh = nodes[m]["size"][1] + GAP_Y + 30
            if cur and h + nh > MAX_COL_H:
                columns.append(cur)
                cur, h = [], 0
            cur.append(m)
            h += nh
        columns.append(cur)
    body_w = sum(max(nodes[m]["size"][0] for m in col) + GAP_X for col in columns) - GAP_X if columns else 0
    note_w = min(max(420, body_w), 900)
    note_lines = st["note"].count("\n") + 1 + len(st["note"]) // 90
    note_h = 40 + 19 * note_lines
    measured = sizes.get(str(note_id))
    if measured:  # the frontend may draw the note larger (text wrap, min size)
        note_w, note_h = max(note_w, measured[0]), max(note_h, measured[1])
    note = {"id": note_id, "type": "Note", "pos": [PAD, PAD + 40], "size": [note_w, note_h],
            "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [],
            "properties": {}, "widgets_values": [st["note"]], "color": NOTE_COLOR, "bgcolor": NOTE_BG,
            "title": st.get("note_title", "Note: " + st["title"])}
    pos = {}
    cx, top = PAD, PAD + 40 + note_h + GAP_Y + 30
    bottom, right = top, PAD + note_w
    for col in columns:
        w = max(nodes[m]["size"][0] for m in col)
        y = top
        for m in col:
            pos[m] = [cx, y]
            y += nodes[m]["size"][1] + GAP_Y + 30  # + title bar
        bottom = max(bottom, y)
        right = max(right, cx + w)
        cx += w + GAP_X
    return pos, note, right + PAD, bottom + PAD


def layout(builder, stages, sizes=None):
    """stages: [{"title", "color", "nodes": [api ids], "note": str}] in pipeline order.
    Every stage becomes a group (note on top, nodes in depth columns); groups are placed left to
    right and wrap into rows so the whole canvas is roughly 16:9. Returns groups, notes."""
    prompt, nodes = builder.prompt, builder.nodes
    dep = depths(prompt)
    sizes = sizes or {}
    for n in nodes.values():
        n["size"] = sizes.get(str(n["id"]), estimate_size(n))
    next_id = max(n["id"] for n in nodes.values()) + 1
    local = []
    for st in stages:
        local.append(_stage_local(st, nodes, prompt, dep, sizes, next_id))
        next_id += 1
    area = sum(w * h for _, _, w, h in local)
    row_w = max(max(w for _, _, w, _ in local), (area * ASPECT) ** 0.5 * 1.35)
    groups, notes = [], []
    x = y = row_h = 0
    for i, (st, (pos, note, w, h)) in enumerate(zip(stages, local)):
        if x > 0 and x + w > row_w:
            x, y, row_h = 0, y + row_h + STAGE_GAP, 0
        for m, (px, py) in pos.items():
            nodes[m]["pos"] = [x + px, y + py]
        note["pos"] = [x + note["pos"][0], y + note["pos"][1]]
        note["order"] = len(nodes) + i
        notes.append(note)
        groups.append({"id": i + 1, "title": st["title"], "bounding": [x, y, w, h],
                       "color": COLORS.get(st.get("color", "pale_blue"), st.get("color", "#3f5159")),
                       "font_size": 24, "flags": {}})
        x += w + STAGE_GAP
        row_h = max(row_h, h)
    return groups, notes


def workflow_json(builder, groups, notes, extra=None):
    nodes = []
    for n in sorted(builder.nodes.values(), key=lambda n: n["id"]):
        n = dict(n)
        n.pop("_widget_specs", None)
        nodes.append(n)
    nodes += notes
    return {
        "last_node_id": max(n["id"] for n in nodes),
        "last_link_id": len(builder.links),
        "nodes": nodes,
        "links": builder.links,
        "groups": groups,
        "config": {},
        "extra": {"ds": {"scale": 0.5, "offset": [0, 0]}, **(extra or {})},
        "version": 0.4,
    }


def convert(object_info, prompt, stages=None, sizes=None, rename=None, extra=None):
    b = Builder(object_info, prompt, rename).build()
    if not stages:  # generic: one group per band of 3 depth levels
        dep = depths(prompt)
        bands = {}
        for n in prompt:
            bands.setdefault(dep[n] // 3, []).append(n)
        stages = [{"title": f"Step {i + 1}", "nodes": bands[k], "note": f"Depth {k * 3}-{k * 3 + 2}."}
                  for i, k in enumerate(sorted(bands))]
    missing = [n for n in prompt if not any(n in s["nodes"] for s in stages)]
    if missing:
        raise SystemExit(f"nodes not assigned to a stage: {missing}")
    groups, notes = layout(b, stages, sizes)
    return workflow_json(b, groups, notes, extra), b.missing


# --------------------------------------------------------------------------- recipes

def _recipes():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from workflow_recipes import RECIPES  # noqa: E402
    return RECIPES


def main():
    ap = argparse.ArgumentParser(description="API-format prompt -> UI workflow")
    ap.add_argument("api", nargs="?", help="API prompt JSON (omit with --all)")
    ap.add_argument("out", nargs="?", help="output workflow JSON")
    ap.add_argument("--object-info", required=True, help="ComfyUI URL or saved /object_info JSON")
    ap.add_argument("--all", action="store_true", help="build every recipe into examples/workflows/")
    ap.add_argument("--sizes", help="measured node sizes from tools/check_workflows.js --measure")
    args = ap.parse_args()
    oi = load_object_info(args.object_info)
    sizes = json.load(open(args.sizes, encoding="utf-8")) if args.sizes else {}
    if args.all:
        os.makedirs(OUT_DIR, exist_ok=True)
        for r in _recipes():
            src = r["api"]
            with open(os.path.join(API_DIR, src), encoding="utf-8") as f:
                prompt = json.load(f)
            wf, missing = convert(oi, prompt, r["stages"], sizes.get(r["out"]), r.get("rename"),
                                  {"shotplanner": {"source_api": f"examples/api/{src}",
                                                   "third_party_nodes": r.get("third_party", [])}})
            path = os.path.join(OUT_DIR, r["out"])
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(wf, f, indent=1, ensure_ascii=False)
                f.write("\n")
            print(f"{r['out']}: {len(prompt)} nodes, {len(wf['groups'])} groups"
                  + (f", not installed here: {', '.join(missing)}" if missing else ""))
        return
    if not (args.api and args.out):
        ap.error("give api and out, or --all")
    with open(args.api, encoding="utf-8") as f:
        prompt = json.load(f)
    wf, missing = convert(oi, prompt, sizes=sizes.get(os.path.basename(args.out)))
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(wf, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}" + (f" (not installed here: {', '.join(missing)})" if missing else ""))


if __name__ == "__main__":
    main()
