"""Load-test the pack inside a real ComfyUI (CPU is enough, no models needed).

What it checks
  1. ComfyUI imports the pack without errors (no "IMPORT FAILED", no traceback naming the pack).
  2. Every class in the pack's NODE_CLASS_MAPPINGS is listed by /object_info, with the same
     category as in the code, and no other node is registered from the pack.
  3. Every input of every node has a tooltip.
  4. Every API-format prompt in examples/api/ passes /prompt validation, except for errors that
     only come from missing model / input files. Third-party nodes that are not installed
     (SAM 3, GGUF loader, CR Prompt List, Detail Fixer...) are replaced by validation-only stubs
     that accept any input, so the validator still walks through them into this pack's nodes.
     The stubs raise if they are ever executed.

Setup (once; any OS, Python 3.10-3.12)

    git clone https://github.com/comfyanonymous/ComfyUI
    python -m venv venv && . venv/bin/activate        # Windows: venv\\Scripts\\activate
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -r ComfyUI/requirements.txt

Run (from this repository, with that venv active)

    python tools/comfy_smoke_test.py --comfy-dir path/to/ComfyUI

The script links the pack into ComfyUI/custom_nodes (symlink, or a copy where symlinks are not
allowed), writes the stub pack next to it, starts `main.py --cpu --listen 127.0.0.1` on a free
port, runs the checks, stops the server and removes the stub pack. Exit code 0 = all passed.

Options: --port N, --keep-server (leave ComfyUI running at the end), --no-stubs (send the prompts
as they are; missing third-party nodes are then reported, not failed), --url http://host:port
(test an already running ComfyUI instead of starting one; it must have the pack loaded).

Nothing is generated: API-key environment variables are blanked for the server it starts, and if a
prompt is accepted (some of its outputs are valid) it is interrupted and removed from the queue at once.
"""

import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK_DIR_NAME = "ComfyUI-ShotPlanner"
STUB_DIR_NAME = "zz_shotplanner_smoke_stubs"  # "zz" so it loads after real packs
API_DIR = os.path.join(ROOT, "examples", "api")
KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
            "ELEVENLABS_API_KEY")

# inputs whose values are file names ComfyUI looks up in models/ or input/: a "not in list" or
# "invalid file" error there only means the file is not on this machine
FILE_INPUTS = ("ckpt_name", "lora_name", "unet_name", "vae_name", "clip_name", "clip_name1",
               "clip_name2", "model_name", "gguf_name", "control_net_name", "style_model_name",
               "upscale_model", "image", "audio", "video", "file")


# --------------------------------------------------------------------------- helpers

def log(msg=""):
    print(msg, flush=True)


def http(url, data=None, timeout=30):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw.decode("utf-8", "replace")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load_pack_mappings():
    """The pack's own NODE_CLASS_MAPPINGS, imported as a package like ComfyUI does."""
    name = "shotplanner_smoke_pkg"
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "__init__.py"),
                                                  submodule_search_locations=[ROOT])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    return pkg.NODE_CLASS_MAPPINGS


def load_prompts(api_dir=API_DIR):
    out = {}
    for fn in sorted(os.listdir(api_dir)) if os.path.isdir(api_dir) else []:
        if fn.endswith(".json"):
            with open(os.path.join(api_dir, fn), encoding="utf-8") as f:
                out[fn] = json.load(f)
    return out


# --------------------------------------------------------------------------- setup

def link_pack(comfy_dir):
    target = os.path.join(comfy_dir, "custom_nodes", PACK_DIR_NAME)
    if os.path.lexists(target):
        real = os.path.realpath(target)
        if real != os.path.realpath(ROOT):
            log(f"note: {target} exists and is not this checkout ({real}); testing that copy")
        return target
    try:
        os.symlink(ROOT, target, target_is_directory=True)
        log(f"linked {target} -> {ROOT}")
    except (OSError, NotImplementedError):
        shutil.copytree(ROOT, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "output"))
        log(f"copied the pack to {target} (symlinks not allowed here)")
    return target


STUB_TEMPLATE = '''"""Validation-only stand-ins written by tools/comfy_smoke_test.py. Safe to delete."""
import nodes as _nodes

_SPECS = {specs!r}


class _AnyType(str):
    def __ne__(self, other):
        return False


ANY = _AnyType("*")


def _make(name, inputs):
    class Stub:
        @classmethod
        def INPUT_TYPES(cls):
            return {{"required": {{}}, "optional": {{k: (ANY, {{}}) for k in inputs}}}}

        RETURN_TYPES = (ANY,) * 12
        FUNCTION = "run"
        CATEGORY = "_smoke_test_stubs"

        def run(self, **kw):
            raise RuntimeError(f"{{name}} is a smoke-test stub; install the real node pack.")

    Stub.__name__ = "Stub_" + "".join(c if c.isalnum() else "_" for c in name)
    return Stub


NODE_CLASS_MAPPINGS = {{n: _make(n, i) for n, i in _SPECS.items()
                       if n not in _nodes.NODE_CLASS_MAPPINGS}}
'''


def write_stubs(comfy_dir, prompts, known):
    """Stubs for every class_type in the prompts that is neither core nor from this pack.
    `known` = class types that core ComfyUI provides (filled after a first look) or None."""
    specs = {}
    for prompt in prompts.values():
        for node in prompt.values():
            ct = node["class_type"]
            if known is not None and ct in known:
                continue
            specs.setdefault(ct, set()).update(node.get("inputs", {}).keys())
    d = os.path.join(comfy_dir, "custom_nodes", STUB_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(STUB_TEMPLATE.format(specs={k: sorted(v) for k, v in sorted(specs.items())}))
    return d


class Server:
    def __init__(self, comfy_dir, port, python):
        self.comfy_dir, self.port, self.python = comfy_dir, port, python
        self.proc, self.log_path = None, None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout=600):
        fd, self.log_path = tempfile.mkstemp(prefix="comfy_smoke_", suffix=".log")
        os.close(fd)
        env = dict(os.environ)
        for k in KEY_VARS:
            env[k] = ""
        cmd = [self.python, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(self.port),
               "--disable-auto-launch"]
        log(f"starting ComfyUI: {' '.join(cmd)}  (log: {self.log_path})")
        self.logf = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=self.comfy_dir, env=env, stdout=self.logf,
                                     stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise SystemExit(f"ComfyUI exited with code {self.proc.returncode}:\n{self.read_log()[-4000:]}")
            try:
                status, _ = http(self.url + "/system_stats", timeout=5)
                if status == 200:
                    log(f"ComfyUI is up after {time.time() - t0:.0f}s")
                    return
            except (urllib.error.URLError, ConnectionError, socket.timeout):
                pass
            time.sleep(1)
        raise SystemExit("ComfyUI did not start in time:\n" + self.read_log()[-4000:])

    def read_log(self):
        if not self.log_path:
            return ""
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return f.read()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.proc:
            self.logf.close()


# --------------------------------------------------------------------------- checks

class Report:
    def __init__(self):
        self.failures, self.notes = [], []

    def fail(self, msg):
        self.failures.append(msg)
        log("  FAIL " + msg)

    def ok(self, msg):
        log("  ok   " + msg)

    def note(self, msg):
        self.notes.append(msg)
        log("  note " + msg)


def _is_pack_line(line):
    norm = line.replace("\\", "/")
    return f"custom_nodes/{PACK_DIR_NAME}" in norm and not norm.rstrip().endswith(".py")


def check_import(rep, server_log):
    log("\n[1] pack import")
    lines = server_log.splitlines()
    # ComfyUI logs "Cannot import <path> module for custom nodes: <error>" and later lists the
    # pack under "Import times for custom nodes" with "(IMPORT FAILED)"
    bad = [ln for ln in lines if _is_pack_line(ln) and ("IMPORT FAILED" in ln or "Cannot import" in ln)]
    timing = [ln.strip() for ln in lines if _is_pack_line(ln) and "seconds" in ln]
    if bad:
        rep.fail("ComfyUI could not import the pack:\n      " + "\n      ".join(bad))
    elif not timing:
        rep.fail("the pack is not in ComfyUI's custom node import list (was it linked into custom_nodes?)")
    else:
        rep.ok("imported without errors (" + timing[0].split("seconds")[0].split()[-1] + " s)")


def check_object_info(rep, info, mappings):
    log(f"\n[2] /object_info: {len(mappings)} nodes in NODE_CLASS_MAPPINGS")
    for key, cls in mappings.items():
        if key not in info:
            rep.fail(f"{key} is missing from /object_info")
            continue
        if info[key].get("category") != cls.CATEGORY:
            rep.fail(f"{key}: category {info[key].get('category')!r}, code says {cls.CATEGORY!r}")
    ours = {k for k, v in info.items() if PACK_DIR_NAME in str(v.get("python_module", ""))}
    extra = ours - set(mappings)
    if extra:
        rep.fail(f"registered from the pack but not in NODE_CLASS_MAPPINGS: {sorted(extra)}")
    if not any(f.startswith(f"{k} ") for f in rep.failures for k in mappings):
        rep.ok(f"all {len(mappings)} nodes listed with the right category")

    log("\n[3] tooltips")
    missing = []
    for key in mappings:
        for section in ("required", "optional"):
            for name, spec in (info.get(key, {}).get("input", {}).get(section) or {}).items():
                opts = spec[1] if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict) else {}
                if not str(opts.get("tooltip", "")).strip():
                    missing.append(f"{key}.{name}")
    if missing:
        rep.fail(f"{len(missing)} inputs without a tooltip: {', '.join(missing)}")
    else:
        rep.ok("every input has a tooltip")


def classify(prompt, node_id, err):
    """'file' when the error only means a model / input file is not on this machine."""
    inp = (err.get("extra_info") or {}).get("input_name") or str(err.get("details", "")).split(":")[0]
    ct = prompt.get(node_id, {}).get("class_type", "")
    if err.get("type") == "value_not_in_list" and inp in FILE_INPUTS:
        return "file"
    if err.get("type") == "custom_validation_failed" and ct.startswith("Load") and \
            any(inp.startswith(f) for f in FILE_INPUTS):
        return "file"
    return "real"


def check_prompts(rep, url, prompts, stubbed):
    log(f"\n[4] /prompt validation of {len(prompts)} example prompts")
    for fn, prompt in prompts.items():
        status, r = http(url + "/prompt", {"prompt": prompt, "client_id": "smoke-" + uuid.uuid4().hex})
        if isinstance(r, dict) and r.get("prompt_id"):  # some outputs were valid: never run it
            http(url + "/interrupt", {})
            http(url + "/queue", {"delete": [r["prompt_id"]]})
        if not isinstance(r, dict):
            rep.fail(f"{fn}: unexpected reply {status}: {str(r)[:300]}")
            continue
        top = r.get("error")
        if isinstance(top, dict) and top.get("type") == "missing_node_type":
            ct = (top.get("extra_info") or {}).get("class_type")
            rep.note(f"{fn}: stops at missing node type {ct!r} (third-party pack not installed; "
                     "run without --no-stubs to validate the rest)")
            continue
        node_errors = r.get("node_errors") or {}
        files, real = [], []
        for nid, ne in node_errors.items():
            for e in ne.get("errors", []):
                (files if classify(prompt, nid, e) == "file" else real).append(
                    f"{nid} {ne.get('class_type')}: {e.get('message')}: {e.get('details')}")
        if top and not node_errors and top.get("type") != "prompt_outputs_failed_validation":
            real.append(f"{top.get('type')}: {top.get('message')} {top.get('details', '')}")
        third = sorted({prompt[n]["class_type"] for n in prompt if prompt[n]["class_type"] in stubbed})
        summary = (f"{len(files)} missing-file errors"
                   + (f", third-party nodes stubbed: {', '.join(third)}" if third else ""))
        if real:
            rep.fail(f"{fn}: {len(real)} real validation errors ({summary}):\n      " + "\n      ".join(real))
        else:
            rep.ok(f"{fn}: valid apart from {summary}")
            for f in files:
                log(f"         - {f}")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Load-test the pack inside a real ComfyUI (CPU).")
    ap.add_argument("--comfy-dir", help="ComfyUI checkout (needed unless --url is given)")
    ap.add_argument("--url", help="test an already running ComfyUI instead of starting one")
    ap.add_argument("--port", type=int, default=0, help="port for the ComfyUI it starts (default: free port)")
    ap.add_argument("--python", default=sys.executable, help="interpreter for ComfyUI (default: this one)")
    ap.add_argument("--keep-server", action="store_true", help="leave ComfyUI running at the end")
    ap.add_argument("--no-stubs", action="store_true", help="do not stub missing third-party nodes")
    ap.add_argument("--api-dir", default=API_DIR, help="folder of API prompts to validate (default: examples/api)")
    args = ap.parse_args()
    if not args.url and not args.comfy_dir:
        ap.error("give --comfy-dir (or --url of a running ComfyUI)")

    rep = Report()
    try:
        mappings = load_pack_mappings()
    except Exception as e:  # still start ComfyUI so its own import report is shown too
        import traceback
        rep.fail("the pack does not import outside ComfyUI either:\n" + traceback.format_exc(limit=3))
        mappings = {}
    prompts = load_prompts(args.api_dir)
    server, stub_dir, stubbed = None, None, set()
    try:
        if args.url:
            url, server_log = args.url.rstrip("/"), ""
            log("testing a running ComfyUI: import errors can only be seen in its own console")
        else:
            comfy_dir = os.path.abspath(args.comfy_dir)
            if not os.path.exists(os.path.join(comfy_dir, "main.py")):
                raise SystemExit(f"{comfy_dir} is not a ComfyUI checkout (no main.py)")
            link_pack(comfy_dir)
            if not args.no_stubs:
                # stub every class type that is not in this pack; the stub module itself skips
                # the ones core ComfyUI (or a real installed pack) already registers
                stub_dir = write_stubs(comfy_dir, prompts, set(mappings))
            server = Server(comfy_dir, args.port or free_port(), args.python)
            server.start()
            url = server.url
            server_log = server.read_log()
        status, info = http(url + "/object_info", timeout=120)
        if status != 200:
            raise SystemExit(f"/object_info returned {status}")
        stubbed = {k for k, v in info.items() if v.get("category") == "_smoke_test_stubs"}
        if stubbed:
            log(f"third-party nodes not installed, stubbed for validation: {', '.join(sorted(stubbed))}")
        if server:
            check_import(rep, server_log)
        if mappings:
            check_object_info(rep, info, mappings)
            check_prompts(rep, url, prompts, stubbed)
    finally:
        if server and not args.keep_server:
            server.stop()
        if stub_dir and not args.keep_server:
            shutil.rmtree(stub_dir, ignore_errors=True)
    log("")
    if rep.failures:
        log(f"FAILED: {len(rep.failures)} problem(s)")
        sys.exit(1)
    log("PASSED" + (f" ({len(rep.notes)} note(s))" if rep.notes else ""))
    if args.keep_server and server:
        log(f"ComfyUI is still running at {server.url} (pid {server.proc.pid})")


if __name__ == "__main__":
    main()
