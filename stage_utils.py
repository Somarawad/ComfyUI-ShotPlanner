"""Shared helpers for the video / voice / beat / reframe stages.

Python stdlib + numpy only (no ComfyUI, no torch), so every module that uses it stays
unit-testable on its own. HTTP goes through urllib; API keys never appear in messages.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np


class StageError(ValueError):
    """A clear, user-facing problem with the inputs or with an API call."""


class HttpError(StageError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


ENV_KEYS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "elevenlabs": ("ELEVENLABS_API_KEY",),
}

# patched in tests so retries do not wait
sleep = time.sleep


# --------------------------------------------------------------------------- parsing

def parse_number_list(text, what="durations", positive=True):
    """"3, 2.5 4s" -> [3.0, 2.5, 4.0]. Commas, semicolons or whitespace separate values."""
    out = []
    for tok in re.split(r"[,;\s]+", str(text or "").strip()):
        if not tok:
            continue
        t = tok.lower().rstrip("s")
        try:
            v = float(t)
        except ValueError as e:
            raise StageError(f"{what}: {tok!r} is not a number") from e
        if positive and v <= 0:
            raise StageError(f"{what}: {v:g} must be > 0")
        out.append(v)
    return out


def parse_int_list(text, what="list"):
    out = []
    for v in parse_number_list(text, what):
        if abs(v - round(v)) > 1e-9:
            raise StageError(f"{what}: {v:g} is not a whole number")
        out.append(int(round(v)))
    return out


def load_json(value, what):
    """Parse a JSON string (or pass a dict through). '' / None -> None."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except ValueError as e:
        raise StageError(f"{what} is not valid JSON ({e})") from e


def load_shot_list(shot_list_json):
    plan = load_json(shot_list_json, "shot_list_json")
    if plan is None:
        return None
    if not isinstance(plan, dict) or not isinstance(plan.get("shots"), list):
        raise StageError("shot_list_json is not a Shot Planner list (no \"shots\" array)")
    return plan


def load_timing(timing_json):
    t = load_json(timing_json, "timing_json")
    if t is None:
        return None
    if not isinstance(t, dict) or "shots" not in t or "fps" not in t:
        raise StageError("timing_json is not Ad Timeline output (needs \"fps\" and \"shots\")")
    return t


def extract_json_object(text):
    """First {...} object in a model reply (handles ```json fences and chatter)."""
    if not isinstance(text, str):
        raise StageError("empty reply")
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        v = json.loads(s)
        if isinstance(v, dict):
            return v
    except ValueError:
        pass
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(s[start:i + 1])
                        if isinstance(v, dict):
                            return v
                    except ValueError:
                        pass
                    break
        start = s.find("{", start + 1)
    raise StageError("reply does not contain a JSON object")


# --------------------------------------------------------------------------- keys & HTTP

def resolve_key(provider, api_key=""):
    key = (api_key or "").strip()
    if key:
        return key
    for name in ENV_KEYS.get(provider, ()):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    names = " or ".join(ENV_KEYS.get(provider, ("an API key",)))
    raise StageError(f"No API key for {provider}. Set {names} before starting ComfyUI, or "
                     f"paste a key into the api_key field (it is then saved in the workflow).")


def http_post(url, body, headers=None, timeout=180):
    """POST and return (bytes, content_type). body: dict (sent as JSON) or bytes.

    Raises HttpError with .status on HTTP errors; the message holds the server's reply but
    never the request headers (where the key lives)."""
    hdrs = dict(headers or {})
    if isinstance(body, (dict, list)):
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    else:
        data = body
    req = urllib.request.Request(url, data=data, method="POST", headers=hdrs)
    host = re.sub(r"^https?://([^/]+).*$", r"\1", url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "") if resp.headers else ""
            return resp.read(), ctype
    except urllib.error.HTTPError as e:
        try:
            msg = e.read().decode("utf-8", "replace")[:400]
        except Exception:  # noqa: BLE001 - body is best effort
            msg = ""
        raise HttpError(f"{host} answered HTTP {e.code}: {msg}", e.code) from e
    except urllib.error.URLError as e:
        raise HttpError(f"could not reach {host}: {e.reason}") from e
    except TimeoutError as e:
        raise HttpError(f"{host} did not answer within {timeout} s") from e


def http_post_json(url, body, headers=None, timeout=180):
    raw, _ = http_post(url, body, headers, timeout)
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise HttpError(f"API returned something that is not JSON: {raw[:200]!r}") from e


def post_with_retry(fn, retries=1, codes=(429, 500, 502, 503, 529), backoff=2.0):
    """Call fn(); retry `retries` times on busy-server HTTP codes."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except HttpError as e:
            if attempt < retries and e.status in codes:
                sleep(backoff * (attempt + 1))
                continue
            raise


# --------------------------------------------------------------------------- arrays

def as_numpy(x):
    """torch tensor or array-like -> float32 numpy array (no torch import needed)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def resample_linear(wave, sr_in, sr_out):
    """Resample [C,S] float audio by linear interpolation (with a light box pre-filter when
    downsampling). Good enough for speech and for analysis."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 1:
        wave = wave[None]
    sr_in, sr_out = int(sr_in), int(sr_out)
    n = wave.shape[-1]
    if sr_in == sr_out or n == 0:
        return wave.copy()
    if sr_out < sr_in:
        k = int(round(sr_in / sr_out))
        if k > 1:
            kern = np.ones(k, dtype=np.float32) / k
            wave = np.stack([np.convolve(c, kern, mode="same") for c in wave])
    n_out = max(1, int(round(n * sr_out / sr_in)))
    t = np.arange(n_out, dtype=np.float64) * (sr_in / sr_out)
    src = np.arange(n, dtype=np.float64)
    return np.stack([np.interp(t, src, c) for c in wave]).astype(np.float32)


def to_stereo(wave):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 1:
        wave = wave[None]
    if wave.shape[0] == 1:
        return np.repeat(wave, 2, axis=0)
    return wave[:2]


def audio_to_numpy(audio):
    """ComfyUI AUDIO dict -> ([C,S] float32, sample_rate)."""
    if audio is None:
        raise StageError("no audio connected")
    w = as_numpy(audio["waveform"])
    if w.ndim == 3:
        w = w[0]
    return w, int(audio["sample_rate"])
