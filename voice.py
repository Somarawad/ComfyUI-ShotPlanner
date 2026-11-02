"""Ad Voiceover (TTS) and Ad Voice Fit: pure logic.

stdlib + numpy only. Audio is float32 [channels, samples]. Providers:
  gemini      generateContent with responseModalities AUDIO -> base64 16-bit PCM (24 kHz)
  openai      /v1/audio/speech, response_format wav
  elevenlabs  /v1/text-to-speech/{voice}?output_format=pcm_44100 -> raw 16-bit PCM
Decoding uses struct/numpy (no ffmpeg). Lines can be placed on the edit using Ad Timeline's
timing_json, and Ad Voice Fit time-stretches (WSOLA) lines that overrun their shot.
"""

import base64
import json
import re
import struct
import urllib.parse

import numpy as np

try:
    from . import stage_utils as su
except ImportError:  # imported as a top-level module (tests)
    import stage_utils as su

StageError = su.StageError

PROVIDERS = ["gemini", "openai", "elevenlabs"]
DEFAULT_VOICES = {"gemini": "Kore", "openai": "coral", "elevenlabs": "21m00Tcm4TlvDq8ikWAM"}
DEFAULT_MODELS = {"gemini": "gemini-2.5-flash-preview-tts", "openai": "gpt-4o-mini-tts",
                  "elevenlabs": "eleven_multilingual_v2"}
TEXT_MODES = ["auto", "numbered lines", "one line per shot", "plain text"]
OUTPUT_SR = 48000
LINE_OFFSET_S = 0.3

_NUMBERED = re.compile(r"^\s*(?:shot\s*)?(\d+)\s*[|:]\s*(.*?)\s*$", re.IGNORECASE)


# --------------------------------------------------------------------------- text

def _clean_lines(text):
    return [l.strip() for l in str(text or "").splitlines()
            if l.strip() and not l.strip().startswith("#")]


def parse_vo_lines(text, mode="auto", shot_count=None):
    """-> (lines, per_shot). lines: [{"shot": int|None, "text": str}].

    numbered lines: "2 | Line for shot 2" (also "shot 2: ...").
    one line per shot: line i belongs to shot i (Ad Script From Shots writes this).
    plain text: everything is one read.
    auto: numbered if every line is numbered; one-per-shot if the line count equals the
    number of shots in timing_json; otherwise plain."""
    if mode not in TEXT_MODES:
        raise StageError(f"unknown text mode {mode!r}")
    lines = _clean_lines(text)
    if not lines:
        raise StageError("the voiceover text is empty")
    matches = [_NUMBERED.match(l) for l in lines]
    numbered = all(matches)
    if mode == "numbered lines" or (mode == "auto" and numbered):
        bad = [l for l, m in zip(lines, matches) if not m]
        if bad:
            raise StageError(f"line {bad[0]!r} has no shot number (use \"shot_number | text\")")
        out = [{"shot": int(m.group(1)), "text": m.group(2)} for m in matches if m.group(2)]
        if not out:
            raise StageError("all numbered lines are empty")
        return out, True
    if mode == "one line per shot" or (mode == "auto" and shot_count and len(lines) == shot_count):
        return [{"shot": i + 1, "text": l} for i, l in enumerate(lines)], True
    return [{"shot": None, "text": " ".join(lines)}], False


def placement_times(lines, timing=None, offset_s=LINE_OFFSET_S, durations=None, gap_s=0.3):
    """Start time (s) per line. With timing: the shot's start + offset. Without timing,
    lines follow each other with gap_s between them (durations needed)."""
    starts = []
    if timing is not None:
        by_shot = {int(s["shot"]): s for s in timing["shots"]}
        for l in lines:
            if l["shot"] is None:
                starts.append(float(offset_s))
                continue
            if l["shot"] not in by_shot:
                raise StageError(f"VO line for shot {l['shot']} but timing_json has shots "
                                 f"{', '.join(map(str, sorted(by_shot)))}")
            starts.append(float(by_shot[l["shot"]]["start_s"]) + float(offset_s))
        return starts
    t = 0.0
    for i, _ in enumerate(lines):
        starts.append(t)
        t += float(durations[i] if durations else 0.0) + gap_s
    return starts


# --------------------------------------------------------------------------- decoding

def decode_pcm16(raw, channels=1):
    """Little-endian signed 16-bit PCM bytes -> float32 [C,S]."""
    raw = bytes(raw)
    if len(raw) % 2:
        raw = raw[:-1]
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    channels = max(1, int(channels))
    n = x.size // channels
    return x[:n * channels].reshape(n, channels).T.copy()


def decode_wav(raw):
    """RIFF/WAVE bytes -> (float32 [C,S], sample_rate). PCM 8/16/24/32-bit and float32.
    Tolerates streamed WAVs whose size fields are 0 or 0xFFFFFFFF."""
    raw = bytes(raw)
    if len(raw) < 12 or raw[:4] not in (b"RIFF", b"RF64") or raw[8:12] != b"WAVE":
        raise StageError("audio is not a WAV file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", raw[body:body + 16])
        elif cid == b"data":
            end = len(raw) if size in (0, 0xFFFFFFFF) or body + size > len(raw) else body + size
            data = raw[body:end]
            break
        pos = body + size + (size & 1)
    if fmt is None or data is None:
        raise StageError("WAV file has no fmt or data chunk")
    tag, ch, sr, _, _, bits = fmt
    ch = max(1, ch)
    if tag == 0xFFFE:   # WAVE_FORMAT_EXTENSIBLE: assume PCM unless 32-bit float
        tag = 3 if bits == 32 and b"\x03\x00\x00\x00\x00\x00\x10\x00" in raw[:200] else 1
    if tag == 3 and bits == 32:
        x = np.frombuffer(data[:len(data) // 4 * 4], dtype="<f4").astype(np.float32)
    elif tag == 1 and bits == 16:
        x = np.frombuffer(data[:len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 8:
        x = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(data[:len(data) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    elif tag == 1 and bits == 32:
        x = np.frombuffer(data[:len(data) // 4 * 4], dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise StageError(f"unsupported WAV format (tag {tag}, {bits}-bit)")
    n = x.size // ch
    return x[:n * ch].reshape(n, ch).T.copy(), int(sr)


def encode_wav16(wave, sr):
    """float [C,S] -> 16-bit WAV bytes (used in tests and for debugging)."""
    wave = np.atleast_2d(np.asarray(wave, np.float32))
    pcm = (np.clip(wave.T, -1, 1) * 32767).astype("<i2").tobytes()
    ch = wave.shape[0]
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, sr * ch * 2, ch * 2, 16)
    return hdr + b"data" + struct.pack("<I", len(pcm)) + pcm


def _rate_from_mime(mime, default=24000):
    m = re.search(r"rate=(\d+)", mime or "")
    return int(m.group(1)) if m else default


def decode_audio_bytes(raw, content_type="", default_sr=24000):
    if raw[:4] in (b"RIFF", b"RF64"):
        return decode_wav(raw)
    return decode_pcm16(raw), _rate_from_mime(content_type, default_sr)


# --------------------------------------------------------------------------- providers

def _speed_phrase(speed):
    if speed >= 1.15:
        return "at a brisk, quick pace"
    if speed >= 1.05:
        return "at a slightly faster than normal pace"
    if speed <= 0.85:
        return "slowly, with an unhurried pace"
    if speed <= 0.95:
        return "at a slightly slower than normal pace"
    return ""


def gemini_tts(text, voice, model, style_prompt, speed, key, post=None, timeout=180):
    post = post or su.http_post_json
    how = ", ".join(x for x in (style_prompt.strip() if style_prompt else "", _speed_phrase(speed)) if x)
    prompt = f"Read this commercial voiceover {how}:\n{text}" if how else text
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig":
                                                                  {"voiceName": voice}}}}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    resp = su.post_with_retry(lambda: post(url, body, {"x-goog-api-key": key}, timeout))
    try:
        part = next(p for p in resp["candidates"][0]["content"]["parts"]
                    if "inlineData" in p or "inline_data" in p)
    except (KeyError, IndexError, TypeError, StopIteration) as e:
        raise StageError("Gemini TTS returned no audio (check the model name: it must be a "
                         "TTS model such as gemini-2.5-flash-preview-tts)") from e
    inline = part.get("inlineData") or part.get("inline_data")
    raw = base64.b64decode(inline["data"])
    mime = inline.get("mimeType") or inline.get("mime_type") or ""
    return decode_audio_bytes(raw, mime, 24000)


def openai_tts(text, voice, model, style_prompt, speed, key, post=None, timeout=180):
    post = post or su.http_post
    body = {"model": model, "voice": voice, "input": text, "response_format": "wav",
            "speed": float(min(4.0, max(0.25, speed)))}
    if style_prompt and style_prompt.strip():
        body["instructions"] = style_prompt.strip()
    raw, ctype = su.post_with_retry(lambda: post("https://api.openai.com/v1/audio/speech", body,
                                                 {"Authorization": f"Bearer {key}"}, timeout))
    return decode_audio_bytes(raw, ctype, 24000)


def elevenlabs_tts(text, voice, model, style_prompt, speed, key, post=None, timeout=180):
    post = post or su.http_post
    url = ("https://api.elevenlabs.io/v1/text-to-speech/"
           f"{urllib.parse.quote(voice, safe='')}?output_format=pcm_44100")
    body = {"text": text, "model_id": model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75,
                               "speed": float(min(1.2, max(0.7, speed)))}}
    raw, ctype = su.post_with_retry(lambda: post(url, body, {"xi-api-key": key}, timeout))
    return decode_audio_bytes(raw, ctype or "audio/pcm;rate=44100", 44100)


SYNTH = {"gemini": gemini_tts, "openai": openai_tts, "elevenlabs": elevenlabs_tts}


def synthesize(text, provider, voice="", model="", style_prompt="", speed=1.0, key="",
               post=None):
    """One TTS call -> (float [C,S], sample_rate)."""
    if provider not in SYNTH:
        raise StageError(f"unknown TTS provider {provider!r}")
    voice = (voice or "").strip() or DEFAULT_VOICES[provider]
    model = (model or "").strip() or DEFAULT_MODELS[provider]
    try:
        wave, sr = SYNTH[provider](text, voice, model, style_prompt or "", float(speed), key, post)
    except su.HttpError as e:
        raise StageError(f"{provider} TTS failed: {e}") from e
    if wave.size == 0:
        raise StageError(f"{provider} TTS returned empty audio")
    return wave, sr


# --------------------------------------------------------------------------- assembly

def trim_silence(wave, sr, threshold_db=-45.0, pad_ms=30.0):
    """Cut leading/trailing silence (TTS often adds 0.1-0.4 s), keeping pad_ms."""
    mono = np.abs(wave).max(axis=0)
    thr = 10 ** (threshold_db / 20.0)
    idx = np.where(mono > thr)[0]
    if idx.size == 0:
        return wave[:, :0]
    pad = int(sr * pad_ms / 1000.0)
    return wave[:, max(0, idx[0] - pad):min(wave.shape[1], idx[-1] + 1 + pad)]


def place_lines(segments, starts, sr=OUTPUT_SR, min_seconds=0.0):
    """Sum stereo segments into one buffer at their start times. Returns [2,S]."""
    end = max([int(round(s * sr)) + seg.shape[1] for seg, s in zip(segments, starts)] + [0])
    n = max(end, int(round(min_seconds * sr)))
    out = np.zeros((2, n), dtype=np.float32)
    for seg, s in zip(segments, starts):
        a = int(round(s * sr))
        out[:, a:a + seg.shape[1]] += su.to_stereo(seg)
    return out


def build_voiceover(text, provider="gemini", voice="", model="", style_prompt="", speed=1.0,
                    timing_json=None, api_key="", text_mode="auto", line_offset_s=LINE_OFFSET_S,
                    trim=True, post=None):
    """-> (wave [2,S] at 48 kHz, vo_timing dict)."""
    timing = su.load_timing(timing_json)
    lines, per_shot = parse_vo_lines(text, text_mode, len(timing["shots"]) if timing else None)
    key = su.resolve_key(provider, api_key)
    segs = []
    for l in lines:
        w, sr = synthesize(l["text"], provider, voice, model, style_prompt, speed, key, post)
        w = su.to_stereo(su.resample_linear(w, sr, OUTPUT_SR))
        if trim:
            w = trim_silence(w, OUTPUT_SR)
        segs.append(w)
    durs = [s.shape[1] / OUTPUT_SR for s in segs]
    if per_shot and timing:
        starts = placement_times(lines, timing, line_offset_s)
    elif per_shot:
        starts = placement_times(lines, None, durations=durs)
    else:
        starts = [0.0]
    wave = place_lines(segs, starts, OUTPUT_SR,
                       float(timing["total_seconds"]) if timing and per_shot and "total_seconds" in timing else 0.0)
    by_shot = {int(s["shot"]): s for s in timing["shots"]} if timing else {}
    info = []
    for l, s, d in zip(lines, starts, durs):
        e = {"shot": l["shot"], "text": l["text"], "start_s": round(s, 4),
             "end_s": round(s + d, 4), "duration_s": round(d, 4)}
        if l["shot"] in by_shot:
            e["shot_start_s"] = float(by_shot[l["shot"]]["start_s"])
            e["shot_end_s"] = float(by_shot[l["shot"]]["end_s"])
        info.append(e)
    vo_timing = {"sample_rate": OUTPUT_SR, "provider": provider,
                 "placed_on_timeline": bool(per_shot and timing),
                 "total_seconds": round(wave.shape[1] / OUTPUT_SR, 4), "lines": info}
    return wave, vo_timing


# --------------------------------------------------------------------------- WSOLA

def wsola(wave, rate, sr, frame_ms=40.0, tolerance_ms=10.0):
    """Time-stretch [C,S] without changing pitch. rate > 1 = faster / shorter
    (output length = S / rate). Waveform-similarity overlap-add with a Hann window;
    the similarity search runs on a decimated mono copy so it stays fast."""
    wave = np.atleast_2d(np.asarray(wave, np.float32))
    C, S = wave.shape
    rate = float(rate)
    if rate <= 0:
        raise StageError("stretch rate must be > 0")
    n_out = max(1, int(round(S / rate)))
    if abs(rate - 1.0) < 1e-6 or S < 4:
        return wave.copy()
    N = max(16, int(sr * frame_ms / 1000.0)) // 2 * 2
    Hs = N // 2
    Ha = Hs * rate
    tol = max(1, int(sr * tolerance_ms / 1000.0))
    win = np.hanning(N).astype(np.float32)
    pad = N + tol + 2
    x = np.pad(wave, ((0, 0), (pad, pad + int(N * rate) + N)))
    mono = x.mean(axis=0)
    dec = 4 if sr >= 16000 else 1
    n_frames = int(np.ceil(n_out / Hs)) + 2
    out = np.zeros((C, n_frames * Hs + N), np.float32)
    norm = np.zeros(n_frames * Hs + N, np.float32)
    prev = pad              # input position of the previous chosen frame
    for k in range(n_frames):
        ideal = pad + int(round(k * Ha))
        if k == 0:
            pos = ideal
        else:
            target = mono[prev + Hs: prev + Hs + N][::dec]     # natural continuation
            lo = max(0, ideal - tol)
            region = mono[lo: ideal + tol + N][::dec]
            if target.size and region.size >= target.size:
                corr = np.correlate(region, target, mode="valid")
                pos = lo + int(np.argmax(corr)) * dec
            else:
                pos = ideal
        seg = x[:, pos:pos + N]
        if seg.shape[1] < N:
            seg = np.pad(seg, ((0, 0), (0, N - seg.shape[1])))
        o = k * Hs
        out[:, o:o + N] += seg * win
        norm[o:o + N] += win
        prev = pos
    norm[norm < 1e-3] = 1.0
    out = out / norm
    # the input was padded by `pad` samples; output frame 0 corresponds to input `pad`
    return out[:, :n_out].astype(np.float32)


def fit_voiceover(wave, sr, vo_timing_json, timing_json, max_stretch=0.12, tail_s=0.05):
    """Compress lines that overrun their shot by up to max_stretch (rate 1 + max_stretch).
    Returns (wave, report, new vo_timing)."""
    vo = su.load_json(vo_timing_json, "vo_timing_json")
    if not isinstance(vo, dict) or "lines" not in vo:
        raise StageError("vo_timing_json is not Ad Voiceover output")
    timing = su.load_timing(timing_json)
    if timing is None:
        raise StageError("timing_json is required (connect Ad Timeline's timing_json)")
    by_shot = {int(s["shot"]): s for s in timing["shots"]}
    wave = np.atleast_2d(np.asarray(wave, np.float32))
    segs, starts, report, lines = [], [], [], []
    ordered = sorted(vo["lines"], key=lambda l: l["start_s"])
    for a, b in zip(ordered, ordered[1:]):
        if a["end_s"] > b["start_s"] + 1e-3:
            report.append(f"WARNING: lines for shots {a.get('shot')} and {b.get('shot')} overlap "
                          f"in the voiceover by {a['end_s'] - b['start_s']:.2f} s; they cannot be "
                          "separated, so the overlapping part is left as is")
    for l in vo["lines"]:
        a = int(round(l["start_s"] * sr))
        nxt = [o["start_s"] for o in vo["lines"] if o["start_s"] > l["start_s"] + 1e-6]
        end = min([l["end_s"]] + nxt)          # never copy the start of the next line
        b = min(wave.shape[1], int(round(end * sr)))
        seg = wave[:, a:b]
        dur = seg.shape[1] / sr
        new = dict(l)
        shot = l.get("shot")
        if shot is None or int(shot) not in by_shot:
            segs.append(seg)
            starts.append(l["start_s"])
            lines.append(new)
            report.append(f"line {l.get('text', '')[:30]!r}: no shot, left as is")
            continue
        avail = float(by_shot[int(shot)]["end_s"]) - float(l["start_s"]) - tail_s
        tag = f"shot {shot}"
        if avail <= 0:
            report.append(f"{tag}: starts after its shot ends; OVERRUN {dur:.2f} s")
        elif dur <= avail:
            report.append(f"{tag}: fits ({dur:.2f} s of {avail:.2f} s)")
        else:
            need = dur / avail
            rate = min(need, 1.0 + float(max_stretch))
            seg = wsola(seg, rate, sr)
            nd = seg.shape[1] / sr
            if need <= 1.0 + float(max_stretch) + 1e-9:
                report.append(f"{tag}: sped up {100 * (rate - 1):.1f}% ({dur:.2f} -> {nd:.2f} s)")
            else:
                report.append(f"{tag}: OVERRUN by {nd - avail:.2f} s even at +{100 * (rate - 1):.0f}% "
                              f"({dur:.2f} -> {nd:.2f} s, shot has {avail:.2f} s); shorten the line "
                              "or lengthen the shot")
            new["stretch_rate"] = round(rate, 4)
            new["end_s"] = round(l["start_s"] + nd, 4)
            new["duration_s"] = round(nd, 4)
        segs.append(seg)
        starts.append(l["start_s"])
        lines.append(new)
    out = place_lines(segs, starts, sr, wave.shape[1] / sr)
    vo2 = dict(vo, lines=lines)
    return out, "\n".join(report), vo2


def vo_timing_text(vo_timing):
    return json.dumps(vo_timing, indent=2)
