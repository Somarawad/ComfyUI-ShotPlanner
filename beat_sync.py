"""Ad Beat Sync: tempo / beat detection and beat-aligned cut points. numpy only.

1. Onset strength: spectral flux of the log-magnitude STFT (mono, 22.05 kHz, hop 64).
2. Tempo: autocorrelation of the onset envelope, weighted by a log-normal prior around
   120 bpm (to avoid octave errors), refined with a 4-pulse comb on the autocorrelation.
3. Beats: the phase of a regular beat grid that collects the most onset energy.
4. Downbeat: the beat of the 4/4 bar that is accented most; music_offset_s trims the
   music so that downbeat lands on frame 0.
5. Cuts: dynamic programming over beat-grid points (every n beats) so each shot is as
   close as possible to its preferred duration and at least min_shot_s long.
"""

import math

import numpy as np

try:
    from . import stage_utils as su
except ImportError:  # imported as a top-level module (tests)
    import stage_utils as su

StageError = su.StageError

ANALYSIS_SR = 22050
N_FFT = 1024
HOP = 64
BPM_MIN, BPM_MAX = 60.0, 200.0
BEATS_PER_CUT = ["auto", "1", "2", "4", "8"]
# the log-flux peaks as soon as an onset enters the analysis window, i.e. about one
# window (minus two hops) before the frame's start time catches up with it
ONSET_LATENCY_S = (N_FFT - 2 * HOP) / ANALYSIS_SR


def mono(wave):
    wave = np.atleast_2d(np.asarray(wave, np.float32))
    return wave.mean(axis=0)


def onset_envelope(wave, sr):
    """-> (envelope, frames per second). wave: [C,S] or [S]."""
    x = mono(wave)
    if sr != ANALYSIS_SR:
        x = su.resample_linear(x[None], sr, ANALYSIS_SR)[0]
    if x.size < N_FFT * 4:
        raise StageError("the music is too short to detect a tempo (need a few seconds)")
    n_frames = 1 + (x.size - N_FFT) // HOP
    win = np.hanning(N_FFT).astype(np.float32)
    mags = np.empty((n_frames, N_FFT // 2 + 1), np.float32)
    chunk = 2048
    for c0 in range(0, n_frames, chunk):
        idx = np.arange(c0, min(n_frames, c0 + chunk))[:, None] * HOP + np.arange(N_FFT)[None]
        mags[c0:c0 + idx.shape[0]] = np.abs(np.fft.rfft(x[idx] * win, axis=1))
    logm = np.log1p(100.0 * mags)
    flux = np.maximum(0.0, np.diff(logm, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux]).astype(np.float64)
    rate = ANALYSIS_SR / HOP
    # remove the slowly varying part (crescendos) and keep the peaks
    k = max(1, int(rate * 0.4))
    local = np.convolve(flux, np.ones(k) / k, mode="same")
    env = np.maximum(0.0, flux - local)
    if env.max() <= 1e-9:
        raise StageError("no rhythmic onsets found in the music (silence?)")
    return env / env.max(), rate


def _interp(arr, pos):
    pos = np.asarray(pos, np.float64)
    return np.interp(pos, np.arange(arr.size), arr, left=0.0, right=0.0)


def estimate_tempo(env, rate, bpm_min=BPM_MIN, bpm_max=BPM_MAX, prior_bpm=120.0):
    """-> (bpm, period in envelope frames)."""
    e = env - env.mean()
    n = e.size
    f = np.fft.rfft(e, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf / (acf[0] + 1e-12)
    lag_min = int(math.floor(60.0 * rate / bpm_max))
    lag_max = min(n - 1, int(math.ceil(60.0 * rate / bpm_min)))
    if lag_max <= lag_min + 2:
        raise StageError("the music is too short for tempo detection")
    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * rate / lags
    prior = np.exp(-0.5 * (np.log2(bpms / prior_bpm) / 1.0) ** 2)
    p0 = float(lags[int(np.argmax(np.maximum(acf[lags], 0) * prior))])
    # refine with a comb over the first four multiples of the period
    cand = np.arange(p0 - 2.0, p0 + 2.0, 0.01)
    ks = np.arange(1, 5)
    usable = ks[ks * (p0 + 2) < n - 1]
    if usable.size == 0:
        usable = np.array([1])
    scores = np.array([_interp(acf, usable * c).sum() for c in cand])
    period = float(cand[int(np.argmax(scores))])
    return 60.0 * rate / period, period


def beat_phase(env, period):
    """Phase (frames) of the regular beat grid that collects the most onset energy."""
    n_beats = int((env.size - 1) / period)
    best, best_s = 0.0, -1.0
    for ph in np.arange(0.0, period, 0.25):
        s = _interp(env, ph + period * np.arange(n_beats)).sum()
        if s > best_s:
            best, best_s = float(ph), s
    return best


def detect_beats(wave, sr, beats_per_bar=4):
    """-> dict(bpm, period_s, beat_times (s), downbeat_index, downbeat_s)."""
    env, rate = onset_envelope(wave, sr)
    bpm, period = estimate_tempo(env, rate)
    ph = beat_phase(env, period)
    n_beats = int((env.size - 1 - ph) / period) + 1
    pos = ph + period * np.arange(n_beats)
    strength = np.array([env[max(0, int(round(p)) - 2):int(round(p)) + 3].max() for p in pos])
    times = pos / rate + ONSET_LATENCY_S
    keep = times >= -0.02
    times, strength = np.maximum(times[keep], 0.0), strength[keep]
    if times.size == 0:
        raise StageError("no beats detected")
    acc = [strength[p::beats_per_bar].mean() if strength[p::beats_per_bar].size else 0
           for p in range(min(beats_per_bar, times.size))]
    down = int(np.argmax(acc))
    return {"bpm": float(bpm), "period_s": float(period / rate),
            "beat_times": [float(t) for t in times], "downbeat_index": down,
            "downbeat_s": float(times[down])}


# --------------------------------------------------------------------------- cut planning

def preferred_durations(shot_count, target_total_s, shot_list_json=None):
    """Preferred seconds per shot, scaled to add up to target_total_s."""
    shot_count = int(shot_count)
    if shot_count < 1:
        raise StageError("shot_count must be at least 1")
    pref = [1.0] * shot_count
    plan = su.load_shot_list(shot_list_json)
    if plan:
        for i, s in enumerate(plan["shots"][:shot_count]):
            try:
                d = float(s.get("duration", 0))
            except (TypeError, ValueError, AttributeError):
                d = 0
            if d > 0:
                pref[i] = d
        known = [p for i, p in enumerate(pref) if i < len(plan["shots"])]
        fill = (sum(known) / len(known)) if known else 1.0
        pref = [p if i < len(plan["shots"]) else fill for i, p in enumerate(pref)]
    total = sum(pref)
    return [p * float(target_total_s) / total for p in pref]


def plan_cuts(grid_s, shot_count, end_index, prefs, min_shot_s):
    """DP: choose shot_count segments on grid points 0 = g[0] < ... < g[end_index].
    Returns the list of grid indices (len shot_count + 1) or None if impossible."""
    S, J = int(shot_count), int(end_index)
    INF = float("inf")
    cost = np.full((S + 1, J + 1), INF)
    back = np.zeros((S + 1, J + 1), dtype=int)
    cost[0, 0] = 0.0
    g = np.asarray(grid_s, np.float64)
    for i in range(1, S + 1):
        for j in range(1, J + 1):
            best, arg = INF, -1
            for jp in range(0, j):
                if cost[i - 1, jp] == INF:
                    continue
                d = g[j] - g[jp]
                if d < min_shot_s - 1e-6:
                    continue
                c = cost[i - 1, jp] + (d - prefs[i - 1]) ** 2
                if c < best:
                    best, arg = c, jp
            cost[i, j], back[i, j] = best, arg
    if cost[S, J] == INF:
        return None
    idx = [J]
    for i in range(S, 0, -1):
        idx.append(int(back[i, idx[-1]]))
    return idx[::-1]


def beat_cut_durations(period_s, shot_count, target_total_s, min_shot_s=1.5,
                       beats_per_cut="auto", prefs=None, fps=24.0):
    """Shot durations whose cuts all land on beats (grid every n beats from the first
    downbeat at 0 s). Returns dict(durations, cut_times, beats_per_cut, total_seconds)."""
    S = int(shot_count)
    T = float(target_total_s)
    if T <= 0 or period_s <= 0:
        raise StageError("target_total_s and the beat period must be > 0")
    prefs = list(prefs) if prefs else [T / S] * S
    if S * float(min_shot_s) > T + 1e-6:
        raise StageError(f"{S} shots of at least {min_shot_s:g} s do not fit in {T:g} s; lower "
                         "min_shot_s or shot_count, or raise target_total_s")
    if str(beats_per_cut) == "auto":
        avg = T / S
        lim = max(float(min_shot_s), 0.5 * avg)
        options = [n for n in (8, 4, 2, 1) if n * period_s <= lim + 1e-9] or [1]
        options += [n for n in (4, 2, 1) if n < options[-1]]
    else:
        options = [int(beats_per_cut)]
    last_err = ""
    for n in options:
        step = n * period_s
        J = int(round(T / step))
        if J < S:
            last_err = (f"{S} shots do not fit in {J * step:.2f} s when cutting every {n} beats "
                        f"({step:.2f} s)")
            continue
        grid = [j * step for j in range(J + 1)]
        idx = plan_cuts(grid, S, J, prefs, float(min_shot_s))
        if idx is None:
            last_err = (f"no way to place {S} cuts every {n} beats ({step:.2f} s) in "
                        f"{J * step:.2f} s with shots of at least {min_shot_s:g} s")
            continue
        cuts = [grid[i] for i in idx]
        frames = [int(round(c * float(fps))) for c in cuts]
        durs = [(b - a) / float(fps) for a, b in zip(frames, frames[1:])]
        if min(np.diff(frames)) < 1:
            last_err = "a shot would be shorter than one frame"
            continue
        return {"durations": durs, "cut_times": cuts, "beats_per_cut": n,
                "total_seconds": frames[-1] / float(fps)}
    raise StageError(last_err + "; try beats_per_cut auto / a smaller value or fewer shots")


def format_durations(durs):
    return ", ".join(f"{d:.3f}".rstrip("0").rstrip(".") for d in durs)


def beat_sync(wave, sr, shot_count, target_total_s, min_shot_s=1.5, beats_per_cut="auto",
              shot_list_json=None, fps=24.0):
    """Full analysis. Returns dict(durations (string), durations_list, bpm, beat_times,
    music_offset_s, total_seconds, beats_per_cut, cut_times)."""
    beats = detect_beats(wave, sr)
    offset = beats["downbeat_s"]
    aligned = [round(t - offset, 4) for t in beats["beat_times"] if t >= offset - 1e-6]
    prefs = preferred_durations(shot_count, target_total_s, shot_list_json)
    cuts = beat_cut_durations(beats["period_s"], shot_count, target_total_s, min_shot_s,
                              beats_per_cut, prefs, fps)
    return {"durations": format_durations(cuts["durations"]),
            "durations_list": cuts["durations"], "bpm": round(beats["bpm"], 2),
            "period_s": beats["period_s"], "music_offset_s": round(offset, 4),
            "beat_times": [round(t, 4) for t in beats["beat_times"]],
            "aligned_beat_times": aligned, "cut_times": [round(c, 4) for c in cuts["cut_times"]],
            "beats_per_cut": cuts["beats_per_cut"], "total_seconds": cuts["total_seconds"]}
