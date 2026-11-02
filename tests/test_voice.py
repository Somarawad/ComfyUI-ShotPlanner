"""Tests for voice.py (Ad Voiceover, Ad Voice Fit). All HTTP is mocked."""

import base64
import json
import os
import struct
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stage_utils as su  # noqa: E402
import voice as vo  # noqa: E402

TIMING = {"fps": 24.0, "total_frames": 216, "total_seconds": 9.0, "shots": [
    {"shot": 1, "start_frame": 0, "end_frame": 72, "start_s": 0.0, "end_s": 3.0, "duration_s": 3.0},
    {"shot": 2, "start_frame": 72, "end_frame": 120, "start_s": 3.0, "end_s": 5.0, "duration_s": 2.0},
    {"shot": 3, "start_frame": 120, "end_frame": 216, "start_s": 5.0, "end_s": 9.0, "duration_s": 4.0},
]}


def tone(seconds, sr, freq=220.0, amp=0.5):
    t = np.arange(int(round(seconds * sr))) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None]


def pcm16(wave):
    return (np.clip(wave[0], -1, 1) * 32767).astype("<i2").tobytes()


_saved_sleep = su.sleep


def setUpModule():
    su.sleep = lambda s: None


def tearDownModule():
    su.sleep = _saved_sleep


class ParseTests(unittest.TestCase):
    def test_numbered(self):
        lines, per = vo.parse_vo_lines("1 | Hello there.\n# note\n\n3 | Novara. Run free.")
        self.assertTrue(per)
        self.assertEqual(lines, [{"shot": 1, "text": "Hello there."},
                                 {"shot": 3, "text": "Novara. Run free."}])
        lines, _ = vo.parse_vo_lines("Shot 2: Go")
        self.assertEqual(lines[0]["shot"], 2)

    def test_one_per_shot_and_plain(self):
        text = "Morning.\nThe bridge.\nNovara."
        lines, per = vo.parse_vo_lines(text, "auto", shot_count=3)
        self.assertTrue(per)
        self.assertEqual([l["shot"] for l in lines], [1, 2, 3])
        lines, per = vo.parse_vo_lines(text, "auto", shot_count=5)
        self.assertFalse(per)
        self.assertEqual(lines[0]["text"], "Morning. The bridge. Novara.")
        lines, per = vo.parse_vo_lines("1 | a\n2 | b", "plain text")
        self.assertFalse(per)

    def test_errors(self):
        with self.assertRaises(vo.StageError):
            vo.parse_vo_lines("   \n# only a comment")
        with self.assertRaises(vo.StageError):
            vo.parse_vo_lines("1 | ok\nno number", "numbered lines")
        with self.assertRaises(vo.StageError):
            vo.parse_vo_lines("x", "weird mode")

    def test_placement_offsets(self):
        lines = [{"shot": 1, "text": "a"}, {"shot": 3, "text": "b"}]
        self.assertEqual(vo.placement_times(lines, TIMING), [0.3, 5.3])
        self.assertEqual(vo.placement_times(lines, TIMING, offset_s=0.5), [0.5, 5.5])
        self.assertEqual(vo.placement_times(lines, None, durations=[1.0, 2.0], gap_s=0.3), [0.0, 1.3])
        with self.assertRaises(vo.StageError):
            vo.placement_times([{"shot": 7, "text": "x"}], TIMING)


class DecodeTests(unittest.TestCase):
    def test_pcm16(self):
        w = vo.decode_pcm16(struct.pack("<4h", 0, 16384, -32768, 32767))
        np.testing.assert_allclose(w[0], [0, 0.5, -1, 32767 / 32768], atol=1e-6)
        st = vo.decode_pcm16(struct.pack("<4h", 1, 2, 3, 4), channels=2)
        self.assertEqual(st.shape, (2, 2))
        self.assertEqual(vo.decode_pcm16(b"\x00\x00\x01").shape, (1, 1))     # odd byte dropped

    def test_wav_roundtrip(self):
        w = tone(0.1, 24000)
        dec, sr = vo.decode_wav(vo.encode_wav16(w, 24000))
        self.assertEqual(sr, 24000)
        np.testing.assert_allclose(dec, w, atol=1e-4)
        stereo = np.concatenate([w, -w])
        dec, _ = vo.decode_wav(vo.encode_wav16(stereo, 16000))
        self.assertEqual(dec.shape, (2, w.shape[1]))

    def test_streamed_wav_size(self):
        raw = bytearray(vo.encode_wav16(tone(0.05, 8000), 8000))
        raw[40:44] = b"\xff\xff\xff\xff"                    # data size unknown (streamed)
        dec, _ = vo.decode_wav(bytes(raw))
        self.assertEqual(dec.shape[1], 400)

    def test_float_and_24bit(self):
        x = np.array([0.0, 0.25, -0.5], np.float32)
        def wav(tag, bits, payload):
            ba = bits // 8
            fmt = struct.pack("<IHHIIHH", 16, tag, 1, 8000, 8000 * ba, ba, bits)
            return b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVEfmt " + fmt + \
                b"data" + struct.pack("<I", len(payload)) + payload
        dec, _ = vo.decode_wav(wav(3, 32, x.astype("<f4").tobytes()))
        np.testing.assert_allclose(dec[0], x)
        ints = [0, 1 << 21, -(1 << 22)]
        payload = b"".join(struct.pack("<i", v)[:3] for v in ints)
        dec, _ = vo.decode_wav(wav(1, 24, payload))
        np.testing.assert_allclose(dec[0], [0, 0.25, -0.5])
        with self.assertRaises(vo.StageError):
            vo.decode_wav(b"OggS....")

    def test_rate_from_mime(self):
        raw = pcm16(tone(0.01, 24000))
        _, sr = vo.decode_audio_bytes(raw, "audio/L16;codec=pcm;rate=24000")
        self.assertEqual(sr, 24000)
        _, sr = vo.decode_audio_bytes(raw, "", 44100)
        self.assertEqual(sr, 44100)


class ProviderTests(unittest.TestCase):
    def test_gemini(self):
        audio_b64 = base64.b64encode(pcm16(tone(0.5, 24000))).decode()
        resp = {"candidates": [{"content": {"parts": [{"inlineData": {
            "mimeType": "audio/L16;codec=pcm;rate=24000", "data": audio_b64}}]}}]}
        post = mock.Mock(return_value=resp)
        w, sr = vo.synthesize("Hello", "gemini", "", "", "warm, Italian-accented", 1.2, "GK", post)
        self.assertEqual((sr, w.shape[1]), (24000, 12000))
        url, body, headers, _ = post.call_args[0]
        self.assertIn("gemini-2.5-flash-preview-tts:generateContent", url)
        self.assertEqual(headers, {"x-goog-api-key": "GK"})
        cfg = body["generationConfig"]
        self.assertEqual(cfg["responseModalities"], ["AUDIO"])
        self.assertEqual(cfg["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"], "Kore")
        text = body["contents"][0]["parts"][0]["text"]
        self.assertIn("warm, Italian-accented", text)
        self.assertIn("brisk", text)
        self.assertTrue(text.endswith("Hello"))

    def test_gemini_no_audio(self):
        post = mock.Mock(return_value={"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
        with self.assertRaises(vo.StageError):
            vo.synthesize("x", "gemini", key="k", post=post)

    def test_openai(self):
        post = mock.Mock(return_value=(vo.encode_wav16(tone(0.25, 24000), 24000), "audio/wav"))
        w, sr = vo.synthesize("Hi", "openai", "", "", "calm", 1.1, "OK", post)
        self.assertEqual((sr, w.shape[1]), (24000, 6000))
        url, body, headers, _ = post.call_args[0]
        self.assertEqual(url, "https://api.openai.com/v1/audio/speech")
        self.assertEqual(headers, {"Authorization": "Bearer OK"})
        self.assertEqual((body["model"], body["voice"], body["response_format"]),
                         ("gpt-4o-mini-tts", "coral", "wav"))
        self.assertEqual((body["instructions"], body["speed"]), ("calm", 1.1))

    def test_elevenlabs(self):
        post = mock.Mock(return_value=(pcm16(tone(0.2, 44100)), "application/octet-stream"))
        w, sr = vo.synthesize("Hi", "elevenlabs", "voice id/1", "", "", 1.5, "EK", post)
        self.assertEqual((sr, w.shape[1]), (44100, 8820))
        url, body, headers, _ = post.call_args[0]
        self.assertIn("/v1/text-to-speech/voice%20id%2F1?output_format=pcm_44100", url)
        self.assertEqual(headers, {"xi-api-key": "EK"})
        self.assertEqual(body["model_id"], "eleven_multilingual_v2")
        self.assertEqual(body["voice_settings"]["speed"], 1.2)       # clamped to the API range

    def test_retry_then_error(self):
        post = mock.Mock(side_effect=[su.HttpError("busy", 503),
                                      (vo.encode_wav16(tone(0.1, 8000), 8000), "audio/wav")])
        w, _ = vo.synthesize("Hi", "openai", key="k", post=post)
        self.assertEqual(post.call_count, 2)
        post = mock.Mock(side_effect=su.HttpError("bad request", 400))
        with self.assertRaises(vo.StageError) as cm:
            vo.synthesize("Hi", "openai", key="SECRETKEY", post=post)
        self.assertNotIn("SECRETKEY", str(cm.exception))
        with self.assertRaises(vo.StageError):
            vo.synthesize("Hi", "nope", key="k")

    def test_missing_key(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaises(vo.StageError) as cm:
                vo.build_voiceover("hi", "openai")
            self.assertIn("OPENAI_API_KEY", str(cm.exception))


class BuildTests(unittest.TestCase):
    def fake_openai(self, seconds_per_call):
        it = iter(seconds_per_call)
        def post(url, body, headers, timeout):
            s = next(it)
            w = np.concatenate([np.zeros((1, 2400), np.float32), tone(s, 24000)], axis=1)  # 0.1 s lead
            return vo.encode_wav16(w, 24000), "audio/wav"
        return post

    def test_placed_on_timeline(self):
        wave, t = vo.build_voiceover("1 | First line\n3 | Last line", "openai", timing_json=json.dumps(TIMING),
                                     api_key="k", post=self.fake_openai([1.0, 2.0]))
        self.assertEqual(wave.shape[0], 2)
        self.assertEqual(wave.shape[1], 9 * 48000)                 # padded to the spot length
        self.assertTrue(t["placed_on_timeline"])
        a, b = t["lines"]
        self.assertEqual((a["shot"], a["start_s"], b["start_s"]), (1, 0.3, 5.3))
        self.assertAlmostEqual(a["duration_s"], 1.0 + 0.03, delta=0.01)   # 0.1 s lead trimmed to the 30 ms pad
        self.assertEqual((b["shot_start_s"], b["shot_end_s"]), (5.0, 9.0))
        first = np.where(np.abs(wave[0]) > 0.01)[0][0] / 48000
        self.assertAlmostEqual(first, 0.3 + 0.03, delta=0.01)

    def test_sequential_and_plain(self):
        wave, t = vo.build_voiceover("1 | a\n2 | b", "openai", api_key="k",
                                     post=self.fake_openai([0.5, 0.5]))
        self.assertFalse(t["placed_on_timeline"])
        self.assertAlmostEqual(t["lines"][1]["start_s"], t["lines"][0]["duration_s"] + 0.3, places=3)
        wave, t = vo.build_voiceover("Just one read.", "openai", api_key="k",
                                     post=self.fake_openai([0.5]), trim=False)
        self.assertEqual(len(t["lines"]), 1)
        self.assertAlmostEqual(wave.shape[1] / 48000, 0.6, delta=0.01)

    def test_trim_silence(self):
        w = np.concatenate([np.zeros((2, 4800)), np.ones((2, 480)) * 0.5, np.zeros((2, 4800))], 1)
        out = vo.trim_silence(w, 48000, pad_ms=10)
        self.assertEqual(out.shape[1], 480 + 2 * 480)
        self.assertEqual(vo.trim_silence(np.zeros((2, 100)), 48000).shape[1], 0)


class WsolaTests(unittest.TestCase):
    def test_length_within_1_percent(self):
        sr = 16000
        x = tone(2.0, sr, 330) + tone(2.0, sr, 523, 0.2)
        for rate in (0.8, 0.9, 1.06, 1.12, 1.3):
            y = vo.wsola(x, rate, sr)
            want = x.shape[1] / rate
            self.assertLess(abs(y.shape[1] - want) / want, 0.01, rate)

    def test_pitch_and_level_kept(self):
        sr = 16000
        x = tone(2.0, sr, 400)
        y = vo.wsola(x, 1.12, sr)
        mid = y[0, 2000:-2000]
        spec = np.abs(np.fft.rfft(mid * np.hanning(mid.size)))
        peak_hz = np.argmax(spec) * sr / mid.size
        self.assertAlmostEqual(peak_hz, 400, delta=8)
        self.assertAlmostEqual(np.sqrt((mid ** 2).mean()), 0.5 / np.sqrt(2), delta=0.05)

    def test_stereo_and_identity(self):
        x = np.concatenate([tone(0.5, 8000), tone(0.5, 8000, 300)])
        self.assertEqual(vo.wsola(x, 1.1, 8000).shape[0], 2)
        np.testing.assert_array_equal(vo.wsola(x, 1.0, 8000), x)
        with self.assertRaises(vo.StageError):
            vo.wsola(x, 0, 8000)


class FitTests(unittest.TestCase):
    def make(self, line_secs):
        sr = 16000
        segs, lines = [], []
        for shot, secs in line_secs:
            start = TIMING["shots"][shot - 1]["start_s"] + 0.3
            segs.append(tone(secs, sr, 300))
            lines.append({"shot": shot, "text": f"line {shot}", "start_s": start,
                          "end_s": start + secs, "duration_s": secs})
        wave = vo.place_lines(segs, [l["start_s"] for l in lines], sr, 9.0)
        return wave, sr, json.dumps({"sample_rate": sr, "lines": lines})

    def test_fits_stretch_overrun(self):
        # shot 1 has 3 - 0.3 - 0.05 = 2.65 s; shot 2: 1.65 s; shot 3: 3.65 s
        wave, sr, vt = self.make([(1, 2.0), (2, 1.8), (3, 5.0)])
        out, report, vt2 = vo.fit_voiceover(wave, sr, vt, json.dumps(TIMING), 0.12)
        lines = report.split("\n")
        self.assertIn("fits", lines[0])
        self.assertIn("sped up", lines[1])
        self.assertIn("OVERRUN", lines[2])
        l2 = vt2["lines"][1]
        self.assertLessEqual(l2["end_s"], 5.0 - 0.05 + 0.01)
        self.assertAlmostEqual(vt2["lines"][2]["stretch_rate"], 1.12, places=4)
        self.assertEqual(out.shape[1], wave.shape[1])

    def test_overlap_warning(self):
        wave, sr, vt = self.make([(1, 3.5), (2, 1.0)])         # line 1 runs into line 2
        out, report, _ = vo.fit_voiceover(wave, sr, vt, json.dumps(TIMING), 0.12)
        self.assertTrue(report.startswith("WARNING: lines for shots 1 and 2 overlap"))
        self.assertEqual(out.shape, wave.shape)

    def test_needs_timing(self):
        wave, sr, vt = self.make([(1, 1.0)])
        with self.assertRaises(vo.StageError):
            vo.fit_voiceover(wave, sr, vt, None)
        with self.assertRaises(vo.StageError):
            vo.fit_voiceover(wave, sr, "{}", json.dumps(TIMING))


if __name__ == "__main__":
    unittest.main()
