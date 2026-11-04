"""Tests for stage_utils.py (parsing, keys, HTTP with a mocked urlopen, resampling)."""

import io
import json
import os
import sys
import unittest
import urllib.error
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stage_utils as su  # noqa: E402


class FakeResp(io.BytesIO):
    def __init__(self, data, ctype="application/json"):
        super().__init__(data)
        self.headers = {"Content-Type": ctype}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class ParseTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(su.parse_number_list("3, 2.5 4s;1"), [3.0, 2.5, 4.0, 1.0])
        self.assertEqual(su.parse_number_list(""), [])
        self.assertEqual(su.parse_int_list("81, 49"), [81, 49])
        for bad in ("3, x", "0", "-1"):
            with self.assertRaises(su.StageError):
                su.parse_number_list(bad)
        with self.assertRaises(su.StageError):
            su.parse_int_list("2.5")

    def test_json_loaders(self):
        self.assertIsNone(su.load_json("  ", "x"))
        self.assertEqual(su.load_shot_list('{"shots": []}'), {"shots": []})
        with self.assertRaises(su.StageError):
            su.load_shot_list('{"title": 1}')
        with self.assertRaises(su.StageError):
            su.load_timing('{"shots": []}')
        with self.assertRaises(su.StageError):
            su.load_json("{bad", "verdicts_json")

    def test_extract_json_object(self):
        self.assertEqual(su.extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(su.extract_json_object('Here: {"a": "}{", "b": {"c": 2}} done'),
                         {"a": "}{", "b": {"c": 2}})
        with self.assertRaises(su.StageError):
            su.extract_json_object("nothing")


class KeyTests(unittest.TestCase):
    def test_resolve(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "g2"}):
            self.assertEqual(su.resolve_key("gemini"), "g2")
            self.assertEqual(su.resolve_key("gemini", " typed "), "typed")
        with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}):
            with self.assertRaises(su.StageError) as cm:
                su.resolve_key("elevenlabs")
            self.assertIn("ELEVENLABS_API_KEY", str(cm.exception))


class HttpTests(unittest.TestCase):
    def test_post_json(self):
        with mock.patch("urllib.request.urlopen", return_value=FakeResp(b'{"ok": true}')) as m:
            self.assertEqual(su.http_post_json("https://x.test/a", {"q": 1}, {"k": "SECRET"}), {"ok": True})
        req = m.call_args[0][0]
        self.assertEqual(json.loads(req.data), {"q": 1})
        self.assertEqual(req.get_header("Content-type"), "application/json")

    def test_http_error_hides_key(self):
        err = urllib.error.HTTPError("https://x.test/a", 503, "busy", {}, io.BytesIO(b"overloaded"))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(su.HttpError) as cm:
                su.http_post("https://x.test/a", {}, {"x-api-key": "SECRET"})
        self.assertEqual(cm.exception.status, 503)
        self.assertIn("overloaded", str(cm.exception))
        self.assertNotIn("SECRET", str(cm.exception))

    def test_url_error_and_bad_json(self):
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")):
            with self.assertRaises(su.HttpError):
                su.http_post("https://x.test/a", b"raw")
        with mock.patch("urllib.request.urlopen", return_value=FakeResp(b"<html>")):
            with self.assertRaises(su.HttpError):
                su.http_post_json("https://x.test/a", {})

    def test_retry(self):
        calls = []
        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise su.HttpError("busy", 429)
            return "ok"
        with mock.patch.object(su, "sleep", lambda s: None):
            self.assertEqual(su.post_with_retry(fn), "ok")
            with self.assertRaises(su.HttpError):
                su.post_with_retry(mock.Mock(side_effect=su.HttpError("bad", 400)))


class AudioTests(unittest.TestCase):
    def test_resample_length_and_tone(self):
        sr = 24000
        t = np.arange(sr) / sr
        x = np.sin(2 * np.pi * 440 * t).astype(np.float32)[None]
        y = su.resample_linear(x, sr, 48000)
        self.assertEqual(y.shape, (1, 48000))
        spec = np.abs(np.fft.rfft(y[0]))
        self.assertAlmostEqual(np.argmax(spec) * 48000 / y.shape[1], 440, delta=2)
        self.assertEqual(su.resample_linear(x, sr, 8000).shape, (1, 8000))
        self.assertEqual(su.to_stereo(x).shape, (2, sr))
        self.assertEqual(su.to_stereo(np.zeros((3, 5))).shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
