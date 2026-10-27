import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shot_planner as sp  # noqa: E402

REPLY = {"title": "Dragon Coffee", "style": "cinematic", "shots": [
    {"id": 7, "description": "Office at dawn", "camera": "wide", "duration": "4.6",
     "image_prompt": "An empty office at dawn", "video_prompt": "Slow push-in"},
    {"description": "Dragon peeks out", "camera": "close-up"},
]}


class FakeResp:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Tests(unittest.TestCase):
    def test_offline_counts_and_fields(self):
        brief = "One. Two. Three. Four. Five. Six. Seven."
        for n in (1, 3, 7, 10):
            p = sp.plan_shots(brief, num_shots=n)
            self.assertEqual(len(p["shots"]), n)
            for i, s in enumerate(p["shots"], 1):
                self.assertEqual(s["id"], i)
                self.assertTrue(s["image_prompt"] and s["video_prompt"])

    def test_extract_json_with_fence_and_chatter(self):
        txt = 'Sure! Here you go:\n```json\n' + json.dumps(REPLY) + '\n```\nEnjoy {not json}'
        self.assertEqual(sp.extract_json(txt)["title"], "Dragon Coffee")
        self.assertEqual(sp.extract_json('x {"a": "b}c"} y')["a"], "b}c")

    def test_normalize_fills_and_renumbers(self):
        p = sp.normalize(REPLY, 6, "cinematic", 5)
        self.assertEqual([s["id"] for s in p["shots"]], [1, 2])
        self.assertEqual(p["shots"][0]["duration"], 5)
        self.assertEqual(p["shots"][1]["image_prompt"], "Dragon peeks out")
        self.assertEqual(p["shots"][1]["duration"], 5)

    def _run_provider(self, provider, body):
        seen = {}

        def fake_urlopen(req, timeout=0):
            seen["url"] = req.full_url
            seen["headers"] = {k.lower(): v for k, v in req.header_items()}
            seen["payload"] = json.loads(req.data)
            return FakeResp(body)

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            p = sp.plan_shots("A brief.", provider=provider, api_key="k-123", num_shots=2)
        return p, seen

    def test_anthropic(self):
        p, seen = self._run_provider("anthropic", {"content": [{"type": "text", "text": json.dumps(REPLY)}]})
        self.assertIn("api.anthropic.com", seen["url"])
        self.assertEqual(seen["headers"]["x-api-key"], "k-123")
        self.assertEqual(seen["payload"]["model"], sp.DEFAULT_MODELS["anthropic"])
        self.assertEqual(len(p["shots"]), 2)

    def test_openai(self):
        p, seen = self._run_provider("openai", {"choices": [{"message": {"content": json.dumps(REPLY)}}]})
        self.assertEqual(seen["headers"]["authorization"], "Bearer k-123")
        self.assertEqual(p["title"], "Dragon Coffee")

    def test_gemini(self):
        p, seen = self._run_provider("gemini", {"candidates": [{"content": {"parts": [{"text": json.dumps(REPLY)}]}}]})
        self.assertIn("generateContent", seen["url"])
        self.assertEqual(seen["headers"]["x-goog-api-key"], "k-123")
        self.assertEqual(len(p["shots"]), 2)

    def test_missing_key_message(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(sp.ShotPlannerError) as cm:
                sp.plan_shots("A brief.", provider="anthropic")
        self.assertIn("ANTHROPIC_API_KEY", str(cm.exception))

    def test_env_key_used(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            _, seen = self._run_provider("openai", {"choices": [{"message": {"content": json.dumps(REPLY)}}]})
        # explicit api_key wins over env
        self.assertEqual(seen["headers"]["authorization"], "Bearer k-123")

    def test_empty_brief(self):
        with self.assertRaises(sp.ShotPlannerError):
            sp.plan_shots("   ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
