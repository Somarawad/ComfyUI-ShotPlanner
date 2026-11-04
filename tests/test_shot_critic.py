import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shot_critic as sc  # noqa: E402

SHOTS = [{"image_prompt": "A barista hands over a NOVARA cup"}, {"image_prompt": "Close-up of the cup"},
         {"image_prompt": "Wide shot of the cafe"}]


class CriticTests(unittest.TestCase):
    def test_normalize_fills_missing_and_applies_threshold(self):
        raw = {"shots": [{"shot": 1, "score": 8.6, "issues": [], "fixed_prompt": ""},
                         {"shot": "2", "score": 4, "issues": ["six fingers"], "fixed_prompt": "relaxed hand, five fingers"}],
               "continuity_notes": "cup colour changes"}
        v = sc.normalize_verdicts(raw, SHOTS, 7)
        self.assertEqual([s["score"] for s in v["shots"]], [9, 4, 0])
        self.assertEqual([s["pass"] for s in v["shots"]], [True, False, False])
        self.assertEqual([s["route"] for s in v["shots"]], ["pass", "reroll", "reroll"])
        self.assertEqual(v["shots"][0]["fixed_prompt"], SHOTS[0]["image_prompt"])  # empty rewrite keeps original
        self.assertEqual(v["shots"][1]["fixed_prompt"], "relaxed hand, five fingers")
        self.assertIn("no verdict", v["shots"][2]["issues"][0]["issue"])

    def test_user_text_lists_shots_rules_and_refs(self):
        t = sc.build_user_text(SHOTS, "", "Logo reads NOVARA", 1)
        self.assertIn("Shot 2: prompt: Close-up of the cup", t)
        self.assertIn("Brand rules", t)
        self.assertIn("Reference N", t)
        self.assertIn("Anatomy", t)  # default checks when empty

    def test_critique_falls_back_to_next_model_when_busy(self):
        busy = sc.ShotPlannerError("API error 503")
        busy.status = 503
        reply = json.dumps({"shots": [{"shot": 1, "score": 9, "issues": [], "fixed_prompt": "x"}]})
        calls = []

        def fake(model, key, system, text, images):
            calls.append(model)
            if len(calls) == 1:
                raise busy
            self.assertEqual([lbl for lbl, _ in images], ["Reference 1:", "Shot 1:"])
            return reply

        with mock.patch.dict(sc.CALLERS, {"gemini": fake}), mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}):
            v = sc.critique([b"png"], SHOTS[:1], references_png=[b"ref"])
        self.assertEqual(calls, [sc.CRITIC_MODELS["gemini"][0]] * 2)  # busy -> second try on the same model
        self.assertTrue(v["shots"][0]["pass"])

    def test_routes_detail_vs_reroll(self):
        raw = {"shots": [
            {"shot": 1, "score": 5, "issues": [{"issue": "logo reads NOVAPA", "fix": "detail", "target": "Logo"}]},
            {"shot": 2, "score": 5, "issues": [{"issue": "logo garbled", "fix": "detail", "target": "text"},
                                                {"issue": "she is outside, should be inside", "fix": "reroll"}]},
            {"shot": 3, "score": 9, "issues": []}]}
        v = sc.normalize_verdicts(raw, SHOTS, 7)
        self.assertEqual([s["route"] for s in v["shots"]], ["detail", "reroll", "pass"])
        self.assertEqual(v["shots"][0]["detail_targets"], ["logo"])
        self.assertTrue(v["shots"][0]["pass"])  # detail-only shots are kept, not re-rendered
        self.assertIn("1 need a detail fix", sc.report_text(v))

    def test_consistency_route_needs_valid_reference(self):
        raw = {"shots": [
            {"shot": 1, "score": 9, "issues": []},
            {"shot": 2, "score": 6, "issues": [{"issue": "barista wears white shirt", "fix": "consistency",
                                                "target": "Barista", "reference": "Shot 1"}],
             "edit_instruction": "Make the barista match Picture 2."},
            {"shot": 3, "score": 6, "issues": [{"issue": "cup differs", "fix": "consistency", "target": "cup",
                                                "reference": "shot 3"},  # self-reference is useless
                                               {"issue": "logo garbled", "fix": "detail", "target": "logo"}]}]}
        v = sc.normalize_verdicts(raw, SHOTS, 7)
        self.assertEqual([s["route"] for s in v["shots"]], ["pass", "consistency", "detail"])
        self.assertEqual(v["shots"][1]["reference"], ["shot", 1])
        self.assertEqual(v["shots"][1]["edit_instruction"], "Make the barista match Picture 2.")
        self.assertIn("1 need a consistency fix", sc.report_text(v))
        self.assertIn("Edit with shot 1 as reference", sc.report_text(v))

    def test_majority_vote_outvotes_random_miss(self):
        def review(routes):
            return {"shots": [{"shot": i + 1, "route": r, "score": 8 if r == "pass" else 5, "issues": [],
                               "pass": r != "reroll"} for i, r in enumerate(routes)],
                    "pass_score": 7, "model": "m", "seconds": 7}
        v = sc.combine_reviews([review(["pass", "consistency", "reroll"]),
                                review(["pass", "consistency", "pass"]),
                                review(["detail", "pass", "pass"])])
        self.assertEqual([s["route"] for s in v["shots"]], ["pass", "consistency", "pass"])
        self.assertEqual(v["shots"][1]["votes"], ["consistency", "consistency", "pass"])
        self.assertEqual(v["reviews"], 3)

    def test_parallel_reviews_are_combined(self):
        reply = json.dumps({"shots": [{"shot": 1, "seen": "a cup", "score": 9, "issues": []}]})
        with mock.patch.dict(sc.CALLERS, {"gemini": lambda *a: reply}), \
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}):
            v = sc.critique([b"png"], SHOTS[:1], reviews=3)
        self.assertEqual(v["reviews"], 3)
        self.assertEqual(v["shots"][0]["seen"], "a cup")
        self.assertIn("majority of 3", sc.report_text(v))

    def test_prior_votes_count_for_unchanged_shots_only(self):
        def review(routes, votes=None):
            r = {"shots": [{"shot": i + 1, "route": x, "score": 7, "issues": [], "pass": x != "reroll"}
                           for i, x in enumerate(routes)], "pass_score": 7, "reviews": 3}
            if votes:
                for sh, vv in zip(r["shots"], votes):
                    sh["votes"] = vv
            return r
        prior = review(["reroll", "consistency"], votes=[["reroll"] * 3, ["consistency", "consistency", "pass"]])
        now = [review(["pass", "detail"]), review(["pass", "detail"]), review(["pass", "consistency"])]
        v = sc.combine_reviews(now, prior)
        # shot 1 was re-rendered: old votes ignored. shot 2 unchanged: 3 consistency votes out of 6 -> consistency
        self.assertEqual([s["route"] for s in v["shots"]], ["pass", "consistency"])

    def test_malformed_json_is_retried(self):
        replies = iter(['{"shots": [{"shot": 1 "score": 9}]}', json.dumps({"shots": [{"shot": 1, "score": 9, "issues": []}]})])
        with mock.patch.dict(sc.CALLERS, {"gemini": lambda *a: next(replies)}), \
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}):
            v = sc.critique([b"png"], SHOTS[:1])
        self.assertEqual(v["shots"][0]["route"], "pass")

    def test_report(self):
        v = sc.normalize_verdicts({"shots": [{"shot": 1, "score": 3, "issues": ["logo misspelled"],
                                              "fixed_prompt": "cup with NOVARA logo"}]}, SHOTS[:1], 7)
        r = sc.report_text(v)
        self.assertIn("1 need a re-render", r)
        self.assertIn("RE-RENDER", r)
        self.assertIn("New prompt: cup with NOVARA logo", r)


if __name__ == "__main__":
    unittest.main()
