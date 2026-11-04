"""Whole-pack checks (needs torch, not ComfyUI): registration, node metadata, the docs, and the
node wrappers that have no other tests (animatic, critic helpers, video engine).

The pack is imported as a package, exactly as ComfyUI imports it.
"""

import importlib.util
import json
import os
import subprocess
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = "shotplanner_pack_test"

EXPECTED = {
    "ShotPlanner", "ShotSelector", "ShotListEditor",                       # planning
    "ShotAnimatic", "ClipStitcher",                                        # animatic
    "ShotCritic", "ShotRerollMerge", "ShotConsistencyPrep", "ShotKeepBetter",
    "ShotContinuityPrompts", "ShotJoin", "ShotSelect",                     # critic
    "AdTimeline", "AdSupers", "AdEndCard", "AdAudioMix", "AdScriptFromShots",
    "ShotVideoEngine", "ShotMotionPrompts", "ShotVideoQA", "ClipRerollMerge", "ClipsFromBatch",
    "ClipLastFrame", "ClipResampleFPS",                                    # video
    "AdVoiceover", "AdVoiceFit", "AdBeatSync", "AdReframe", "AdCutdown",
    "TreatmentBuilder", "PitchDeckBuilder", "StoryboardSheet", "PaletteFromShots",
}


def load_pkg():
    if PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(PKG, os.path.join(ROOT, "__init__.py"),
                                                      submodule_search_locations=[ROOT])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[PKG] = pkg
        spec.loader.exec_module(pkg)
    return sys.modules[PKG]


def node(key):
    return load_pkg().NODE_CLASS_MAPPINGS[key]()


def imgs(n, h=8, w=8, value=None):
    """A list of single-image batches, image i filled with i/10 (or value)."""
    return [torch.full((1, h, w, 3), (i / 10) if value is None else value) for i in range(n)]


class Registration(unittest.TestCase):
    def test_every_node_registered_once_with_display_name(self):
        pkg = load_pkg()
        self.assertEqual(set(pkg.NODE_CLASS_MAPPINGS), EXPECTED)
        self.assertEqual(set(pkg.NODE_DISPLAY_NAME_MAPPINGS), EXPECTED)

    def test_metadata_complete(self):
        for key, cls in load_pkg().NODE_CLASS_MAPPINGS.items():
            with self.subTest(node=key):
                self.assertTrue(cls.CATEGORY.startswith("Shot Planner"))
                self.assertTrue(getattr(cls, "DESCRIPTION", "").strip())
                self.assertTrue(callable(getattr(cls, cls.FUNCTION)))
                it = cls.INPUT_TYPES()
                for section in ("required", "optional"):
                    for name, spec in (it.get(section) or {}).items():
                        opts = spec[1] if len(spec) > 1 else {}
                        self.assertTrue(str(opts.get("tooltip", "")).strip(), f"{key}.{name} has no tooltip")
                n = len(cls.RETURN_TYPES)
                self.assertEqual(len(getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES)), n)
                self.assertEqual(len(getattr(cls, "OUTPUT_TOOLTIPS", ())), n, f"{key} OUTPUT_TOOLTIPS")
                if hasattr(cls, "OUTPUT_IS_LIST"):
                    self.assertEqual(len(cls.OUTPUT_IS_LIST), n)

    def test_nodes_md_up_to_date(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_nodes_md.py"), "--check"],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

    def test_example_prompts_are_consistent_with_the_nodes(self):
        """Links point at existing nodes; this pack's nodes get exactly the inputs they declare."""
        ours = set(load_pkg().NODE_CLASS_MAPPINGS)
        api = os.path.join(ROOT, "examples", "api")
        for fn in sorted(os.listdir(api)):
            with open(os.path.join(api, fn), encoding="utf-8") as f:
                prompt = json.load(f)
            for nid, n in prompt.items():
                for k, v in n["inputs"].items():
                    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                        self.assertIn(v[0], prompt, f"{fn}: {nid}.{k} links to missing node {v[0]}")
                ct = n["class_type"]
                if ct in ours:
                    spec = load_pkg().NODE_CLASS_MAPPINGS[ct].INPUT_TYPES()
                    known = set(spec["required"]) | set(spec.get("optional", {}))
                    self.assertLessEqual(set(n["inputs"]), known, f"{fn}: {nid} has unknown inputs")
                    self.assertLessEqual(set(spec["required"]), set(n["inputs"]),
                                         f"{fn}: {nid} misses a required input")


class AnimaticNodes(unittest.TestCase):
    PLAN = json.dumps({"shots": [{"duration": 1, "camera": "push-in"}, {"duration": 0, "camera": "pan left"}]})

    def test_clip_stitcher_trims_only_shots_with_a_duration(self):
        clips = [torch.rand(20, 8, 8, 3), torch.rand(20, 8, 8, 3)]
        frames, count, fps = node("ClipStitcher").stitch(clips, [10], [0], [True], [self.PLAN])
        # shot 1: 1 s at 10 fps -> 10 frames; shot 2 has no duration -> all 20 kept (was cut to 2)
        self.assertEqual(count, 30)
        self.assertEqual(fps, 10.0)

    def test_clip_stitcher_without_trim(self):
        clips = [torch.rand(5, 8, 8, 3), torch.rand(7, 16, 16, 3)]
        frames, count, _ = node("ClipStitcher").stitch(clips, [16], [2], [False])
        self.assertEqual(tuple(frames.shape), (10, 8, 8, 3))  # 5 + 7 - 2 overlap, resized to clip 1

    def test_animatic_lengths_follow_shot_list(self):
        frames, count, fps, timing = node("ShotAnimatic").build(
            imgs(2, 32, 48), [10], [320], ["16:9"], [0.1], [0], [3], [self.PLAN])
        self.assertEqual(count, 10 + 30)  # shot 2: duration 0 -> default_seconds
        self.assertEqual(tuple(frames.shape[1:]), (180, 320, 3))
        self.assertIn("Shot 2", timing)


def verdicts(routes, scores=None, candidates=1):
    scores = scores or [8] * len(routes)
    return json.dumps({"candidates": candidates, "pass_score": 7, "shots": [
        {"shot": i + 1, "route": r, "pass": r != "reroll", "score": s, "detail_targets": ["logo"] if r == "detail" else [],
         "reference": ["shot", 1] if r == "consistency" else None, "edit_instruction": "match",
         "consistency_target": "woman"}
        for i, (r, s) in enumerate(zip(routes, scores))]})


class CriticNodes(unittest.TestCase):
    def test_select_and_merge_round_trip(self):
        shots = imgs(3)
        v = verdicts(["pass", "detail", "pass"])
        picked, targets, summary = node("ShotSelect").select(shots, [v], ["detail"])
        self.assertEqual(len(picked), 1)
        self.assertTrue(torch.equal(picked[0], shots[1]))
        self.assertEqual(targets, ["logo"])
        fixed = [torch.ones(1, 8, 8, 3)]
        out, note = node("ShotRerollMerge").merge(shots, [v], ["detail"], fixed)
        self.assertTrue(torch.equal(out[1], fixed[0]))
        self.assertTrue(torch.equal(out[0], shots[0]))

    def test_select_placeholder_and_lazy_skip_when_nothing_matches(self):
        shots = imgs(2)
        v = verdicts(["pass", "pass"])
        picked, _, summary = node("ShotSelect").select(shots, [v], ["detail"])
        self.assertEqual(len(picked), 1)
        self.assertIn("No shots", summary)
        merge = node("ShotRerollMerge")
        self.assertEqual(merge.check_lazy_status(shots, [v], ["detail"], (None,)), [])
        out, note = merge.merge(shots, [v], ["detail"], (None,))
        self.assertEqual(len(out), 2)

    def test_merge_requests_lazy_inputs_when_needed(self):
        v = verdicts(["reroll", "pass"])
        merge = node("ShotRerollMerge")
        self.assertEqual(merge.check_lazy_status(imgs(2), [v], ["reroll"], (None,), (None,)),
                         ["rerolled", "candidate_verdicts_json"])
        self.assertEqual(merge.check_lazy_status(imgs(2), [v], ["reroll"], (None,)), ["rerolled"])

    def test_merge_keeps_best_candidate(self):
        v = verdicts(["reroll", "pass"], candidates=2)
        cands = [torch.full((1, 8, 8, 3), 0.3), torch.full((1, 16, 16, 3), 0.9)]
        cv = json.dumps({"shots": [{"score": 5}, {"score": 9}]})
        out, note = node("ShotRerollMerge").merge(imgs(2), [v], ["reroll"], cands, [cv])
        self.assertAlmostEqual(float(out[0].mean()), 0.9, places=4)
        self.assertEqual(tuple(out[0].shape), (1, 8, 8, 3))  # resized to the original
        self.assertIn("candidate 2/2", note)

    def test_keep_better_never_makes_it_worse(self):
        before, after = imgs(3), imgs(3, value=0.5)
        vb = verdicts(["detail", "detail", "pass"], [5, 6, 9])
        va = verdicts(["pass", "reroll", "pass"], [8, 3, 9])
        out, merged, summary = node("ShotKeepBetter").choose(before, [vb], after, [va])
        self.assertTrue(torch.equal(out[0], after[0]))   # better: kept
        self.assertTrue(torch.equal(out[1], before[1]))  # worse: rejected
        self.assertEqual([s["score"] for s in json.loads(merged)["shots"]], [8, 6, 9])
        self.assertIn("rejected", summary)

    def test_consistency_prep_pairs_shot_with_reference(self):
        shots = imgs(3)
        v = verdicts(["pass", "consistency", "pass"])
        s, r, instr, targets, summary = node("ShotConsistencyPrep").prep(shots, [v])
        self.assertTrue(torch.equal(s[0], shots[1]))
        self.assertTrue(torch.equal(r[0], shots[0]))
        self.assertEqual(targets, ["woman"])

    def test_continuity_prompts_and_join(self):
        plan = json.dumps({"shots": [{"image_prompt": "a"}, {"image_prompt": "b"}, {"image_prompt": "c"}]})
        first, rest, n = node("ShotContinuityPrompts").split(plan, "same people")
        self.assertEqual((first, rest, n), ("a", ["b same people", "c same people"], 2))
        (joined,) = node("ShotJoin").join([torch.rand(1, 8, 8, 3)], [torch.rand(2, 16, 16, 3)])
        self.assertEqual([tuple(j.shape) for j in joined], [(1, 8, 8, 3)] * 3)


class VideoEngineNode(unittest.TestCase):
    def test_only_the_chosen_branch_is_requested(self):
        eng = node("ShotVideoEngine")
        self.assertEqual(eng.check_lazy_status(["local: Wan"], wan_clips=(None,), h3_clips=(None,)), ["wan_clips"])
        self.assertEqual(eng.check_lazy_status(["local: MiniMax H3"], wan_clips=(None,), h3_clips=(None,)),
                         ["h3_clips"])
        clips, fps, summary = eng.run(["local: Wan"], wan_clips=[torch.rand(5, 8, 8, 3)], fps_override=[0.0])
        self.assertEqual((len(clips), fps), (1, 16.0))


if __name__ == "__main__":
    unittest.main()
