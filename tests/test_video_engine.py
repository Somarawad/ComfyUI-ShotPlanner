import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import video_engine as ve

class FakeComp:
    def __init__(self, n): self.images, self.frame_rate = [0]*n, 30
class FakeVideo:
    def __init__(self, n): self.n = n
    def get_components(self): return FakeComp(self.n)
class Clip:
    def __init__(self, n): self.shape = (n, 4, 4, 3)

class T(unittest.TestCase):
    def test_needed(self):
        self.assertEqual(ve.needed_input("local: Wan"), "wan_clips")
        with self.assertRaises(ValueError): ve.needed_input("x")
    def test_lazy_only_selected(self):
        n = ve.ShotVideoEngine()
        self.assertEqual(n.check_lazy_status(["local: MiniMax H3"], wan_clips=[Clip(3)], h3_clips=None), ["h3_clips"])
        self.assertEqual(n.check_lazy_status(["local: Wan"], wan_clips=[Clip(3)]), [])
        self.assertEqual(n.check_lazy_status(["local: Wan"], wan_clips=[None]), ["wan_clips"])
    def test_run_wan(self):
        clips, fps, s = ve.ShotVideoEngine().run(["local: Wan"], wan_clips=[Clip(5), Clip(7)])
        self.assertEqual((len(clips), fps), (2, 16.0)); self.assertIn("5, 7", s)
    def test_api_video(self):
        clips, fps, _ = ve.ShotVideoEngine().run(["api (VIDEO input)"], api_videos=[FakeVideo(4)], fps_override=[0.0])
        self.assertEqual((len(clips[0]), fps), (4, 30.0))
    def test_missing_branch(self):
        with self.assertRaises(ValueError): ve.ShotVideoEngine().run(["local: Wan"])
    def test_override(self):
        _, fps, _ = ve.ShotVideoEngine().run(["local: Wan"], wan_clips=[Clip(2)], fps_override=[48.0])
        self.assertEqual(fps, 48.0)

if __name__ == "__main__": unittest.main()
