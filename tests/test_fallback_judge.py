"""Offline regression for the judge warm-up guard. Run: python -B -m unittest discover -s tests.

The app log has a wedged decider-2b load (547 s, with warm-ups of 547/165/163/117 s)
holding the first real message's prejudge on the load lock for 542 s — minutes of dead
panel. These tests pin the contract that bounds it, with fakes instead of models:

  · Judge._load(wait_s) raises JudgeNotReady once the bound expires, and skips the lock
    entirely when the model is already loaded (the steady-state path pays nothing);
  · FallbackJudge treats JudgeNotReady as transient — that one call goes to the fallback
    and the primary keeps its seat — while a real failure still flips permanently.
"""
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import judge


class FakePrimary:
    label = "fake-primary"

    def __init__(self):
        self.calls = 0
        self.mode = "ok"          # ok | not_ready | broken

    def judge(self, message, context=None):
        self.calls += 1
        self._maybe_raise()
        return {"intent": "帮忙", "backend": "fake-primary"}

    def rank_candidates(self, message, intent, candidates):
        self.calls += 1
        self._maybe_raise()
        return [{"text": c, "prob": 1.0} for c in candidates]

    def _maybe_raise(self):
        if self.mode == "not_ready":
            raise judge.JudgeNotReady("fake repo 仍在加载（>0s）")
        if self.mode == "broken":
            raise RuntimeError("boom")


class FakeFallback:
    label = "fake-fallback"

    def __init__(self):
        self.calls = 0

    def judge(self, message, context=None):
        self.calls += 1
        return {"intent": "闲聊", "backend": "fake-fallback"}

    def rank_candidates(self, message, intent, candidates):
        self.calls += 1
        return [{"text": c, "prob": 0.5} for c in candidates]


class FallbackJudgeWarmUpGuard(unittest.TestCase):
    def test_not_ready_is_transient_not_permanent(self):
        primary, fallback = FakePrimary(), FakeFallback()
        wrapped = judge.FallbackJudge(primary, lambda: fallback, "fake-primary")

        primary.mode = "not_ready"
        out = wrapped.judge("在吗")
        self.assertEqual(out["intent"], "闲聊")
        self.assertIn("预热超时", out["backend"])
        self.assertFalse(wrapped.fell_back)          # the primary keeps its seat

        primary.mode = "ok"                          # warm-up finished
        out = wrapped.judge("明天带份早饭")
        self.assertEqual(out["intent"], "帮忙")
        self.assertEqual(out["backend"], "fake-primary")
        self.assertEqual(fallback.calls, 1)          # and steady state never pays it again

    def test_not_ready_rank_is_transient_too(self):
        primary, fallback = FakePrimary(), FakeFallback()
        wrapped = judge.FallbackJudge(primary, lambda: fallback, "fake-primary")

        primary.mode = "not_ready"
        ranked = wrapped.rank_candidates("在吗", "帮忙", ["a", "b"])
        self.assertEqual([r["prob"] for r in ranked], [0.5, 0.5])
        self.assertFalse(wrapped.fell_back)

        primary.mode = "ok"
        ranked = wrapped.rank_candidates("在吗", "帮忙", ["a", "b"])
        self.assertEqual([r["prob"] for r in ranked], [1.0, 1.0])

    def test_real_failure_still_flips_permanently(self):
        primary, fallback = FakePrimary(), FakeFallback()
        wrapped = judge.FallbackJudge(primary, lambda: fallback, "fake-primary")

        primary.mode = "broken"
        out = wrapped.judge("在吗")
        self.assertEqual(out["intent"], "闲聊")
        self.assertTrue(wrapped.fell_back)
        self.assertIn("不可用", out["backend"])

        primary.mode = "ok"                          # a permanent flip is permanent
        out = wrapped.judge("在吗")
        self.assertEqual(out["intent"], "闲聊")
        self.assertEqual(primary.calls, 1)           # the primary is never consulted again
        self.assertEqual(fallback.calls, 2)


class JudgeLoadWait(unittest.TestCase):
    def test_load_wait_times_out_when_another_thread_holds_the_lock(self):
        j = judge.Judge(device="cpu")                # torch import only; no model load
        loaded = threading.Event()

        def holder():
            with j._load_lock:
                loaded.set()
                time.sleep(0.5)

        t = threading.Thread(target=holder)
        t.start()
        self.assertTrue(loaded.wait(2))
        try:
            with self.assertRaises(judge.JudgeNotReady):
                j._load(wait_s=0.05)
        finally:
            t.join()
        self.assertFalse(j._loaded)

    def test_loaded_fast_path_never_touches_the_lock(self):
        j = judge.Judge(device="cpu")
        j._loaded = True
        held = threading.Event()

        def holder():
            with j._load_lock:
                held.set()
                time.sleep(0.3)

        t = threading.Thread(target=holder)
        t.start()
        self.assertTrue(held.wait(2))
        try:
            j._load(wait_s=0.05)                     # must return, not raise
        finally:
            t.join()


if __name__ == "__main__":
    unittest.main()
