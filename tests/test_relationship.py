"""Offline tests for the cautious relationship-signal reader.

The point of these tests is not to prove what a real person means.  It is to pin the
contract that matters in the app: the reader is deterministic, explains itself through
literal cues, never turns a neutral/task message into a romantic verdict, and always
carries the disclaimer that keeps the panel honest.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import relationship  # noqa: E402


class RelationshipSignals(unittest.TestCase):
    def test_invitation_is_recognised_without_calling_it_a_date(self):
        out = relationship.analyze("周末有空吗")
        self.assertEqual(out["label"], "发出邀约")
        self.assertIn("抛出见面或一起做的安排", out["cues"])
        self.assertIn("邀约", out["label"])
        self.assertIn(relationship.DISCLAIMER, out["disclaimer"])

    def test_probe_has_a_cautious_read(self):
        out = relationship.analyze("你觉得我怎么样")
        self.assertEqual(out["label"], "试探态度")
        self.assertIn("试探你的态度或感情状态", out["cues"])
        self.assertIn("留", out["read"])

    def test_emotional_message_gets_support_advice_not_romance(self):
        out = relationship.analyze("今天好累啊")
        self.assertEqual(out["label"], "寻求情绪价值")
        self.assertIn("先接住感受", out["read"])
        self.assertIn("不自动等于喜欢", out["read"])

    def test_avoidance_and_boundary_are_negative(self):
        avoid = relationship.analyze("下次吧")
        boundary = relationship.analyze("别多想，我们只是朋友")
        self.assertEqual(avoid["label"], "回避/降温")
        self.assertEqual(boundary["label"], "边界/拒绝")
        self.assertLess(avoid["score"], 0)
        self.assertLess(boundary["score"], 0)

    def test_dry_reply_is_low_signal_not_a_verdict(self):
        out = relationship.analyze("嗯嗯")
        self.assertEqual(out["label"], "简短回应")
        self.assertEqual(out["score"], 0)
        self.assertIn("看下一轮谁主动延续话题", out["read"])
        self.assertTrue(any(c.startswith("回复很短") for c in out["cues"]))

    def test_task_help_is_not_mistaken_for_an_invitation(self):
        out = relationship.analyze("明天早上能帮我带份早饭不")
        self.assertEqual(out["label"], "中性闲聊")
        self.assertEqual(out["cues"], [])
        self.assertEqual(out["temperature"], "中性")

    def test_context_double_text_adds_an_active_cue(self):
        out = relationship.analyze("你睡了吗", context="我: 刚忙完\n对方: 在吗\n对方: 今天好玩吗")
        self.assertEqual(out["label"], "主动靠近")
        self.assertIn("连续发消息（不代表好感或投入程度）", out["cues"])
        self.assertGreater(out["score"], 0)

    def test_empty_message_is_explicitly_unknown(self):
        out = relationship.analyze("")
        self.assertEqual(out["label"], "信息不足")
        self.assertEqual(out["cues"], [])
        self.assertEqual(out["score"], 0)

    def test_question_alone_is_engagement_not_a_direction(self):
        out = relationship.analyze("然后呢？")
        self.assertEqual(out["label"], "中性闲聊")
        self.assertIn("用问句把话题递回给你", out["cues"])



class RelationshipJudgeIntegration(unittest.TestCase):
    """The app's judge verdict must carry the hint without another model forward."""

    def test_verdict_has_relationship_and_still_scores_intent(self):
        import numpy as np
        import judge

        j = judge.Judge(device="cpu")
        j._loaded = True
        j._load = lambda wait_s=None: None
        j._forward = lambda prompt, n_slots: ([0] * n_slots, [0, 1])
        j._slot_probs = lambda logits_by_slot, n_options, slot: np.array(
            [1.0] + [0.0] * (n_options - 1), dtype=np.float32)

        out = j.judge("周末有空吗", context="我: 在忙\n对方: 刚忙完")
        self.assertEqual(out["intent"], list(judge.INTENTS)[0])
        self.assertEqual(out["relationship"]["label"], "发出邀约")
        self.assertIn("disclaimer", out["relationship"])


class RelationshipMoreCues(unittest.TestCase):
    def test_direct_affection_is_not_left_as_neutral(self):
        out = relationship.analyze("我好像喜欢你")
        self.assertEqual(out["label"], "直接表达好感")
        self.assertIn("直接表达好感或想念", out["cues"])
        self.assertIn("不用把一句话立刻升级成关系承诺", out["read"])

    def test_existing_partner_is_treated_as_a_boundary(self):
        out = relationship.analyze("我有男朋友了")
        self.assertEqual(out["label"], "边界/拒绝")

    def test_asking_about_your_partner_is_probe_not_boundary(self):
        out = relationship.analyze("你有女朋友吗")
        self.assertEqual(out["label"], "试探态度")

    def test_loneliness_and_goodnight_have_distinct_reads(self):
        lonely = relationship.analyze("好无聊")
        goodnight = relationship.analyze("晚安")
        self.assertEqual(lonely["label"], "寻求情绪价值")
        self.assertEqual(goodnight["label"], "主动靠近")


if __name__ == "__main__":
    unittest.main()
