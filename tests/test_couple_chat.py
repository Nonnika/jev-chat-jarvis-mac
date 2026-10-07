"""Offline semantic boundaries for couple/flirting cues and generation requests.

All messages are synthetic; generation is captured before any HTTP or key lookup.
This does not grade a provider's generated text or infer anyone's actual feelings.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import relationship
import styles
from generate import Generator


SIGNAL_CASES = [
    ("别再熬夜啦，我会担心", "主动靠近"),
    ("不要再熬夜了，早点睡", "主动靠近"),
    ("别再联系我了", "边界/拒绝"),
    ("不要再纠缠我", "边界/拒绝"),
    ("我们先做朋友吧", "边界/拒绝"),
    ("我不喜欢你", "边界/拒绝"),
    ("我不爱你了", "边界/拒绝"),
    ("我有男朋友了", "边界/拒绝"),
    ("暂时不想谈恋爱", "边界/拒绝"),
    ("我需要一点自己的空间", "边界/拒绝"),
    ("我好像喜欢你", "直接表达好感"),
    ("我爱你", "直接表达好感"),
    ("我想你了", "直接表达好感"),
    ("我不想你了", "中性闲聊"),
    ("我没有喜欢你", "中性闲聊"),
    ("不是不想见你，是今天发烧了", "中性闲聊"),
    ("不想一起去", "中性闲聊"),
    ("不想和你一起看电影", "中性闲聊"),
    ("我不需要安慰，也没有不开心", "回避/降温"),
    ("你有女朋友吗", "试探态度"),
    ("你觉得我怎么样", "试探态度"),
    ("今晚一起看电影吗", "发出邀约"),
    ("今天不方便，周六下午一起喝咖啡吧", "发出邀约"),
    ("不用来接我啦，我已经到家了", "中性闲聊"),
    ("这双鞋不合适", "中性闲聊"),
    ("我想多了", "中性闲聊"),
    ("想不想我呀", "试探态度"),
    ("我不想和你在一起", "边界/拒绝"),
    ("下次吧", "回避/降温"),
    ("今天好累啊", "寻求情绪价值"),
    ("我今天有点委屈", "寻求情绪价值"),
    ("抱抱我嘛", "寻求情绪价值"),
    ("我今天发烧了", "寻求情绪价值"),
    ("你每次都只顾着玩手机，根本不听我说话", "关系不满"),
    ("说好陪我的，你又放我鸽子", "关系不满"),
    ("我不喜欢你拿我跟前任比较", "关系不满"),
    ("你把我们的聊天发给别人，太不尊重我了", "关系不满"),
    ("你根本不在乎我", "关系不满"),
    ("对不起，刚才我的语气不好", "尝试和好"),
    ("我们好好聊聊吧", "尝试和好"),
    ("对不起，但别再联系我了", "边界/拒绝"),
    ("你可真厉害，又放我鸽子", "关系不满"),
    ("嗯嗯", "简短回应"),
    ("好的", "简短回应"),
    ("明天早上能帮我带份早饭不", "中性闲聊"),
    ("然后呢？", "中性闲聊"),
]


class CoupleSignals(unittest.TestCase):
    def test_literal_boundary_cases(self):
        for message, gold in SIGNAL_CASES:
            with self.subTest(message=message):
                out = relationship.analyze(message)
                self.assertEqual(out["label"], gold)
                self.assertEqual(out["disclaimer"], relationship.DISCLAIMER)
                for snippets in out["evidence"].values():
                    for snippet in snippets:
                        self.assertIn(snippet, message)
                self.assertNotIn("她", out["read"])

    def test_consecutive_messages_do_not_score_attraction(self):
        plain = relationship.analyze("地址发给你了")
        consecutive = relationship.analyze("地址发给你了", "对方: 这是快递单号")
        self.assertEqual(consecutive["label"], plain["label"])
        self.assertEqual(consecutive["score"], plain["score"])

    def test_empty_or_unknown_direction_has_no_continuity_cue(self):
        self.assertEqual(relationship.analyze("", "对方: 在吗")["cues"], [])
        out = relationship.analyze("地址发给你了", "方向未确认: 单号在这里")
        self.assertEqual(out["cues"], [])


class CoupleGenerationRequests(unittest.TestCase):
    def capture(self, message, tone, context=None, streamed=False):
        gen = Generator(model="test", api="openai")
        lines = []
        def reply(prompt, on_delta=None):
            if on_delta:
                on_delta("我听见了\n我们慢慢聊")
            return "我听见了\n我们慢慢聊"
        with patch.object(gen, "_call", side_effect=reply) as call:
            texts, error = gen._one_tone(message, "", tone, context,
                                        lines.append if streamed else None)
        self.assertFalse(error)
        self.assertEqual(texts, ["我听见了", "我们慢慢聊"])
        if streamed:
            self.assertEqual(lines, texts)
        return call.call_args.args[0]

    def test_boundary_and_complaint_override_playfulness(self):
        for tone in styles.RELATION_TONES:
            with self.subTest(tone=tone):
                prompt = self.capture("别再联系我了", tone)
                self.assertIn("先尊重", prompt)
                self.assertIn("不以玩笑或情话绕过拒绝", prompt)
                self.assertNotIn("更皮、更夸张", prompt)
                self.assertIn("不默认我全错", prompt)

    def test_ambiguous_relationship_is_not_assumed_and_context_is_preserved(self):
        context = "我: 今天面试怎么样\n对方: 不太顺利"
        prompt = self.capture("我今天有点委屈", "自然关心", context, streamed=True)
        self.assertIn(context, prompt)
        self.assertIn("先接住感受", prompt)
        self.assertIn("不能仅凭称呼", prompt)
        self.assertIn("不编造自己的位置、经历、过错", prompt)

    def test_existing_comic_tone_keeps_its_variation(self):
        prompt = self.capture("今晚吃什么", "贴吧老哥 v1.0")
        self.assertIn("更皮、更夸张", prompt)
        self.assertNotIn("回应原则：", prompt)

    def test_new_labels_are_stripped_from_streamed_or_numbered_output(self):
        for tone in styles.RELATION_TONES:
            self.assertEqual(Generator._parse(f"1. {tone}：我听见了"), ["我听见了"])


class ChatSceneSelection(unittest.TestCase):
    def test_primary_and_fallback_share_explicit_question(self):
        import numpy as np
        import judge
        import judge_laya
        for scene in ("general", "relationship"):
            with self.subTest(scene=scene):
                primary = judge.Judge(device="cpu", scene=scene)
                primary._load = Mock()
                primary._forward = Mock(return_value=([0, 0], [0, 1]))
                primary._slot_probs = lambda logits, n, slot: np.array([1.] + [0.] * (n - 1))
                primary.judge("我想你了")
                prompt = primary._forward.call_args.args[0]
                self.assertIn(judge.intent_question(scene), prompt)
                self.assertIn(judge.RISK_QUESTION, prompt)
                fallback = judge_laya.LayaJudge(model="test", scene=scene)
                fallback._load = Mock()
                fallback._agent = Mock()
                fallback._agent.predict.return_value = {"answers": {}}
                fallback.judge("我想你了")
                question = fallback._agent.predict.call_args.args[1]
                self.assertEqual(question["intent"]["instructions"], judge.intent_question(scene))
                self.assertEqual(question["risk"]["criteria"], judge.RISK_LEVELS)


if __name__ == "__main__":
    unittest.main()
