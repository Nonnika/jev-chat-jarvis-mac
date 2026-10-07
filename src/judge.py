"""Judge: local Jev-shaped model reads a Chinese student chat message and returns
intent + risk.

One forward pass answers both slots (decider's documented multi-question layout:
append further `Question k: ... Answer k: (` blocks and read logits at each slot).

Historical intent-only probe: decider-2b 53/54 = 98.1%. The current app-shaped
two-slot probe measures 49/54 for general chat; keep those measurements separate.
The opt-in relationship scene improves synthetic couple/flirting cases from 29/36
to 31/36 on development and 18/20 to 19/20 on frozen confirmation, but scores
48/54 on student cases, so it must not replace the general scene. Label descriptions
and risk levels stay unchanged. The laya-coreml fallback reads formal sentences
and goes near-uniform on short
colloquial ones, which is why it lost the primary seat (2026-09). 嘲讽 (sarcasm) was
dropped from the label set — tone labels were exactly where the weak backend
coin-flipped.
"""

from __future__ import annotations

import threading

import numpy as np
# Imported for its side effect: userconfig points HF_HOME / LAYA_COREML_CACHE at the
# checkout's .models/ before anything can import huggingface_hub and pin the default
# (details on prefer_repo_model_cache). This module is the entry point of every path
# that loads a judge model, so the ordering holds for the app, the CLIs and the probes.
import userconfig  # noqa: F401
import relationship

# 描述保持这个长度是有实测依据的，别为了省 prefill 时间去瘦身：两轮压缩措辞
# （保语义锚点、每条砍 ~1/3 字符）在 22 条回归上分别是 81.8% 和 77.3%，都低于
# 原文的 86.4%——批评/要解释 的边界对措辞极敏感。省下的 ~100 ms 判断又藏在
# 停稳窗口里基本不可见，不划算（2026-09 实测，judge_zh_test.py 已改为直接
# import 这份 INTENTS，改这里必须重跑回归）。2026-09 学生化那轮又试了「保长度、
# 只换学生场景」的重写：40 条调词集上 35/40，和原描述持平而没赚，所以说描述的
# 措辞不是这轮的瓶颈（细节见下面 INTENT_QUESTION 上方的实测表）。
INTENTS = {
    "帮忙": "对方想让我帮他做点事，但不好意思直说",
    "问进度": "对方在询问某件事的进展或状态",
    "批评": "对方对我做的事或交的东西表达不满、指出错误",
    "要解释": "对方要求我说明原因或给出解释",
    "闲聊": "对方只是在闲聊、分享日常或表达感受，没有要求也没针对我",
    "约时间": "对方想约个时间碰面、语音或一起做某事",
    "夸奖": "对方真心肯定、称赞我做的事",
}

# 问句只有这一份：decider、laya、jev、回归脚本都从这里取。此前四份手抄，
# 改一处其余三处不会跟着变，回归就可能在测一个线上不发的配置（和 INTENTS 同理）。
# 措辞按「学生微信」写，并带两条从误判里总结的判定规则（规则里点名了 约时间/
# 要解释 两个标签，改标签名要一起改这里）。
#
# 这版措辞是量出来的，别凭感觉改（probe/judge_prompt_ab.py，54 条学生用例：
# dev 19 条老回归 + 21 条 A/B 调词集 + 14 条写于调词之前、只用于确认的确认集）：
#   原文（这句话的真实意图是什么？）        46/54，确认集 11/14
#   只换学生化问句（v1）                    47/54，确认集 10/14 ← 单靠「同学发来的」不够
#   只重写 7 条标签描述（v2/v3）            35/40（= 原文同分），没赚还有风险：
#       描述里给 问进度 塞「报告」会把「实验报告能发我参考下不」从 帮忙 拽走
#   本版问句 + 两条判定规则（v6）           53/54，确认集 14/14
#   本版问句 + 描述加提示（v7）             53/54，确认集 13/14
# v6/v7 总量打平，选 v6：标签描述一字未动（措辞敏感，见下），改动面最小。
INTENT_QUESTION = ("这是同学发来的微信消息，对方想让我做什么？"
                   "（约我一起做某事的算约时间；问我为什么没做或没来的算要解释）")
RELATIONSHIP_INTENT_QUESTION = (
    "这是微信聊天消息，对方想让我做什么？"
    "（约我一起做某事的算约时间；问我为什么没做或没来的算要解释；"
    "对我的行为表达不满算批评；肯定我的表现算夸奖；只表达感受或确认上文算闲聊）")


def intent_question(scene: str) -> str:
    return RELATIONSHIP_INTENT_QUESTION if scene == "relationship" else INTENT_QUESTION


RISK_QUESTION = "如果直接回复这句话，风险有多大？"

# 第 8 级原本是「涉及责任或利益」（职场措辞）。16 条排序 sanity（probe/judge_prompt_ab.py
# --cases fresh）在同一问句下换掉这一级：mae 1.96 → 1.87、rho 0.881 → 0.905；但问句
# 学生化本身会把 risk 的 mae 从 1.86 推到 1.96，两条合起来对基线是 1.86 → 1.87、
# rho 0.883 → 0.905——都在 16 条的噪声里。所以这级是「不差 + 去掉最后一处职场口径」
# 才落的，不是「变准了」。风险这层**没有任何回归**，sanity 只查单调性不查刻度：
# 实测模型把高风险消息压在 3–5 分（「你把我们组的实验数据弄丢了？」4.7、「这格式全
# 不对，重做一版」2.5），要改刻度得另立一套带金标的风险回归。
RISK_LEVELS = [
    "完全没风险，怎么回都行",
    "基本没风险",
    "平淡，正常回就好",
    "需要稍微留神",
    "有点敏感，措辞注意",
    "需要谨慎，可能被挑刺",
    "比较危险，容易得罪人或踩坑",
    "很危险，说错要出问题",
    "非常危险，涉及责任、评优或同学关系",
    "极度危险，先别回，想清楚再说",
]

# V0: actions are a static derivation, no generation involved
ACTION_MAP = {
    "帮忙": ["先问清要帮什么", "能帮就给个准话", "帮不了就直说"],
    "问进度": ["直接说事实", "给个明确时间", "有卡点就说卡点"],
    "批评": ["先听清不满", "核实事实", "商量改善"],
    "要解释": ["说清原因", "别找借口", "说下一步怎么办"],
    "闲聊": ["接住感受", "自然回应", "温和确认"],
    "约时间": ["先确认时间", "问清要干什么", "去不了提前说"],
    "夸奖": ["接住并感谢", "别过度谦虚", "可以顺带聊下去"],
}


class JudgeNotReady(RuntimeError):
    """The primary model is still loading; the caller may serve this one call from the
    fallback backend instead of waiting for a load whose end it cannot see."""


# How long a real message waits for the load lock before that happens. A normal
# decider-2b load is 6-15 s (README quotes 10-20 s cold), so a healthy warm-up never
# trips this. But the app log has recorded wedged loads of 547/165/163/117 s that held
# the first real message's prejudge on the lock for 542 s — minutes of dead panel for
# one message. 30 s bounds that at "one slow reply" instead.
LOAD_WAIT_S = 30.0


class Judge:
    """Wraps a decoder-only decision model; lazy-loads on first use."""

    label = "本地 decider-2b"

    def __init__(self, repo: str = "Mapika/decider-2b", device: str | None = None,
                 scene: str | None = None):
        import torch

        self.torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.repo = repo
        self.scene = scene if scene in userconfig.CHAT_SCENES else userconfig.chat_scene()
        self.temperature = 1.3
        self._loaded = False
        # RLock, not Lock: warm() holds it across the whole dummy forward, and judge()
        # inside that same call re-enters _load(). One lock guards both the load and the
        # first forward, so a warm-up and a real judgment can never run a forward at the
        # same time — they queue up instead.
        self._load_lock = threading.RLock()

    def _load(self, wait_s: float | None = None):
        """Block until the model is loaded; wait_s bounds how long a caller waits.

        Real messages pass LOAD_WAIT_S so a wedged load (a slow download, a stuck
        Core ML compile) costs them one fallback verdict instead of minutes on the
        lock. warm() passes no bound — it runs on the HUD's background thread and is
        exactly the caller that should absorb the whole load.
        """
        if self._loaded:
            return
        # Double-checked: the warm-up thread and the first real message can both get here
        # at once, and two concurrent from_pretrained calls would load the model twice.
        # The loser of the race just waits on the lock until the winner is done.
        if wait_s is None:
            self._load_lock.acquire()
        elif not self._load_lock.acquire(timeout=wait_s):
            raise JudgeNotReady(f"{self.repo} 仍在加载（>{wait_s:.0f}s）")
        try:
            if self._loaded:
                return
            from transformers import AutoModelForCausalLM, AutoTokenizer

            t = self.torch
            self.tok = AutoTokenizer.from_pretrained(self.repo)
            # float16, not bfloat16: MPS takes the slow path for bf16 (limited op coverage) and
            # it costs exactly 2x here — measured on this model, same prompt, three runs each:
            # bf16 1352/1393/1467 ms vs fp16 734/745/827 ms. The judge is the single biggest
            # steady-state cost in the pipeline, so this is the difference between a ~3 s and a
            # ~4 s reply. CPU has no fp16 win, so it stays fp32.
            dtype = t.float16 if self.device == "mps" else t.float32
            self.model = AutoModelForCausalLM.from_pretrained(self.repo, dtype=dtype).to(self.device).eval()
            self._letters = [self.tok.encode(c, add_special_tokens=False)[0]
                             for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
            self._loaded = True
        finally:
            self._load_lock.release()

    def warm(self) -> None:
        """Load the model and run one real-shaped forward, so no real message pays for it.

        decider-2b's first load costs 9-15 s and lands inside whichever judge() call gets
        there first — the HUD starts this in the background right after launch, so that
        call is ours, not the user's first message. The whole thing runs under the load
        lock: if a real message arrives mid-warm-up, its judge() waits at most LOAD_WAIT_S
        before serving that one message from the fallback; once the warm-up is done,
        every later message runs here at steady state.
        """
        with self._load_lock:
            self._load()
            self.judge("预热")

    def _slot_probs(self, logits_by_slot: list, n_options: int, slot: int) -> np.ndarray:
        logits = logits_by_slot[slot]
        ids = self._letters[:n_options]
        probs = self.torch.softmax(logits[ids].float() / self.temperature, -1)
        return probs.cpu().numpy()

    def _forward(self, prompt: str, n_slots: int):
        """One forward pass; returns (logits, [token index per 'Answer: (' slot]).

        logits[i] is the distribution for position i+1, so reading at the token that
        contains "(" gives the letter distribution for that slot.
        """
        import re

        ids = self.tok(prompt, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        offsets = ids.pop("offset_mapping")[0].tolist()
        with self.torch.no_grad():
            out = self.model(**ids)
        slot_token_idx = []
        for m in re.finditer(r"Answer: \(", prompt):
            char_pos = m.start() + len("Answer: ")
            for i, (s, e) in enumerate(offsets):
                if s <= char_pos < e:
                    slot_token_idx.append(i)
                    break
        if len(slot_token_idx) < n_slots:
            raise RuntimeError(f"expected {n_slots} answer slots, found {len(slot_token_idx)}")
        return out.logits[0], slot_token_idx

    def rank_candidates(self, message: str, intent: str,
                        candidates: list[str]) -> list[dict]:
        """Rank reply candidates by asking which one fits best.

        The candidates are the options, so one forward pass yields the distribution the
        phone demo shows as 89% / 9% / 2%.
        """
        self._load(LOAD_WAIT_S)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        prompt = f"Context:\n收到：「{message}」\n判断出的意图：{intent}\n\n"
        prompt += "Question: 哪一条回复最合适？\nOptions:\n"
        for i, c in enumerate(candidates):
            prompt += f"({letters[i]}) {c}\n"
        prompt += "Answer: ("

        logits, slots = self._forward(prompt, 1)
        probs = self._slot_probs([logits[slots[0]]], len(candidates), 0)
        ranked = sorted(
            ({"text": c, "prob": float(p)} for c, p in zip(candidates, probs)),
            key=lambda r: -r["prob"])
        return ranked

    def judge(self, message: str, context: str | None = None) -> dict:
        self._load(LOAD_WAIT_S)
        intents = list(INTENTS)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        prompt = f"Context:\n{context + chr(10) + chr(10) if context else ''}{message}\n\n"
        # slot 0: intent
        prompt += f"Question: {intent_question(self.scene)}\nOptions:\n"
        for i, name in enumerate(intents):
            prompt += f"({letters[i]}) {name} - {INTENTS[name]}\n"
        prompt += "Answer: ("
        # slot 1: risk
        prompt += f"\n\nQuestion: {RISK_QUESTION}\nOptions:\n"
        for i, lv in enumerate(RISK_LEVELS):
            prompt += f"({letters[i]}) {lv}\n"
        prompt += "Answer: ("

        logits, slot_token_idx = self._forward(prompt, 2)

        intent_probs = self._slot_probs([logits[slot_token_idx[0]]], len(intents), 0)
        risk_probs = self._slot_probs([logits[slot_token_idx[1]]], len(RISK_LEVELS), 0)

        intent_idx = int(np.argmax(intent_probs))
        risk_value = float((np.arange(len(RISK_LEVELS)) * risk_probs).sum())

        return {
            "intent": intents[intent_idx],
            "confidence": float(intent_probs[intent_idx]),
            "intent_probs": {n: float(p) for n, p in zip(intents, intent_probs)},
            "risk": round(risk_value, 1),
            "risk_probs": {str(i): float(p) for i, p in enumerate(risk_probs)},
            "actions": ACTION_MAP.get(intents[intent_idx], []),
            "message": message,
            "relationship": relationship.analyze(message, context),
            "backend": "local decider-2b",
        }


if __name__ == "__main__":
    import json
    import sys

    j = Judge()
    msg = sys.argv[1] if len(sys.argv) > 1 else "明天早上能帮我带份早饭不"
    print(json.dumps(j.judge(msg), ensure_ascii=False, indent=1))


class FallbackJudge:
    """Run the primary local judge; on its first failure switch permanently to the
    fallback.

    A judgment layer that dies because the model failed to load or a forward pass
    blew up would take the whole panel down, so the first failure flips permanently
    to the other local backend and the verdict carries which backend produced it.

    JudgeNotReady is the one non-permanent case: a warm-up still holding the load lock
    is slow, not broken, so that single call goes to the fallback and the primary keeps
    its seat — the log has a wedged 547 s load that otherwise held the first message's
    prejudge for 542 s.
    """

    def __init__(self, primary, fallback_factory, name: str):
        self.primary = primary
        self.label = primary.label
        self.name = name
        self._fallback_factory = fallback_factory
        self._fallback = None
        self.fell_back = False
        self.reason = ""

    def _other(self):
        if self._fallback is None:
            self._fallback = self._fallback_factory()
        return self._fallback

    def judge(self, message: str, context: str | None = None) -> dict:
        if not self.fell_back:
            try:
                return self.primary.judge(message, context)
            except JudgeNotReady:
                out = self._other().judge(message, context)
                out["backend"] = f"local 兜底（{self.name} 预热超时）"
                return out
            except Exception as e:
                self.fell_back = True
                self.reason = f"{type(e).__name__}: {str(e)[:80]}"
        out = self._other().judge(message, context)
        out["backend"] = f"local ({self.name} 不可用: {self.reason})"
        return out

    def rank_candidates(self, message: str, intent: str, candidates: list[str]) -> list[dict]:
        if not self.fell_back:
            try:
                return self.primary.rank_candidates(message, intent, candidates)
            except JudgeNotReady:
                # no per-call decoration here: candidates carry no backend string
                return self._other().rank_candidates(message, intent, candidates)
            except Exception as e:
                self.fell_back = True
                self.reason = f"{type(e).__name__}: {str(e)[:80]}"
        return self._other().rank_candidates(message, intent, candidates)

    def warm(self) -> None:
        # Only the primary: warming both would pay two model loads at launch. If the
        # primary is broken, hud._warm logs it and the first real judge() flips before
        # the fallback ever loads.
        self.primary.warm()


def make_judge():
    """decider-2b primary, laya-coreml as the on-failure fallback; plain decider-2b
    alone if the laya-coreml package is missing (broken install).

    Reversed from laya-primary (2026-09): on the 22-case student regression laya
    measured 22.7% — it handles complete formal sentences but goes near-uniform on
    short colloquial ones (「6」「睡了吗」 top prob 0.1–0.4, the argmax coin-flips
    between the tone labels). laya keeps the fallback seat because it loads without
    torch, so a broken torch install still leaves a working (if blunt) judge.
    """
    judge = Judge()
    try:
        import judge_laya
        if judge_laya.laya_available():
            return FallbackJudge(judge, lambda: judge_laya.LayaJudge(scene=judge.scene), "decider-2b")
    except Exception:
        pass
    return judge
