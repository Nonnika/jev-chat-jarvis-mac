"""Judge backed by laya-coreml — Laya's typed-decision model exported to Core ML.

Same shape as the TypeSafe Jev API this layer was built around (`state` + typed
`questions` -> calibrated probabilities, never text), but fully local: a 322M
multilingual encoder, one forward pass per question, no key and no network. That
makes it the primary judge, replacing the cloud Jev call; decider-2b stays as the
on-failure fallback (judge.FallbackJudge).

    import laya_coreml as laya
    agent = laya.load("aac6fef/laya-multilingual-coreml")
    agent.predict(state, {"intent": {"type": "choice", ...},
                          "risk":   {"type": "score", ...}})

Config (src/userconfig.py):
    LAYA_COREML_MODEL    default aac6fef/laya-multilingual-coreml

The model id is the package's own default: the 1024-token multilingual export. The
96-token ANE exports are short-decision specials — this layer's 8 intent options +
10 risk levels + message state do not fit and `prepare` rejects them outright. First
`laya.load` downloads the bundle into the shared HF cache and compiles the Core ML
package once; warm() pays that at HUD startup, same contract as decider-2b's warm.
"""

from __future__ import annotations

import json
import threading
import time

import userconfig
from judge import ACTION_MAP, INTENTS, RISK_LEVELS

DEFAULT_MODEL = "aac6fef/laya-multilingual-coreml"


def laya_model() -> str:
    return userconfig.get("LAYA_COREML_MODEL") or DEFAULT_MODEL


def laya_available() -> bool:
    """laya-coreml is a hard dependency; this only fails on a broken install."""
    try:
        import laya_coreml  # noqa: F401
    except ImportError:
        return False
    return True


class LayaJudge:
    """Same surface as judge.Judge: judge(message, context) -> dict."""

    name = "laya-coreml"

    def __init__(self, model: str | None = None):
        self.model = model or laya_model()
        self._agent = None
        # Same shape as judge.Judge's load lock: warm() and the first real message can
        # both get here at once, and the loser waits instead of loading the model twice.
        self._load_lock = threading.RLock()

    def _load(self):
        if self._agent is None:
            with self._load_lock:
                if self._agent is None:
                    import laya_coreml as laya

                    self._agent = laya.load(self.model)

    def judge(self, message: str, context: str | None = None) -> dict:
        self._load()
        state = f"{context}\n\n{message}" if context else message
        # One predict call, two forwards (batch 1 per question, insertion order).
        result = self._agent.predict(state, {
            "intent": {"type": "choice",
                       "instructions": "这句话的真实意图是什么？",
                       "criteria": INTENTS},
            "risk": {"type": "score",
                     "instructions": "如果直接回复这句话，风险有多大？",
                     "criteria": RISK_LEVELS},
        })
        answers = result.get("answers") or {}
        intent_ans = answers.get("intent") or {}
        risk_ans = answers.get("risk") or {}

        intent = intent_ans.get("choice") or "闲聊"
        if intent not in INTENTS:
            for name in INTENTS:
                if name in str(intent):
                    intent = name
                    break
            else:
                intent = "闲聊"
        # score answers report the expected zero-based category index — the same
        # semantics judge.py computes by hand for decider-2b
        risk = risk_ans.get("score")
        risk = float(risk) if isinstance(risk, (int, float)) else 0.0

        return {
            "intent": intent,
            "confidence": float(intent_ans.get("confidence") or 0.0),
            "intent_probs": intent_ans.get("probabilities") or {},
            "risk": round(risk, 1),
            "risk_probs": risk_ans.get("probabilities") or {},
            "actions": ACTION_MAP.get(intent, []),
            "message": message,
            "backend": f"laya-coreml/{self.model}",
        }

    def rank_candidates(self, message: str, intent: str,
                        candidates: list[str]) -> list[dict]:
        """Rank reply candidates — just another `choice` question with the texts as options."""
        if not candidates:
            return []
        self._load()
        # dict.fromkeys dedupes in order: laya rejects duplicate choice labels, and a
        # repeated candidate would otherwise flip the whole judge to the fallback.
        texts = list(dict.fromkeys(candidates))
        result = self._agent.predict(
            f"收到的消息：{message}\n判断出的意图：{intent}",
            {"best": {"type": "choice",
                      "instructions": "哪一条回复最合适？",
                      "criteria": dict.fromkeys(texts)}})
        probs = ((result.get("answers") or {}).get("best") or {}).get("probabilities") or {}
        ranked = [{"text": c, "prob": float(probs.get(c, 0.0))} for c in texts]
        ranked.sort(key=lambda r: -r["prob"])
        return ranked

    def warm(self) -> None:
        """Load + compile the Core ML package, then run one real-shaped judge."""
        with self._load_lock:
            self._load()
            self.judge("预热")


if __name__ == "__main__":
    import sys

    j = LayaJudge()
    msg = sys.argv[1] if len(sys.argv) > 1 else "这个需求你今天跟一下"
    t0 = time.perf_counter()
    j._load()
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    out = j.judge(msg)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"耗时 {time.perf_counter() - t1:.2f}s（首次加载另付 {load_s:.1f}s）")
