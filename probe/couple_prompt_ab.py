"""Measure couple/flirting prompts through the real Judge.judge path.

Uses cached local models only (set HF_HUB_OFFLINE=1); no screen/API/credentials.
Default: development cases + original student tuning cases. --confirm runs the
frozen couple confirmation set and all 54 student cases, after choosing wording.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import judge
from judge_zh_test import GROUPS
from couple_cases import DEV, CONFIRM

BASELINE = ("这是同学发来的微信消息，对方想让我做什么？"
            "（约我一起做某事的算约时间；问我为什么没做或没来的算要解释）")
NEUTRAL = ("这是微信聊天消息，对方想让我做什么？"
           "（约我一起做某事的算约时间；问我为什么没做或没来的算要解释）")
RELATION = ("这是微信聊天消息，结合上文，对方这条消息主要想让我做什么？"
            "（约我一起做某事的算约时间；问我为什么没做或没来的算要解释；"
            "表达喜欢、想念或倾诉感受但没有要求的算闲聊）")
CLOSE = ("这是情侣或互相了解的人发来的微信消息，对方最新这句话主要想让我做什么？"
         "（约我一起做某事的算约时间；问我为什么没做或没来的算要解释；"
         "表达喜欢、想念或倾诉感受但没有要求的算闲聊）")
ACTS = ("这是微信聊天消息，结合上文判断对方最新一句的主要意图。"
        "（明确请求我做事算帮忙；询问事情完成情况或行动状态算问进度；"
        "对我的行为表达不满算批评；追问原因算要解释；"
        "约我一起做某事算约时间；肯定我的表现算夸奖；"
        "只表达感受、确认收到或答应上文的算闲聊）")
BALANCED = ("这是微信聊天消息，对方想让我做什么？"
            "（约我一起做某事的算约时间；问我为什么没做或没来的算要解释；"
            "对我的行为表达不满算批评；肯定我的表现算夸奖；只表达感受或确认上文算闲聊）")
LATEST = (BASELINE + "只判断最新消息，上文只用于理解，不把上文的邀请当成当前消息的意图。")
VARIANTS = {"baseline": BASELINE, "neutral": NEUTRAL,
            "relation": RELATION, "close": CLOSE, "acts": ACTS,
            "balanced": BALANCED, "latest": LATEST,
            "shipped": judge.RELATIONSHIP_INTENT_QUESTION}


def run(names: list[str], confirm: bool) -> dict:
    pools = {"couple_dev": DEV}
    if confirm:
        pools["couple_confirm"] = CONFIRM
    for name, cases in GROUPS.items():
        if name != "confirm" or confirm:
            pools[f"student_{name}"] = [(t, g, None) for t, g in cases]
    j = judge.Judge(scene="general")
    j.warm()
    original = judge.INTENT_QUESTION
    report = {}
    try:
        for name in names:
            # The final shipped variant exercises the actual scene selector.
            j.scene = "relationship" if name == "shipped" else "general"
            judge.INTENT_QUESTION = VARIANTS[name]
            groups = {}
            for group, cases in pools.items():
                rows = []
                for text, gold, context in cases:
                    out = j.judge(text, context)
                    rows.append({"text": text, "context": context, "gold": gold,
                                 "pred": out["intent"], "confidence": out["confidence"]})
                correct = sum(r["gold"] == r["pred"] for r in rows)
                groups[group] = {"correct": correct, "n": len(rows), "rows": rows}
                print(f"{name} {group}: {correct}/{len(rows)}", flush=True)
                for r in rows:
                    if r["gold"] != r["pred"]:
                        print(f"  {r['text']} gold={r['gold']} pred={r['pred']}", flush=True)
            report[name] = {"question": judge.INTENT_QUESTION, "groups": groups}
    finally:
        judge.INTENT_QUESTION = original
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="baseline,shipped")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--output", default="results/couple_prompt_ab.json")
    args = parser.parse_args()
    names = args.variants.split(",")
    if any(n not in VARIANTS for n in names):
        parser.error("unknown variant")
    report = run(names, args.confirm)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved {path}")
