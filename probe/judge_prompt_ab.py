"""A/B the judge prompt wording: does a student-flavored prompt actually score better?

Prompt wording is a measured artifact in this repo (see the INTENTS comment in
src/judge.py: two slimming passes lost accuracy), so a wording change ships only with
a before/after number. This probe loads decider-2b once, swaps the shipped prompt
constants through judge's module globals — so it measures the real code path, not a
re-implementation — and scores every variant on the three case groups defined in
judge_zh_test.py:

  dev       orig    — the original 19-case regression
  holdout   ab      — the 21 cases the 2026-09 wording pass was tuned on
  fresh     confirm — 14 cases written before that wording was drafted and never used
                      to design it; this group decided v6 over v7 (14/14 vs 13/14)

The `shipped` variant reads judge.py live, so a future wording change starts from
what is actually live. Usage:

  uv run python -B probe/judge_prompt_ab.py                      # decider, all variants
  uv run python -B probe/judge_prompt_ab.py --only shipped,v0_shipped
  uv run python -B probe/judge_prompt_ab.py --backend laya       # fallback judge
  uv run python -B probe/judge_prompt_ab.py --cases fresh        # confirmation only

Writes results/judge_prompt_ab.json (--tag adds a suffix).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import judge  # noqa: E402
from judge_zh_test import GROUPS  # noqa: E402

# Case groups live in judge_zh_test (single source with the shipped regression):
#   dev      orig    — the original 19-case regression
#   holdout  ab      — the A/B tuning set (written before the wording was chosen)
#   fresh    confirm — written before the wording was drafted, never used to design
#                      it; this is the group that arbitrates close candidates.
POOL = {"dev": GROUPS["orig"], "holdout": GROUPS["ab"], "fresh": GROUPS["confirm"]}

# Risk has no gold labels anywhere in the repo (no risk regression exists), so this is
# an ordering check, not an accuracy claim: 16 student messages with a rough expected
# severity, scored by MAE and Spearman rho against the model's 0-9 risk value. Banter
# must stay low, blame/责任 must stay high; absolute values are not the point.
RISK_SANITY: list[tuple[str, float]] = [
    ("明天降温，记得多穿点", 1.0),
    ("晚上一起吃饭不", 1.0),
    ("哈哈哈笑死我了", 0.5),
    ("周末我回家了", 1.0),
    ("笔记借我看看呗", 2.5),
    ("在吗，想找你帮个小忙", 3.0),
    ("你那个报告啥时候能给我", 3.5),
    ("这块能不能改一下", 4.0),
    ("为什么用这个数据？", 4.5),
    ("你怎么又没来上课", 5.0),
    ("怎么就你一个人没交", 6.0),
    ("这格式全不对，重做一版", 6.0),
    ("跟你说了多少遍了还错", 6.5),
    ("老师问你怎么又没交作业", 7.0),
    ("你把我们组的实验数据弄丢了？", 7.0),
    ("你把我论文原稿发群里了？", 8.0),
]


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else 0.0


def risk_metrics(rows: list[dict]) -> dict:
    expected = [r["expected"] for r in rows]
    got = [r["risk"] for r in rows]
    return {
        "mae": round(sum(abs(e - g) for e, g in zip(expected, got)) / len(rows), 2),
        "spearman": _spearman(expected, got),
        "values": {r["text"]: r["risk"] for r in rows},
    }

Q0 = "这句话的真实意图是什么？"
RISK0 = "如果直接回复这句话，风险有多大？"

D0 = dict(judge.INTENTS)
R0 = list(judge.RISK_LEVELS)

# Student-anchored descriptions: same labels, same order, wording lifted out of
# workplace register (交付/成果) into things students actually say to each other.
D1 = {
    "帮忙": "对方想让我帮他做点事，比如借笔记、带饭、代拿快递、搭把手",
    "问进度": "对方在问小组作业、报告或分工做到哪了，想催我交东西",
    "批评": "对方嫌我做的部分（作业、材料、分工）做得差，或指出我犯的错",
    "要解释": "对方质问我为什么没做、没来或这么做，要我给个说法",
    "闲聊": "对方只是在闲聊、分享日常或吐槽，没有要我做的事",
    "约时间": "对方想约个时间一起吃饭、自习、打球、开黑或碰面",
    "夸奖": "对方真心觉得我做得好，在称赞我",
}

# The question itself carries the setting; no label descriptions changed here.
Q1 = "这是同学发来的微信消息，对方想让我做什么？"
Q2 = "这是同学或室友发来的微信消息，对方主要是想让我做什么？（只是分享、没有要我做的事就算闲聊）"

# Round 2: keep v1's question and try to close its three remaining misses — the
# 「走不走」invitations (约时间) and a 「为什么还没交」question (要解释) — with cues added
# to the question and/or to just those two descriptions. Adding, never slimming.
D2 = dict(D0)
D2["约时间"] = D0["约时间"] + "，「走不走」「来不来」「一起」这类约着做某事的都算"
D2["要解释"] = D0["要解释"] + "，比如问我为什么没做、没来、没交"

D3 = dict(D0)
D3["约时间"] = D0["约时间"] + "，「走不走」「来不来」「一起」这类约着做某事的都算"

Q5 = "这是同学发来的微信消息，对方想让我做什么？（约我一起做某事的算约时间）"
Q6 = "这是同学发来的微信消息，对方想让我做什么？（约我一起做某事的算约时间；问我为什么没做或没来的算要解释）"

# risk side: only the one workplace-register rung changes (责任或利益 → 责任、评优或
# 同学关系); the question text stays as-is because "直接回复这句话" already fits both
# registers and there is no evidence a student-framed risk question helps.
R1 = list(R0)
R1[8] = "非常危险，涉及责任、评优或同学关系"

VARIANTS: dict[str, dict] = {
    # the live config, read from judge.py — always the thing to beat
    "shipped": {"question": judge.INTENT_QUESTION, "intents": dict(judge.INTENTS),
                "risk_question": judge.RISK_QUESTION, "risk": list(judge.RISK_LEVELS)},
    # records of the 2026-09 pass that picked the shipped wording (v6 == what shipped)
    "v0_shipped": {"question": Q0, "intents": D0, "risk_question": RISK0, "risk": R0},
    "v6_two_hints": {"question": Q6, "intents": D0, "risk_question": RISK0, "risk": R0},
    # rejected alternatives, kept so the table can be re-verified
    "v1_question": {"question": Q1, "intents": D0, "risk_question": RISK0, "risk": R0},
    "v2_descs": {"question": Q0, "intents": D1, "risk_question": RISK0, "risk": R0},
    "v3_both": {"question": Q1, "intents": D1, "risk_question": RISK0, "risk": R0},
    "v4_both_hint": {"question": Q2, "intents": D1, "risk_question": RISK0, "risk": R0},
    "v5_time_hint": {"question": Q5, "intents": D0, "risk_question": RISK0, "risk": R0},
    "v7_cue_both": {"question": Q1, "intents": D2, "risk_question": RISK0, "risk": R0},
    "v8_cue_time": {"question": Q1, "intents": D3, "risk_question": RISK0, "risk": R0},
    # risk-only change, same intent prompt as shipped
    "v9_risk_student": {"question": Q6, "intents": D0, "risk_question": RISK0, "risk": R1},
}


def apply_variant(v: dict) -> None:
    """Patch the shipped constants — judge() reads them as module globals at call time.

    judge_laya bound the names at import, so the laya backend needs a reload; the
    decider path (used for the intent metric) picks the patch up directly.
    """
    judge.INTENT_QUESTION = v["question"]
    judge.RISK_QUESTION = v["risk_question"]
    judge.INTENTS = v["intents"]
    judge.RISK_LEVELS = v["risk"]
    if "judge_laya" in sys.modules:
        importlib.reload(sys.modules["judge_laya"])


def score(cases, results) -> dict:
    by_gold: dict[str, list[bool]] = {}
    for (text, gold), pred in zip(cases, results):
        by_gold.setdefault(gold, []).append(pred == gold)
    per_set = {}
    for name, group in POOL.items():
        idx = [i for i, c in enumerate(cases) if c in group]
        if idx:
            per_set[name] = f"{sum(results[i] == cases[i][1] for i in idx)}/{len(idx)}"
    n = len(cases)
    return {
        "n": n,
        "acc": round(sum(1 for (t, g), p in zip(cases, results) if p == g) / n, 4),
        "per_set": per_set,
        "per_intent": {k: round(sum(v) / len(v), 2) for k, v in by_gold.items()},
        "misses": [{"text": t, "gold": g, "pred": p}
                   for (t, g), p in zip(cases, results) if p != g],
    }


def run_decider(variants: dict, cases: list, risk_cases: list) -> dict:
    j = judge.Judge(scene="general")
    t0 = time.perf_counter()
    j._load()
    load_s = time.perf_counter() - t0
    j.judge("预热")  # first forward pays MPS kernel compile; keep it out of timings
    out = {}
    for name, v in variants.items():
        apply_variant(v)
        t1 = time.perf_counter()
        preds = [j.judge(text)["intent"] for text, _ in cases]
        elapsed = time.perf_counter() - t1
        rows = [{"text": t, "expected": e, "risk": j.judge(t)["risk"]} for t, e in risk_cases]
        s = score(cases, preds)
        print(f"  {name:16s} acc={s['acc']:.3f} {s['per_set']}  risk={risk_metrics(rows)}",
              flush=True)
        out[name] = {"question": v["question"], "intents": v["intents"],
                     "risk_question": v["risk_question"], "risk": v["risk"],
                     "elapsed_s": round(elapsed, 1), "results": preds,
                     "risk_sanity": risk_metrics(rows), **s}
    out["_load_s"] = round(load_s, 1)
    return out


def run_laya(variants: dict, cases: list, risk_cases: list) -> dict:
    import judge_laya

    out = {}
    for name, v in variants.items():
        apply_variant(v)
        judge_laya = importlib.reload(judge_laya)
        j = judge_laya.LayaJudge()
        t0 = time.perf_counter()
        preds = [j.judge(text)["intent"] for text, _ in cases]
        elapsed = time.perf_counter() - t0
        rows = [{"text": t, "expected": e, "risk": j.judge(t)["risk"]} for t, e in risk_cases]
        s = score(cases, preds)
        print(f"  {name:16s} acc={s['acc']:.3f} {s['per_set']}  risk={risk_metrics(rows)}",
              flush=True)
        out[name] = {"question": v["question"], "intents": v["intents"],
                     "risk_question": v["risk_question"], "risk": v["risk"],
                     "elapsed_s": round(elapsed, 1), "results": preds,
                     "risk_sanity": risk_metrics(rows), **s}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated variant names")
    ap.add_argument("--cases", default="dev,holdout")
    ap.add_argument("--backend", default="decider", choices=("decider", "laya"))
    ap.add_argument("--tag", default="", help="suffix for the results file")
    args = ap.parse_args()

    names = [n for n in (args.only.split(",") if args.only else VARIANTS) if n]
    variants = {n: VARIANTS[n] for n in names}
    cases = [c for k in args.cases.split(",") if k for c in POOL[k]]
    run = run_decider if args.backend == "decider" else run_laya
    report = run(variants, cases, RISK_SANITY)

    print(f"\nbackend={args.backend}  cases={len(cases)}")
    for name, r in report.items():
        if name.startswith("_"):
            continue
        print(f"\n== {name}  acc={r['acc']:.3f}  ({r['n']} cases, {r['elapsed_s']}s)")
        print(f"   Q: {r['question']}")
        print("   per-set:", r["per_set"])
        print("   per-intent:", r["per_intent"])
        for m in r["misses"]:
            print(f"   XX {m['text'][:24]:26s} gold={m['gold']:4s} pred={m['pred']}")
        rs = r["risk_sanity"]
        print(f"   risk sanity: mae={rs['mae']} rho={rs['spearman']}")
        print("     " + "  ".join(f"{v:.1f}({t[:10]})" for t, v in rs["values"].items()))

    out_path = ROOT / "results" / f"judge_prompt_ab{args.tag}.json"
    out_path.parent.mkdir(exist_ok=True)
    payload = {"backend": args.backend,
               "cases": [{"text": t, "gold": g, "set": next(
                   k for k, v in POOL.items() if (t, g) in v)}
                         for t, g in cases],
               "variants": report}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
