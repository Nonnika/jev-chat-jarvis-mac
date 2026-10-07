"""Chinese intent-judgment test: can a local Jev-shaped model read a student message?

Zero-shot, no training — this measures how much labeling work the app will need.
Cases are realistic student WeChat messages (group projects, borrowed notes, dorm
banter); judge.INTENTS is written for student chats, so gold labels use that set.

The 54 cases come in three groups with different provenance, and the report scores
them separately — a tuning set and a set that never influenced the wording must not
be added into one flattering number (probe/judge_prompt_ab.py relies on the split):

  orig      the original 19-case regression, the historical README caliber
  ab        written before the current INTENT_QUESTION wording was chosen, used to
            pick it (the A/B tuning set)
  confirm   written before that wording was drafted and never used to design it,
            so it arbitrates between close candidates on its own
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# 直接 import 判断层的 INTENTS：回归测的必须是线上真正发出的那份 prompt。
# 此前这里是一份手工同步的副本，judge.py 改了描述这里不会跟着变，回归就在
# 测一个没人用的配置（与 dtype 那条注释是同一个原则）。
from judge import INTENTS, INTENT_QUESTION
from judge_laya import laya_model

# (message text, gold intent) — realistic student-chat cases. 嘲讽 intentionally has
# no cases: the label was removed from judge.INTENTS (tone labels were where the weak
# backends went near-uniform); if it comes back, add its cases back here.
CASES_ORIG: list[tuple[str, str]] = [
    ("在吗，想找你帮个小忙", "帮忙"),
    ("那个…你高数的笔记能借我看看吗", "帮忙"),
    ("明天早上能帮我带份早饭不", "帮忙"),
    ("你那部分写完了吗", "问进度"),
    ("作业你做到哪了", "问进度"),
    ("报告啥时候能发我", "问进度"),
    ("你这做的也太敷衍了吧", "批评"),
    ("这格式全不对，重做一版", "批评"),
    ("跟你说了多少遍了还错", "批评"),
    ("你昨天怎么没来开会", "要解释"),
    ("怎么就你一个人没交", "要解释"),
    ("为什么用这个数据？", "要解释"),
    ("哈哈哈笑死我了", "闲聊"),
    ("今天食堂新出的那个真不错", "闲聊"),
    ("困死了，不想上课", "闲聊"),
    ("明晚八点开黑，来不来", "约时间"),
    ("周六下午图书馆自习走不走", "约时间"),
    ("牛啊，这你都能做出来", "夸奖"),
    ("你笔记记得也太全了吧", "夸奖"),
]

# A/B tuning set: the 「走不走」invitations and 「为什么还没交」questions that the old
# wording read as 闲聊/问进度.
CASES_AB: list[tuple[str, str]] = [
    ("帮我带瓶水呗，我在实验室走不开", "帮忙"),
    ("你上次的大物实验报告能发我参考下不", "帮忙"),
    ("选课的时候顺手帮我占个名额行吗", "帮忙"),
    ("小组Pre的PPT你那边做完了吗", "问进度"),
    ("你那段代码推到仓库了吗", "问进度"),
    ("论文初稿大概啥时候能给我", "问进度"),
    ("你交的这版数据全算错了", "批评"),
    ("排版乱成这样，能不能认真点", "批评"),
    ("这写的啥啊，根本没法用", "批评"),
    ("你怎么又没来上课", "要解释"),
    ("为什么你的部分拖到现在还没交", "要解释"),
    ("你上次那个数据是怎么算的，老师问我了", "要解释"),
    ("今天热死了，完全不想动", "闲聊"),
    ("刚刷到一个特别搞笑的视频", "闲聊"),
    ("困了，先睡了啊", "闲聊"),
    ("晚上一起去食堂吃饭不", "约时间"),
    ("周末一起打球不", "约时间"),
    ("明早八点图书馆占座走不走", "约时间"),
    ("牛啊，这么快就写完了", "夸奖"),
    ("你这次考得也太高了吧", "夸奖"),
    ("你做的这个封面真好看", "夸奖"),
]

# Confirmation set: same 「怎么没做/没交」and 「走不走」patterns in fresh phrasings,
# plus ones the tuning pass never touched (「这数据一看就不对」).
CASES_CONFIRM: list[tuple[str, str]] = [
    ("帮我占个座呗，我下课晚", "帮忙"),
    ("你做的图能借我用一下吗", "帮忙"),
    ("分工那部分你弄到哪了", "问进度"),
    ("实验报告写得怎么样了", "问进度"),
    ("这数据一看就不对", "批评"),
    ("你这排版看着好乱", "批评"),
    ("你怎么没按说好的做", "要解释"),
    ("这次怎么又没交上去", "要解释"),
    ("今天食堂排队排了好久", "闲聊"),
    ("哈哈哈你也太惨了", "闲聊"),
    ("明晚一起开黑不", "约时间"),
    ("晚上去操场跑步走不走", "约时间"),
    ("可以啊，这都能被你想到", "夸奖"),
    ("你写的这段我想直接抄了", "夸奖"),
]

CASES: list[tuple[str, str]] = CASES_ORIG + CASES_AB + CASES_CONFIRM

# group -> cases, for the per-group breakdown (and for probe/judge_prompt_ab.py)
GROUPS: dict[str, list[tuple[str, str]]] = {
    "orig": CASES_ORIG, "ab": CASES_AB, "confirm": CASES_CONFIRM,
}
CASE_GROUP = {case: name for name, group in GROUPS.items() for case in group}


def run_decider() -> dict:
    """Use the real two-slot app path, not a separate intent-only prompt.

    The old probe omitted the risk slot and reported 53/54; the real judge's
    two-slot forward on the same cached model scored 49/54 before this pass.
    Keep that historical number separate instead of hiding a pipeline difference.
    """
    from judge import Judge
    backend = Judge()
    backend.warm()

    t0 = time.perf_counter()
    results = []
    for text, gold in CASES:
        out = backend.judge(text)
        results.append({"text": text, "gold": gold, "pred": out["intent"],
                        "conf": out["confidence"]})
    return {"model": "decider-2b", "elapsed_s": time.perf_counter() - t0, "results": results}


def run_laya_coreml() -> dict:
    """laya-coreml (aac6fef/laya-multilingual-coreml) — the shipped judge backend.

    Must build its question dict exactly like judge_laya.LayaJudge.judge: a regression
    measuring a different prompt is measuring a configuration nobody ships.
    """
    from judge_laya import LayaJudge
    backend = LayaJudge()

    t0 = time.perf_counter()
    results = []
    for text, gold in CASES:
        try:
            out = backend.judge(text)
            pred = out["intent"]
            conf = out["confidence"]
        except Exception as e:
            pred, conf = f"ERR:{type(e).__name__}", 0.0
        results.append({"text": text, "gold": gold, "pred": pred, "conf": conf})
    return {"model": f"laya-coreml ({laya_model()})", "elapsed_s": time.perf_counter() - t0,
            "results": results}


def summarize(run: dict) -> dict:
    res = run["results"]
    n = len(res)
    correct = sum(1 for r in res if r["pred"] == r["gold"])
    by_gold: dict[str, list[bool]] = {}
    by_group: dict[str, list[bool]] = {}
    for r in res:
        ok = r["pred"] == r["gold"]
        by_gold.setdefault(r["gold"], []).append(ok)
        by_group.setdefault(CASE_GROUP[(r["text"], r["gold"])], []).append(ok)
    return {
        "model": run["model"],
        "acc": correct / n,
        "n": n,
        "elapsed_s": round(run["elapsed_s"], 1),
        "per_intent": {k: round(sum(v) / len(v), 2) for k, v in by_gold.items()},
        # 分项报分：orig 是历史口径，ab 是调词集，confirm 只用于确认最终措辞
        "per_group": {k: f"{sum(v)}/{len(v)}" for k, v in by_group.items()},
        "majority_baseline": round(max(
            sum(1 for r in res if r["gold"] == g) for g in set(r["gold"] for r in res)) / n, 3),
    }


def main() -> None:
    out_path = Path("results/judge_zh.json")
    out_path.parent.mkdir(exist_ok=True)
    report = {}

    for name, fn in (("decider", run_decider), ("laya_coreml", run_laya_coreml)):
        print(f"\n===== {name} =====", flush=True)
        try:
            run = fn()
            s = summarize(run)
            report[name] = {"summary": s, "results": run["results"]}
            print(f"acc={s['acc']:.3f}  n={s['n']}  elapsed={s['elapsed_s']}s  "
                  f"majority_baseline={s['majority_baseline']}")
            print("per-intent:", s["per_intent"])
            print("per-group:", s["per_group"])
            for r in run["results"]:
                flag = "OK " if r["pred"] == r["gold"] else "XX "
                print(f"  {flag}{r['text'][:22]:24s} gold={r['gold']:5s} "
                      f"pred={str(r['pred']):6s} conf={r['conf']:.2f}")
        except Exception as e:
            import traceback
            report[name] = {"error": f"{type(e).__name__}: {e}"}
            print("FAILED:", e)
            traceback.print_exc()

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
