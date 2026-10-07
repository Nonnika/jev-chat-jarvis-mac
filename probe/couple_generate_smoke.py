"""Generate synthetic relationship replies through the configured provider.

This deliberately calls the generation API; no screen/WeChat/fill interaction.
Never prints credentials. Structural checks aren't a semantic accuracy claim:
review the printed replies for respect of boundaries and unsupported assumptions.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from generate import Generator

CASES = [
    ("安慰", "我今天有点委屈", "我: 今天面试怎么样\n对方: 感觉没有发挥好"),
    ("边界", "暂时不想谈恋爱，我们慢慢了解吧", "我: 要不要试着在一起"),
    ("矛盾", "你总是只顾着玩手机，不听我说话", "我: 我们聊聊昨天的事吧"),
]

if __name__ == "__main__":
    gen = Generator()
    failed = False
    for scene, message, context in CASES:
        out = gen.generate(message, "", ["自然关心", "轻松甜一点", "认真沟通"], context)
        print(f"{scene}:", flush=True)
        for group in out["groups"]:
            texts = group["texts"]
            ok = len(texts) == 2 and all(len(t) <= 30 for t in texts)
            failed |= not ok
            print(f"  {group['tone']} {'PASS' if ok else 'FAIL'}: {texts}", flush=True)
        if not out["groups"]:
            print("  FAIL: 没有可用候选，请检查生成层配置", flush=True)
            failed = True
    raise SystemExit(1 if failed else 0)
