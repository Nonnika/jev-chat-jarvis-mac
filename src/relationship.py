"""Read cautious relationship signals from the current message + recent turns.

The intent judge answers "what does this message ask me to do"; this module answers a
different, fuzzier question: "what might the other person be signalling about the
relationship, based only on the words on screen".  It is deliberately *not* a trained
classifier and not another model forward.  It is a small, deterministic cue reader
that sits beside the local judge, so adding it cannot move the intent/risk regression
and its reasoning is inspectable:

  * every returned label comes with the literal cues (`cues`) that fired;
  * a single neutral message is reported as 中性闲聊 / 信息不足, never as a confident
    "she likes you";
  * every payload carries a disclaimer because reverse-reading chat text is easy to
    overfit and hard to verify.

Cues are computed locally, with no model or network call and no logging. The reply
generator may include the cautious read alongside the chat context it already uses.
"""

from __future__ import annotations

import re

DISCLAIMER = "只是文字线索，不是读心；会受反讽、表情包、分段和 OCR 漏字影响。"

# Cue keys are what the panel/CLI shows as evidence.  Patterns are intentionally
# conservative: a false positive ("他只是在聊正事") is worse here than a missed signal.
_ACTIVE = "主动找你/问你的状态"
_INVITE = "抛出见面或一起做的安排"
_CARE = "关心你的作息或身体"
_SHARE = "主动分享自己的日常"
_PROBE = "试探你的态度或感情状态"
_EMOTION = "向你倾诉情绪、求安慰"
_PRAISE = "主动夸你"
_CONFESS = "直接表达好感或想念"
_DOUBLE = "连续发消息（不代表好感或投入程度）"
_CONFLICT = "表达对相处方式的不满"
_REPAIR = "道歉或提出好好沟通"

_DRY = "回复很短、语气平"
_AVOID = "回避或推迟"
_BOUNDARY = "边界或拒绝信号"
_OTHER = "提到其他人或感情对象"

_POSITIVE_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (_ACTIVE, re.compile(
        r"在干嘛|干嘛呢|在忙什么|在忙吗|睡了吗|睡没|忙不忙|今天干嘛|在做什么|"
        r"在吗[?？~～。. ]*$")),
    (_INVITE, re.compile(
        r"有空吗|有空不|一起(?:去|吃|玩|走|看|喝|自习|运动)?|"
        r"出来(?:吃饭|玩|走走|坐坐|喝|看|聊)?|见面|去不去|要不要一起|"
        r"来不来|走不走|想不想(?:一起|去|吃|看|玩)|约(?:你|个|一下|饭|电影|咖啡|起来|吗|不)")),
    (_CARE, re.compile(
        r"早点睡|别(?:再)?熬夜|不要再熬夜|注意身体|多穿|带伞|"
        r"到家(?:了吗|了没|没|记得|说)|吃饭没|吃了吗|"
        r"照顾好自己|多喝热水|记得吃饭|按时吃饭|晚安|好梦|睡个好觉")),
    (_CONFESS, re.compile(
        r"我(?:好像|可能|真的|越来越)?(?:喜欢|爱上)(?:上)?你(?:了)?"
        r"(?:$|[\uFF01，。？\s])|想和你在一起|做我(?:女朋友|男朋友)|"
        r"我爱你|爱你[，。！!\s]|想你了|好想你")),
    (_SHARE, re.compile(
        r"我今天|我刚刚|我刚|我发现|我最近|我昨天|给你看|跟你说|拍给你|"
        r"买了|刚看到|路过")),
    (_PROBE, re.compile(
        r"你是不是.*(?:喜欢|想|在意)|你觉得我|你对我|有没有(?:女朋友|男朋友)|"
        r"单身|喜欢(?:我|你)吗|喜不喜欢我|有(?:女朋友|男朋友)吗|想不想我|想我|想我没|在跟谁|"
        r"和谁(?:一起|吃饭|玩)|谁啊|吃醋|在意吗")),
    (_EMOTION, re.compile(
        r"好累|累死|烦死|难受|不开心|委屈|emo|睡不着|害怕|焦虑|压力|崩溃|"
        r"想哭|心情不好|好无聊|无聊|没人陪|抱抱我|陪陪我|"
        r"我(?:今天|有点|好像)?(?:发烧|生病|感冒)")),
    (_REPAIR, re.compile(
        r"对不起|刚才.*(?:语气|态度).*(?:不好|太重|凶)|"
        r"我们.*(?:好好聊|好好说|好好沟通)|不想和你吵|和好吧")),
    (_PRAISE, re.compile(
        r"你.*(?:厉害|牛|好看|可爱|优秀|有趣|靠谱|细心|幽默|帅|美)|"
        r"拍得不错|写得好|做得好|太强了")),
)

_NEGATIVE_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (_BOUNDARY, re.compile(
        r"别多想|你想多了|^想多了|我们.*不合适|^不合适(?:$|[，。！!\s])|只是朋友|做朋友|"
        r"(?:别再|不要再)(?:联系|找我|纠缠|打扰|这样)|自重|过分了|"
        r"别这样|算了吧|我(?:已经)?有(?:男朋友|女朋友|喜欢的人)|"
        r"我不(?:喜欢|爱)你(?:了)?(?:$|[，。！!\s])|不想(?:谈恋爱|继续这段关系|和你在一起)|"
        r"分手吧|需要.*(?:自己|个人).*空间")),
    (_CONFLICT, re.compile(
        r"不(?:回|理)我|不听我说|放我鸽子|不尊重我|"
        r"(?:你|跟你).*(?:敷衍|骗我|忘了.*约定|拿我.*比|擅自.*决定)|"
        r"你(?:根本|一点都|都)?不(?:在乎|关心|爱)我|"
        r"(?:我不喜欢|我讨厌)你(?:拿我|总是|每次)")),
    (_AVOID, re.compile(
        r"再说吧|下次吧|改天|最近.*忙|看情况|不方便|不用了|不需要|先这样|"
        r"回头说|有空再说|考虑一下")),
    (_OTHER, re.compile(
        r"前任|前女友|前男友|那个女生|那个男生|别的女生|别的男生|相亲|"
        r"喜欢的人|暧昧对象")),
)

# Low-information replies.  Not an insult and not proof of disinterest: many people
# text this way all day.  It only means "the current bubble carries little signal".
_DRY_RE = re.compile(
    r"^(?:嗯+|哦+|噢+|好的?|好|行|可以|收到|哈哈+|嘿嘿+|ok|OK|没事|没事儿|"
    r"行吧|好的吧)[。.~～\s]*$")

_ME = {"我", "me", "self", "自己"}

_READS = {
    "发出邀约": "对方在把话题往见面或一起做点什么上带；这不等于约会邀请，但值得给一个明确、轻松的回应。",
    "试探态度": "对方在询问你的态度或感情状态，但话还留着退路；可以坦诚，不用急着定义关系。",
    "直接表达好感": "这句话在直接表达喜欢或想念；回应可以真诚，但不用把一句话立刻升级成关系承诺。",
    "主动靠近": "对方在开话题、分享或关心你；单靠一两句话不能判断好感或关系阶段。",
    "寻求情绪价值": "对方在表达难受或希望得到陪伴；先接住感受，别急着讲道理。这不自动等于喜欢。",
    "边界/拒绝": "这句话有边界或拒绝意味；先尊重，不要硬推或连续确认。",
    "回避/降温": "对方在拒绝或推迟当前安排；可能只是忙，不能据此判断感情降温，先尊重安排。",
    "简短回应": "这句话信息量很低；可能是确认、习惯或暂时忙，看下一轮谁主动延续话题，不据此判断冷淡。",
    "关系不满": "对方明确表达了对相处方式的不满；先听清具体感受，核实事实，再谈怎么改善，别用玩笑带过。",
    "尝试和好": "对方在道歉或提出沟通；可以回应具体问题和自己的感受，不代表矛盾已经解决。",
    "中性闲聊": "当前语言线索均衡；关系推断容易变成投射，建议再看两轮互动。",
    "信息不足": "目前可用的线索不够，别靠单句脑补。",
    "提到其他感情对象": "这句话提到其他人或感情对象；可能是随口一提，别只凭这一点吃醋或下结论。",
}

_PRIORITY = {
    _BOUNDARY: "边界/拒绝",
    _CONFLICT: "关系不满",
    _AVOID: "回避/降温",
    _REPAIR: "尝试和好",
    _CONFESS: "直接表达好感",
    _INVITE: "发出邀约",
    _EMOTION: "寻求情绪价值",
    _PROBE: "试探态度",
    _ACTIVE: "主动靠近",
    _CARE: "主动靠近",
    _SHARE: "主动靠近",
    _PRAISE: "主动靠近",
    _OTHER: "提到其他感情对象",
    _DRY: "简短回应",
}


def _temperature(score: int) -> str:
    """Plain-language reading of how many positive/negative cues fired."""
    if score >= 2:
        return "偏主动"
    if score <= -2:
        return "偏冷"
    if score == 1:
        return "略主动"
    if score == -1:
        return "略冷"
    return "中性"


def _last_speaker(context: str | None) -> str:
    lines = [line for line in (context or "").splitlines() if line.strip()]
    if not lines:
        return ""
    speaker, sep, _ = lines[-1].partition(":")
    return speaker.strip() if sep else ""


def _has_double_text(context: str | None) -> bool:
    """True when the newest message follows another message from the other side."""
    speaker = _last_speaker(context)
    return bool(speaker) and speaker not in _ME and speaker != "方向未确认"


def _matches(pattern: re.Pattern[str], text: str) -> list[str]:
    """Ignore a literal cue when locally negated, without guessing sarcasm.

    Scope stops at punctuation: 「不是不想见你，是今天发烧」 must not become
    an invitation. This intentionally errs on the side of missing an ambiguous cue.
    """
    hits = []
    for match in pattern.finditer(text):
        prefix = re.split(r"[，。！？!?；;\n]", text[:match.start()])[-1]
        if re.search(r"(?:不|没|别|不是|不要|不想|没有)(?:再|太|很|真的|那么|这么)?"
                     r"(?:和你|跟你|陪你)?$", prefix):
            continue
        hits.append(match.group())
    return hits


def analyze(message: str, context: str | None = None) -> dict:
    """Return a cautious, explainable relationship read for one incoming message.

    `context` is the same speaker-prefixed transcript the judge receives, for example
    a line like ``"我: 在忙"`` followed by ``"对方: 刚忙完"``.  The function never
    raises on missing or odd input: an empty message yields the 信息不足 payload.
    """
    text = (message or "").strip()
    found: list[str] = []
    evidence: dict[str, list[str]] = {}
    score = 0

    for cue, pattern in _POSITIVE_CUES:
        hits = _matches(pattern, text)
        if hits:
            found.append(cue)
            evidence[cue] = hits
            score += 1
    for cue, pattern in _NEGATIVE_CUES:
        hits = _matches(pattern, text)
        # A concrete replacement invitation is engagement, not a cooling verdict.
        if cue == _AVOID and _INVITE in found and re.search(
                r"(?:今天|明天|今晚|明晚|周[一二三四五六日天末]|星期|\d+[点号])", text):
            continue
        # Declining help that is already unnecessary doesn't signal a relationship.
        if cue == _AVOID and re.search(r"(?:已经|已)(?:到家|到了|吃过|吃完|买好|办好)", text):
            continue
        if hits:
            found.append(cue)
            evidence[cue] = hits
            score -= 1
    if text and _DRY_RE.match(text):
        found.append(_DRY)
        evidence[_DRY] = [text]
    if text and _has_double_text(context):
        found.append(_DOUBLE)

    if not text:
        label = "信息不足"
    else:
        label = "中性闲聊"
        for cue, name in _PRIORITY.items():
            if cue in found:
                label = name
                break

    if not found and text and text.endswith(("?", "？")):
        # A question with no other cue is engagement, but too weak to pick a direction.
        found.append("用问句把话题递回给你")
        evidence[found[-1]] = [text]

    read = _READS["信息不足"] if not text else _READS.get(label, _READS["中性闲聊"])
    return {
        "label": label,
        "temperature": _temperature(score),
        "score": max(-3, min(3, score)),
        "cues": found,
        "evidence": evidence,
        "read": read,
        "disclaimer": DISCLAIMER,
    }

if __name__ == "__main__":
    import json
    import sys

    msg = sys.argv[1] if len(sys.argv) > 1 else "周末有空吗"
    ctx = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(analyze(msg, ctx), ensure_ascii=False, indent=1))
