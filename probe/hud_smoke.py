"""HUD panel smoke test: synthetic payloads, offscreen render, no screen access.

Run: uv run python -B probe/hud_smoke.py
Renders the real panel to /tmp/jev-hud-smoke.png and checks collapse behaviour.
No WeChat window, no OCR, no model calls, no credentials: the apply* callbacks
are pure UI, and the read loop only starts in main().
"""
import sys
from pathlib import Path

import AppKit
from Foundation import NSDate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hud import HudController, PANEL_W  # noqa: E402
from judge import ACTION_MAP

OUT = Path('/tmp/jev-hud-smoke.png')

# One turn per visual line, the way _context_text(preview=True) flattens them:
# a group-chat sender name, an own reply, a folded turn (line breaks -> " / "),
# and an over-long line that must ellipsize inside the field, not wrap.
RECOGNIZED = (
    '张三: 这个方案周五下班前能给到吗\n'
    '我: 尽量，周末前一定\n'
    '张三: 老板原话是「这版不行 / 重做 / 周五前我要看到」\n'
    '李四: 一条特别特别特别长的消息用来检查超宽单行是否按预期截断而不是把布局撑坏'
)

VERDICT = {
    'intent': '批评', 'confidence': 0.86, 'risk': 6.0,
    'message': '这做的什么玩意，重做。',
    'actions': ACTION_MAP['批评'],
    'backend': 'local decider-2b',
    'relationship': {
        'label': '试探态度', 'temperature': '略主动', 'score': 1,
        'cues': ['试探你的态度或感情状态'],
        'evidence': {'试探你的态度或感情状态': ['你觉得我怎么样']},
        'read': '对方在询问你的态度，但还留着退路。',
        'disclaimer': '只是文字线索，不是读心。',
    },
}


def render(controller, path):
    view = controller.panel.contentView()
    rect = view.bounds()
    rep = view.bitmapImageRepForCachingDisplayInRect_(rect)
    view.cacheDisplayInRect_toBitmapImageRep_(rect, rep)
    rep.representationUsingType_properties_(
        AppKit.NSBitmapImageFileTypePNG, {}).writeToFile_atomically_(str(path), True)


app = AppKit.NSApplication.sharedApplication()
app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
controller = HudController.alloc().init()
controller.applyChat_('产品研发群')
controller.applyPending_(('这做的什么玩意，重做。', '张三', '尽量，周末前一定', RECOGNIZED))
controller.applyJudgment_((VERDICT, '张三', '尽量，周末前一定', RECOGNIZED))
assert '关系信号：试探态度' in controller.rows['sender'].stringValue()
assert '你觉得我怎么样' in controller.rows['sender'].toolTip()
# 功耗读数: a synthetic reading — the real meter needs IOReport and is verified by
# `uv run python src/power.py` plus tests/test_power.py. Here only the wiring matters:
# the tick path calls _update_power(), which renders and sets the tooltip.
controller._power.sample = lambda: ('电池 8.4 W', '电池 8.4 W（放电 ≈ 整机功耗）\nGPU 1.2 W')
controller._update_power()
assert controller.rows['power'].stringValue() == '电池 8.4 W', controller.rows['power'].stringValue()
AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
    NSDate.dateWithTimeIntervalSinceNow_(0.2))

# Layout sanity: the recognized field sits inside the first surface card and above
# the intent section — catch an accidental overlap without eyeballing pixels.
# (Before collapsing: the window resize re-parents the content view's frame.)
card_bottom = 54 + 118
rec = controller.rows['recognized'].frame()
view_h = controller.panel.contentView().frame().size.height
rec_top = view_h - rec.origin.y - rec.size.height     # AppKit bottom-origin -> top
rec_bottom = rec_top + rec.size.height
assert rec_top >= 96 + 14, f'recognized overlaps sender row ({rec_top:.0f})'
assert rec_bottom <= card_bottom, f'recognized escapes its card ({rec_bottom:.0f})'
intent = controller.rows['intent'].frame()
intent_top = view_h - intent.origin.y - intent.size.height
assert intent_top >= card_bottom + 8, f'intent section overlaps card ({intent_top:.0f})'
assert controller.rows['recognized'].stringValue() == RECOGNIZED

# 功耗 sits between the chat name and the gear, on the same visual row.
chat = controller.rows['chat'].frame()
power = controller.rows['power'].frame()
gear = controller.settings_button.frame()
chat_top = view_h - chat.origin.y - chat.size.height
power_top = view_h - power.origin.y - power.size.height
assert power.origin.x >= chat.origin.x + chat.size.width, 'power overlaps the chat name'
assert power.origin.x + power.size.width <= gear.origin.x, 'power runs under the gear'
assert abs(power_top - chat_top) <= 4, f'power is not on the chat row ({power_top:.0f})'
assert power_top + power.size.height <= 54, 'power spilled into the first card'

# 生成中占位: after the verdict lands both active slots are waiting for candidates —
# each reserved row area must show its spinner placeholder, and the header must be busy.
assert controller._pending_slots == {0, 1}, controller._pending_slots
assert controller.rows['cand_header'].stringValue() == '候选回复 · 生成中…'
for slot in (0, 1):
    spin, ph = controller._group_placeholders[slot]
    assert not spin.isHidden() and not ph.isHidden(), f'slot {slot} placeholder missing'
render(controller, OUT)

# The first streamed line retires its slot's placeholder and shows the row instead.
controller.applyStreamLine_((controller._gen_epoch, 0, '在的在的，明早九点前给你，不用等今晚'))
assert 0 not in controller._pending_slots, 'streamed line did not retire the placeholder'
spin0, ph0 = controller._group_placeholders[0]
assert spin0.isHidden() and ph0.isHidden(), 'slot 0 placeholder still up after its line'
assert not controller._rows[0][0]['text'].isHidden()
assert 1 in controller._pending_slots, "slot 1 is still generating and must stay pending"
AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
    NSDate.dateWithTimeIntervalSinceNow_(0.2))
render(controller, Path('/tmp/jev-hud-smoke-streaming.png'))

# The latest incoming message must display all wrapped lines, including explicit
# newlines. Growing this card must move the context and every later section down;
# replacing it with a short message must release that space again.
short_h = controller.panel.frame().size.height
long_message = ('今晚想和你说件事，刚才发送的那条消息可能没表达清楚。\n'
                '我搜索了附近的餐厅，想问问你有没有空一起吃饭。\n'
                '如果你今天累了，我们也可以改天再约，你先好好休息。')
controller.applyIncoming_((long_message, None, None, RECOGNIZED))
message_frame = controller.rows['message'].frame()
assert message_frame.size.height > 60, 'latest message still has a fixed/clipped height'
assert controller.rows['message'].toolTip() == long_message
extra = message_frame.size.height - 30
assert abs(controller.panel.frame().size.height - short_h - extra) < .5
rec = controller.rows['recognized'].frame()
assert rec.origin.y + rec.size.height < message_frame.origin.y, 'context overlaps message'
assert controller._message_surface.frame().size.height == 118 + extra
assert controller.rows['message'].cell().lineBreakMode() == AppKit.NSLineBreakByWordWrapping
render(controller, Path('/tmp/jev-hud-smoke-long-message.png'))
controller.applyIncoming_(('好呀', None, None, RECOGNIZED))
assert controller.rows['message'].frame().size.height == 30
assert abs(controller.panel.frame().size.height - short_h) < .5, 'short message did not shrink'

# Collapsed: the new row hides with the other details and the window shrinks.
# (No PNG here — an offscreen render after the window resize comes out blank;
# the hiding itself is what matters.)
controller._set_collapsed(True)
AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
    NSDate.dateWithTimeIntervalSinceNow_(0.2))
assert controller.rows['recognized'].isHidden()
assert not controller.rows['power'].isHidden(), 'readout must survive collapsing'
assert controller.rows['power'].stringValue() == '电池 8.4 W'
assert controller.panel.frame().size.height < controller._expanded_h
collapsed_h = controller.panel.frame().size.height
controller.applyIncoming_((long_message, None, None, RECOGNIZED))
assert controller.panel.frame().size.height == collapsed_h, 'incoming message expanded collapsed HUD'
# the still-pending slot's placeholder must be gone from the rolled-up strip, and a
# candidates push while collapsed must not grow the window back
spin1, ph1 = controller._group_placeholders[1]
assert spin1.isHidden() and ph1.isHidden(), 'placeholder visible in the collapsed strip'
controller.applyCandidates_([(0, controller.slot_tones[0], [
    {'text': '在的在的，明早九点前给你', 'prob': 0.82},
    {'text': '收到，明早给', 'prob': 0.64}])])
assert controller.panel.frame().size.height < controller._expanded_h, \
    'a late payload grew the collapsed window back'
# the surviving rows must sit INSIDE the strip, not at their old full-height coordinates
strip_h = controller.panel.contentView().frame().size.height
for key in ('chat', 'status', 'power'):
    f = controller.rows[key].frame()
    assert f.origin.y >= 0 and f.origin.y + f.size.height <= strip_h + 0.5, \
        f'{key} is outside the collapsed strip (y={f.origin.y:.0f}, strip {strip_h:.0f})'
controller._set_collapsed(False)
AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
    NSDate.dateWithTimeIntervalSinceNow_(0.2))
assert not controller.rows['recognized'].isHidden()

controller.panel.orderOut_(None)
print(f'PASS: panel rendered -> {OUT} + streaming state (panel width {PANEL_W})')
