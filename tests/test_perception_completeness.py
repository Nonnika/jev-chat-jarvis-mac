"""Message completeness regressions using synthetic OCR, without screen/API access."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import perception as p


def line(text, top, height=16, window_height=700, x=.40, width=.25):
    return p.TextBlock(text, 1.0, x, 1 - (top + height) / window_height,
                       width, height / window_height)


class CompletenessTests(unittest.TestCase):
    def test_ui_words_inside_body_do_not_discard_whole_line(self):
        texts = ['我搜索了附近的餐厅', '刚才发送的照片你看到了吗',
                 '我不知道该输入文字还是发语音', '我们一共3个人', '明天搜索一下吧']
        for text in texts:
            with self.subTest(text=text):
                messages = p.extract_messages([line(text, 200, height=25)])
                self.assertEqual([m.text for m in messages], [text])

    def test_wrapped_body_with_ui_word_keeps_every_line(self):
        messages = p.extract_messages([
            line('我想和你说件事', 180, height=25),
            line('刚才发送的那条消息', 208, height=25),
            line('是想问你今晚有没有空', 236, height=25),
        ])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].text,
                         '我想和你说件事\n刚才发送的那条消息\n是想问你今晚有没有空')

    def test_literal_label_or_time_inside_bubble_is_a_message(self):
        for text in ('发送', '搜索', '输入文字', '10:30'):
            with self.subTest(text=text):
                messages = p.extract_messages([line(text, 200)], window_height=700)
                self.assertEqual([m.text for m in messages], [text])
        self.assertFalse(p.extract_messages([line('10:30', 200, x=.62, width=.08)],
                                           window_height=700))

    def test_short_body_survives_window_height_changes(self):
        for height in (400, 679, 900, 1200):
            with self.subTest(window_height=height):
                messages = p.extract_messages([line('好呀', height / 2,
                    window_height=height)], window_height=height)
                self.assertEqual([m.text for m in messages], ['好呀'])
                self.assertEqual(messages[0].side, 'them')

    def test_wrapped_message_survives_small_and_tall_windows(self):
        for height in (400, 679, 900, 1200):
            with self.subTest(window_height=height):
                blocks = [line(text, height / 3 + i * 22, window_height=height)
                          for i, text in enumerate(['今晚要不要一起吃饭', '我搜索了附近的餐厅', '你想吃哪家呀'])]
                messages = p.extract_messages(blocks, window_height=height)
                self.assertEqual(len(messages), 1)
                self.assertEqual(messages[0].lines,
                                 ['今晚要不要一起吃饭', '我搜索了附近的餐厅', '你想吃哪家呀'])

    def test_small_sender_attaches_without_folding_into_body(self):
        blocks = [line('小王', 180, height=12), line('今晚有空吗', 215)]
        messages = p.extract_messages(blocks, window_height=700)
        self.assertEqual([m.text for m in messages], ['今晚有空吗'])
        self.assertEqual(messages[0].sender, '小王')
        self.assertEqual(p.extract_messages([blocks[0]], window_height=700)[0].text, '小王')

    def test_ocr_height_fluctuation_does_not_delete_body(self):
        messages = p.extract_messages([line('第一行完整消息', 180, height=13),
            line('第二行完整消息', 202, height=18)], window_height=700)
        self.assertEqual([m.text for m in messages], ['第一行完整消息\n第二行完整消息'])
        self.assertIsNone(messages[0].sender)
        messages = p.extract_messages([line('好呀', 450, height=13,
                                           window_height=900)], window_height=900)
        self.assertEqual([m.text for m in messages], ['好呀'])

    def test_separate_bubbles_and_opposite_sides_stay_separate(self):
        blocks = [line('好呀', 180), line('你想吃什么', 222),
                  line('我来选', 244, x=.80, width=.08)]
        messages = p.extract_messages(blocks, window_height=700)
        self.assertEqual([m.text for m in messages], ['好呀', '你想吃什么', '我来选'])
        self.assertEqual([m.side for m in messages], ['them', 'them', 'me'])

    def test_extraction_does_not_mutate_vision_coordinates(self):
        blocks = [line('完整消息', 180, height=25)]
        original_y = blocks[0].y
        first = p.extract_messages(blocks)
        second = p.extract_messages(blocks)
        self.assertEqual(blocks[0].y, original_y)
        self.assertEqual(first, second)

    def test_read_conversation_uses_actual_window_height(self):
        win = p.WindowInfo(1, 1, 'WeChat', 0, 0, 1000, 900)
        with patch.object(p, 'find_wechat_window', return_value=win), \
             patch.object(p, 'capture_image', return_value=object()), \
             patch.object(p, '_fingerprint', return_value=b'new'), \
             patch.object(p, 'ocr_image', return_value=[line('好呀', 450, window_height=900)]):
            result = p.read_conversation()
        self.assertEqual([m.text for m in result['messages']], ['好呀'])


if __name__ == '__main__':
    unittest.main()
