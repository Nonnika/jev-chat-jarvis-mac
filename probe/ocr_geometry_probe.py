"""Compare Vision ROI/full-image coordinates on a synthetic 1x Chinese image.

No screen capture, WeChat interaction, API or credentials. Exit nonzero if the
same recognized text changes geometry just because OCR used a region of interest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import AppKit as A
from Foundation import NSString, NSMakePoint, NSMakeRect
from perception import ocr_image, extract_messages

TEXTS = ('我搜索了附近的餐厅', '刚才发送的照片', '好呀', '自己发出的消息')


def synthetic_image(font_size=16, window_height=700, scale=1):
    width, height = 1000 * scale, window_height * scale
    rep = A.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, width, height, 8, 4, True, False, A.NSDeviceRGBColorSpace, width * 4, 32)
    ctx = A.NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    A.NSGraphicsContext.saveGraphicsState()
    try:
        A.NSGraphicsContext.setCurrentContext_(ctx)
        A.NSColor.whiteColor().setFill()
        A.NSRectFill(NSMakeRect(0, 0, width, height))
        attrs = {A.NSFontAttributeName: A.NSFont.systemFontOfSize_(font_size * scale),
                 A.NSForegroundColorAttributeName: A.NSColor.blackColor()}
        for text, x, y in [(TEXTS[0], 400, window_height * .68),
                           (TEXTS[1], 400, window_height * .68 - 22),
                           (TEXTS[2], 400, window_height * .40),
                           (TEXTS[3], 780, window_height * .50)]:
            NSString.stringWithString_(text).drawAtPoint_withAttributes_(
                NSMakePoint(x * scale, y * scale), attrs)
    finally:
        A.NSGraphicsContext.restoreGraphicsState()
    return rep.CGImage()


if __name__ == '__main__':
    image = synthetic_image()
    full = {b.text: b for b in ocr_image(image, chat_only=False)}
    cropped = {b.text: b for b in ocr_image(image, chat_only=True)}
    assert len(full) == 4 and full.keys() == cropped.keys(), 'synthetic text was missed'
    delta = max(abs(getattr(full[t], key) - getattr(cropped[t], key))
                for t in full for key in ('x', 'y', 'w', 'h'))
    print(f'synthetic blocks={len(full)}, max ROI/full coordinate delta={delta:.4f}')
    assert delta < .005, 'ROI coordinates differ from full-image coordinates'
    for height in (400, 679, 900):
        for scale in (1, 2):
            blocks = ocr_image(synthetic_image(14, height, scale))
            assert len(blocks) == 4, 'Vision missed a synthetic text block'
            recognized = {('me' if b.x >= .66 else 'first' if b.y > .55
                           else 'last'): [] for b in blocks}
            for b in sorted(blocks, key=lambda b: -b.y):
                key = 'me' if b.x >= .66 else 'first' if b.y > .55 else 'last'
                recognized[key].append(b.text)
            messages = extract_messages(blocks, window_height=height)
            # Vision itself can omit a glyph. Extraction must retain every recognized
            # character, without pretending a downstream guess is an OCR correction.
            assert [m.text for m in messages] == ['\n'.join(recognized['first']),
                recognized['me'][0], recognized['last'][0]], \
                f'body lost/split at window height {height}, scale {scale}'
            assert [m.side for m in messages] == ['them', 'me', 'them']
            exact = sum(b.text in TEXTS for b in blocks)
            print(f'extraction: height={height}, scale={scale}, messages=3/3, '
                  f'Vision exact text blocks={exact}/4')
    print('PASS: Vision geometry and full-message extraction at 1x/2x')
