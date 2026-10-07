"""Native settings smoke test on macOS, using only temporary files and a local HTTP server.

Run: uv run python -B probe/settings_smoke.py
Renders the real window offscreen to /tmp/jev-settings-smoke.png.
Does not read messages, real credentials, or modify the user's configuration.
"""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import patch

import AppKit as A
from Foundation import NSDate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from test_settings import Server, SettingsNetwork
import userconfig
import settings_config
from settings import SettingsController


def wait_for_request(controller):
    deadline = time.monotonic() + 5
    while controller.busy and time.monotonic() < deadline:
        A.NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.02))
    assert not controller.busy, 'UI never received request completion'
    assert controller.save_button.isEnabled()


def request_button(controller, prefix, title):
    item = next(i for i in controller.tabs.tabViewItems() if i.identifier() == prefix)
    return next(v for v in item.view().subviews() if isinstance(v, A.NSButton) and v.title() == title)


app = A.NSApplication.sharedApplication()
app.setActivationPolicy_(A.NSApplicationActivationPolicyRegular)
SettingsNetwork.setUpClass()
try:
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), patch.object(userconfig, '_startup_sources', None), patch.object(userconfig, 'env_files', return_value=[Path(directory) / 'env']), patch.object(userconfig, 'PROJECT_ENV', Path(directory) / '.env'):
        path = Path(directory) / 'env'
        path.write_text('# keep\nJEV_TONES="名字=说明"\n')
        userconfig.load()
        c = SettingsController.alloc().init().build()
        c.show()
        assert c.scene_field.titleOfSelectedItem() == '通用聊天'
        c.scene_field.selectItemAtIndex_(c.scene_keys.index('relationship'))
        c.sceneChanged_(c.scene_field)
        fields = c.fields['OPENAI']
        fields['API_KEY'].setStringValue_('test-only-key')
        fields['BASE_URL'].setStringValue_(SettingsNetwork.base + '/v1')
        fields['MODEL'].setStringValue_('typed-model')
        Server.response = {'data': [{'id': 'served-model'}]}
        Server.code = 200
        # Hold the synthetic request until disabled controls have been observed;
        # a loopback reply can otherwise finish during performClick's animation.
        release = threading.Event()
        list_models = settings_config.list_models
        def delayed_list(*args):
            assert release.wait(5), 'request was not released'
            return list_models(*args)
        with patch.object(settings_config, 'list_models', side_effect=delayed_list):
            try:
                request_button(c, 'OPENAI', '获取模型列表').performClick_(None)
                assert not c.save_button.isEnabled()
                assert not c.scene_field.isEnabled()
            finally:
                release.set()
            wait_for_request(c)
        assert fields['MODEL'].objectValues() == ['served-model']
        assert fields['MODEL'].stringValue() == 'typed-model', 'must not silently switch model'
        Server.response = {'choices': [{'message': {'content': '连接成功'}}]}
        request_button(c, 'OPENAI', '测试连接').performClick_(None)
        wait_for_request(c)
        assert '连接成功' in c.status.stringValue(), c.status.stringValue()
        Server.code = 401
        request_button(c, 'OPENAI', '获取模型列表').performClick_(None)
        wait_for_request(c)
        assert '401' in c.status.stringValue() and '手填' in c.status.stringValue()
        assert fields['MODEL'].isEnabled()
        c.save_button.performClick_(None)
        assert '已保存' in c.status.stringValue(), c.status.stringValue()
        assert userconfig.parse_env_file(path)['OPENAI_MODEL'] == 'typed-model'
        assert userconfig.parse_env_file(path)['JEV_CHAT_SCENE'] == 'relationship'
        assert '# keep\nJEV_TONES="名字=说明"\n' in path.read_text()
        assert path.stat().st_mode & 0o777 == 0o600
        assert not userconfig.get('OPENAI_API_KEY'), 'must not hot reload'
        assert userconfig.chat_scene() == 'general', 'scene must not hot reload'
        assert not c.changed()
        c.tabs.selectTabViewItemAtIndex_(1)
        A.NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.2))
        c.window.display()
        view = c.window.contentView()
        # The NSWindow background isn't part of contentView's cached bitmap.
        background = A.NSBox.alloc().initWithFrame_(view.bounds())
        background.setBoxType_(A.NSBoxCustom)
        background.setBorderType_(A.NSNoBorder)
        background.setFillColor_(A.NSColor.whiteColor())
        view.addSubview_positioned_relativeTo_(background, A.NSWindowBelow, None)
        rect = view.bounds()
        rep = view.bitmapImageRepForCachingDisplayInRect_(rect)
        view.cacheDisplayInRect_toBitmapImageRep_(rect, rep)
        rep.representationUsingType_properties_(A.NSBitmapImageFileTypePNG, {}).writeToFile_atomically_('/tmp/jev-settings-smoke.png', True)
        c.window.close()
        reopened = SettingsController.alloc().init().build()
        assert reopened.fields['OPENAI']['MODEL'].stringValue() == 'typed-model'
        assert reopened.scene_field.titleOfSelectedItem() == '情侣与暧昧'
        # Existing keychain expression remains byte-for-byte when editing only the model.
        path.write_text('export OPENAI_API_KEY="$(security find-generic-password -w)" # keep expression\nOPENAI_MODEL=old\n')
        shell = SettingsController.alloc().init().build()
        shell.fields['OPENAI']['MODEL'].setStringValue_('new-model')
        shell.save_button.performClick_(None)
        assert 'export OPENAI_API_KEY="$(security find-generic-password -w)" # keep expression\n' in path.read_text()
        assert shell.fields['OPENAI']['API_KEY'].stringValue() == ''
        print('PASS: native buttons, async completion, models/manual entry, HTTP failure, secure save, scene/restart isolation, reopen, shell-expression preservation')
finally:
    SettingsNetwork.tearDownClass()
