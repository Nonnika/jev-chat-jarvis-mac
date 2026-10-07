"""Offline regression for the judge model cache location. Run: python -B -m unittest discover -s tests.

The models are ~8 GB, so where they land is a user-visible decision: a source checkout
keeps them in <project>/.models/ (gitignored), while a built .app must NOT — its copy
would be wiped by every app update, so it keeps the user-level Hugging Face cache.

These tests only exercise the pure decision function; importing userconfig also applies
the default, which is why the env assertions restore os.environ afterwards.
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import userconfig


class ModelCacheRoot(unittest.TestCase):
    def test_source_checkout_uses_dot_models(self):
        self.assertEqual(userconfig.model_cache_root(Path('/Users/x/Code/jev-chat-jarvis-mac')),
                         Path('/Users/x/Code/jev-chat-jarvis-mac/.models'))

    def test_app_bundle_keeps_the_user_cache(self):
        bundle = Path('/Applications/jev-jarvis.app/Contents/Resources/app')
        self.assertIsNone(userconfig.model_cache_root(bundle))

    def test_repo_root_matches_this_checkout(self):
        self.assertEqual(userconfig.model_cache_root(ROOT), ROOT / '.models')
        self.assertEqual(userconfig.REPO_MODEL_CACHE, ROOT / '.models')

    def test_defaults_do_not_override_an_explicit_choice(self):
        saved = {k: os.environ.get(k) for k in ('HF_HOME', 'LAYA_COREML_CACHE')}
        try:
            os.environ['HF_HOME'] = '/tmp/my-hf'
            os.environ.pop('LAYA_COREML_CACHE', None)
            userconfig.prefer_repo_model_cache(ROOT)
            self.assertEqual(os.environ['HF_HOME'], '/tmp/my-hf')
            self.assertEqual(os.environ['LAYA_COREML_CACHE'], str(ROOT / '.models' / 'laya-coreml'))
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


if __name__ == '__main__':
    unittest.main()
