import os
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
WEB_DIR = os.path.join(REPO_ROOT, "output", "web")
INDEX_PATH = os.path.join(WEB_DIR, "index.html")
CSS_PATH = os.path.join(WEB_DIR, "hmi.css")
JS_PATH = os.path.join(WEB_DIR, "hmi.js")


class TestWebAssets(unittest.TestCase):
    def test_split_asset_files_exist(self):
        self.assertTrue(os.path.isfile(INDEX_PATH), f"missing file: {INDEX_PATH}")
        self.assertTrue(os.path.isfile(CSS_PATH), f"missing file: {CSS_PATH}")
        self.assertTrue(os.path.isfile(JS_PATH), f"missing file: {JS_PATH}")

    def test_index_references_static_assets(self):
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn('href="/static/hmi.css"', html)
        self.assertIn('src="/static/hmi.js"', html)
        # Ensure index stays as a thin shell and does not regress to inline blobs.
        self.assertNotIn("<style>", html)
        self.assertNotIn("<script>\n", html)
