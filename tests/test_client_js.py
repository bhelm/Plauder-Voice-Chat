"""Client-side asset checks: the split static/js modules must parse and the
pure ones (i18n table, markdown renderer, VCT frame parsers) must behave —
run in node, which is the only JS runtime available here. Skipped when node
is not installed (the rest of the suite stays green without it)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_client_js_files_parse():
    for js in sorted((ROOT / "static/js").glob("*.js")):
        r = subprocess.run([NODE, "--check", str(js)], capture_output=True, text=True)
        assert r.returncode == 0, f"{js.name}: {r.stderr}"


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_client_js_inline_blocks_parse(tmp_path):
    html = (ROOT / "static/index.html").read_text(encoding="utf-8")
    # Capture the opening tag too, so non-JS blocks (e.g. the three-vrm
    # <script type="importmap"> which is JSON) can be skipped — node --check
    # would choke on them.
    blocks = re.findall(r"(<script(?![^>]*\bsrc=)[^>]*>)(.*?)</script>", html,
                        flags=re.S | re.I)
    assert blocks, "no inline script blocks found"
    JS_TYPES = {"", "text/javascript", "application/javascript", "module"}
    i = 0
    for tag, block in blocks:
        if not block.strip():
            continue
        m = re.search(r'\btype\s*=\s*["\']([^"\']*)["\']', tag, flags=re.I)
        if m and m.group(1).strip().lower() not in JS_TYPES:
            continue   # importmap / speculationrules / other non-JS payloads
        p = tmp_path / f"block{i}.js"
        p.write_text(block, encoding="utf-8")
        r = subprocess.run([NODE, "--check", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, f"inline block {i}: {r.stderr}"
        i += 1


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_client_pure_modules():
    r = subprocess.run(
        [NODE, str(ROOT / "tests/client/pure_modules.test.mjs")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
