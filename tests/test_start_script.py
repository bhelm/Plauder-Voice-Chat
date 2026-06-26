"""start.sh existiert, ist ausführbar und syntaktisch valide."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "start.sh"


def test_script_exists():
    assert SCRIPT.is_file(), f"start.sh fehlt unter {SCRIPT}"


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), "start.sh ist nicht ausführbar"


def test_script_has_shebang():
    first = SCRIPT.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("#!"), "kein Shebang"
    assert "bash" in first


def test_script_bash_syntax_valid():
    bash = shutil.which("bash")
    if not bash:
        return  # ohne bash nicht prüfbar
    res = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert res.returncode == 0, f"Syntaxfehler: {res.stderr}"


def test_script_references_server_and_requirements():
    body = SCRIPT.read_text(encoding="utf-8")
    assert "server.py" in body
    assert "requirements.txt" in body
    assert ".venv" in body


def test_requirements_file_exists_and_lists_core_deps():
    req = ROOT / "requirements.txt"
    assert req.is_file()
    body = req.read_text(encoding="utf-8").lower()
    for dep in ("aiohttp", "openai", "numpy"):
        assert dep in body, f"{dep} fehlt in requirements.txt"
