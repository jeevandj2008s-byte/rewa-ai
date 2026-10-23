"""test_rewa.py - Root test launcher forwarding to .vscode/test_rewa.py."""
import runpy
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
vscode_dir = root_dir / ".vscode"
if str(vscode_dir) not in sys.path:
    sys.path.insert(0, str(vscode_dir))

test_file = vscode_dir / "test_rewa.py"
runpy.run_path(str(test_file), run_name="__main__")

