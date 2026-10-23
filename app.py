"""app.py - Root entry point for ReWA.AI Streamlit application."""
import runpy
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
vscode_dir = root_dir / ".vscode"
if str(vscode_dir) not in sys.path:
    sys.path.insert(0, str(vscode_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

app_file = vscode_dir / "app1.py"
runpy.run_path(str(app_file), run_name="__main__")

