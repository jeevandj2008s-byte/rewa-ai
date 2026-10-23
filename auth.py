"""auth.py - Root entry forwarder to .vscode/auth.py."""
import importlib.util
from pathlib import Path
import sys

_vscode_auth = Path(__file__).resolve().parent / ".vscode" / "auth.py"
_spec = importlib.util.spec_from_file_location("_vscode_auth", str(_vscode_auth))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Export all attributes to this module namespace
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
