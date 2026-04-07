"""
native_dialogs.py - OS-native file/folder picker dialogs.

On Linux: uses zenity (GNOME/GTK) or kdialog (KDE), falls back to Tkinter.
On Windows: uses the native Windows file dialog via Tkinter (already native).
"""

import os
import shutil
import subprocess
import sys


def _which(cmd: str) -> bool:
    """Return True if cmd is available on PATH."""
    return shutil.which(cmd) is not None


def _zenity_open_file(title: str, filetypes: list[tuple], multiple: bool = False) -> list[str]:
    """Open file(s) via zenity."""
    args = ["zenity", "--file-selection", f"--title={title}"]
    if multiple:
        args.append("--multiple")
        args.append("--separator=|")
    for label, patterns in filetypes:
        if patterns == "*.*" or patterns == "*":
            args.append(f"--file-filter={label} | *")
        else:
            args.append(f"--file-filter={label} | {patterns}")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return result.stdout.strip().split("|")


def _zenity_open_dir(title: str) -> str:
    """Open directory via zenity."""
    args = ["zenity", "--file-selection", "--directory", f"--title={title}"]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    return result.stdout.strip()


def _kdialog_open_file(title: str, filetypes: list[tuple], multiple: bool = False) -> list[str]:
    """Open file(s) via kdialog."""
    # Build filter string: "Label (*.ext *.ext2)"
    filter_parts = []
    for label, patterns in filetypes:
        if patterns in ("*.*", "*"):
            filter_parts.append(f"{label} (*)")
        else:
            exts = " ".join(f"*{e.lstrip('*')}" for e in patterns.split())
            filter_parts.append(f"{label} ({exts})")
    filter_str = "\n".join(filter_parts) if filter_parts else "*"

    if multiple:
        args = ["kdialog", "--getopenfilenames", os.path.expanduser("~"), filter_str, f"--title={title}"]
    else:
        args = ["kdialog", "--getopenfilename", os.path.expanduser("~"), filter_str, f"--title={title}"]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return result.stdout.strip().split("\n")


def _kdialog_open_dir(title: str) -> str:
    """Open directory via kdialog."""
    args = ["kdialog", "--getexistingdirectory", os.path.expanduser("~"), f"--title={title}"]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    return result.stdout.strip()


def _tk_open_file(title: str, filetypes: list[tuple], multiple: bool = False):
    """Fallback: open file(s) via Tkinter filedialog."""
    from tkinter import filedialog
    tk_filetypes = [(label, patterns) for label, patterns in filetypes]
    if multiple:
        paths = filedialog.askopenfilenames(title=title, filetypes=tk_filetypes)
        return list(paths)
    else:
        path = filedialog.askopenfilename(title=title, filetypes=tk_filetypes)
        return [path] if path else []


def _tk_open_dir(title: str) -> str:
    """Fallback: open directory via Tkinter filedialog."""
    from tkinter import filedialog
    return filedialog.askdirectory(title=title) or ""


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def askopenfilename(title: str, filetypes: list[tuple]) -> str:
    """Return a single selected file path, or '' if cancelled."""
    paths = _pick_files(title, filetypes, multiple=False)
    return paths[0] if paths else ""


def askopenfilenames(title: str, filetypes: list[tuple]) -> list[str]:
    """Return a list of selected file paths (may be empty)."""
    return _pick_files(title, filetypes, multiple=True)


def askdirectory(title: str) -> str:
    """Return a selected directory path, or '' if cancelled."""
    if sys.platform == "win32":
        return _tk_open_dir(title)

    if _which("zenity"):
        return _zenity_open_dir(title)
    if _which("kdialog"):
        return _kdialog_open_dir(title)
    return _tk_open_dir(title)


def _pick_files(title: str, filetypes: list[tuple], multiple: bool) -> list[str]:
    if sys.platform == "win32":
        return _tk_open_file(title, filetypes, multiple)

    if _which("zenity"):
        return _zenity_open_file(title, filetypes, multiple)
    if _which("kdialog"):
        return _kdialog_open_file(title, filetypes, multiple)
    return _tk_open_file(title, filetypes, multiple)
