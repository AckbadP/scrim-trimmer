#!/usr/bin/env python3
"""
Build script for EVE AT Practice Trimmer.

Creates a one-click executable for the current platform (Linux or Windows).
Run this script on the platform you want to build for.

System prerequisites (must be installed separately — not bundled):
  Ubuntu/Debian:
    sudo apt install python3 python3-pip python3-tk tesseract-ocr ffmpeg
  Windows:
    - Python 3.10+: https://python.org
    - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
    - ffmpeg: https://ffmpeg.org/download.html  (add to PATH)

Usage:
  python build.py

Output:
  dist/eve-at-trimmer          (Linux — single executable)
  dist/eve-at-trimmer.exe      (Windows — single executable)
"""

import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")


def run(cmd):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def pip(*pkgs):
    run([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])


def check_system_deps():
    missing = [t for t in ("ffmpeg", "tesseract") if shutil.which(t) is None]
    if missing:
        print(
            f"\nWARNING: {', '.join(missing)} not found in PATH.\n"
            "The executable will still build, but users must install these tools:\n"
        )
        if platform.system() == "Linux":
            print("  sudo apt install ffmpeg tesseract-ocr\n")
        else:
            print(
                "  ffmpeg:    https://ffmpeg.org/download.html\n"
                "  tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n"
            )


def main():
    system = platform.system()
    print(f"Building EVE AT Practice Trimmer for {system}...\n")

    check_system_deps()

    pip("pyinstaller", "pyinstaller-hooks-contrib")
    pip("opencv-python", "Pillow", "numpy", "pytesseract", "tkinterdnd2")

    sep = os.pathsep  # ':' on Linux, ';' on Windows

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "--onefile",
        "--name", "eve-at-trimmer",
        # Bundle data files that gui.py loads at runtime
        "--add-data", f"{os.path.join(SRC, 'no_log_warning.txt')}{sep}.",
        "--add-data", f"{os.path.join(ROOT, 'README.md')}{sep}.",
        # tkinterdnd2 ships native DLLs/SOs that must be collected
        "--collect-all", "tkinterdnd2",
        # cv2 ships its own native libs
        "--collect-all", "cv2",
        # Imports not detected by static analysis
        "--hidden-import", "PIL._tkinter_finder",
        "--hidden-import", "pytesseract",
        "--hidden-import", "numpy",
        "--hidden-import", "config",
        "--hidden-import", "native_dialogs",
        "--hidden-import", "frame_extractor",
        "--hidden-import", "ocr_processor",
        "--hidden-import", "chat_analyzer",
        "--hidden-import", "chat_log_parser",
        "--hidden-import", "log_matcher",
        "--hidden-import", "video_clipper",
        # Tell PyInstaller where to find our src/ modules
        "--paths", SRC,
    ]

    if system == "Windows":
        # Suppress the console window on Windows
        cmd += ["--windowed"]

    cmd.append(os.path.join(SRC, "gui.py"))

    run(cmd)

    out = os.path.join(ROOT, "dist", "eve-at-trimmer")
    if system == "Windows":
        out += ".exe"

    print(f"\nBuild complete!  ->  {out}")

    if system == "Linux":
        print(
            "\nTarget machines need:\n"
            "  sudo apt install ffmpeg tesseract-ocr\n"
        )
    elif system == "Windows":
        print(
            "\nTarget machines need:\n"
            "  ffmpeg:    https://ffmpeg.org/download.html  (add to PATH)\n"
            "  tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n"
        )


if __name__ == "__main__":
    main()
