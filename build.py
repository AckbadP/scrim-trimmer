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
import stat
import subprocess
import sys
import urllib.request

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


def _make_placeholder_icon(path):
    """Generate a simple 256x256 PNG icon using Pillow."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (256, 256), (30, 90, 160, 255))
    draw = ImageDraw.Draw(img)
    draw.text((60, 108), "EVE\nTrim", fill=(255, 255, 255, 255))
    img.save(path)


def get_appimagetool():
    """Return path to appimagetool, downloading it to the repo root if not in PATH."""
    tool = shutil.which("appimagetool")
    if tool:
        return tool

    dest = os.path.join(ROOT, "appimagetool-x86_64.AppImage")
    if not os.path.exists(dest):
        print("  Downloading appimagetool...")
        urllib.request.urlretrieve(
            "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage",
            dest,
        )
    os.chmod(dest, os.stat(dest).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def build_appimage():
    """Wrap the PyInstaller binary in an AppImage and return its path."""
    app_dir = os.path.join(ROOT, "AppDir")
    if os.path.exists(app_dir):
        shutil.rmtree(app_dir)
    os.makedirs(app_dir)

    # Binary
    src_bin = os.path.join(ROOT, "dist", "eve-at-trimmer")
    dest_bin = os.path.join(app_dir, "eve-at-trimmer")
    shutil.copy2(src_bin, dest_bin)
    os.chmod(dest_bin, os.stat(dest_bin).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # AppRun launcher
    apprun = os.path.join(app_dir, "AppRun")
    with open(apprun, "w") as f:
        f.write(
            '#!/bin/bash\n'
            'HERE="$(dirname "$(readlink -f "$0")")"\n'
            'exec "$HERE/eve-at-trimmer" "$@"\n'
        )
    os.chmod(apprun, 0o755)

    # .desktop file
    with open(os.path.join(app_dir, "eve-at-trimmer.desktop"), "w") as f:
        f.write(
            "[Desktop Entry]\n"
            "Name=EVE AT Practice Trimmer\n"
            "Exec=eve-at-trimmer\n"
            "Icon=eve-at-trimmer\n"
            "Type=Application\n"
            "Categories=Utility;\n"
        )

    # Icon
    icon_dest = os.path.join(app_dir, "eve-at-trimmer.png")
    candidates = [
        os.path.join(ROOT, "icon.png"),
        os.path.join(SRC, "icon.png"),
        os.path.join(ROOT, "assets", "icon.png"),
    ]
    src_icon = next((p for p in candidates if os.path.exists(p)), None)
    if src_icon:
        shutil.copy2(src_icon, icon_dest)
    else:
        _make_placeholder_icon(icon_dest)

    # Run appimagetool (APPIMAGE_EXTRACT_AND_RUN avoids requiring FUSE on CI)
    output = os.path.join(ROOT, "dist", "eve-at-trimmer-x86_64.AppImage")
    tool = get_appimagetool()
    env = {**os.environ, "ARCH": "x86_64", "APPIMAGE_EXTRACT_AND_RUN": "1"}
    print(f"  $ {tool} --no-appstream {app_dir} {output}")
    subprocess.run([tool, "--no-appstream", app_dir, output], check=True, env=env)

    shutil.rmtree(app_dir)
    return output


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

    if system == "Linux":
        print("\nPackaging AppImage...")
        out = build_appimage()
    else:
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
