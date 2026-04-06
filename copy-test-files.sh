#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <name>"
    echo "  Copies the most recent .mkv and Local*.txt files into src/resources/<name>.{mkv,txt}"
    exit 1
fi

NAME="$1"
DEST="$(dirname "$0")/src/resources"

MKV_SRC=$(find ~/Videos/AG7 -maxdepth 1 -name "*.mkv" -printf "%T@ %p\n" 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-)

TXT_SRC=$(find ~/.local/share/Steam/steamapps/compatdata/8500/pfx/drive_c/users/steamuser/Documents/EVE/logs/Chatlogs \
    -maxdepth 1 -name "*Local*.txt" -printf "%T@ %p\n" 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-)

if [[ -z "$MKV_SRC" ]]; then
    echo "Error: no .mkv file found in ~/Videos/AG7"
    exit 1
fi

if [[ -z "$TXT_SRC" ]]; then
    echo "Error: no Local*.txt file found in the EVE Chatlogs directory"
    exit 1
fi

cp "$MKV_SRC" "$DEST/$NAME.mkv"
echo "Copied: $MKV_SRC -> $DEST/$NAME.mkv"

cp "$TXT_SRC" "$DEST/$NAME.txt"
echo "Copied: $TXT_SRC -> $DEST/$NAME.txt"
