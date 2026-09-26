#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Frame fingerprints of every .rpl file, for tools/detect_stock.py and tools/align_languages.py.
# Usage: fingerprint.sh
# Reads <lang>/**.rpl in $WZSEQ_RPL (default: rpl/ in this folder), with the ffmpeg in $WZSEQ_FFMPEG (default: the PATH).
# Writes analysis/fp/<lang>/<name>.fp: raw 32x24 8-bit gray, one frame after another, at the file's own frame rate.
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
ff="${WZSEQ_FFMPEG:+$WZSEQ_FFMPEG/}ffmpeg"
cd "${WZSEQ_RPL:-$root/rpl}"
find eng fre ger -name '*.rpl' | sort | while read -r f; do
    out="$root/analysis/fp/${f%.rpl}.fp"
    mkdir -p "$(dirname "$out")"
    "$ff" -v error -i "$f" -an -vf "scale=32:24:flags=area,format=gray" -f rawvideo -y "$out" < /dev/null
done
