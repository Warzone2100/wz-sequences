#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Keyframe list for the encodes: every shared-video frame an edit list jumps TO needs a forced keyframe,
so the engine's seek (previous keyframe, then decode forward) lands on it directly.

Jump targets per effective plan (manual rows win per (sequence, language), as in consistency_check.py):
  tail_loop    'loop the exact cycle ending at the shared end (a-b s, P)'    -> a
               'loop English frames a-b ...'                                  -> a
  tail_segment 'play English|shared a-b s (repeat as needed)'                -> a
               'then loop English 0.00-2.00s (frames 0-49)'                   -> 0
  mid_loop     'extend the exact cycle x-b s (P) by d s' (loop inserted at b) -> b - P (loop start), and b (resume point,
               a jump unless d is a whole number of cycles, forced either way)
  loop         'segment=frames a-b ...; continue at b+1'                      -> a
  mid_skip     'skip shared a-b s'                                            -> b
Holds, black and fades are presentation-only (no seek). Per-language videos play straight through.
Frame 0 is always a keyframe. A target less than MIN_GAP s after another kept keyframe is not forced
(the seek then decodes at most a few frames forward). The CSV lists it as 'covered by' that keyframe.

Output: analysis/keyframes.csv (sequence, frame, time_s, fps, reasons) and recipes/keyframes.json
{sequence: [times in seconds]} for the encoder (ffmpeg -force_key_frames with these times, on the shared video).
"""
import csv, json, os, re, sys

from paths import RECIPES, ANALYSIS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared_video

MIN_GAP = 0.2

inv = {(r['lang'], r['name']): r for r in csv.DictReader(open(f'{RECIPES}/inventory.csv'))}
M = list(csv.DictReader(open(f'{RECIPES}/sequence_edits.csv')))
A = list(csv.DictReader(open(f'{RECIPES}/sequence_edits_auto.csv')))


def effective():
    """(sequence, lang) -> the rows the build uses."""
    man = {}
    for r in M:
        for l in ('fre', 'ger'):
            if l in r['applies_to'] or 'fre + ger' in r['applies_to']:
                man.setdefault((r['sequence'], l), []).append(r)
    eff = dict(man)
    for r in A:
        k = (r['sequence'], r['applies_to'].split()[0])
        if k not in man:
            eff.setdefault(k, []).append(r)
    return eff


def targets(r, fps):
    """[(frame, reason)] jump targets of one row, in shared-video frames."""
    op, p = r['op'], r['params']
    t2f = lambda t: t * fps
    if op == 'tail_loop':
        m = re.search(r'loop English frames (\d+)-', p)
        if m:
            return [(int(m.group(1)), 'tail loop start')]
        a = float(re.search(r'\(([\d.]+)-[\d.]+s, P=', p).group(1))
        return [(t2f(a), 'tail loop start')]
    if op == 'tail_segment':
        m = re.search(r'loop English ([\d.]+)-[\d.]+s', p) or re.search(r'play (?:English|shared) ([\d.]+)-[\d.]+s', p)
        out = [(t2f(float(m.group(1))), 'tail segment start (jump back)')]
        ent = re.search(r'play (?:English|shared) ([\d.]+)-[\d.]+s, then loop', p)
        if ent:
            out.append((t2f(float(ent.group(1))), 'tail entry (jump back)'))
        return out
    if op == 'mid_loop':
        x, b, P = map(float, re.search(r'cycle ([\d.]+)-([\d.]+)s \(P=([\d.]+)s\)', p).groups())
        d = float(re.search(r'by ([\d.]+)s', p).group(1))
        whole = abs(d / P - round(d / P)) < 1e-6
        out = [(t2f(b - P), 'mid loop start')]
        out.append((t2f(b), 'mid loop resume' + (' (sequential: whole cycles)' if whole else ' (jump)')))
        return out
    if op == 'loop':
        a = int(re.search(r'segment=frames (\d+)-', p).group(1))
        return [(a, 'loop start')]
    if op == 'mid_skip':
        b = float(re.search(r'skip shared [\d.]+-([\d.]+)s', p).group(1))
        return [(t2f(b), 'skip landing (forward jump)')]
    return []


if __name__ == '__main__':
    eff = effective()
    kf = {}                 # sequence -> {frame: set(reasons)}
    problems = []
    for (seq, lang), rows in sorted(eff.items()):
        if any(r['op'] == 'per_language_video' for r in rows):
            continue
        fps = float(inv[('eng', seq)]['fps'])
        n = len(shared_video.frames(seq, lambda c: int(inv[('eng', c)]['frames'])))
        for r in rows:
            for f, why in targets(r, fps):
                if abs(f - round(f)) > 1e-6:
                    problems.append(f'{seq} {lang}: target {f / fps:.2f}s is not on a frame ({f:.2f} @{fps:g}fps)')
                f = int(round(f))
                if not 0 <= f < n:
                    problems.append(f'{seq} {lang}: target frame {f} outside the shared video (0-{n - 1})')
                kf.setdefault(seq, {}).setdefault(f, set()).add(f'{lang} {why}')
    seqs = sorted({n for (l, n) in inv if l == 'eng'})
    rows_out, js = [], {}
    for seq in seqs:
        fps = float(inv[('eng', seq)]['fps'])
        fr = {0: {'start'}} | kf.get(seq, {})
        kept = []
        for f in sorted(fr):
            cover = next((k for k in reversed(kept) if (f - k) / fps < MIN_GAP), None)
            if cover is None:
                kept.append(f)
            rows_out.append(dict(sequence=seq, frame=f, time_s=f'{f / fps:.2f}', fps=f'{fps:g}',
                                 forced='yes' if cover is None else f'no (covered by frame {cover})', reasons='; '.join(sorted(fr[f]))))
        js[seq] = [round(f / fps, 4) for f in kept]
    os.makedirs(ANALYSIS, exist_ok=True)
    with open(f'{ANALYSIS}/keyframes.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['sequence', 'frame', 'time_s', 'fps', 'forced', 'reasons']); w.writeheader(); w.writerows(rows_out)
    json.dump(js, open(f'{RECIPES}/keyframes.json', 'w'), indent=1)
    extra = sum(len(v) - 1 for v in js.values())
    print(f'{len(seqs)} sequences; {sum(len(v) > 1 for v in js.values())} with forced keyframes beyond frame 0 ({extra} frames)')
    for r in rows_out:
        if r['forced'] != 'yes':
            print(f"  not forced: {r['sequence']} frame {r['frame']} ({r['time_s']}s) {r['forced']}")
    for seq in sorted(kf):
        print(f'  {seq}: ' + ', '.join(f"{f} ({f / float(inv[('eng', seq)]['fps']):.2f}s)" for f in sorted(kf[seq])))
    print(f'problems: {len(problems)}'); [print('  ' + x) for x in problems]
    sys.exit(1 if problems else 0)
