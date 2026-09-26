#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Consistency check of the edit lists (manual sequence_edits.csv + automatic sequence_edits_auto.csv).

Checks:
  1. coverage: every localized track (inventory) has a plan (manual rows win per (sequence, language))
  2. no auto-check rows left, and every manual row confirmed
  3. shared_video.SPECS in sync with the CSV (shared_video.check())
  4. English: shared video length vs English audio length (after trims/pads)
  5. each localized track: planned picture length (shared video + mid inserts + tail, minus skipped English frames)
     vs planned audio length (localized audio + pads - cuts/trims)
  6. references: 'English X-Ys' segment rows on sequences whose shared video differs from the English clip
Usage: consistency_check.py   (prints a report, and exits with code 1 on any problem, not on notes)
"""
import csv, os, re, subprocess, sys

from paths import RECIPES
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared_video

TOL = 0.085                        # a bit over 2 frames @25 fps
inv = {(r['lang'], r['name']): r for r in csv.DictReader(open(f'{RECIPES}/inventory.csv'))}
M = list(csv.DictReader(open(f'{RECIPES}/sequence_edits.csv')))
A = list(csv.DictReader(open(f'{RECIPES}/sequence_edits_auto.csv')))
problems, notes = [], []


def langs_of(r):
    a = r['applies_to']
    return [l for l in ('fre', 'ger') if l in a or 'fre + ger' in a]


def num(pat, s, g=1):
    m = re.search(pat, s)
    return float(m.group(g)) if m else None


def adur(lang, name):
    r = inv[(lang, name)]
    return float(r['audio_dur'] or r['video_dur'])


# 2. statuses
for r in M:
    if not r['status'].startswith('confirmed'):
        problems.append(f"manual row not confirmed: {r['sequence']} {r['applies_to']} {r['op']} ({r['status']})")
for r in A:
    if r['status'] == 'auto-check':
        problems.append(f"auto-check left: {r['sequence']} {r['applies_to']} {r['op']}")

# 3. SPECS
try:
    shared_video.check()
except SystemExit as e:
    problems.append(str(e))

# shared video lengths
nfr = {r['name']: int(r['frames']) for (l, n), r in inv.items() if l == 'eng'}
seqs = sorted(nfr)
S = {}
for n in seqs:
    S[n] = len(shared_video.frames(n, lambda c: nfr[c])) / float(inv[('eng', n)]['fps'])

# 4. English
for n in seqs:
    ea = adur('eng', n)
    for r in M:
        if r['sequence'] != n:
            continue
        a, p = r['applies_to'], r['params']
        if a.startswith('shared video') and r['op'] == 'trim_head':
            ea -= int(re.search(r'audio_samples=(\d+)', p).group(1)) / 22050
        elif 'eng audio' in a:
            ea += num(r'eng audio:? pad ([\d.]+)s', p) or num(r'silence ([\d.]+)s at start', p) or 0.0
    d = S[n] - ea
    if abs(d) > TOL:
        notes.append(f"eng {n}: shared video {S[n]:.2f}s vs English audio {ea:.2f}s ({d:+.2f}s picture {'longer' if d > 0 else 'shorter'})")

# 1 + 5. localized tracks
man = {}
for r in M:
    for l in langs_of(r):
        man.setdefault((r['sequence'], l), []).append(r)
aut = {}
for r in A:
    aut.setdefault((r['sequence'], r['applies_to'].split()[0]), []).append(r)

counts = dict(manual=0, auto=0, per_language_video=0)
for (lang, n) in sorted(k for k in inv if k[0] in ('fre', 'ger')):
    k = (n, lang)
    if n not in nfr:
        problems.append(f'{lang} {n}: no English clip'); continue
    rows = man.get(k) or aut.get(k)
    if not rows:
        problems.append(f'{lang} {n}: NO PLAN'); continue
    counts['manual' if k in man else 'auto'] += 1
    if any(r['op'] == 'per_language_video' for r in rows):
        counts['per_language_video'] += 1; continue
    if any(r['op'] == 'identical' for r in rows):
        if abs(adur(lang, n) - adur('eng', n)) > TOL:
            problems.append(f'{lang} {n}: marked identical but durations differ')
        continue
    au = adur(lang, n); pic = S[n]; total = None; approx = False
    # audio trims first (their times are in original localized time), then pads
    for r in sorted(rows, key=lambda r: r['op'] not in ('trim_tail', 'trim_head')):
        op, p = r['op'], r['params']
        # audio
        if op == 'pad_head':
            au += num(r'silence=?\s*([\d.]+)s', p)
        elif op == 'trim_head':
            au -= num(r'([\d.]+)s', p)
        elif op == 'pad_tail':
            x = num(r'silence ([\d.]+)s', p)
            if x is None:
                au = max(au, num(r'silence to ([\d.]+)s', p))
            else:
                au += x
        elif op == 'pad_gap':
            au += sum(float(x) for x in re.findall(r'(?:insert |and )([\d.]+)s', p))
        elif op == 'cut_gap':
            au -= num(r'\(([\d.]+)s\)', p)
        elif op == 'trim_tail':
            c = num(r'keep ~([\d.]+)s', p) or num(r'\(~([\d.]+)s\)', p)
            approx |= c is not None or '~' in p
            au = c if c is not None else num(r'cut at ([\d.]+)s', p)
        # picture
        if op in ('mid_hold', 'mid_loop'):
            pic += num(r'for ([\d.]+)s', p) if op == 'mid_hold' else num(r'by ([\d.]+)s', p)
            x = num(r'audio pad ([\d.]+)s at end', p)
            if x:
                au += x
        elif op == 'loop':
            pic += num(r'\(\+([\d.]+)s\)', p)
        elif op == 'mid_skip':
            a, b = map(float, re.search(r'skip shared ([\d.]+)-([\d.]+)s', p).groups()); pic -= b - a
        elif op.startswith('tail_'):
            at = num(r'at shared ([\d.]+)s', p)
            if at is not None:
                pic = at
            if 'play 0-' in p and '(+' in p:                        # cam32alp form
                pic = num(r'play 0-([\d.]+)s', p) + num(r'\(\+([\d.]+)s\)', p)
            elif 'play 0-' in p and 'frames =' in p:                # brfcom form: whole timeline
                pic = num(r'= ([\d.]+)s\)', p)
            else:
                x = (num(r'last frame ([\d.]+)s', p) or num(r'final frame ([\d.]+)s', p) or num(r'\) ([\d.]+)s;', p)
                     or num(r'over ([\d.]+)s', p) or num(r'black ([\d.]+)s', p) or num(r'for \+?([\d.]+)s', p)
                     or num(r'frame ~?([\d.]+)s', p))
                if x is None:
                    problems.append(f'{lang} {n}: cannot parse {op}: {p[:90]}'); continue
                approx |= '~' in p
                pic += x
            t = num(r'total ([\d.]+)s', p)
            total = t if t is not None else total
    d = pic - au
    tag = 'manual' if k in man else 'auto'
    if abs(d) > (0.45 if approx else TOL):
        # picture longer than audio: the audio ends early (silence). Audio longer: it would be cut off.
        (problems if d < 0 and (tag == 'manual' or abs(d) > 0.5) else notes).append(
            f"{lang} {n} [{tag}]: picture {pic:.2f}s vs audio {au:.2f}s ({d:+.2f}s)  ops: "
            + '; '.join(r['op'] for r in rows))
    if total is not None and abs(total - pic) > TOL:
        notes.append(f'{lang} {n} [{tag}]: row states total {total:.2f}s, computed picture {pic:.2f}s')

# 6. English-referenced segments on changed shared videos
changed = set(shared_video.SPECS) | set(shared_video.trims())
for r in M + A:
    if r['sequence'] in changed and re.search(r'English [\d.]+-[\d.]+s', r['params']) and r['op'] in ('tail_segment', 'tail_loop', 'mid_loop', 'loop'):
        notes.append(f"{r['sequence']} {r['applies_to']} {r['op']}: refers to English times on a changed shared video: {r['params'][:80]}")

print(f"tracks: {counts}  (inventory localized tracks: {sum(1 for k in inv if k[0] != 'eng')})")
print(f'\nPROBLEMS ({len(problems)}):'); [print('  ' + x) for x in problems]
print(f'\nNOTES ({len(notes)}):'); [print('  ' + x) for x in notes]
sys.exit(1 if problems else 0)
