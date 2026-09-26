#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Per-track static gains for the loudness policy: partial normalization, K = 0.5.

  English target  = T + (1 - K) * (L_eng - T),  T = the English median integrated loudness (LUFS)
  every track of a sequence (eng, fre, ger) gets that target: gain = target - L_track
One static gain per track, no compression. Loudness is measured after DC removal (tools/audio_analysis.py).
A track whose true peak after the gain would exceed CEIL (-1 dBTP) gets its gain capped (with a note).
Output: recipes/audio_gains.csv (lang, name, lufs, target, gain_db, tp_after, note)
"""
import csv, os
import numpy as np

from paths import RECIPES, ANALYSIS
K, CEIL = 0.5, -1.0

rows = list(csv.DictReader(open(f'{ANALYSIS}/audio_analysis.csv')))
val = lambda r, k: float(r[k]) if r.get(k) not in (None, '', 'nan', '-inf') else float('nan')
ok = lambda L: np.isfinite(L) and L > -70
eng = {r['name']: r for r in rows if r['lang'] == 'eng'}
T = float(np.median([val(r, 'lufs') for r in eng.values() if ok(val(r, 'lufs'))]))

out, notes = [], []
for r in sorted(rows, key=lambda r: (r['name'], r['lang'])):
    L, tp = val(r, 'lufs'), val(r, 'tp_dbtp')
    e = eng.get(r['name'])
    Le = val(e, 'lufs') if e else float('nan')
    if not ok(L):
        out.append(dict(lang=r['lang'], name=r['name'], lufs=r['lufs'], note='silent or no audio: no gain')); continue
    if ok(Le):
        target = T + (1 - K) * (Le - T); note = ''
    else:
        target = T + (1 - K) * (L - T); note = 'no English loudness: own level used'
    g = target - L
    if np.isfinite(tp) and tp + g > CEIL:
        g2 = CEIL - tp
        note = (note + '; ' if note else '') + f'gain capped {g:+.1f} -> {g2:+.1f} dB (true peak)'
        g = g2
    out.append(dict(lang=r['lang'], name=r['name'], lufs=f'{L:.1f}', target=f'{target:.1f}', gain_db=f'{g:+.2f}',
                    tp_after=f'{tp + g:.1f}' if np.isfinite(tp) else '', note=note))

with open(f'{RECIPES}/audio_gains.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, ['lang', 'name', 'lufs', 'target', 'gain_db', 'tp_after', 'note']); w.writeheader(); w.writerows(out)
g = {l: np.array([float(o['gain_db']) for o in out if o['lang'] == l and o.get('gain_db')]) for l in ('eng', 'fre', 'ger')}
print(f'T (English median) = {T:.2f} LUFS, K = {K}')
for l, a in g.items():
    print(f'{l}: {len(a)} tracks, gain median {np.median(a):+.1f} dB, range {a.min():+.1f} .. {a.max():+.1f} dB')
for o in out:
    if o.get('note'):
        print(f"  {o['lang']} {o['name']}: {o['note']}")
