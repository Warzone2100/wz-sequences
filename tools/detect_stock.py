#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Find frames of stock "elastic" animations in every sequence: loops whose on-screen length follows the dialogue
(the rotating atoms shown during voice transmissions, the rotating NEXUS logo).

Each seed is a known stretch of one animation.
The set grows by nearest-neighbor closure over all 16x12 fingerprints of all languages.
Dark frames are excluded and the number of closure steps is capped (later steps chain into unrelated content).
Input: analysis/fp/ (tools/fingerprint.sh) and the frame rates in recipes/inventory.csv.
Output: analysis/stock_frames.pkl = {(lang, name): uint8 array per frame, 0 = none, else index into KINDS}.
"""
import csv, glob, os, pickle, sys
import numpy as np

from paths import RECIPES, ANALYSIS
THR = 8.0   # RMS distance at 16x12
# kind: (lang, sequence, start s, end s, max closure steps)
SEEDS = {
    'atom':  [('ger', 'cam3/c3ad2pl1', 15.4, 16.6, 7),    # 6-spoke atom
              ('ger', 'cam1/sub1_1', 22.0, 26.0, 4)],     # 3-ring atom
    'nexus': [('fre', 'cam3/c3ad2pl3', 5.0, 9.5, 7)],     # rotating NEXUS logo
}
KINDS = [None] + list(SEEDS)


def small(a):
    return a.reshape(-1, 12, 2, 16, 2).mean((2, 4)).reshape(len(a), -1)


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0) / A.shape[1])


FPS = {(r['lang'], r['name']): float(r['fps']) for r in csv.DictReader(open(f'{RECIPES}/inventory.csv'))}


def fps_of(lang, name):
    return FPS[(lang, name)]


F = {}
for p in sorted(glob.glob(f'{ANALYSIS}/fp/*/**/*.fp', recursive=True)):
    lang = os.path.relpath(p, f'{ANALYSIS}/fp').split('/')[0]
    name = os.path.relpath(p, f'{ANALYSIS}/fp/{lang}')[:-3]
    F[(lang, name)] = small(np.fromfile(p, np.uint8).reshape(-1, 768).astype(np.float32))
keys = list(F)
allf = np.concatenate([F[k] for k in keys])
owner = np.concatenate([[i] * len(F[k]) for i, k in enumerate(keys)])
bright = allf.max(1) >= 40      # every stock animation has a bright core (dark frames let the closure drift)

label = np.zeros(len(allf), np.uint8)
for kind, seeds in SEEDS.items():
    for lang, name, t0, t1, steps in seeds:
        fps = fps_of(lang, name)
        S = F[(lang, name)][int(t0 * fps):int(t1 * fps)]
        m = np.zeros(len(allf), bool)
        for _ in range(steps):
            new = (dist(allf, S).min(1) < THR) & ~m & bright
            if not new.any():
                break
            m |= new; S = allf[m][::3]
        label[m & (label == 0)] = KINDS.index(kind)
        print(f'{kind:6s} seed {lang} {name} {t0}-{t1}s: {m.sum()} frames')

out = {k: label[owner == i] for i, k in enumerate(keys)}
pickle.dump(dict(kinds=KINDS, frames=out), open(f'{ANALYSIS}/stock_frames.pkl', 'wb'))
for i, k in enumerate(KINDS[1:], 1):
    print(f'{k}: {(label == i).sum()} frames in {sum((v == i).any() for v in out.values())} files')
