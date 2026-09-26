#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""The *shared* video of each sequence: the English frames after the confirmed shared-video edits.

Each sequence's shared video is a list of (English clip, frame index) pairs at the English clip's frame rate.
Without edits it is every frame of the English clip.
SPECS mirrors the confirmed rows of recipes/sequence_edits.csv whose applies_to is 'shared video ...'
(check() fails unless every such row has an entry here, and every entry a row).

Used by align_languages.py (to plan the localized tracks against the picture that ships) and the stage (b) build.
"""
import csv, os, re

from paths import RECIPES
EDITS = f'{RECIPES}/sequence_edits.csv'

# sequence -> list of (clip, first_frame, last_frame_inclusive or None = end, step[, 'fade'])
# A part marked 'fade' is faded linearly to black over its length (baked into the shared video).
SPECS = {
    # continue the final rotating module with English cam1/cam1 (25 fps, every 2nd frame)
    'cam1/cam1ccf': [('cam1/cam1ccf', 0, None, 1), ('cam1/cam1', 362, 400, 2)],
    # continue the 2.0 s atom-composite cycle by 5 frames
    'cam1/cam1out1': [('cam1/cam1out1', 0, None, 1), ('cam1/cam1out1', 313, 317, 1)],
    # 6 more frames of the exact opening cycle before the hard cut at frame 59
    'cam1/sub1_3': [('cam1/sub1_3', 0, 58, 1), ('cam1/sub1_3', 34, 39, 1), ('cam1/sub1_3', 59, None, 1)],
    # replace the 2.04 s end freeze with the opening atom cycle, baked to the German length
    'cam3/cam3bpl': [('cam3/cam3bpl', 0, 249, 1), ('cam3/cam3bpl', 24, 101, 1)],
    # continue the atom the English ends on with the same rendition from English cam3cpl1
    # (cam3abpl's last frame ~ cam3cpl1 frame 75), 16 frames = 0.64 s, baked to the German length
    'cam3/cam3abpl': [('cam3/cam3abpl', 0, None, 1), ('cam3/cam3cpl1', 76, 91, 1)],
    # continue the atom seamlessly with English sub1_2 (cam1clz's last 3 frames = sub1_2 frames 168-170)
    # for 1.04 s, fading to black. French and German then hold black.
    'cam1/cam1clz': [('cam1/cam1clz', 0, None, 1), ('cam1/sub1_2', 171, 183, 1, 'fade')],
    # the same pattern: c3ad1pl2's last frame = c3ad1pl1 frame 49 (pixel-identical).
    # Continue with c3ad1pl1 frames 50-75 (1.04 s @25 fps), fading to black.
    'cam3/c3ad1pl2': [('cam3/c3ad1pl2', 0, None, 1), ('cam3/c3ad1pl1', 50, 75, 1, 'fade')],
    # the same pattern
    'cam3/c3ad1pl1': [('cam3/c3ad1pl1', 0, None, 1), ('cam3/c3ad1pl2', 166, 191, 1, 'fade')],   # pixel-exact join
    'cam3/cam3int': [('cam3/cam3int', 0, None, 1), ('cam3/c003', 231, 243, 1, 'fade')],          # pixel-exact join
    # near match, 2.38 s (long French tail)
    'cam2/cam26afm': [('cam2/cam26afm', 0, None, 1), ('cam1/sub1_3', 19, 48, 1, 'fade')],
    'cam3/cam3intb': [('cam3/cam3intb', 0, None, 1), ('cam1/cam1out3', 43, 67, 2, 'fade')],      # weak match
    # c3ad2pl4 is the same atom loop as cam31bpl, with heavy cumulative compression artifacts.
    # cam31bpl's clean English video replaces it, continued along its exact 2.0 s end cycle (frame 250 = frame 200)
    # to the English audio length, 11.40 s.
    'cam3/c3ad2pl4': [('cam3/cam31bpl', 0, None, 1), ('cam3/cam31bpl', 201, 234, 1)],
}
OPS_WITH_SPEC = {'extend_tail_from_clip', 'extend_tail_cycle', 'bake_cycle_extend', 'patch_tail', 'extend_tail_fade', 'replace_video'}


def _rows():
    return [r for r in csv.DictReader(open(EDITS))
            if r['applies_to'].startswith('shared video') and r['status'].startswith('confirmed')]


def trims():
    """Confirmed head trims of the shared video: sequence -> number of frames."""
    out = {}
    for r in _rows():
        if r['op'] == 'trim_head':
            out[r['sequence']] = int(re.search(r'frames=(\d+)', r['params']).group(1))
    return out


def check():
    have = {r['sequence'] for r in _rows() if r['op'] in OPS_WITH_SPEC}
    missing, extra = have - set(SPECS), set(SPECS) - have
    if missing or extra:
        raise SystemExit(f'shared_video.SPECS out of sync with {EDITS}: missing {sorted(missing)}, extra {sorted(extra)}')


def frames(name, n_frames_of):
    """List of (clip, frame) making up the shared video of `name`.
    n_frames_of(clip) -> number of frames of that English clip."""
    return [(c, k) for c, k, _ in _build(name, n_frames_of)]


def gains(name, n_frames_of):
    """Per-frame brightness gain of the shared video (1.0 = untouched, < 1 inside a baked fade to black)."""
    return [g for _, _, g in _build(name, n_frames_of)]


def _build(name, n_frames_of):
    check()
    parts = SPECS.get(name, [(name, 0, None, 1)])
    out = []
    for part in parts:
        clip, a, b, step = part[:4]; fade = len(part) > 4 and part[4] == 'fade'
        b = n_frames_of(clip) - 1 if b is None else b
        ks = list(range(a, b + 1, step))
        out += [(clip, k, (1 - (i + 1) / len(ks)) if fade else 1.0) for i, k in enumerate(ks)]
    t = trims().get(name, 0)
    return out[t:]


if __name__ == '__main__':
    check()
    print('shared-video specs in sync with', EDITS)
    print('head trims:', trims())
