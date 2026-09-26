#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Structured edit model: the confirmed and automatic edit rows (free text) -> per-track operations,
used by the stage (b) build.

model(seq) -> {
  'fps', 'shared_frames' (count), 'shared_dur', 'warnings' (messages, each starting with its language),
  'eng':  {'audio': [audio ops], 'audio_len': the conformed length in seconds},
  <lang>: {'audio': [audio ops] | None (same as English), 'same_as': 'eng' | None, 'audio_len': as for English,
           'picture': [picture ops on the shared-video timeline] | None (plays the shared video straight),
           'video': per-language video id | None}
}
Picture ops (seconds on the shared-video timeline, on the frame grid):
  {'op': 'play', 'from': a, 'to': b}               play [a, b) (a seek when a isn't where the previous op ended:
                                                   a mid_skip row gives two plays with a forward jump between them)
  {'op': 'hold', 'duration': d}                    repeat the frame shown last
  {'op': 'loop', 'from': a, 'to': b, 'duration': d}  play [a, b) repeatedly for d seconds (may end mid-cycle)
  {'op': 'black', 'duration': d}
  {'op': 'hold_fade', 'duration': d}               hold the last frame, fading it linearly to black over d
Audio ops (seconds on the track's own original timeline, applied after the 10 Hz high-pass, before the gain):
  ('trim_head', samples) ('trim_tail', t, fade_s) ('cut', a, b, xfade_s) ('insert', at, dur, (ra, rb) room tone | None)
  ('pad_head', s) ('pad_tail', s) ('pad_to', t)
Tail picture ops take their duration from the conformed audio length (the text value is only a cross-check).
"""
import csv, os, re, sys

from paths import RECIPES
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared_video

SR = 22050
TRIM_FADE, XFADE = 0.03, 0.03
inv = {(r['lang'], r['name']): r for r in csv.DictReader(open(f'{RECIPES}/inventory.csv'))}
M = list(csv.DictReader(open(f'{RECIPES}/sequence_edits.csv')))
A = list(csv.DictReader(open(f'{RECIPES}/sequence_edits_auto.csv')))
LANGS = ('fre', 'ger')


def num(pat, s, g=1):
    m = re.search(pat, s)
    return float(m.group(g)) if m else None


def effective():
    """(sequence, lang) -> the rows the build uses (manual rows win per track)."""
    man = {}
    for r in M:
        for l in LANGS:
            if l in r['applies_to'] or 'fre + ger' in r['applies_to']:
                man.setdefault((r['sequence'], l), []).append(r)
    eff = dict(man)
    for r in A:
        k = (r['sequence'], r['applies_to'].split()[0])
        if k not in man:
            eff.setdefault(k, []).append(r)
    return eff


EFF = effective()


def audio_dur(lang, seq):
    r = inv[(lang, seq)]
    return int(r['audio_samples']) / int(r['arate']) if r['audio_samples'] else 0.0


def audio_ops(rows):
    ops = []
    for r in rows:
        op, p = r['op'], r['params']
        if op == 'pad_head':
            ops.append(('pad_head', num(r'silence=?\s*([\d.]+)s', p)))
        elif op == 'pad_tail':
            x = num(r'silence ([\d.]+)s', p)
            ops.append(('pad_tail', x) if x is not None else ('pad_to', num(r'silence to ([\d.]+)s', p)))
        elif op == 'trim_tail':
            t = num(r'keep ~([\d.]+)s', p) or num(r'\(~([\d.]+)s\)', p) or num(r'cut at ([\d.]+)s', p)
            ops.append(('trim_tail', t, TRIM_FADE))
        elif op == 'cut_gap':
            a, b = map(float, re.search(r'remove ([\d.]+)-([\d.]+)s', p).groups())
            ops.append(('cut', a, b, XFADE))
        elif op == 'pad_gap':
            m = re.findall(r'insert ([\d.]+)s at ([\d.]+)s \(room tone looped from the \w+ pause ([\d.]+)-([\d.]+)s', p)
            m += re.findall(r'and ([\d.]+)s at ([\d.]+)s \(room tone from ([\d.]+)-([\d.]+)s', p)
            for d, at, ra, rb in m:
                ops.append(('insert', float(at), float(d), (float(ra), float(rb))))
            m2 = re.search(r'insert ([\d.]+)s of silence \(or room tone\) in the pause at ([\d.]+)-([\d.]+)s', p)
            if m2:
                d, ra, rb = map(float, m2.groups())
                ops.append(('insert', (ra + rb) / 2, d, (ra, rb)))
        elif op == 'mid_hold':
            x = num(r'audio pad ([\d.]+)s at end', p)
            if x:
                ops.append(('pad_tail', x))
    return ops


def conformed_len(ops, dur):
    """Length of the conformed audio (seconds)."""
    L = dur
    for o in ops:
        if o[0] == 'trim_head': L -= o[1] / SR
        elif o[0] == 'trim_tail': L = min(L, o[1])
    for o in ops:
        if o[0] == 'cut': L -= o[2] - o[1]
        elif o[0] == 'insert': L += o[2]
        elif o[0] in ('pad_head', 'pad_tail'): L += o[1]
    for o in ops:
        if o[0] == 'pad_to': L = max(L, o[1])
    return L


def picture_ops(rows, S, fps, audio_len, warn):
    """Picture timeline for one localized track, or None if it plays the shared video straight."""
    head = [('play', 0.0, S)]              # (op, from, to) segments before the tail
    tail = None; tail_hold_end = False; tail_entry = None
    for r in rows:
        op, p = r['op'], r['params']
        if op == 'mid_hold':
            f = int(re.search(r'hold English frame (\d+)', p).group(1)); d = num(r'for ([\d.]+)s', p)
            head = _insert(head, (f + 1) / fps, [('hold', d)])
        elif op == 'mid_loop':
            x, b, P = map(float, re.search(r'cycle ([\d.]+)-([\d.]+)s \(P=([\d.]+)s\)', p).groups())
            d = num(r'by ([\d.]+)s', p)
            head = _insert(head, b, [('loop', b - P, b, d)])
        elif op == 'mid_skip':
            a, b = map(float, re.search(r'skip shared ([\d.]+)-([\d.]+)s', p).groups())
            head = _skip(head, a, b)
        elif op == 'loop':
            a, b = map(int, re.search(r'segment=frames (\d+)-(\d+)', p).groups())
            d = num(r'\(\+([\d.]+)s\)', p)
            head = _insert(head, (b + 1) / fps, [('loop', a / fps, (b + 1) / fps, d)])
        elif op.startswith('tail_'):
            at = num(r'at shared ([\d.]+)s', p) or num(r'^play 0-([\d.]+)s', p)
            if at is not None:
                head = _cut(head, at)
            text_d = (num(r'last frame ([\d.]+)s', p) or num(r'final frame ([\d.]+)s', p) or num(r'\) ([\d.]+)s;', p)
                      or num(r'over ([\d.]+)s', p) or num(r'black ([\d.]+)s', p) or num(r'for \+?([\d.]+)s', p)
                      or num(r'\(\+([\d.]+)s\)', p) or num(r'frame ~([\d.]+)s', p))
            if op in ('tail_hold',):
                tail = ['hold']
            elif op == 'tail_black':
                tail = ['black']
            elif op == 'tail_hold_fade':
                tail = ['hold_fade']
            elif op == 'tail_loop':
                m = re.search(r'loop English frames (\d+)-(\d+)', p)
                if m:
                    tail = ['loop', int(m.group(1)) / fps, (int(m.group(2)) + 1) / fps]
                else:
                    a, b = map(float, re.search(r'\(([\d.]+)-([\d.]+)s, P=', p).groups()); tail = ['loop', a, b]
            elif op == 'tail_segment':
                m = re.search(r'(?:play|loop) (?:English|shared) ([\d.]+)-([\d.]+)s', p.split('; whole cycles')[0].split('then')[-1])
                tail = ['loop', float(m.group(1)), float(m.group(2))]
                ent = re.search(r'play (?:English|shared) ([\d.]+)-([\d.]+)s, then loop', p)
                if ent:                                            # enter the loop at another point of the cycle
                    tail_entry = (float(ent.group(1)), float(ent.group(2)))
            tail.append(text_d)
            tail_hold_end = 'whole cycles only, then hold the last frame' in p
    ops = [dict(op='play', **{'from': a, 'to': b}) if o == 'play' else _opdict(o, a, b) for o, a, b in
           [(s[0], s[1], s[2] if len(s) > 2 else None) if s[0] == 'play' else (s[0], s, None) for s in head]]
    ops = [_fix(o) for o in ops]
    plen = sum(_dur(o) for o in ops)
    if tail:
        d = round(audio_len - plen, 3)
        if tail[-1] is not None and abs(tail[-1] - d) > 0.1:
            warn(f'tail {tail[0]} text says {tail[-1]:.2f}s, audio needs {d:.2f}s')
        if tail_entry and d > 1e-3:
            e0, e1 = tail_entry
            ops.append({'op': 'play', 'from': e0, 'to': e1}); d = round(d - (e1 - e0), 3)
        if d > 1e-3:
            if tail[0] == 'loop' and tail_hold_end:
                # whole cycles only, then hold the last frame: no jump back in the final frames
                P = tail[2] - tail[1]; nc = int(d / P + 1e-6)
                if nc == 0:                                   # shorter than one cycle: no wrap anyway
                    ops.append({'op': 'loop', 'from': tail[1], 'to': tail[2], 'duration': d})
                else:
                    rest = round(d - nc * P, 3)
                    ops.append({'op': 'loop', 'from': tail[1], 'to': tail[2], 'duration': round(nc * P, 3)})
                    if rest > 1e-3:
                        ops.append({'op': 'hold', 'duration': rest})
            elif tail[0] == 'loop':
                ops.append({'op': 'loop', 'from': tail[1], 'to': tail[2], 'duration': d})
            else:
                ops.append({'op': tail[0], 'duration': d})
    if len(ops) == 1 and ops[0]['op'] == 'play' and ops[0]['from'] == 0 and abs(ops[0]['to'] - S) < 1e-6:
        return None
    return ops


def _opdict(o, s, _):
    return s


def _fix(o):
    if isinstance(o, dict):
        return o
    if o[0] == 'hold':
        return {'op': 'hold', 'duration': o[1]}
    if o[0] == 'loop':
        return {'op': 'loop', 'from': o[1], 'to': o[2], 'duration': o[3]}
    raise ValueError(o)


def _dur(o):
    return o['to'] - o['from'] if o['op'] == 'play' else o['duration']


def _insert(head, t, items):
    """Split the play segment containing time t and insert items there."""
    out = []
    for s in head:
        if s[0] == 'play' and s[1] < t < s[2]:
            out += [('play', s[1], t)] + items + [('play', t, s[2])]
        elif s[0] == 'play' and abs(s[2] - t) < 1e-9:
            out += [s] + items
        else:
            out.append(s)
    return out


def _skip(head, a, b):
    """Leave out shared [a, b) (a forward jump)."""
    out = []
    for s in head:
        if s[0] == 'play' and s[1] < a and b < s[2]:
            out += [('play', s[1], a), ('play', b, s[2])]
        else:
            out.append(s)
    return out


def _cut(head, t):
    """End the timeline at shared time t (drop everything after it)."""
    out = []
    for s in head:
        if s[0] == 'play':
            if s[1] >= t:
                break
            out.append(('play', s[1], min(s[2], t)))
        else:
            out.append(s)
    return out


def model(seq):
    fps = float(inv[('eng', seq)]['fps'])
    n = len(shared_video.frames(seq, lambda c: int(inv[('eng', c)]['frames'])))
    S = n / fps
    warnings = []
    res = dict(fps=fps, shared_frames=n, shared_dur=S, warnings=warnings)
    # English audio
    eops = []
    for r in M:
        if r['sequence'] != seq:
            continue
        a, p = r['applies_to'], r['params']
        if a.startswith('shared video') and r['op'] == 'trim_head':
            eops.append(('trim_head', int(re.search(r'audio_samples=(\d+)', p).group(1))))
        if 'eng audio' in a:
            x = num(r'eng audio:? pad ([\d.]+)s', p)
            if x:
                eops.append(('pad_tail', x))
            if r['op'] == 'pad_head':
                eops.append(('pad_head', num(r'silence ([\d.]+)s at start', p)))
    res['eng'] = {'audio': eops, 'audio_len': conformed_len(eops, audio_dur('eng', seq)) if inv[('eng', seq)]['audio_samples'] else 0.0}
    for lang in LANGS:
        if (lang, seq) not in inv:
            continue
        rows = EFF.get((seq, lang), [])
        warn = lambda m, l=lang: warnings.append(f'{l}: {m}')
        t = {'same_as': None, 'video': None}
        if any(r['op'] == 'identical' for r in rows):
            t.update(same_as='eng', audio=None, picture=None, audio_len=res['eng']['audio_len'])
        else:
            ops = audio_ops(rows)
            t['audio'] = ops
            t['audio_len'] = conformed_len(ops, audio_dur(lang, seq))
            if any(r['op'] == 'per_language_video' for r in rows):
                t['video'] = f"{seq}.{'loc' if seq == 'cam3/c3_d1_b' else lang}"
                t['picture'] = None
            else:
                t['picture'] = picture_ops(rows, S, fps, t['audio_len'], warn)
        res[lang] = t
    return res


def grid_problems(m):
    """Picture-op times that are not on the shared video's frame grid."""
    out = []
    for lang in LANGS:
        for o in (m.get(lang) or {}).get('picture') or []:
            for k in ('from', 'to', 'duration'):
                if k in o and abs(o[k] * m['fps'] - round(o[k] * m['fps'])) > 1e-6 and not (k == 'duration' and o['op'] in ('hold', 'black', 'hold_fade', 'loop')):
                    out.append(f"{lang} {o['op']} {k}={o[k]}")
    return out


if __name__ == '__main__':
    import json
    seqs = sorted({n for (l, n) in inv if l == 'eng'})
    nw = 0
    for s in (sys.argv[1:] or seqs):
        m = model(s)
        if sys.argv[1:]:
            print(json.dumps({k: v for k, v in m.items()}, indent=1, default=str))
        for w in m['warnings'] + grid_problems(m):
            print(f'{s}: {w}'); nw += 1
    print(f'{len(sys.argv[1:] or seqs)} sequences, {nw} warnings')
