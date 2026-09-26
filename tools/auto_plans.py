#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Automatic per-track plans for every localized track without a manual decision.

Reads   analysis/alignment_segments.json (tools/align_languages.py, against the *shared* video),
        recipes/sequence_edits.csv (manual decisions: pairs they cover are skipped),
        stage_a/audio/<lang>/<name>.flac (ffmpeg >= 8.1 decode) for pauses and tail levels.
Writes  recipes/sequence_edits_auto.csv (same columns as sequence_edits.csv, can be regenerated any time).
The manual file is the authority: the build uses a manual row whenever one covers the (sequence, language).

Rules (seconds are on the shared-video timeline, offset = localized time - shared time):
  start   offset < 0: pad the localized audio at the start.
          offset > 0: trim the start if that audio is quiet, else a head extension (flagged).
  change  localized gains d: hold the shared frame if the picture is still there, else loop an exact cycle,
          else cut d from a quiet gap (audio only), else loop an approximate cycle or hold (both flagged:
          an approximate cycle may repeat one-off events).
          Localized loses d: insert d of silence at a pause there, else flag.
          Timing is measured between anchor segments (matched stretches outside English cycles).
          Jumps inside a cycle are loop phase, and a change with a cycle between anchors is a real cycle extension.
  end     shorter: pad with silence.
          Longer: trim if the extra audio is effectively silent, else black (English ends on black),
          loop (English ends in an exact cycle) or hold (English ends frozen, or the localized version froze itself).
          Anything else is flagged.
Status: 'auto' = straightforward, 'auto-check' = needs a look (reason in the note).
"""
import csv, json, os, subprocess
import numpy as np

from paths import STAGE_A, RECIPES, ANALYSIS, FF
EPS = 0.10            # s: offsets or changes smaller than this are ignored (about one 12.5 fps frame)
QUIET = -45.0         # dBFS (DC removed): effectively silent
PAUSE_REL = 15.0      # dB below the track's median level counts as a pause
COLS = ['sequence', 'applies_to', 'op', 'params', 'source', 'status', 'note']
SRC = 'auto (alignment: brightness-normalized, low-info frames skipped, cycles, shared video)'


def manual_coverage():
    cov = set()
    for r in csv.DictReader(open(f'{RECIPES}/sequence_edits.csv')):
        for l in ('fre', 'ger'):
            if l in r['applies_to'] or 'fre + ger' in r['applies_to']:
                cov.add((r['sequence'], l))
    return cov


def audio_levels(lang, name, hop=0.02):
    p = f'{STAGE_A}/audio/{lang}/{name}.flac'
    if not os.path.exists(p):
        return None, hop
    a = np.frombuffer(subprocess.run([FF, '-v', 'error', '-i', p, '-ac', '1', '-af', 'highpass=f=20', '-f', 's16le', '-'],
                                     capture_output=True).stdout, np.int16).astype(float)
    n = int(hop * 22050); k = len(a) // n
    return 20 * np.log10(np.sqrt((a[:k * n].reshape(k, n) ** 2).mean(1)) / 32768 + 1e-9), hop


def level(lv, hop, t0, t1):
    if lv is None:
        return -120.0
    s = lv[int(max(0, t0) / hop):max(int(max(0, t0) / hop) + 1, int(t1 / hop))]
    return float(10 * np.log10((10 ** (s / 10)).mean())) if len(s) else -120.0


def find_pause(lv, hop, t0, t1, need):
    """Quietest stretch of at least `need` s within [t0, t1] (localized time): (start, end) or None."""
    if lv is None:
        return None
    thr = min(QUIET + 10, float(np.median(lv)) - PAUSE_REL)
    a, b = int(max(0, t0) / hop), min(len(lv), int(t1 / hop) + 1)
    best = None; i = a      # longest qualifying pause in the window
    while i < b:
        if lv[i] < thr:
            e = i
            while e < b and lv[e] < thr:
                e += 1
            if (e - i) * hop >= need and (best is None or e - i > best[1] - best[0]):
                best = (i, e)
            i = e
        else:
            i += 1
    return (best[0] * hop, best[1] * hop) if best else None


def pauses(lv, hop, t0, t1, need):
    """All pauses of at least `need` s within [t0, t1] (localized time)."""
    if lv is None:
        return []
    thr = min(QUIET + 10, float(np.median(lv)) - PAUSE_REL)
    a, b = int(max(0, t0) / hop), min(len(lv), int(t1 / hop) + 1); out = []; i = a
    while i < b:
        if lv[i] < thr:
            e = i
            while e < b and lv[e] < thr:
                e += 1
            if (e - i) * hop >= need:
                out.append((round(i * hop, 2), round(e * hop, 2)))
            i = e
        else:
            i += 1
    return out


def cycle_at(cycles, t, margin=0.25):
    for a, b, P, q in cycles:
        if a - margin <= t <= b + margin and P > 0:
            return a, b, P, q
    return None


def plan(j):
    out = []
    name, lang = j['name'], j['lang']
    add = lambda op, params, status='auto', note='': out.append(
        dict(sequence=name, applies_to=f'{lang}' + (' audio' if op in ('pad_head', 'trim_head', 'pad_tail', 'trim_tail', 'pad_gap', 'cut_gap') else ''),
             op=op, params=params, source=SRC, status=status, note=note))
    if j['mux'] == 'identical':
        add('identical', 'use the English audio for this track (byte-identical file)', 'auto',
            'identical files: the localized audio IS the English audio; apply the same audio edits as English' +
            (' (this sequence has shared-video edits)' if j.get('shared_modified') else ''))
        return out
    segs = j.get('segs') or []
    if not segs:
        add('check', 'no alignment segments', 'auto-check', f"mux {j['mux']}")
        return out
    efps, edur, ldur = j['efps'], j['edur'], j['ldur']
    lv, hop = audio_levels(lang, name)
    ecyc = j.get('eng_cycles', [])
    # ---- anchors: matched stretches outside English cycles.
    # Offsets inside a cycle are only meaningful up to whole periods, so all timing is measured between anchors
    # (a change with a cycle in between is a real cycle extension).
    def outside_cycles(sg):          # seconds of the segment's English range that lie outside every English cycle
        inside = sum(max(0.0, min(sg['u1'], c[1]) - max(sg['u0'], c[0])) for c in ecyc)
        return (sg['u1'] - sg['u0']) - inside
    # still stretches (frozen picture) also carry no timing: a still frame matches anywhere within the still
    cand = [sg for sg in segs if outside_cycles(sg) >= min(1.0, 0.5 * (sg['u1'] - sg['u0'])) and sg['t1'] - sg['t0'] >= 0.5
            and not sg.get('static')] or [sg for sg in segs if not sg.get('static')][:1] or segs[:1]
    # keep the longest (by duration) chain that moves forward in both timelines
    # (stray matches that jump back into an earlier part of the English are dropped)
    best = [(sg['t1'] - sg['t0'], -1) for sg in cand]
    for k, sg in enumerate(cand):
        for q in range(k):
            if cand[q]['t1'] <= sg['t0'] + 0.3 and cand[q]['u1'] <= sg['u0'] + 0.3 and best[q][0] + sg['t1'] - sg['t0'] > best[k][0]:
                best[k] = (best[q][0] + sg['t1'] - sg['t0'], q)
    k = max(range(len(cand)), key=lambda x: best[x][0]); chain = []
    while k >= 0:
        chain.append(cand[k]); k = best[k][1]
    anchors = chain[::-1]
    # ---- start
    o0 = anchors[0]['off']
    if o0 < -EPS:
        add('pad_head', f'silence {-o0:.2f}s at start')
    elif o0 > EPS:
        if level(lv, hop, 0, o0) < QUIET:
            add('trim_head', f'cut the first {o0:.2f}s (effectively silent, < {QUIET:.0f} dBFS)')
        else:
            c = cycle_at(ecyc, 0.0)
            how = (f'extend the opening {c[3]} cycle ({c[0]:.2f}-{c[1]:.2f}s, P={c[2]:.2f}s) by {o0:.2f}s (edit list, or bake + pad the other tracks)'
                   if c else f'head hold of the first frame {o0:.2f}s (edit list)')
            add('head_extend', how, 'auto-check' if not (c and c[3] == 'exact') else 'auto',
                f'localized has {o0:.2f}s more picture before the shared start, with audio')
    # ---- changes between anchors
    for a, b in zip(anchors, anchors[1:]):
        d = b['off'] - a['off']
        if abs(d) <= EPS:
            continue
        u = a['u1']
        cq = next((c for c in ecyc if c[1] >= a['u1'] - 0.25 and c[0] <= b['u0'] + 0.25), None)   # a cycle between them
        win0, win1 = min(a['t1'], b['t0']) - 0.6, max(a['t1'], b['t0']) + 0.6
        # inside an English still, a timing change is invisible: search the whole still for the best pause
        st = next((x for x in j.get('eng_stills', []) if x[0] - 0.3 <= u <= x[1] + 0.3 or x[0] - 0.3 <= b['u0'] <= x[1] + 0.3), None)
        if st:
            win0, win1 = st[0] + a['off'], max(st[1] + a['off'], b['t0'])
            if d > 0:
                add('mid_hold', f'hold the still (shared frame {int(round(u * efps))}, {u:.2f}s) for {d:.2f}s',
                    'auto', f'inside an English still ({st[0]:.2f}-{st[1]:.2f}s), so the hold is invisible')
                continue
            ps = pauses(lv, hop, win0, win1, 0.25)
            if ps:
                # spread the insertion over the pauses, in proportion to their length
                tot = sum(e - s0 for s0, e in ps); parts = [(s0, e, -d * (e - s0) / tot) for s0, e in ps]
                add('pad_gap', 'insert ' + ' + '.join(f'{x:.2f}s in the pause at {s0:.2f}-{e:.2f}s' for s0, e, x in parts) +
                    ' (localized time; room tone from each pause, short crossfades)',
                    'auto', f'the {lang} version cuts {-d:.2f}s of an English still ({st[0]:.2f}-{st[1]:.2f}s)')
                continue
        if d > 0:            # localized shows more picture here
            p = find_pause(lv, hop, win0, win1, d)
            if cq and cq[3] == 'exact':
                add('mid_loop', f'extend the exact cycle {cq[0]:.2f}-{cq[1]:.2f}s (P={cq[2]:.2f}s) by {d:.2f}s',
                    'auto', 'edit list (or bake + pad the other tracks if small)')
            elif p:
                add('cut_gap', f'remove {d:.2f}s from the pause at {p[0]:.2f}-{p[1]:.2f}s (localized time), short crossfade',
                    'auto', f'localized gains {d:.2f}s at shared {u:.2f}s')
            elif cq:
                add('mid_loop', f'extend the approx cycle {cq[0]:.2f}-{cq[1]:.2f}s (P={cq[2]:.2f}s) by {d:.2f}s',
                    'auto-check', 'approximate cycle: looping may repeat one-off events (ex. the c002 eye closing); consider a hold')
            else:
                add('mid_hold', f'hold shared frame {int(round(u * efps))} ({u:.2f}s) for {d:.2f}s',
                    'auto-check', f'no pause or cycle there; check the picture is still at {u:.2f}s, or accept the drift if tiny')
        else:                # localized shows less picture: needs silence inserted
            p = find_pause(lv, hop, win0, win1, 0.05)
            if p:
                add('pad_gap', f'insert {-d:.2f}s of silence (or room tone) in the pause at {p[0]:.2f}-{p[1]:.2f}s (localized time)',
                    'auto', f'localized skips {-d:.2f}s of shared picture at {u:.2f}s')
            else:
                add('pad_gap', f'insert {-d:.2f}s at ~{a["t1"]:.2f}s (localized time), or accept the drift',
                    'auto-check', 'no pause found nearby (speech continuous)')
    # ---- end (from the last anchor: cyclic segments after it are loop phase)
    last = anchors[-1]
    covered_end = edur + last['off']          # localized time at which the shared video ends
    tail = ldur - covered_end
    if tail < -EPS:
        add('pad_tail', f'silence {-tail:.2f}s at end (to the shared video length {edur:.2f}s)')
    elif tail > EPS:
        tl = level(lv, hop, covered_end, ldur)
        if tl < QUIET:
            add('trim_tail', f'cut at {covered_end:.2f}s (localized time; the remaining {tail:.2f}s is effectively silent, {tl:.0f} dBFS); remove DC first, short fade')
            return out
        ee = j.get('eng_end', {}); le = j.get('loc_end', {})
        # jumps back into earlier English
        back = [sg for sg in segs if sg['t0'] >= last['t1'] - 0.3 and sg['u0'] < last['u1'] - 1.0]
        c_end = cycle_at(ecyc, edur - 0.05, 0.0)
        if c_end:   # replaying earlier turns of the cycle the English ends in is just that cycle continuing
            back = [sg for sg in back if not (c_end[0] - 0.25 <= sg['u0'] and sg['u1'] <= c_end[1] + 0.25)]
        if back:
            u0 = min(sg['u0'] for sg in back); u1 = max(sg['u1'] for sg in back)
            # jump where the localized version jumps if the English after that point is only a freeze
            # or a short remainder (up to 1 s, or the English freeze + 0.15 s if that is longer)
            rest = edur - last['u1']
            if 0.05 < rest <= max(ee.get("frozen_s", 0) + 0.15, 1.0):
                kind = 'freeze' if rest <= ee.get('frozen_s', 0) + 0.15 else 'frames'
                add('tail_segment', f'at shared {last["u1"]:.2f}s (skipping the English final {rest:.2f}s {kind}), play English {u0:.2f}-{u1:.2f}s (repeat as needed) for {ldur - last["t1"]:.2f}s',
                    'auto-check', f'the {lang} version jumps back into English {u0:.2f}-{u1:.2f}s there itself (as cam32alp/cam3bpl); check the jump')
            else:
                add('tail_segment', f'after the shared end, play English {u0:.2f}-{u1:.2f}s (repeat as needed) for {tail:.2f}s',
                    'auto-check', f'the {lang} version jumps back into English {u0:.2f}-{u1:.2f}s there itself (as cam32alp/cam3bpl); check the jump'
                    + (f'; note: {rest:.2f}s of moving English picture plays before the jump' if rest > 0.3 else ''))
            return out
        c = cycle_at(ecyc, edur - 0.05, 0.0)
        note = '' if tl > -35 else f'extra audio is quiet ({tl:.0f} dBFS): maybe only an effect/music tail, so a trim could be an option (listen)'
        if ee.get('black'):
            add('tail_black', f'black {tail:.2f}s after the shared end', 'auto' if not note else 'auto-check', note)
        elif c and c[3] == 'exact':
            add('tail_loop', f'loop the exact cycle ending at the shared end ({max(c[0], edur - c[2]):.2f}-{edur:.2f}s, P={c[2]:.2f}s) for {tail:.2f}s',
                'auto' if not note else 'auto-check', note)
        elif c:
            add('tail_loop', f'loop the approx cycle ({max(c[0], edur - c[2]):.2f}-{edur:.2f}s, P={c[2]:.2f}s) for {tail:.2f}s', 'auto-check',
                ('approximate cycle: may repeat one-off events; ' + note).strip('; '))
        elif ee.get('frozen_s', 0) >= 0.3:
            add('tail_hold', f'hold the last frame {tail:.2f}s (the English already ends frozen for {ee.get("frozen_s", 0):.2f}s)',
                'auto' if not note else 'auto-check', note)
        elif le.get('frozen_s', 0) >= 0.8 * tail:
            add('tail_hold', f'hold the last frame {tail:.2f}s (the {lang} version itself freezes its last {le["frozen_s"]:.2f}s)',
                'auto' if not note else 'auto-check', note)
        elif (le.get('cyclic') or le.get('stock')) and [c for c in ecyc if c[3] == 'exact']:
            c = max((c for c in ecyc if c[3] == 'exact'), key=lambda c: c[1] - c[0])
            add('tail_segment', f'loop the English exact cycle {c[1] - c[2]:.2f}-{c[1]:.2f}s (P={c[2]:.2f}s) for {tail:.2f}s after the shared end',
                'auto-check', f'the {lang} version continues an animation/atom past the English end, but the English ends without one; '
                'loop a cycle from elsewhere in the clip (as cam32alp/cam3bpl) - check the jump. ' + note)
        else:
            add('tail_hold', f'hold the last frame {tail:.2f}s', 'auto-check',
                ('the English ends on moving picture: a hold may look stalled; look for a cycle or use black. ' + note).strip())
    if not out:
        add('none', f'fit: offset {o0:+.2f}s (ignored), same length', 'auto')
    return out


if __name__ == '__main__':
    cov = manual_coverage()
    js = json.load(open(f'{ANALYSIS}/alignment_segments.json'))
    rows = []
    for j in sorted(js, key=lambda j: (j['name'], j['lang'])):
        if (j['name'], j['lang']) in cov:
            continue
        rows += plan(j)
    with open(f'{RECIPES}/sequence_edits_auto.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
    from collections import Counter
    tracks = {(r['sequence'], r['applies_to'].split()[0]) for r in rows}
    chk = {(r['sequence'], r['applies_to'].split()[0]) for r in rows if r['status'] == 'auto-check'}
    print(f'{len(tracks)} tracks planned, {len(rows)} rows -> recipes/sequence_edits_auto.csv')
    print('ops:', dict(Counter(r['op'] for r in rows).most_common()))
    print(f'tracks needing a check: {len(chk)}')
