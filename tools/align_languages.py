#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Map each localized (fre/ger) sequence onto its English counterpart, frame by frame.

Inputs  (in analysis/):  fp/<lang>/<name>.fp       32x24 uint8 grayscale per frame (tools/fingerprint.sh)
                         stock_frames.pkl          stock-animation frames (tools/detect_stock.py)
        recipes/inventory.csv                      the frame rate of every file
Outputs (in analysis/):  alignment.csv             one row per (sequence, language)
                         alignment_detail.txt      runs and jumps per pair
                         alignment_segments.json   segments, cycles and ending state per pair (for tools/auto_plans.py)
The .rpl files (--src) are read only to find localized files that are byte-identical to the English.
The English side is the *shared video* (the English after the confirmed shared-video edits, tools/shared_video.py).

Method: for each localized frame, find the best English frame of the same sequence (brightness-normalized).
Among near-ties (static scenes), prefer the one that continues the previous match, so the path stays stable.
The path is split into runs of constant offset.
Jumps between runs are classified as repeats (backward), skips (forward)
or holds/insertions (loc gap longer than eng gap).
The same is done English->localized to catch content the localized version repeats fewer times.
Unmatched stretches are searched in all English clips.
"""
import argparse, csv, glob, hashlib, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RECIPES, ANALYSIS, RPL
import shared_video
from multiprocessing import Pool

# Matching uses per-frame normalized fingerprints (zero mean, unit spread, spread floored at 10 levels),
# so a global brightness or contrast difference between language versions doesn't break matches
# (ex. fre/ger cam1/c001 is much darker).
# The distances below are in those normalized units.
THR = 0.40       # normalized RMS distance below which two 32x24 fingerprints are "the same frame"
TIE = 0.04       # candidates within this of the best are ties (static or near-static scenes)
MIN_RUN = 0.4    # seconds: shorter runs are fragments
JUMP = 0.3       # seconds: an offset change larger than this starts a new run
MIN_REPEAT = 0.5 # seconds of reference content a run must show again to count as a repeat (loop)
HOLD_THR = 4.0   # RMS distance to the first/last frame below which the picture counts as frozen (held)
FLAT_STD = 3.0   # fingerprint std (levels) below which a frame is blank (black or flat padding)
LOWINFO_STD = 6.0  # fingerprint std below which a frame carries no identity (black, white-outs, flat fades):
                   # skipped when matching, and doesn't break runs
CYC_THR = 2.5    # raw RMS distance (32x24) between frames one period apart for an 'exact' cycle (loops seamlessly)
CYC_APPROX_ABS = 12.0  # 'approx' cycle (repeats with drift or inset changes): median one-period distance over a period
CYC_APPROX_MIN_P = 0.8 # shortest 'approx' period (s) (approx cycles also need >= 2 periods and >= 2 s of repeats)
CYC_APPROX_REL = 0.7   # ... and a median one-period distance below this fraction of the picture's own motion
CYC_MIN_P, CYC_MAX_P = 0.4, 4.2   # searched cycle periods (s)

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=RPL)
args = ap.parse_args()


def load(lang, name):
    return np.fromfile(f'{ANALYSIS}/fp/{lang}/{name}.fp', np.uint8).reshape(-1, 768).astype(np.float32)


FPS = {(r['lang'], r['name']): float(r['fps']) for r in csv.DictReader(open(f'{RECIPES}/inventory.csv'))}

NAMES = sorted(os.path.relpath(p, f'{ANALYSIS}/fp/eng')[:-3] for p in glob.glob(f'{ANALYSIS}/fp/eng/**/*.fp', recursive=True))


def znorm(a):
    return (a - a.mean(1, keepdims=True)) / np.maximum(a.std(1, keepdims=True), 10.0)


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0) / A.shape[1])


def track(D, ratio, skip=None):
    """Best-match path through distance matrix D (rows = query frames). Returns match index and distance.
    Rows in `skip` (low-information frames) are left unmatched without breaking the path's continuity."""
    n = len(D); j = np.full(n, -1); d = np.full(n, np.inf)
    prev = None; last_i = None
    for i in range(n):
        if skip is not None and skip[i]:
            continue
        if prev is not None and last_i is not None:
            prev = prev + ratio * (i - last_i - 1)   # advance the prediction across skipped rows
        last_i = i
        row = D[i]; m = row.min()
        if m >= THR:
            prev = None; d[i] = m; continue
        cand = np.where(row <= m + TIE)[0]
        k = cand[np.abs(cand - (prev + ratio)).argmin()] if prev is not None else row.argmin()
        j[i] = k; d[i] = row[k]; prev = k
    return j, d


def runs_of(j, qfps, rfps, skip=None):
    """Split a path into runs of (roughly) constant offset. Offsets in seconds: query time - reference time.
    Frames in `skip` (low-information) are transparent: a run continues across them."""
    skip = np.zeros(len(j), bool) if skip is None else skip
    info_before = np.r_[0, np.cumsum(~skip)]        # informative frames before index i
    runs = []
    for i in range(len(j)):
        if j[i] < 0:
            continue
        off = i / qfps - j[i] / rfps
        contiguous = runs and info_before[i] - info_before[runs[-1]['q1'] + 1] == 0
        if runs and contiguous and abs(off - runs[-1]['off']) <= JUMP:
            r = runs[-1]; r['q1'] = i; r['r1'] = max(r['r1'], j[i]); r['r0'] = min(r['r0'], j[i])
        else:
            runs.append(dict(q0=i, q1=i, r0=j[i], r1=j[i], off=off))
    # merge runs with the same offset split only by up to 3 unmatched frames (low-information frames don't count)
    merged = []
    for r in runs:
        gap_info = info_before[r['q0']] - info_before[merged[-1]['q1'] + 1] if merged else 99
        if merged and gap_info <= 3 and abs(r['off'] - merged[-1]['off']) <= JUMP:
            m = merged[-1]; m['q1'] = r['q1']; m['r0'] = min(m['r0'], r['r0']); m['r1'] = max(m['r1'], r['r1'])
        else:
            merged.append(dict(r))
    for r in merged:
        r['dur'] = (r['q1'] - r['q0'] + 1) / qfps
        r['t0'], r['t1'] = r['q0'] / qfps, (r['q1'] + 1) / qfps
        r['u0'], r['u1'] = r['r0'] / rfps, (r['r1'] + 1) / rfps
    return merged


def structure(runs, qdur, rdur, qatom):
    """Summarize a run list: repeats (content shown again), skips, insertions, coverage.
    Runs that are mostly stock-atom animation (qatom = per-query-frame mask) don't count as loop repeats
    (stock-animation length differences are measured separately)."""
    big = [r for r in runs if r['dur'] >= MIN_RUN] or runs
    covered = np.zeros(int(rdur * 100) + 2, bool)
    repeat_t = 0.0; repeats = []; skips = []; inserts = []
    for k, r in enumerate(big):
        a, b = int(r['u0'] * 100), int(r['u1'] * 100)
        seg = covered[a:b]
        # a repeat shows >= MIN_REPEAT of reference content again (shorter is held or static frames, not a loop)
        is_atom = qatom[r['q0']:r['q1'] + 1].mean() > 0.5
        if len(seg) and seg.mean() > 0.5 and r['u1'] - r['u0'] >= MIN_REPEAT and not is_atom:
            repeat_t += r['dur']; repeats.append(r)
        covered[a:b] = True
        if k:
            p = big[k - 1]
            ref_gap = r['u0'] - p['u1']; q_gap = r['t0'] - p['t1']
            if ref_gap < -JUMP:
                pass  # backward jump: counted via repeats
            elif ref_gap > JUMP:
                skips.append((p['u1'], r['u0']))
            if q_gap - max(ref_gap, 0) > JUMP:
                inserts.append((p['t1'], r['t0'], q_gap - max(ref_gap, 0)))
    period = None
    if repeats:
        spans = [r['u1'] - r['u0'] for r in repeats]
        period = float(np.median(spans))
    return dict(cov=covered[:int(rdur * 100)].mean() if rdur else 0, repeat_t=repeat_t, n_repeat=len(repeats),
                period=period, skips=skips, inserts=inserts, big=big, repeats=repeats)


# 16x12 fingerprints, and a global English index of them (GE, GL) for finding unmatched localized frames in other clips
def small(a):
    return a.reshape(-1, 12, 2, 16, 2).mean((2, 4)).reshape(len(a), -1)


_ST = pickle.load(open(f'{ANALYSIS}/stock_frames.pkl', 'rb'))   # tools/detect_stock.py
STOCK = _ST['frames']; KINDS = _ST['kinds']

GE, GL = [], []
for n in NAMES:
    e = znorm(small(load('eng', n))); GE.append(e); GL += [n] * len(e)
GE = np.concatenate(GE); GL = np.array(GL)


def flat(F):
    """Frames with (almost) no picture: black or blank padding.
    They match every other blank frame, so they carry no identity."""
    return F.std(1) < FLAT_STD


def lowinfo(F):
    return F.std(1) < LOWINFO_STD


def cycles(F, fps):
    """Stretches where the picture repeats with a period between CYC_MIN_P and CYC_MAX_P seconds.
    Returns (mask, period_s per frame (0 = none), segments [(start_s, end_s, period_s, quality)]).
    quality is 'exact' (frames one period apart within CYC_THR: loops seamlessly)
    or 'approx' (within CYC_APPROX_ABS: repeats with drift, ex. the cam3abf rotation whose tint drifts).
    A stretch counts only if frames one period apart match for a full period
    (or, in a clip shorter than two periods, the whole available overlap, ex. brfcom: 2.4 s with a 2.0 s cycle),
    it lasts >= 1 s, and the picture really moves (a frozen or flickering still is a hold, not a cycle)."""
    n = len(F); mask = np.zeros(n, bool); per = np.zeros(n); qual = np.zeros(n, np.int8)   # 2 exact, 1 approx
    motion = np.r_[0, np.sqrt(((F[1:] - F[:-1]) ** 2).mean(1))]
    from numpy.lib.stride_tricks import sliding_window_view as swv
    for tier in (2, 1):
        for P in range(max(2, int(round(CYC_MIN_P * fps))), min(n - 1, int(round(CYC_MAX_P * fps))) + 1):
            dP = np.sqrt(((F[P:] - F[:-P]) ** 2).mean(1))                  # dP[k]: frame k+P vs frame k
            if tier == 2:
                ok = dP < CYC_THR
            else:
                # drifting cycles: over one period, the repeat distance stays well below the picture's own motion
                w = min(P, len(dP))
                med = np.median(swv(dP, w), 1); mot = swv(motion[1:len(dP) + 1], w).mean(1) if len(motion) > len(dP) else med
                okw = (med < CYC_APPROX_ABS) & (med < CYC_APPROX_REL * mot)
                ok = np.zeros(len(dP), bool)
                for q in np.where(okw)[0]:
                    ok[q:q + w] = True
                # ...but every frame must itself stay close to its repeat, so a hard cut always ends a cycle
                ok &= dP < CYC_APPROX_ABS
            k = 0
            while k < len(ok):
                if not ok[k]:
                    k += 1; continue
                e = k
                while e < len(ok) and ok[e]:
                    e += 1
                a, b = k, e + P                                            # frames a..b-1 are cyclic
                long_enough = (b - a) / fps >= 1.0 if tier == 2 else ((b - a) >= 2 * P and (b - a) / fps >= 2.0)
                if tier == 1 and P / fps < CYC_APPROX_MIN_P:
                    long_enough = False                                    # short 'approx' periods are flicker or pans
                if e - k >= min(P, len(ok)) and long_enough and motion[a + 1:b].mean() > 2.0:
                    sel = np.arange(a, b); new = qual[sel] < tier
                    per[sel[new & ~mask[sel]]] = P / fps
                    qual[sel[new]] = tier; mask[sel] = True
                k = e
    segs = []; i = 0
    while i < n:
        if mask[i]:
            e = i
            while e < n and mask[e]:
                e += 1
            q = 'exact' if (qual[i:e] == 2).mean() >= 0.5 else 'approx'
            segs.append((round(i / fps, 2), round(e / fps, 2), round(float(np.median(per[i:e][per[i:e] > 0])) if (per[i:e] > 0).any() else 0, 2), q))
            i = e
        else:
            i += 1
    return mask, per, segs


def stills(F, fps, min_s=1.0):
    """Stretches (>= min_s) where the picture doesn't move: [(start_s, end_s)]."""
    mo = np.r_[0, np.sqrt(((F[1:] - F[:-1]) ** 2).mean(1))] < 1.0
    out = []; i = 0
    while i < len(mo):
        if mo[i]:
            e = i
            while e < len(mo) and mo[e]:
                e += 1
            if (e - i) / fps >= min_s:
                out.append((round(i / fps, 2), round(e / fps, 2)))
            i = e
        else:
            i += 1
    return out


def edge_black(F, fps, end):
    """Seconds of blank frames at the head (end=False) or tail (end=True)."""
    f = flat(F)[::-1] if end else flat(F)
    return (int(np.argmin(f)) if not f.all() else len(f)) / fps


def other_clips(L, mask, lfps, own):
    """Runs (>=1 s) of unmatched localized frames that match a *different* English clip."""
    idx = np.where(mask & ~flat(L))[0]
    if not len(idx):
        return []
    S = znorm(small(L[idx])); D = dist(S, GE); k = D.argmin(1); dd = D[np.arange(len(idx)), k]
    found = []; cur = None
    for i, kk, d in zip(idx, k, dd):
        name = GL[kk] if d < THR else None
        if name == own:
            name = None
        if cur and name == cur[0] and i - cur[2] <= 3:
            cur[2] = i
        else:
            if cur and cur[0] and (cur[2] - cur[1] + 1) / lfps >= 1.0:
                found.append((cur[0], cur[1] / lfps, (cur[2] + 1) / lfps))
            cur = [name, i, i]
    if cur and cur[0] and (cur[2] - cur[1] + 1) / lfps >= 1.0:
        found.append((cur[0], cur[1] / lfps, (cur[2] + 1) / lfps))
    return found


def held(F, fps, end):
    """Seconds of (near-)frozen picture at the head (end=False) or tail (end=True) of a clip."""
    ref = F[-1] if end else F[0]
    d = np.sqrt(((F - ref) ** 2).mean(1))
    seq = d[::-1] if end else d
    n = int(np.argmax(seq >= HOLD_THR)) if (seq >= HOLD_THR).any() else len(seq)
    return max(n - 1, 0) / fps


def mux_plan(fw, E, L, efps, lfps, edur, ldur, la, cls, sx, lcyc=None, ecyc=None):
    """How the localized audio could be put on the English video.
    Segments = consecutive runs with the same offset. Each change of offset is a point where one version shows
    more (or less) picture than the other.
    It is 'elastic' if the localized picture there is stock animation, a repeated loop, or frozen
    (something that could be stretched without new footage).
    head_ext/tail_ext = seconds of picture the localized version needs before English starts / after it ends."""
    if cls in 'ADE':
        return dict(mux={'A': 'identical', 'D': 'merged', 'E': 'different'}[cls])
    big = fw['big']
    segs = []
    for r in big:
        if segs and abs(r['off'] - segs[-1]['off']) <= 0.2 and not any(r is x for x in fw['repeats']):
            segs[-1]['t1'] = r['t1']; segs[-1]['q1'] = r['q1']
        else:
            segs.append(dict(r, rep=any(r is x for x in fw['repeats'])))
    motion = np.r_[np.inf, np.sqrt(((L[1:] - L[:-1]) ** 2).mean(1))]
    hard = 0
    for a, b in zip(segs, segs[1:]):
        w0 = max(0, int((a['t1'] - 1.0) * lfps)); w1 = min(len(L), int((b['t0'] + 1.0) * lfps))
        win = slice(w0, max(w1, w0 + 1))
        e0 = max(0, int((min(a['u1'], b['u0']) - 1.0) * efps)); e1 = min(len(E), int((max(a['u1'], b['u0']) + 1.0) * efps))
        in_cycle = (lcyc is not None and lcyc[win].mean() >= 0.5) or (ecyc is not None and e1 > e0 and ecyc[e0:e1].mean() >= 0.5)
        elastic = b['rep'] or a['rep'] or (la | sx)[win].mean() >= 0.5 or np.median(motion[win]) < 1.0 or in_cycle
        hard += not elastic
    first, last = segs[0], segs[-1]
    head_ext = max(0.0, first['off']) if first['u0'] <= JUMP else max(0.0, first['t0'] - first['u0'])
    tail_ext = max(0.0, ldur - (edur + last['off']))
    kind = ''
    if tail_ext > JUMP:
        tail = slice(max(0, int((edur + last['off']) * lfps)), len(L))
        kind = 'stock' if la[tail].mean() >= 0.5 else 'shot' if sx[tail].mean() >= 0.5 else 'black' if flat(L[tail]).mean() >= 0.5 else ('hold' if np.median(motion[tail][1:] if len(motion[tail]) > 1 else [0]) < 1.0 else 'other')
    if len(segs) == 1:
        mux = 'fit' if head_ext <= JUMP and tail_ext <= JUMP else ('tail_ext' if head_ext <= JUMP else 'head_ext')
    else:
        mux = 'multi_elastic' if hard == 0 else 'multi_hard'
    return dict(mux=mux, n_changes=len(segs) - 1, hard_changes=hard, head_ext=round(head_ext, 2),
                tail_ext=round(tail_ext, 2), tail_kind=kind,
                _segs=[dict(t0=round(s['t0'], 2), t1=round(s['t1'], 2), u0=round(s['u0'], 2), u1=round(s['u1'], 2),
                            off=round(s['off'], 2), rep=bool(s['rep']),
                            # a still English stretch matches anywhere within itself, so its offset carries no timing
                            static=bool((np.r_[0, np.sqrt(((E[1:] - E[:-1]) ** 2).mean(1))][int(s['u0'] * efps):max(int(s['u1'] * efps), int(s['u0'] * efps) + 2)] >= 1.0).mean() < 0.2))
                       for s in segs])


def shared_shot_extensions(others, E, efps, lfps, edur, fw, n_loc):
    """Split 'other clip' matches into genuine merges and shared-shot extensions.
    An other-clip stretch is an extension (elastic) when it lies after all of the English clip's content
    (or before it) AND the English clip's own last (first) second also appears in that other clip.
    The localized version then simply shows more of a shot the English clip ends (starts) with
    (ex. a longer rotation of the same model in cam1/cam1ccf)."""
    if not fw['big'] or not others:
        return others, [], np.zeros(n_loc, bool)
    first, last = fw['big'][0], fw['big'][-1]
    k = max(1, int(efps))
    edge = {'tail': znorm(E[-k:]), 'head': znorm(E[:k])}
    keep, ext = [], []
    mask = np.zeros(n_loc, bool)
    for o, a, b in others:
        pos = 'tail' if (a >= last['t1'] - 0.5 and last['u1'] >= edur - 1.0) else \
              ('head' if (b <= first['t0'] + 0.5 and first['u0'] <= 1.0) else None)
        shared = False
        if pos:
            O = znorm(load('eng', o))
            shared = float(np.median(dist(edge[pos], O).min(1))) < THR
        if shared:
            ext.append((o, a, b, pos)); mask[int(a * lfps):int(round(b * lfps))] = True
        else:
            keep.append((o, a, b))
    return keep, ext, mask


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def analyse(item):
    name, lang = item
    if not os.path.exists(f'{ANALYSIS}/fp/{lang}/{name}.fp'):
        return None
    L = load(lang, name)
    src = shared_video.frames(name, lambda c: len(load('eng', c)))       # shared video = English after confirmed edits
    cache = {}
    E = np.stack([(cache.setdefault(c, load('eng', c)))[k] for c, k in src])
    ek = np.array([STOCK[('eng', c)][k] for c, k in src])
    efps, lfps = FPS[('eng', name)], FPS[(lang, name)]
    edur, ldur = len(E) / efps, len(L) / lfps
    row = dict(name=name, lang=lang, eng_fps=efps, loc_fps=lfps, eng_dur=round(edur, 2), loc_dur=round(ldur, 2),
               dur_diff=round(ldur - edur, 2))
    if md5(f'{args.src}/eng/{name}.rpl') == md5(f'{args.src}/{lang}/{name}.rpl'):
        row.update(cls='A', mux='identical', note='byte-identical'); return row, ''
    D = dist(znorm(L), znorm(E))
    li_l, li_e = lowinfo(L), lowinfo(E)
    D[:, li_e] = 9.0; D[li_l, :] = 9.0             # low-information frames can't match or be matched
    j, d = track(D, efps / lfps, li_l)             # localized -> English
    jr, dr = track(D.T, lfps / efps, li_e)         # English -> localized
    lk = STOCK[(lang, name)]
    la, ea = lk > 0, ek > 0                        # stock "elastic" animation frames (atom, NEXUS logo)
    fw = structure(runs_of(j, lfps, efps, li_l), ldur, edur, la)
    bw = structure(runs_of(jr, efps, lfps, li_e), edur, ldur, ea)
    matched = float(((j >= 0) | li_l).mean())
    atom_loc, atom_eng = la.sum() / lfps, ea.sum() / efps
    unmatched = (j < 0) & ~la & ~li_l              # stock-animation and low-information frames are never 'unmatched'
    others = other_clips(L, unmatched, lfps, name)
    others, shot_ext, sx = shared_shot_extensions(others, E, efps, lfps, edur, fw, len(L))
    other_t = sum(b - a for _, a, b in others)
    shot_t = sum(b - a for _, a, b, _ in shot_ext)
    offs = [r['off'] for r in fw['big']]
    main = max(fw['big'], key=lambda r: r['dur']) if fw['big'] else None
    insert_t = sum(x[2] for x in fw['inserts']); skip_t = sum(b - a for a, b in fw['skips'])
    row.update(matched=round(matched, 3), eng_cov=round(fw['cov'], 3), n_runs=len(fw['big']),
               main_off=round(main['off'], 2) if main else '', off_min=round(min(offs), 2) if offs else '',
               off_max=round(max(offs), 2) if offs else '',
               loc_repeat_t=round(fw['repeat_t'], 2), loc_repeat_period=round(fw['period'], 2) if fw['period'] else '',
               eng_repeat_t=round(bw['repeat_t'], 2), eng_repeat_period=round(bw['period'], 2) if bw['period'] else '',
               insert_t=round(insert_t, 2), skip_t=round(skip_t, 2), other_t=round(other_t, 2),
               others=';'.join(f'{n}@{a:.1f}-{b:.1f}' for n, a, b in others),
               shot_ext=';'.join(f'{p}:{n}@{a:.1f}-{b:.1f}' for n, a, b, p in shot_ext))
    unexplained = unmatched.sum() / lfps - other_t - shot_t
    loop_d = fw['repeat_t'] - bw['repeat_t']; atom_d = atom_loc - atom_eng
    tags = []
    if fw['repeat_t'] >= 1.0 or bw['repeat_t'] >= 1.0:
        tags.append('loop')
    for ki, kind in enumerate(KINDS[1:], 1):
        if abs((lk == ki).sum() / lfps - (ek == ki).sum() / efps) >= 0.5:
            tags.append(kind)
    if fw['big']:
        first, last = fw['big'][0], fw['big'][-1]
        if first['u0'] > JUMP: tags.append('eng_head_not_in_loc')
        if first['t0'] > JUMP and not la[:first['q0']].mean() > 0.5: tags.append('loc_extra_head')
        if last['u1'] < edur - JUMP: tags.append('eng_tail_not_in_loc')
        if last['t1'] < ldur - JUMP and not la[last['q1'] + 1:].mean() > 0.5: tags.append('loc_extra_tail')
    if insert_t > JUMP: tags.append('holds')
    if shot_t: tags.append('shot_ext')
    bh_e, bt_e = edge_black(E, efps, False), edge_black(E, efps, True)
    bh_l, bt_l = edge_black(L, lfps, False), edge_black(L, lfps, True)
    if bt_l - bt_e >= 0.5: tags.append('black_tail')
    if bh_l - bh_e >= 0.5: tags.append('black_head')
    # frozen first/last frame, only counted when not stock animation (that is measured above)
    hh_e = 0 if ea[:2].any() else held(E, efps, False); hh_l = 0 if la[:2].any() else held(L, lfps, False)
    ht_e = 0 if ea[-2:].any() else held(E, efps, True); ht_l = 0 if la[-2:].any() else held(L, lfps, True)
    hold_d = (hh_l - hh_e) + (ht_l - ht_e)
    if abs(ht_l - ht_e) >= 0.5: tags.append('hold_tail')
    if abs(hh_l - hh_e) >= 0.5: tags.append('hold_head')
    row.update(stock_eng=round(atom_eng, 2), stock_loc=round(atom_loc, 2), unexplained_t=round(unexplained, 2),
               hold_eng=round(hh_e + ht_e, 2), hold_loc=round(hh_l + ht_l, 2),
               residual=round(ldur - edur - loop_d - atom_d - hold_d, 2), tags='+'.join(tags))
    # classification (priority order)
    if other_t >= 1.0:
        cls = 'D'
    elif unexplained > max(1.0, 0.2 * ldur):
        cls = 'E'
    elif any(t in tags for t in ['loop', 'hold_tail', 'hold_head', 'shot_ext', 'black_tail', 'black_head'] + KINDS[1:]):
        cls = 'L'
    elif offs and max(offs) - min(offs) <= 0.2 and fw['cov'] >= 0.9 and abs(ldur - edur) <= 0.3:
        cls = 'B'
    else:
        cls = 'C'
    row['cls'] = cls
    lcyc, _, lsegs = cycles(L, lfps); ecyc, _, esegs = cycles(E, efps)
    mp = mux_plan(fw, E, L, efps, lfps, edur, ldur, la, cls, sx, lcyc, ecyc)
    segs_json = mp.pop('_segs', [])
    row.update(mp)
    frz_e = held(E, efps, True)
    row['_json'] = dict(name=name, lang=lang, efps=efps, lfps=lfps, edur=round(edur, 2), ldur=round(ldur, 2),
                        mux=row['mux'], segs=segs_json, eng_cycles=esegs, loc_cycles=lsegs, eng_stills=stills(E, efps),
                        eng_end=dict(black=bool(flat(E[-1:])[0]), frozen_s=round(frz_e, 2),
                                     cyclic=bool(ecyc[-1]), stock=bool(ek[-3:].any())),
                        loc_tail_black=bool(flat(L[-1:])[0]),
                        loc_end=dict(frozen_s=round(held(L, lfps, True), 2), cyclic=bool(lcyc[-1]), stock=bool(lk[-3:].any())),
                        shared_modified=name in shared_video.SPECS or name in shared_video.trims())
    lines = [f'== {lang} {name}  class {cls} [{row["tags"]}]  eng {edur:.2f}s@{efps:g}  {lang} {ldur:.2f}s@{lfps:g}  matched {matched:.0%}  eng covered {fw["cov"]:.0%}  stock anim eng {atom_eng:.2f}s / {lang} {atom_loc:.2f}s']
    for r in fw['big']:
        rep = '  (repeat)' if any(r is x for x in fw['repeats']) else ''
        lines.append(f'   {lang} {r["t0"]:6.2f}-{r["t1"]:6.2f}s  -> eng {r["u0"]:6.2f}-{r["u1"]:6.2f}s  offset {r["off"]:+6.2f}s{rep}')
    if fw['repeat_t']:
        lines.append(f'   {lang} repeats {fw["repeat_t"]:.2f}s of English content (period ~{fw["period"]:.2f}s)')
    if bw['repeat_t']:
        lines.append(f'   eng repeats {bw["repeat_t"]:.2f}s of {lang} content (period ~{bw["period"]:.2f}s)')
    for a, b, g in fw['inserts']:
        lines.append(f'   {lang} extra/held time {g:.2f}s between {a:.2f}s and {b:.2f}s')
    for a, b in fw['skips']:
        lines.append(f'   eng content {a:.2f}-{b:.2f}s not in {lang}')
    for n, a, b in others:
        lines.append(f'   {lang} {a:.2f}-{b:.2f}s matches OTHER English clip {n}')
    for n, a, b, p in shot_ext:
        lines.append(f'   {lang} {a:.2f}-{b:.2f}s extends the {p} shot (also seen in English clip {n}); elastic')
    return row, '\n'.join(lines)


if __name__ == '__main__':
    items = [(n, l) for n in NAMES for l in ['fre', 'ger']]
    with Pool(6) as p:
        res = [r for r in p.map(analyse, items, chunksize=1) if r]
    os.makedirs(ANALYSIS, exist_ok=True)
    cols = ['name', 'lang', 'cls', 'eng_fps', 'loc_fps', 'eng_dur', 'loc_dur', 'dur_diff', 'matched', 'eng_cov', 'n_runs',
            'main_off', 'off_min', 'off_max', 'loc_repeat_t', 'loc_repeat_period', 'eng_repeat_t', 'eng_repeat_period',
            'insert_t', 'skip_t', 'other_t', 'others', 'shot_ext', 'stock_eng', 'stock_loc', 'hold_eng', 'hold_loc', 'mux', 'n_changes', 'hard_changes', 'head_ext', 'tail_ext', 'tail_kind', 'unexplained_t', 'residual', 'tags', 'note']
    with open(f'{ANALYSIS}/alignment.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols); w.writeheader()
        for row, _ in res:
            w.writerow({c: row.get(c, '') for c in cols})
    with open(f'{ANALYSIS}/alignment_detail.txt', 'w') as f:
        f.write('\n\n'.join(t for _, t in res if t) + '\n')
    json.dump([row.get('_json', dict(name=row['name'], lang=row['lang'], mux=row['mux'])) for row, _ in res],
              open(f'{ANALYSIS}/alignment_segments.json', 'w'), indent=0)
    print(f'{len(res)} pairs analyzed -> {ANALYSIS}/alignment.csv, alignment_detail.txt')
