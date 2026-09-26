#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Audio analysis of every decoded track (stage_a/audio/<lang>/<name>.flac, decoded with ffmpeg 8.1).

Per track:
  loudness     EBU R128 integrated loudness (LUFS), loudness range (LU), true peak (dBTP) - after the stage (b)
               10 Hz high-pass (2nd-order Butterworth, removes DC and ADPCM drift)
  peak/clip    sample peak (dBFS), clipped samples (|x| >= 32767 or runs of >= 3 equal samples at >= -0.1 dBFS)
  DC           mean offset (dBFS of |mean|), DC drift (max - min of 0.5 s window means, in sample units),
               subsonic wander (RMS below ~9 Hz, dBFS)
  silence      leading / trailing silence (s, < -60 dBFS after DC removal, 20 ms windows), share of silent windows
  pops         clicks: <= 2 ms bursts of 1st-difference energy > 15 dB above the median of both the 20 ms before
               and after (sound onsets stay loud afterwards, so they don't count), > -40 dBFS (counted, with times)
Also per (sequence, language): loudness difference to English (loc - eng, LU).
Output: analysis/audio_analysis.csv
"""
import csv, glob, json, os, re, subprocess, sys
from multiprocessing import Pool
import numpy as np

from paths import STAGE_A, ANALYSIS, FF, FP


def read(path):
    info = subprocess.run([FP, '-v', 'error', '-show_entries', 'stream=sample_rate,channels',
                           '-of', 'json', path], capture_output=True, text=True)
    st = json.loads(info.stdout)['streams'][0]; sr, ch = int(st['sample_rate']), int(st['channels'])
    raw = subprocess.run([FF, '-v', 'error', '-i', path, '-f', 's16le', '-acodec', 'pcm_s16le', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.int16).reshape(-1, ch).astype(np.float64), sr, ch


def ebur128(path):
    """Integrated loudness, LRA, true peak of the DC-removed signal (ffmpeg ebur128)."""
    p = subprocess.run([FF, '-hide_banner', '-nostats', '-i', path, '-af', 'highpass=f=10:p=2,ebur128=peak=true:framelog=quiet',
                        '-f', 'null', '-'], capture_output=True, text=True).stderr
    g = lambda pat: (lambda m: float(m.group(1)) if m else float('nan'))(re.search(pat, p.split('Summary:')[-1]))
    return g(r'I:\s+(-?[\d.]+|-inf) LUFS'), g(r'LRA:\s+([\d.]+) LU'), g(r'Peak:\s+(-?[\d.]+|-inf) dBFS')


def db(x):
    return 20 * np.log10(max(x, 1e-9) / 32768)


def analyse(path):
    lang, name = path.split('/audio/')[1].split('/', 1); name = name[:-5]
    x, sr, ch = read(path)
    n = len(x); dur = n / sr
    r = dict(lang=lang, name=name, sr=sr, ch=ch, dur=round(dur, 3))
    if n == 0:
        return r
    mean = x.mean(0)
    r['dc'] = round(float(np.abs(mean).max()), 1); r['dc_dbfs'] = round(db(float(np.abs(mean).max())), 1)
    win = int(sr * 0.5)
    if n >= 2 * win:
        wm = x[: n // win * win].reshape(-1, win, ch).mean(1)
        r['dc_drift'] = round(float((wm.max(0) - wm.min(0)).max()), 1)
    peak = float(np.abs(x).max()); r['peak_dbfs'] = round(db(peak), 2)
    clip = int((np.abs(x) >= 32767).sum())
    hot = np.abs(x) >= 32768 * 10 ** (-0.1 / 20)
    runs = 0
    for c in range(ch):
        h = hot[:, c].astype(np.int8); eq = np.r_[False, (np.diff(x[:, c]) == 0) & h[1:].astype(bool)]
        runs += int((np.convolve(eq, [1, 1], 'same') >= 2).sum())
    r['clipped'] = clip + runs
    # DC-removed mono mix for silence/pops
    m = (x - mean).mean(1)
    # subsonic wander (< ~9 Hz: 50 ms moving average of the DC-removed signal), RMS in dBFS
    k50 = int(sr * 0.05)
    if n > 4 * k50:
        c50 = np.cumsum(np.r_[0, m]); r['sub9_dbfs'] = round(db(float(np.sqrt((((c50[k50:] - c50[:-k50]) / k50) ** 2).mean()))), 1)
    fw = int(sr * 0.02); nw = n // fw
    if nw:
        rms = np.sqrt((m[: nw * fw].reshape(nw, fw) ** 2).mean(1)); sil = 20 * np.log10(np.maximum(rms, 1e-9) / 32768) < -60
        lead = int(np.argmax(~sil)) if (~sil).any() else nw; trail = int(np.argmax(~sil[::-1])) if (~sil).any() else nw
        r['lead_sil'] = round(lead * 0.02, 2); r['trail_sil'] = round(trail * 0.02, 2); r['silent_share'] = round(float(sil.mean()), 3)
    # pops/clicks: a burst of high-frequency energy (1st difference) lasting <= 2 ms that stands > 15 dB above the
    # median level of BOTH the 20 ms before and the 20 ms after it (a sound onset stays loud afterwards)
    f1 = max(1, sr // 1000); d = np.diff(m); nf = len(d) // f1
    if nf > 60:
        e = (d[: nf * f1].reshape(nf, f1) ** 2).mean(1)
        from numpy.lib.stride_tricks import sliding_window_view as sw
        med = np.median(sw(np.pad(e, 20, mode='edge'), 18), 1)   # median of 18 frames starting at i-20+j
        before = med[: nf] ; after = med[23: 23 + nf] if len(med) >= 23 + nf else np.pad(med[23:], (0, 23 + nf - len(med)), mode='edge')
        amp = np.sqrt(e)
        cand = np.nonzero((e > 10 ** 1.5 * np.maximum(before, after)) & (amp > 32768 * 10 ** (-40 / 20)))[0]
        pops = []
        for i in cand:
            if not pops or i - pops[-1] > 5:
                pops.append(i)
        r['pops'] = len(pops); r['pop_times'] = ' '.join(f'{i * f1 / sr:.3f}' for i in pops[:8])
    I, LRA, TP = ebur128(path)
    r.update(lufs=I, lra=LRA, tp_dbtp=TP)
    return r


if __name__ == '__main__':
    files = sorted(glob.glob(f'{STAGE_A}/audio/*/**/*.flac', recursive=True))
    with Pool(6) as p:
        rows = p.map(analyse, files, chunksize=4)
    eng = {r['name']: r for r in rows if r['lang'] == 'eng'}
    for r in rows:
        e = eng.get(r['name'])
        if r['lang'] != 'eng' and e and np.isfinite(r.get('lufs', np.nan)) and np.isfinite(e.get('lufs', np.nan)):
            r['d_eng_lu'] = round(r['lufs'] - e['lufs'], 1)
    cols = ['lang', 'name', 'sr', 'ch', 'dur', 'lufs', 'lra', 'tp_dbtp', 'peak_dbfs', 'clipped', 'dc', 'dc_dbfs', 'dc_drift', 'sub9_dbfs',
            'lead_sil', 'trail_sil', 'silent_share', 'pops', 'pop_times', 'd_eng_lu']
    with open(f'{ANALYSIS}/audio_analysis.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} tracks -> analysis/audio_analysis.csv')
