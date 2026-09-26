#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Stage (b): edited masters in stage_b/ (layout: README.md).

  build_masters.py video [seq ...]    shared videos: the English picture after the confirmed edits
                                      (shared_video.frames/gains), FFV1 8-bit, native size and frame rate, BT.601 tags
  build_masters.py langvideo          the per-language videos (PER_LANG), from English footage
  build_masters.py audio [seq ...]    conformed audio per (sequence, language): 10 Hz high-pass (first), the edits
                                      (edit_model audio ops), one static gain reaching the loudness target
                                      (recipes/audio_gains.csv, measured after the edits). FLAC 24-bit,
                                      native rate and channels. Tracks identical to the English are not written.
  build_masters.py edits              edit lists <seq>.edits.json (format below)
  build_masters.py preview [seq ...]  baked per-language previews (the edit list applied to the picture, with the
                                      conformed audio and a subtitle track naming each op) -> stage_b/preview/
  build_masters.py check              verify everything (bit-exact videos, durations, timelines, keyframes) and write
                                      stage_b/manifest.json (checksums of every file and recipe, versions)
Output: stage_b/video/<seq>.mkv, <seq>.<lang>.mkv ('loc' = one file for fre + ger), audio/<seq>.<lang>.flac,
        edits/<seq>.edits.json

Edit list, format version 1 (next to each WebM video in the packs, as <name>.edits.json):
  {"format": "wz2100-sequence-edits", "version": 1,
   "video": {"duration": s, "frames": n, "fps": f},            guard: ignore the edits (with a warning) if the video differs
   "tracks": {<lang>: {                                        only languages that don't simply play the video straight
       "video": "<name>.<variant>.webm", "video_duration": s, "video_frames": n,   optional: a per-language video
       "timeline": [ops],                                      optional: picture ops, in seconds on that video's timeline
       "duration": s}}}                                        the timeline's total length (= this language's audio)
  ops: {"op": "play", "from": a, "to": b} | {"op": "hold", "duration": d} | {"op": "loop", "from": a, "to": b, "duration": d}
       | {"op": "black", "duration": d} | {"op": "hold_fade", "duration": d}
  play seeks when "from" isn't where the previous op ended.
  loop plays [from, to) repeatedly for "duration" (it may end mid-cycle).
  hold repeats the last frame shown, and hold_fade holds it while fading linearly to black.
  The subtitle clock is the video position during play,
  and pauses during hold, black, hold_fade and the repeats of a loop.
"""
import csv, hashlib, json, os, subprocess, sys
import numpy as np

from paths import ROOT, STAGE_A, STAGE_B, RECIPES, FF
B = STAGE_B
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared_video

FS = 320 * 240 * 3 // 2           # the usual frame size (res_* clips are 192x168: use size())
inv = {(r['lang'], r['name']): r for r in csv.DictReader(open(f'{RECIPES}/inventory.csv'))}
TAGS = ['-vf', 'setparams=range=tv:colorspace=smpte170m:color_primaries=smpte170m:color_trc=smpte170m']
# -fflags/-flags +bitexact: no random IDs, dates or FFmpeg version in the file, so a rebuild gives the same bytes
FFV1 = ['-c:v', 'ffv1', '-level', '3', '-slices', '4', '-slicecrc', '1', '-g', '1', '-fflags', '+bitexact', '-flags', '+bitexact']


def size(clip):
    r = inv[('eng', clip)]
    return int(r['width']), int(r['height'])


# Per-language videos, from English footage.
# Parts: (clip, first, last, step[, repeat[, {'fadein': n}]]) | ('hold', n) | ('shared', seq).
# 'fadein' ramps the part's first n output frames up from black.
# 'tails': the edit-list op per language when its audio outlasts the video (default: hold).
# The parts of cam26afm, cam2_di and c3_d1_b match every localized frame to English footage.
# c3_d1_b follows the German structure and serves French too
# (the French edit list ends at 7.28 s, before the German fade).
PER_LANG = {
    'cam2/cam2_2n2.fre': dict(fps=25, parts=[('cam1/cam1out4', 406, 655, 1)]),
    'cam2/cam2_2n2.ger': dict(fps=25, parts=[('cam1/cam1out4', 426, 688, 1), ('hold', 1)]),
    'cam1/sub1_3.ger': dict(fps=12.5, parts=[('shared', 'cam1/sub1_3'), ('hold', 23), ('cam1/sub1_2', 131, 213, 1)]),
    'cam2/cam26afm.ger': dict(fps=25, parts=[
        ('cam1/sub17fmv', 806, 1199, 1), ('cam2/cam26afm', 79, 89, 1, 2), ('hold', 2), ('cam2/cam26afm', 90, 336, 1, 2),
        ('cam2/cam26afm', 338, 374, 1, 2), ('hold', 2), ('cam2/cam26afm', 375, 382, 1, 2), ('hold', 6),
        ('cam2/cam26afm', 383, 386, 1, 2), ('cam2/cam26afm', 391, 435, 1, 2), ('hold', 2), ('cam2/cam26afm', 436, 437, 1, 2),
        # the German fades the c003 atom up from black over 2.08 s (not the English cam26afm fade-in)
        ('cam3/c003', 176, 241, 1, 2, {'fadein': 52}),
        ('cam3/c003', 142, 149, 1, 2), ('cam3/c003', 656, 656, 1, 2), ('cam3/c003', 101, 123, 1, 2)]),
    # the German plays cam2_di with small holds and skips, then a stock atom from English footage
    'cam2/cam2_di.ger': dict(fps=12.5, parts=[
        ('cam1/sub1_2', 0, 1, 1), ('cam2/cam2_di', 15, 27, 1), ('hold', 1), ('cam2/cam2_di', 30, 84, 1), ('hold', 1),
        ('cam2/cam2_di', 85, 110, 1), ('hold', 1), ('cam2/cam2_di', 111, 115, 1), ('cam2/cam2_di', 118, 142, 1), ('hold', 1),
        ('cam2/cam2_di', 143, 150, 1), ('cam2/cam2_di', 158, 161, 1), ('hold', 1), ('cam2/cam2_di', 162, 162, 1), ('hold', 5),
        ('cam2/cam2_di', 163, 165, 1), ('cam3/c003', 655, 672, 1), ('cam3/c003', 598, 598, 1), ('cam3/c003', 624, 672, 1),
        ('cam3/c003', 598, 598, 1), ('cam1/sub1_2', 274, 304, 1), ('cam2/cam26pl1', 101, 195, 2)]),
    'cam2/cam2ca.ger': dict(fps=12.5, parts=[
        ('cam2/cam2ca', 25, 124, 1), ('cam2/cam2ca', 157, 158, 1), ('cam3/c003', 655, 672, 1), ('cam3/c003', 598, 598, 1),
        ('cam1/sub1_2', 274, 315, 1), ('cam2/cam26pl1', 122, 136, 2), ('cam1/sub1_2', 274, 304, 1), ('cam2/cam26pl1', 301, 339, 2),
        ('cam1/sub1_2', 151, 193, 1)]),
    # English to its end, a hold to German 13.68 s, then the German's stock atom
    # (simplified: the German's back-and-forth over English 8.76-9.24 s is left out)
    'transport.ger': dict(fps=25, parts=[('transport', 0, 250, 1), ('hold', 91), ('cam1/sub1_2', 131, 203, 1, 2)]),
    # ends where the picture ends (6.96 s). The tails are edit-list ops: German fades to black, French holds 0.32 s.
    'cam3/c3_d1_b.loc': dict(fps=25, parts=[('cam3/c3_d1_a', 26, 99, 1), ('cam3/c3_d1_b', 0, 99, 1)],
                             tails={'ger': 'hold_fade', 'fre': 'hold'}),
}


def per_lang_check():
    rows = {r['sequence'] for r in csv.DictReader(open(f'{RECIPES}/sequence_edits.csv'))
            if r['op'] == 'per_language_video' and r['status'].startswith('confirmed')}
    have = {k.rsplit('.', 1)[0] for k in PER_LANG}
    if rows != have:
        raise SystemExit(f'PER_LANG out of sync with the CSV: missing {sorted(rows - have)}, extra {sorted(have - rows)}')


def lang_frames(key):
    out = []
    for p in PER_LANG[key]['parts']:
        if p[0] == 'hold':
            out += [out[-1]] * p[1]
        elif p[0] == 'shared':
            out += list(shared_frames(p[1]))
        else:
            clip, a, b, st = p[:4]; rep = p[4] if len(p) > 4 else 1; opt = p[5] if len(p) > 5 else {}
            X = eng_frames(clip); part = []
            for k in range(a, b + 1, st):
                part += [X[k]] * rep
            n = opt.get('fadein', 0)
            part = [fade(f, (i + 1) / n) if i < n else f for i, f in enumerate(part)]
            out += part
    return out


def build_langvideo():
    per_lang_check()
    for key, spec in PER_LANG.items():
        seq = key.rsplit('.', 1)[0]
        frames = lang_frames(key)
        out = f'{B}/video/{key}.mkv'
        write_video(frames, spec['fps'], out, size(seq))
        print(f'{key}: {len(frames)} frames @{spec["fps"]:g} = {len(frames) / spec["fps"]:.2f}s -> {os.path.relpath(out, ROOT)}', flush=True)


# ---------------- audio ----------------
SR = 22050
ROOM_XF = 0.02


def read_hp(lang, seq):
    """Decoded audio after the 10 Hz high-pass, float64 (samples x channels, full scale = 1.0)."""
    r = inv[(lang, seq)]; ch = int(r['ach']); sr = int(r['arate'])
    raw = subprocess.run([FF, '-v', 'error', '-i', f'{STAGE_A}/audio/{lang}/{seq}.flac', '-af', 'highpass=f=10:p=2',
                          '-f', 'f64le', '-acodec', 'pcm_f64le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float64).reshape(-1, ch).copy(), sr


def _ramp(n):
    return np.linspace(0, 1, n, endpoint=False)[:, None] if n else np.zeros((0, 1))


def _xfade_join(a, b, n):
    """a then b, overlapping n samples with a linear crossfade."""
    n = min(n, len(a), len(b))
    if n == 0:
        return np.vstack([a, b])
    r = _ramp(n)
    return np.vstack([a[:-n], a[-n:] * (1 - r) + b[:n] * r, b[n:]])


def room_tone(x, sr, ra, rb, dur):
    """dur seconds of room tone: the segment [ra, rb) looped with ROOM_XF crossfades (its first and last ROOM_XF
    seconds are crossfaded with the audio around the insert point)."""
    seg = x[int(ra * sr):int(rb * sr)]; n = int(round(dur * sr)); xf = int(ROOM_XF * sr)
    out = seg.copy()
    while len(out) < n:
        out = _xfade_join(out, seg, xf)
    return out[:n]


def conform(x, sr, ops):
    """Apply the edit_model audio ops (times in the track's original seconds)."""
    for o in ops:
        if o[0] == 'trim_head':
            x = x[o[1]:]
    for o in ops:
        if o[0] == 'trim_tail':
            n = int(round(o[1] * sr)); f = int(round(o[2] * sr)); x = x[:n].copy()
            x[-f:] *= (1 - _ramp(f))
    # cuts and inserts at original positions, applied back to front so earlier positions stay valid
    mids = sorted([o for o in ops if o[0] in ('cut', 'insert')], key=lambda o: -o[1])
    for o in mids:
        if o[0] == 'cut':
            a, b, xf = int(round(o[1] * sr)), int(round(o[2] * sr)), int(round(o[3] * sr))
            x = _xfade_join(x[:a + xf // 2], x[b - xf // 2:], xf)     # removes exactly b - a samples
        else:
            t = int(round(o[1] * sr)); xf = int(ROOM_XF * sr)
            fill = room_tone(x, sr, o[3][0], o[3][1], o[2]) if o[3] else np.zeros((int(round(o[2] * sr)), x.shape[1]))
            x = _xfade_join(_xfade_join(x[:t + xf], fill, xf), x[t - xf:], xf)   # inserts exactly o[2] seconds
    for o in ops:
        if o[0] == 'pad_head':
            x = np.vstack([np.zeros((int(round(o[1] * sr)), x.shape[1])), x])
    for o in ops:
        if o[0] == 'pad_tail':
            x = np.vstack([x, np.zeros((int(round(o[1] * sr)), x.shape[1]))])
        elif o[0] == 'pad_to':
            n = int(round(o[1] * sr))
            if n > len(x):
                x = np.vstack([x, np.zeros((n - len(x), x.shape[1]))])
    return x


def write_flac24(x, sr, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    y = np.clip(x, -1, 1 - 2 ** -23)
    p = subprocess.run([FF, '-v', 'error', '-y', '-f', 'f64le', '-ar', str(sr), '-ac', str(x.shape[1]), '-i', '-',
                        '-c:a', 'flac', '-sample_fmt', 's32', '-bits_per_raw_sample', '24', out], input=y.astype('<f8').tobytes())
    if p.returncode:
        raise SystemExit(f'ffmpeg failed for {out}')


def loudness(x, sr):
    """Integrated loudness (LUFS) and true peak (dBTP) of a float signal (ffmpeg ebur128)."""
    import re
    p = subprocess.run([FF, '-hide_banner', '-nostats', '-f', 'f64le', '-ar', str(sr), '-ac', str(x.shape[1]), '-i', '-',
                        '-af', 'ebur128=peak=true:framelog=quiet', '-f', 'null', '-'], input=x.astype('<f8').tobytes(),
                       capture_output=True).stderr.decode().split('Summary:')[-1]
    g = lambda pat: (lambda m: float(m.group(1)) if m else float('nan'))(re.search(pat, p))
    return g(r'I:\s+(-?[\d.]+) LUFS'), g(r'Peak:\s+(-?[\d.]+) dBFS')


def build_audio(seqs):
    """The gain brings the conformed track to its loudness target (recipes/audio_gains.csv 'target', measured after
    the edits), capped so the true peak stays <= -1 dBTP."""
    import edit_model
    targets = {(r['lang'], r['name']): float(r['target']) for r in csv.DictReader(open(f'{RECIPES}/audio_gains.csv')) if r.get('target')}
    report = []
    for seq in seqs:
        m = edit_model.model(seq)
        for lang in ('eng', 'fre', 'ger'):
            if (lang, seq) not in inv or not inv[(lang, seq)]['audio_samples']:
                continue
            t = m[lang]
            if lang != 'eng' and t.get('same_as'):
                report.append((seq, lang, 'same as eng', '', '', '')); continue
            x, sr = read_hp(lang, seq)
            y = conform(x, sr, t['audio'])
            g = 0.0; note = ''
            if (lang, seq) in targets:
                I, TP = loudness(y, sr)
                g = targets[(lang, seq)] - I
                if TP + g > -1.0:
                    note = f'gain capped {g:+.2f} -> {-1.0 - TP:+.2f} dB (true peak)'; g = -1.0 - TP
                y = y * 10 ** (g / 20)
            out = f'{B}/audio/{seq}.{lang}.flac'
            write_flac24(y, sr, out)
            report.append((seq, lang, f'{len(y) / sr:.3f}', f"{t['audio_len']:.3f}", f"{g:+.2f}", note))
        print(seq, flush=True)
    # merge with the rows of earlier builds (a partial build must not drop the other tracks)
    path = f'{B}/audio_build.csv'; rows = {}
    if os.path.exists(path):
        for r in csv.reader(open(path)):
            if r and r[0] != 'sequence':
                rows[(r[0], r[1])] = r
    for r in report:
        rows[(r[0], r[1])] = list(r)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['sequence', 'lang', 'duration_s', 'planned_s', 'gain_db', 'note']); w.writerows(sorted(rows.values()))
    bad = [r for r in report if r[3] and abs(float(r[2]) - float(r[3])) > 0.002]
    print(f'{len(report)} tracks; duration mismatches vs plan: {len(bad)}', *bad[:10], sep='\n  ')


def dump_edits(doc):
    """JSON with one line per op or small object (easy to read and edit by hand)."""
    c = lambda o: json.dumps(o, separators=(', ', ': '))
    lines = ['{', f'  "format": {c(doc["format"])}, "version": {doc["version"]},', f'  "video": {c(doc["video"])},', '  "tracks": {']
    tr = list(doc['tracks'].items())
    for i, (lang, e) in enumerate(tr):
        lines.append(f'    {c(lang)}: {{')
        items = [(k, v) for k, v in e.items() if k != 'timeline']
        body = [f'      {c(k)}: {c(v)}' for k, v in items]
        if 'timeline' in e:
            body.insert(0, '      "timeline": [\n' + ',\n'.join(f'        {c(o)}' for o in e['timeline']) + '\n      ]')
        lines.append(',\n'.join(body))
        lines.append('    }' + (',' if i < len(tr) - 1 else ''))
    lines += ['  }', '}']
    out = '\n'.join(lines) + '\n'
    assert json.loads(out) == json.loads(json.dumps(doc))
    return out


def build_edits(seqs):
    import edit_model
    r3 = lambda v: round(v + 0.0, 3)
    n_files = 0; langs = 0
    for seq in seqs:
        m = edit_model.model(seq); base = seq.rsplit('/', 1)[-1]
        tracks = {}
        for lang in ('fre', 'ger'):
            t = m.get(lang)
            if not t or t.get('same_as'):
                continue
            entry = {}
            if t['video']:
                key = t['video']; spec = PER_LANG[key]; nfr = len(lang_frames(key))
                entry.update(video=f"{base}.{key.rsplit('.', 1)[1]}.webm", video_duration=r3(nfr / spec['fps']), video_frames=nfr)
                vdur = nfr / spec['fps']
                if t['audio_len'] < vdur - 1e-6:                 # stop the picture where this language's audio ends
                    entry['timeline'] = [{'op': 'play', 'from': 0.0, 'to': r3(round(t['audio_len'] * spec['fps']) / spec['fps'])}]
                elif t['audio_len'] > vdur + 1e-6:               # the audio outlasts the video: the language's tail op
                    entry['timeline'] = [{'op': 'play', 'from': 0.0, 'to': r3(vdur)},
                                         {'op': spec.get('tails', {}).get(lang, 'hold'), 'duration': r3(t['audio_len'] - vdur)}]
            if t['picture']:
                entry['timeline'] = [{k: (r3(v) if isinstance(v, float) else v) for k, v in o.items()} for o in t['picture']]
            if entry:
                tl = entry.get('timeline')
                entry['duration'] = r3(sum((o['to'] - o['from']) if o['op'] == 'play' else o['duration'] for o in tl)) if tl \
                    else entry.get('video_duration', r3(m['shared_dur']))
                tracks[lang] = entry
        if not tracks:
            continue
        doc = {'format': 'wz2100-sequence-edits', 'version': 1,
               'video': {'duration': r3(m['shared_dur']), 'frames': m['shared_frames'], 'fps': m['fps']}, 'tracks': tracks}
        out = f'{B}/edits/{seq}.edits.json'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, 'w').write(dump_edits(doc)); n_files += 1; langs += len(tracks)
    print(f'{n_files} edit lists ({langs} language entries) -> {os.path.relpath(B, ROOT)}/edits/')


def apply_timeline(frames, fps, timeline):
    """Frames of the picture as the engine shows it, and the cues [(t0, t1, label)]."""
    out, cues = [], []
    black = frames[0].copy(); ny = len(black) * 2 // 3; black[:ny] = 16; black[ny:] = 128
    for o in timeline:
        t0 = len(out) / fps
        if o['op'] == 'play':
            a, b = int(round(o['from'] * fps)), int(round(o['to'] * fps))
            out += list(frames[a:b]); lab = f"play {o['from']:.2f}-{o['to']:.2f}s"
        elif o['op'] == 'loop':
            a, b = int(round(o['from'] * fps)), int(round(o['to'] * fps)); n = int(round(o['duration'] * fps))
            out += [frames[a + i % (b - a)] for i in range(n)]; lab = f"LOOP {o['from']:.2f}-{o['to']:.2f}s for {o['duration']:.2f}s"
        elif o['op'] == 'hold':
            n = int(round(o['duration'] * fps)); out += [out[-1]] * n; lab = f"HOLD {o['duration']:.2f}s"
        elif o['op'] == 'black':
            n = int(round(o['duration'] * fps)); out += [black] * n; lab = f"BLACK {o['duration']:.2f}s"
        elif o['op'] == 'hold_fade':
            n = int(round(o['duration'] * fps)); last = out[-1]
            out += [fade(last, 1 - (i + 1) / n) for i in range(n)]; lab = f"HOLD + FADE {o['duration']:.2f}s"
        cues.append((t0, len(out) / fps, lab))
    return out, cues


def build_preview(seqs):
    os.makedirs(f'{B}/preview', exist_ok=True); n = 0
    fmt = lambda t: f'00:{int(t // 60):02d}:{t % 60:06.3f}'.replace('.', ',')
    for seq in seqs:
        path = f'{B}/edits/{seq}.edits.json'
        if not os.path.exists(path):
            continue
        doc = json.load(open(path))
        for lang, e in doc['tracks'].items():
            if 'video' in e:
                key = f"{seq}.{e['video'].split('.')[-2]}"; fr = lang_frames(key); fps = PER_LANG[key]['fps']
            else:
                fr = list(shared_frames(seq)); fps = doc['video']['fps']
            tl = e.get('timeline') or [{'op': 'play', 'from': 0.0, 'to': len(fr) / fps}]
            frames, cues = apply_timeline(fr, fps, tl)
            out = f"{B}/preview/{seq.replace('/', '_')}.{lang}.mkv"; srt = out[:-4] + '.srt'
            open(srt, 'w').write(''.join(f'{i + 1}\n{fmt(a)} --> {fmt(b)}\n{lab}\n\n' for i, (a, b, lab) in enumerate(cues)))
            wh = size(seq)
            p = subprocess.run([FF, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', f'{wh[0]}x{wh[1]}', '-r', f'{fps:g}', '-i', '-',
                                '-i', f'{B}/audio/{seq}.{lang}.flac', '-i', srt, '-map', '0:v', '-map', '1:a', '-map', '2:s',
                                '-vf', f"scale={wh[0] * 2}:{wh[1] * 2}:flags=lanczos,drawtext=text='{lang.upper()} %{{pts\\:hms}}':x=8:y=8:fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.6",
                                '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast', '-c:a', 'aac', '-b:a', '128k', '-c:s', 'srt',
                                '-disposition:s:0', 'default', out], input=np.stack(frames).tobytes(), capture_output=True)
            os.remove(srt)
            if p.returncode:
                print(f'{seq} {lang}: FAILED {p.stderr.decode()[-200:]}')
            n += 1
    print(f'{n} previews -> {os.path.relpath(B, ROOT)}/preview/')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def decode(path, wh):
    raw = subprocess.run([FF, '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, wh[0] * wh[1] * 3 // 2)


def check(seqs):
    import edit_model
    problems = []; notes = []; files = {}
    # videos: bit-exact against their recipes
    for seq in seqs:
        got = decode(f'{B}/video/{seq}.mkv', size(seq)); exp = np.stack(list(shared_frames(seq)))
        if got.shape != exp.shape or not (got == exp).all():
            problems.append(f'{seq}: shared video differs from its recipe')
    for key in PER_LANG:
        got = decode(f'{B}/video/{key}.mkv', size(key.rsplit('.', 1)[0])); exp = np.stack(lang_frames(key))
        if got.shape != exp.shape or not (got == exp).all():
            problems.append(f'{key}: per-language video differs from its recipe')
    # audio durations, edit-list timelines
    adur = {}
    for r in csv.DictReader(open(f'{B}/audio_build.csv')):
        if r['duration_s'] != 'same as eng':
            adur[(r['sequence'], r['lang'])] = float(r['duration_s'])
            if abs(float(r['duration_s']) - float(r['planned_s'])) > 0.002:
                problems.append(f"{r['sequence']} {r['lang']}: audio {r['duration_s']}s vs planned {r['planned_s']}s")
    for (lang, seq), r in inv.items():
        if r['audio_samples'] and (seq, lang) not in adur and not (lang != 'eng' and edit_model.model(seq)[lang].get('same_as')):
            problems.append(f'{seq} {lang}: no conformed audio in audio_build.csv')
    kf_from_edits = {}
    for seq in seqs:
        m = edit_model.model(seq); fps = m['fps']
        path = f'{B}/edits/{seq}.edits.json'
        doc = json.load(open(path)) if os.path.exists(path) else {'tracks': {}}
        if doc['tracks'] and (doc['video']['frames'] != m['shared_frames']):
            problems.append(f'{seq}: edit list guard {doc["video"]} vs shared video {m["shared_frames"]} frames')
        for lang in ('fre', 'ger'):
            if (lang, seq) not in inv:
                continue
            a = adur.get((seq, lang), adur.get((seq, 'eng')))
            e = doc['tracks'].get(lang)
            vid_len = e['video_duration'] if e and 'video' in e else m['shared_dur']
            if e:
                if a and a - e['duration'] > 0.5 / m['fps']:              # the audio would be cut off
                    problems.append(f'{seq} {lang}: timeline {e["duration"]}s shorter than the audio {a:.3f}s')
                elif a and e['duration'] - a > 0.5 / m['fps']:            # the picture outlasts the audio: fine
                    notes.append(f'{seq} {lang}: picture runs {e["duration"] - a:.2f}s past the audio')
                pos = 0.0; tl = e.get('timeline', [])
                for j, o in enumerate(tl):
                    if o['op'] in ('play', 'loop'):
                        if not (0 <= o['from'] < o['to'] <= vid_len + 1e-6):
                            problems.append(f'{seq} {lang}: {o} outside the video (0-{vid_len}s)')
                        if 'video' not in e and (o['op'] == 'loop' or abs(o['from'] - pos) > 1e-6):
                            kf_from_edits.setdefault(seq, set()).add(round(o['from'] * fps))
                        P = o['to'] - o['from']
                        if (o['op'] == 'loop' and 'video' not in e and j + 1 < len(tl) and tl[j + 1]['op'] == 'play'
                                and abs(o['duration'] / P - round(o['duration'] / P)) > 1e-6):
                            # resume after a partial cycle: a seek
                            kf_from_edits[seq].add(round(tl[j + 1]['from'] * fps))
                        pos = o['to'] if o['op'] == 'play' else (o['to'] if abs(o['duration'] / P - round(o['duration'] / P)) < 1e-6 else -1)
            elif a is not None and a > m['shared_dur'] + 0.085:
                problems.append(f'{seq} {lang}: audio {a:.3f}s runs past the shared video {m["shared_dur"]:.2f}s without an edit list')
    # keyframes: every jump target of the edit lists is in keyframes.json (or covered: < 0.2 s after a kept keyframe)
    kj = json.load(open(f'{RECIPES}/keyframes.json'))
    for seq, frs in kf_from_edits.items():
        fps = float(inv[('eng', seq)]['fps']); kept = [round(t * fps) for t in kj.get(seq, [0.0])]
        n = int(inv[('eng', seq)]['frames']) if seq not in shared_video.SPECS else len(shared_video.frames(seq, lambda c: int(inv[('eng', c)]['frames'])))
        for f in sorted(frs):
            if f >= n:
                continue                                   # resume at the very end = no seek
            if not any(0 <= f - k < 0.2 * fps for k in kept):
                problems.append(f'{seq}: jump target frame {f} ({f / fps:.2f}s) has no forced keyframe')
    # manifest
    for root, dirs, fs in os.walk(B):
        dirs[:] = [d for d in dirs if d != 'preview']      # previews are for checking only, not part of the stage
        for fn in sorted(fs):
            pth = os.path.join(root, fn)
            if fn != 'manifest.json' and not fn.startswith('.'):      # skips file-browser metadata, ex. .DS_Store
                files[os.path.relpath(pth, B)] = sha256(pth)
    recipes = {os.path.relpath(p, ROOT): sha256(p) for p in [f'{RECIPES}/sequence_edits.csv', f'{RECIPES}/sequence_edits_auto.csv',
               f'{RECIPES}/audio_gains.csv', f'{RECIPES}/keyframes.json', f'{RECIPES}/inventory.csv'] + [f'{ROOT}/tools/{t}' for t in
               ('build_masters.py', 'edit_model.py', 'shared_video.py', 'loudness_gains.py', 'keyframes.py',
                'paths.py')]}
    ver = subprocess.run([FF, '-version'], capture_output=True, text=True).stdout.split('\n')[0]
    json.dump({'stage': 'stage_b', 'created': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
               'ffmpeg': ver, 'source': 'stage_a', 'recipes': recipes, 'files': files,
               'checks': {'problems': problems, 'notes': notes}}, open(f'{B}/manifest.json', 'w'), indent=1)
    print(f'{len(files)} files in the manifest; problems: {len(problems)}', *problems[:30], sep='\n  ')
    print(f'notes: {len(notes)}', *notes, sep='\n  ')
    return problems


def eng_frames(clip, cache={}):
    if clip not in cache:
        if len(cache) > 6:
            cache.clear()
        raw = subprocess.run([FF, '-v', 'error', '-i', f'{STAGE_A}/video/eng/{clip}.mkv', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'],
                             capture_output=True, check=True).stdout
        w, h = size(clip)
        cache[clip] = np.frombuffer(raw, np.uint8).reshape(-1, w * h * 3 // 2)
    return cache[clip]


def fade(f, g):
    x = f.astype(np.float32); y = np.empty_like(x); ny = len(f) * 2 // 3
    y[:ny] = 16 + (x[:ny] - 16) * g; y[ny:] = 128 + (x[ny:] - 128) * g
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def write_video(frames, fps, out, wh=(320, 240)):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p = subprocess.Popen([FF, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', f'{wh[0]}x{wh[1]}', '-r', f'{fps:g}', '-i', '-',
                          *FFV1, *TAGS, '-an', out], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait():
        raise SystemExit(f'ffmpeg failed for {out}')


def shared_frames(seq):
    n_of = lambda c: len(eng_frames(c))
    src = shared_video.frames(seq, n_of); gs = shared_video.gains(seq, n_of)
    for (c, k), g in zip(src, gs):
        f = eng_frames(c)[k]
        yield f if g == 1.0 else fade(f, g)


def build_video(seqs):
    for seq in seqs:
        fps = float(inv[('eng', seq)]['fps'])
        out = f'{B}/video/{seq}.mkv'
        frames = list(shared_frames(seq))
        write_video(frames, fps, out, size(seq))
        print(f'{seq}: {len(frames)} frames @{fps:g} -> {os.path.relpath(out, ROOT)}', flush=True)


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    seqs = args or sorted({n for (l, n) in inv if l == 'eng' and inv[(l, n)]['frames']})
    if cmd == 'video':
        shared_video.check()
        build_video(seqs)
    elif cmd == 'langvideo':
        build_langvideo()
    elif cmd == 'audio':
        build_audio(seqs)
    elif cmd == 'edits':
        build_edits(seqs)
    elif cmd == 'preview':
        build_preview(seqs)
    elif cmd == 'check':
        sys.exit(1 if check(seqs) else 0)
    else:
        raise SystemExit(__doc__)
