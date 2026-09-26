#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Stage (d): the WebM encodes (VP9 + Opus) and the .wz pack of a tier (settings.TIERS: standard, enhanced, low).

  build_stage_d.py video <tier> [<video> ...]   VP9 2-pass of the tier's stage (c) masters, keyframes forced where the
                                                edit lists jump (recipes/keyframes.json)
                                                -> stage_d/<tier>/_video/<video>.webm (video only)
  build_stage_d.py audiogain <tier>             measures each track's loudness change through Opus and writes the
                                                gains that undo it -> stage_d/<tier>/opus_gains.json
  build_stage_d.py mux <tier> [<video> ...]     Opus audio from the stage (b) FLAC masters, with the gains from
                                                stage_d/<tier>/opus_gains.json -> stage_d/<tier>/_audio/, then the
                                                video and audio tracks muxed with language tags, and the edit lists
                                                copied -> stage_d/<tier>/sequences/
  build_stage_d.py check <tier> [<video> ...]   streams, tags, frame counts, keyframes, audio length, offset and
                                                loudness against the masters, full decode -> stage_d/<tier>/check.txt
                                                (exit code 1 on a problem)
  build_stage_d.py pack <tier>                  stage_d/<tier>-quality-sequences.wz (zip, stored, with README.txt and
                                                the license files) and stage_d/<tier>-quality-encoding.txt
Run video, audiogain, mux, check and pack in this order (pack needs a clean check of every video).
Files per sequence <seq> (ex. cam1/c001):
  sequences/<seq>.webm            the shared video, with audio tracks eng (default) and every fre/ger track that has
                                  a FLAC master and no per-language video (tracks identical to English aren't in
                                  stage (b), and the game falls back to the English one)
  sequences/<seq>.<lang>.webm     a per-language video (fre, ger, or loc for the languages whose edit list names it),
                                  with that language's audio
  sequences/<seq>.edits.json      the edit list, from stage (b), where the sequence has one
Encodes run in parallel, one thread each.
Needs FFmpeg with libvpx, libopus, zimg and soxr, and the Python packages in tools/requirements.txt (NumPy).
"""
import datetime, glob, hashlib, json, os, shutil, subprocess, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import settings as S
from paths import ROOT, STAGE_B, STAGE_C, STAGE_D, RECIPES, FF, FP

JOBS = os.cpu_count() or 1
KEYS = json.load(open(f'{RECIPES}/keyframes.json'))


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"{' '.join(cmd[:6])}... returned {p.returncode}: {p.stderr[-500:]}")
    return p.stdout


def probe(path, entries, sel='v'):
    return json.loads(run([FP, '-v', 'error', '-select_streams', sel, '-show_entries', entries, '-of', 'json', path]))


def split(name):
    """'cam2/cam2_2n2.fre' -> ('cam2/cam2_2n2', 'fre'), 'cam1/c001' -> ('cam1/c001', None)."""
    base, _, ext = os.path.basename(name).partition('.')
    return os.path.join(os.path.dirname(name), base), (ext or None)


def videos():
    """The video names to encode: '<seq>' (shared) and '<seq>.<lang|loc>' (per-language), without PACK_EXCLUDE."""
    d = f'{STAGE_B}/video'
    names = (os.path.relpath(os.path.join(r, f), d)[:-4] for r, _, fs in os.walk(d) for f in fs if f.endswith('.mkv'))
    return sorted(n for n in names if split(n)[0] not in S.PACK_EXCLUDE)


def edit_list(seq):
    p = f'{STAGE_B}/edits/{seq}.edits.json'
    return json.load(open(p)) if os.path.exists(p) else None


def seeks(timeline):
    """Positions the timeline jumps to (a 'play' that doesn't continue from the previous play's end)."""
    out, pos = [], 0.0
    for op in timeline:
        if op['op'] == 'play':
            if abs(op['from'] - pos) > 1e-6:
                out.append(op['from'])
            pos = op['to']
    return out


def audio_langs(name):
    """The audio tracks (lang, FLAC path) of an output video, in track order."""
    seq, v = split(name); tracks = (edit_list(seq) or {}).get('tracks', {})
    webm = os.path.basename(name) + '.webm'
    if v is None:
        langs = ['eng'] + [l for l in ('fre', 'ger') if 'video' not in tracks.get(l, {})]
    else:
        langs = [l for l in ('eng', 'fre', 'ger') if tracks.get(l, {}).get('video') == webm]
        assert langs, f'{name}: no edit-list track names {webm}'
    return [(l, f'{STAGE_B}/audio/{seq}.{l}.flac') for l in langs if os.path.exists(f'{STAGE_B}/audio/{seq}.{l}.flac')]


def keyframes(name):
    seq, v = split(name)
    if v is None:
        return sorted(set([0.0] + KEYS[seq]))
    for l, t in (edit_list(seq) or {}).get('tracks', {}).items():    # per-language videos play straight through
        if t.get('video') == os.path.basename(name) + '.webm':
            assert not seeks(t.get('timeline', [])), f'{name}: the {l} timeline seeks inside a per-language video'
    return [0.0]


def master(tier, name):
    return f"{STAGE_C}/{S.TIERS[tier]['src']}/video/{name}.mkv"


# ---------------------------------------------------------------------------------------------------------------- video
def encode_one(tier, name):
    t = S.TIERS[tier]; src = master(tier, name); out = f'{STAGE_D}/{tier}/_video/{name}.webm'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log = f'{STAGE_D}/{tier}/_video/{name}.pass'; kf = ','.join(f'{x:.3f}' for x in keyframes(name))
    vp9 = list(S.VP9)
    if name in t['rc']:
        i = vp9.index('-b:v'); vp9[i:i + 2] = t['rc'][name]
    for ps in (1, 2):
        run([FF, '-v', 'error', '-y', '-i', src, '-vf', S.VF, *vp9, '-crf', str(t['crf']), '-threads', '1',
             '-force_key_frames', kf, '-pass', str(ps), '-passlogfile', log, '-an', *S.BITEXACT]
            + (['-f', 'null', os.devnull] if ps == 1 else [out + '.tmp.webm']))
    os.replace(out + '.tmp.webm', out)
    for f in glob.glob(log + '*'):
        os.remove(f)
    return f'{name}: {os.path.getsize(out) / 1e6:.2f} MB'


def encode(tier, names):
    missing = [n for n in names if not os.path.exists(master(tier, n))]
    if missing:
        sys.exit(f"stage (c) masters missing (build_stage_c.py {S.TIERS[tier]['src']}): {missing[:5]}")
    names = sorted(names, key=lambda n: -os.path.getsize(master(tier, n)))       # longest first
    with ThreadPoolExecutor(JOBS) as ex:
        futs = {n: ex.submit(encode_one, tier, n) for n in names}
        for k, (n, f) in enumerate(futs.items(), 1):
            try:
                print(f'[{k}/{len(names)}] {f.result()}', flush=True)
            except Exception as e:
                print(f'[{k}/{len(names)}] {n}: FAILED {e}', flush=True)


# ------------------------------------------------------------------------------------------------------------------ mux
def channels(path):
    return probe(path, 'stream=channels', 'a')['streams'][0]['channels']


def audio_one(tier, seq_lang, gain):
    """One Opus track per ffmpeg process (several soxr resamplers in one ffmpeg 8.1 process can crash it).
    gain None: no gain, into _audio_raw/ (for audiogain)."""
    t = S.TIERS[tier]; fl = f'{STAGE_B}/audio/{seq_lang}.flac'
    raw = gain is None; g = 0.0 if raw else gain.get(seq_lang, 0.0)
    out = f"{STAGE_D}/{tier}/{'_audio_raw' if raw else '_audio'}/{seq_lang}.webm"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    br = t['mono'] if channels(fl) == 1 else t['stereo']
    af = (f'volume={g:+.2f}dB:precision=double,' if g else '') + S.AF
    run([FF, '-v', 'error', '-y', '-i', fl, '-af', af, *S.OPUS, '-b:a', f'{br}k', *S.BITEXACT, out + '.tmp.webm'])
    os.replace(out + '.tmp.webm', out)
    return out


def gains_path(tier):
    return f'{STAGE_D}/{tier}/opus_gains.json'


def audiogain(tier):
    """Encodes every track without gain, measures the loudness change through Opus
    (decoded vs the FLAC master, EBU R128 integrated), and stores the gain that undoes it.
    A second pass corrects the residual, and keeps the decoded peak at or below PEAK_LIMIT."""
    need = sorted({f'{split(n)[0]}.{l}' for n in videos() for l, _ in audio_langs(n)})
    def one(x):
        o = audio_one(tier, x, None)
        a, b = lufs(['-i', o]), lufs(['-i', f'{STAGE_B}/audio/{x}.flac'])
        return x, (None if a is None or b is None else round(b - a, 2))
    with ThreadPoolExecutor(JOBS) as ex:
        res = dict(ex.map(one, need))
    g = {k: v for k, v in res.items() if v}
    def refine(x):
        o = audio_one(tier, x, g)
        a, b = lufs(['-i', o]), lufs(['-i', f'{STAGE_B}/audio/{x}.flac'])
        r = 0.0 if a is None or b is None else b - a
        new = g.get(x, 0.0) + r if abs(r) > 0.05 else g.get(x, 0.0)
        # the decoder output is clipped to full scale
        peak = float(20 * np.log10(np.abs(pcm(['-i', o], channels(f'{STAGE_B}/audio/{x}.flac'))).max() + 1e-20))
        if peak > S.PEAK_LIMIT:
            new -= peak - S.PEAK_LIMIT
        return x, round(new, 2)
    with ThreadPoolExecutor(JOBS) as ex:
        g = {x: v for x, v in ex.map(refine, need) if v}
    os.makedirs(os.path.dirname(gains_path(tier)), exist_ok=True)
    json.dump(dict(tier=tier, note='dB added before the Opus encode = master LUFS - raw Opus-decoded LUFS', gains=g),
              open(gains_path(tier), 'w'), indent=1, sort_keys=True)
    v = sorted(g.values())
    print(f'{len(need)} tracks, {len(g)} with a gain: min {v[0]:+.2f} median {v[len(v) // 2]:+.2f} max {v[-1]:+.2f} dB, '
          f'{sum(1 for x in res.values() if x is None)} silent')


def mux_one(tier, name):
    out = f'{STAGE_D}/{tier}/sequences/{name}.webm'; os.makedirs(os.path.dirname(out), exist_ok=True)
    seq, _ = split(name); tracks = audio_langs(name)
    cmd = [FF, '-v', 'error', '-y', '-i', f'{STAGE_D}/{tier}/_video/{name}.webm']
    for l, _ in tracks:
        cmd += ['-i', f'{STAGE_D}/{tier}/_audio/{seq}.{l}.webm']
    cmd += ['-map', '0:v']
    for k in range(len(tracks)):
        cmd += ['-map', f'{k + 1}:a']
    cmd += ['-c', 'copy']
    for k, (l, _) in enumerate(tracks):
        cmd += [f'-metadata:s:a:{k}', f'language={l}', f'-metadata:s:a:{k}', f'title={S.LANGS[l]}',
                f'-disposition:a:{k}', 'default' if k == 0 else '0']
    run(cmd + ['-metadata:s:v:0', 'language=und', '-disposition:v:0', 'default', *S.BITEXACT, out + '.tmp.webm'])
    os.replace(out + '.tmp.webm', out)
    return f"{name}: {' '.join(l for l, _ in tracks) or '(no audio)'}"


def mux(tier, names):
    names = names or videos()
    need = sorted({f'{split(n)[0]}.{l}' for n in names for l, _ in audio_langs(n)})
    if not os.path.exists(gains_path(tier)):
        sys.exit(f'{gains_path(tier)} is missing: run "build_stage_d.py audiogain {tier}" first')
    gain = json.load(open(gains_path(tier)))['gains']
    with ThreadPoolExecutor(JOBS) as ex:
        list(ex.map(lambda x: audio_one(tier, x, gain), need))
    print(f'{len(need)} audio tracks encoded', flush=True)
    with ThreadPoolExecutor(JOBS) as ex:
        for k, r in enumerate(ex.map(lambda n: mux_one(tier, n), names), 1):
            print(f'[{k}/{len(names)}] {r}', flush=True)
    seqs = {split(n)[0] for n in names}; n = 0
    for p in glob.glob(f'{STAGE_B}/edits/**/*.edits.json', recursive=True):
        rel = os.path.relpath(p, f'{STAGE_B}/edits')
        if rel[:-len('.edits.json')] in seqs:
            dst = f'{STAGE_D}/{tier}/sequences/{rel}'
            os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copyfile(p, dst); n += 1
    print(f'{n} edit lists copied')


# ---------------------------------------------------------------------------------------------------------------- check
def pcm(args, ch=1):
    raw = subprocess.run([FF, '-v', 'error', *args, '-ac', str(ch), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def lufs(args):
    """EBU R128 integrated loudness (LUFS, from loudnorm's measurement) of an ffmpeg input, None for silence."""
    err = subprocess.run([FF, '-nostats', '-v', 'info', *args, '-af', 'loudnorm=print_format=json', '-f', 'null', '-'],
                         capture_output=True, text=True).stderr
    try:
        v = float(json.loads(err[err.rindex('{'):err.rindex('}') + 1])['input_i'])
    except (ValueError, KeyError):
        return None
    return None if not np.isfinite(v) or v <= -69 else v


def lag(a, b, maxlag=960):
    """Offset of a against b in samples (positive: a is late), from the first 10 s, None for near-silence."""
    n = min(len(a), len(b), 480000); a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    if np.sqrt((b ** 2).mean()) < 1e-4:
        return None
    m = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(a, m) * np.conj(np.fft.rfft(b, m)), m)
    xc = np.concatenate([xc[-maxlag:], xc[:maxlag + 1]])
    return int(np.argmax(xc)) - maxlag


def check_one(tier, name):
    f = f'{STAGE_D}/{tier}/sequences/{name}.webm'; src = master(tier, name); probs = []
    s = probe(f, 'stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,color_range,color_space,color_primaries,'
                 'color_transfer')['streams'][0]
    c = probe(src, 'stream=width,height,r_frame_rate')['streams'][0]
    want = dict(codec_name='vp9', profile='Profile 0', width=c['width'], height=c['height'], pix_fmt='yuv420p',
                r_frame_rate=c['r_frame_rate'], color_range='tv', color_space='smpte170m', color_primaries='smpte170m',
                color_transfer='smpte170m')
    probs += [f'{k}={s.get(k)} (want {v})' for k, v in want.items() if s.get(k) != v]
    # frames and keyframes
    pk = probe(f, 'packet=pts_time,flags')['packets']
    nsrc = int(json.loads(run([FP, '-v', 'error', '-select_streams', 'v', '-count_packets', '-show_entries',
                               'stream=nb_read_packets', '-of', 'json', src]))['streams'][0]['nb_read_packets'])
    if len(pk) != nsrc:
        probs.append(f'{len(pk)} video frames (stage (c): {nsrc})')
    num, den = c['r_frame_rate'].split('/'); fps = int(num) / int(den)
    kts = [float(p['pts_time']) for p in pk if 'K' in p['flags']]
    miss = [x for x in keyframes(name) if not any(abs(k - x) < 0.5 / fps for k in kts)]
    if miss:
        probs.append(f'no keyframe at {miss}')
    # audio: track order, tags and default, then per track the length, offset and level against the FLAC master
    a = probe(f, 'stream=index,codec_name,sample_rate,channels:stream_tags=language:stream_disposition=default', 'a')['streams']
    exp = audio_langs(name)
    got = [(x.get('tags', {}).get('language'), x['disposition']['default']) for x in a]
    if got != [(l, int(k == 0)) for k, (l, _) in enumerate(exp)]:
        probs.append(f'audio tracks {got} (want {[l for l, _ in exp]}, the first one default)')
    rows = []
    for k, (l, fl) in enumerate(exp):
        if k >= len(a):
            break
        ch = channels(fl)
        if a[k]['codec_name'] != 'opus' or a[k]['sample_rate'] != '48000' or a[k]['channels'] != ch:
            probs.append(f'{l}: {a[k]}')
        dec = pcm(['-i', f, '-map', f'0:a:{k}'], ch); ref = pcm(['-i', fl, '-af', S.AF], ch)
        n_d, n_r = len(dec) // ch, len(ref) // ch
        lg = lag(dec[::ch], ref[::ch])
        l_d, l_r = lufs(['-i', f, '-map', f'0:a:{k}']), lufs(['-i', fl])
        lvl = None if l_d is None or l_r is None else l_d - l_r
        peak = 20 * np.log10(np.abs(dec).max() + 1e-20)
        if abs(n_d - n_r) > 48:
            probs.append(f'{l}: {n_d} samples decoded, master {n_r} ({(n_d - n_r) / 48:.1f} ms)')
        if lg is not None and abs(lg) > 1:             # +-1 sample: the cross-correlation's resolution
            probs.append(f'{l}: audio offset {lg} samples ({lg / 48:.2f} ms)')
        if lvl is not None and abs(lvl) > 0.2:
            probs.append(f'{l}: loudness {lvl:+.2f} LU vs master')
        if peak > 0:
            probs.append(f'{l}: decoded peak {peak:+.2f} dBFS')
        rows.append(f"{l} {n_d / 48000:.3f}s lag {lg} loudness {'-' if lvl is None else f'{lvl:+.2f}'} LU peak {peak:.1f} dBFS")
    err = subprocess.run([FF, '-v', 'error', '-i', f, '-map', '0', '-f', 'null', '-'], capture_output=True, text=True).stderr.strip()
    if err:
        probs.append(f'decode errors: {err[:300]}')
    return name, os.path.getsize(f), len(pk), rows, probs


def check(tier, names):
    names = names or videos(); lines, bad, total = [], 0, 0
    with ThreadPoolExecutor(JOBS) as ex:
        for name, size, nf, rows, probs in ex.map(lambda n: check_one(tier, n), names):
            total += size; bad += bool(probs)
            lines.append(f"{'PROBLEM' if probs else 'ok':7s} {name}: {size / 1e6:.2f} MB, {nf} frames; " + '; '.join(rows))
            lines += [f'        ! {p}' for p in probs]
    d = f'{STAGE_D}/{tier}/sequences'
    lists = sorted(os.path.relpath(p, d) for p in glob.glob(f'{d}/**/*.edits.json', recursive=True))
    for s in lists:                                          # the videos the edit lists name must exist
        for l, t in json.load(open(f'{d}/{s}'))['tracks'].items():
            v = t.get('video')
            if v and not os.path.exists(f'{d}/{os.path.dirname(s)}/{v}'):
                lines.append(f'PROBLEM {s}: {l} names {v}, which is missing'); bad += 1
    head = f'{tier}: {len(names)} videos, {len(lists)} edit lists, {total / 1e6:.1f} MB, {bad} with problems'
    open(f'{STAGE_D}/{tier}/check.txt', 'w').write(head + '\n' + '\n'.join(lines) + '\n')
    print(head); print('\n'.join(l for l in lines if not l.startswith('ok')))
    sys.exit(1 if bad else 0)


# ----------------------------------------------------------------------------------------------------------------- pack
README = """Warzone 2100 campaign videos - {Tier} quality
{rule}

WebM versions of the Warzone 2100 campaign videos, with English, French and
German audio.

Installation
------------
Rename the downloaded .wz file to "sequences.wz" and put it in your Warzone 2100
configuration directory (for example ~/.local/share/warzone2100-<version>/ on
Linux), or next to the game's other .wz data files. The game detects it
automatically.

These videos need a Warzone 2100 version that plays WebM videos with edit lists.
The game picks the audio track for its language. Where a video has no French or
German track, it plays the English one (those releases used the English audio).

Contents
--------
sequences/<name>.webm
    The video, with its English (default), French and German audio tracks.
sequences/<name>.<lang>.webm
    A separate video for a language whose picture differs ("loc" is shared by
    French and German).
sequences/<name>.edits.json
    The edit list: how the video's picture is played (holds, loops, black) so
    that it fits each language's audio.

License
-------
See COPYING.README (the Warzone 2100 source and data license) and COPYING (the
GNU General Public License, version 2).
"""

ENCODING = """Warzone 2100 campaign videos - {Tier} quality: encoding details
{rule}

Pack: {pack}, {size} bytes
SHA-256: {sha}
Built {date}.

Sources
-------
The original English, French and German RPL files (Escape 130 video, 4-bit ADPCM
audio), decoded losslessly with FFmpeg ({ffmpeg}).
The English picture is the video master. The French and German audio tracks are
fitted to it with edit lists, and a few sequences have a separate per-language
video where the picture differs.

Video
-----
- 2x the original size (320x240 -> 640x480).
- Levels mapped from the decoder's full range to TV range, in 10-bit.
- {chain}
- VP9 profile 0, 8-bit 4:2:0 (error-diffusion dither), tagged BT.601 TV range.
- libvpx-vp9, 2-pass constant quality: -crf {crf} -b:v 0 -deadline good
  -cpu-used 2 -row-mt 1 -tile-columns 0 -auto-alt-ref 1 -lag-in-frames 25 -g 250.
- Keyframes forced at every position the edit lists jump to.{caps}

Audio
-----
- 10 Hz high-pass, then a static gain to a per-track loudness target (partial
  normalization).
- Resampled to 48 kHz with soxr.
- libopus VBR, {mono} kbps mono / {stereo} kbps stereo, CELT only
  (-application lowdelay), 12 kHz audio bandwidth, 20 ms frames.
- A small per-track gain before the encoder keeps each decoded track within
  0.15 LU of its master, with peaks at or below -0.1 dBFS.
"""

CHAINS_TEXT = {
    'standard': 'Scaled with spline36, then lightly debanded\n  (ffmpeg deband 1thr=0.010:2thr=0.012:3thr=0.012:range=8:blur=1).',
    'ml': 'Upscaled with Real-ESRGAN realesr-general-x4v3 (denoise strength 0.25)\n  to 4x, then area-downscaled to 2x. '
          'Videos made from 160x120 content were\n  upscaled from that resolution.',
}


def zip_entry(name, is_dir=False):
    """A zip entry with a fixed date and attributes, so the same files always give the same pack."""
    zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    zi.create_system = 3
    zi.external_attr = (0o40755 << 16) | 0x10 if is_dir else 0o100644 << 16
    zi.compress_type = zipfile.ZIP_STORED
    return zi


def pack(tier):
    t = S.TIERS[tier]; d = f'{STAGE_D}/{tier}'; zpath = f'{STAGE_D}/{tier}-quality-sequences.wz'
    chk = open(f'{d}/check.txt').readline()
    if ' 0 with problems' not in chk or not chk.startswith(f'{tier}: {len(videos())} videos'):
        sys.exit(f'run check on every video first (check.txt: {chk.strip()})')
    ffv = run([FF, '-version']).splitlines()[0].split(' Copyright')[0]
    Tier = tier.capitalize()
    title = f'Warzone 2100 campaign videos - {Tier} quality'
    readme = README.format(Tier=Tier, rule='=' * len(title))
    entries = sorted(os.path.relpath(p, d) for p in glob.glob(f'{d}/sequences/**/*', recursive=True))
    entries = [e for e in entries if split(e[len('sequences/'):].split('.edits.json')[0].removesuffix('.webm'))[0]
               not in S.PACK_EXCLUDE]
    with zipfile.ZipFile(zpath + '.tmp', 'w', zipfile.ZIP_STORED) as z:
        z.writestr(zip_entry('sequences/', True), b'')
        for rel in entries:
            if os.path.isdir(f'{d}/{rel}'):
                z.writestr(zip_entry(rel + '/', True), b'')
            else:
                z.writestr(zip_entry(rel), open(f'{d}/{rel}', 'rb').read())
        z.writestr(zip_entry('README.txt'), readme.replace('\n', '\r\n'))
        for f in ('COPYING', 'COPYING.README'):
            z.writestr(zip_entry(f), open(f'{ROOT}/{f}', 'rb').read())
    os.replace(zpath + '.tmp', zpath)
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        n = len(z.namelist())
    h = hashlib.sha256(open(zpath, 'rb').read()).hexdigest()
    caps = ('\n- Videos dominated by full-screen static (' + ', '.join(sorted(os.path.basename(k) for k in t['rc'])) +
            f")\n  are limited to {t['static_cap']} kbps.") if t['rc'] else ''
    enc = ENCODING.format(Tier=Tier, rule='=' * len(f'{title}: encoding details'), pack=os.path.basename(zpath),
                          size=os.path.getsize(zpath), sha=h, date=datetime.date.today().isoformat(), ffmpeg=ffv,
                          crf=t['crf'], mono=t['mono'], stereo=t['stereo'], caps=caps, chain=CHAINS_TEXT[t['src']])
    epath = f'{STAGE_D}/{tier}-quality-encoding.txt'
    with open(epath, 'w', newline='\r\n') as f:
        f.write(enc)
    print(f'{zpath}: {os.path.getsize(zpath) / 1e6:.1f} MB, {n} entries, SHA-256 {h}\n{epath}')


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) < 2 or a[1] not in S.TIERS or a[0] not in ('video', 'audiogain', 'mux', 'check', 'pack'):
        sys.exit(__doc__)
    cmd, tier, names = a[0], a[1], a[2:]
    if cmd == 'video':
        encode(tier, names or videos())
    elif cmd == 'audiogain':
        audiogain(tier)
    elif cmd == 'mux':
        mux(tier, names)
    elif cmd == 'check':
        check(tier, names)
    else:
        pack(tier)
