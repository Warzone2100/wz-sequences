#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Inventory, decode check and lossless extraction of every .rpl (eng/fre/ger).

Per file:
  * sha256 and ffprobe stream info (needs ffmpeg >= 8.1: older versions use the wrong ADPCM decoder for these files)
  * one full decode that doubles as the integrity check (every warning and error is logged) and writes
      - English video  -> stage_a/video/eng/<name>.mkv   (FFV1, yuv420p, native frame rate, exact)
      - all audio      -> stage_a/audio/<lang>/<name>.flac (native rate and channels)
  * a second video pass: showinfo -> mpdecimate -> showinfo, giving the total frame count, the unique
    (non-duplicate) frame count and whether dropped frames follow an every-other-frame pattern
    (12.5 fps content padded to 25 fps)
Outputs: recipes/inventory.csv, analysis/decode_log.txt, summary on stdout.
"""
import argparse, csv, hashlib, json, os, re, shutil, subprocess, sys
from paths import ROOT, STAGE_A, RECIPES, ANALYSIS, RPL, FF, FP
from multiprocessing import Pool

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=RPL)
ap.add_argument('--jobs', type=int, default=6)
ap.add_argument('--min-free-gb', type=float, default=30.0)
ap.add_argument('--only', help='only process names containing this substring (testing)')
ap.add_argument('--out', default='inventory.csv', help='inventory file name in recipes/')
args = ap.parse_args()
LANGS = ['eng', 'fre', 'ger']
# warnings that are expected and harmless for this material
BENIGN = [re.compile(p) for p in [r'Guessed Channel Layout']]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def probe(path):
    out = subprocess.run([FP, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', path],
                         capture_output=True, text=True)
    return json.loads(out.stdout or '{}'), out.stderr.strip()


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stderr


def job(item):
    lang, name = item
    src = f'{args.src}/{lang}/{name}.rpl'
    info, perr = probe(src)
    vs = next((s for s in info.get('streams', []) if s['codec_type'] == 'video'), None)
    as_ = next((s for s in info.get('streams', []) if s['codec_type'] == 'audio'), None)
    row = dict(lang=lang, name=name, bytes=os.path.getsize(src), sha256=sha256(src))
    if vs:
        num, den = vs['r_frame_rate'].split('/')
        row.update(vcodec=vs['codec_name'], width=vs['width'], height=vs['height'], pix_fmt=vs.get('pix_fmt'),
                   fps=float(num) / float(den))
    if as_:
        row.update(acodec=as_['codec_name'], arate=int(as_['sample_rate']), ach=as_['channels'])

    # --- pass 1: full decode (integrity check) + lossless outputs
    cmd = [FF, '-hide_banner', '-nostats', '-v', 'warning', '-i', src]
    vout = aout = None
    if vs:
        if lang == 'eng':
            vout = f'{STAGE_A}/video/eng/{name}.mkv'; os.makedirs(os.path.dirname(vout), exist_ok=True)
            cmd += ['-map', '0:v', '-c:v', 'ffv1', '-level', '3', '-slices', '4', '-g', '1', '-fps_mode', 'passthrough',
                    '-threads', '2', '-fflags', '+bitexact', '-flags', '+bitexact', '-y', vout]
        else:
            cmd += ['-map', '0:v', '-f', 'null', '-']
    if as_:
        aout = f'{STAGE_A}/audio/{lang}/{name}.flac'; os.makedirs(os.path.dirname(aout), exist_ok=True)
        cmd += ['-map', '0:a', '-c:a', 'flac', '-compression_level', '8', '-y', aout]
    rc, err = run(cmd)
    lines = [l for l in err.splitlines() if l.strip()]
    serious = [l for l in lines if not any(b.search(l) for b in BENIGN)]
    row.update(decode_rc=rc, warnings=len(serious), benign_warnings=len(lines) - len(serious))

    # --- decoded audio length
    if aout and os.path.exists(aout):
        a = json.loads(subprocess.run([FP, '-v', 'error', '-select_streams', 'a', '-show_entries',
                                       'stream=duration_ts,sample_rate', '-of', 'json', aout],
                                      capture_output=True, text=True).stdout)['streams'][0]
        row['audio_samples'] = int(a['duration_ts']); row['audio_dur'] = round(int(a['duration_ts']) / int(a['sample_rate']), 4)

    # --- pass 2: frame count, unique frames, duplicate pattern
    if vs:
        rc2, err2 = run([FF, '-hide_banner', '-nostats', '-v', 'info', '-i', src, '-map', '0:v',
                         '-vf', 'showinfo,mpdecimate,showinfo', '-f', 'null', '-'])
        pts_all = [int(m.group(1)) for m in re.finditer(r'Parsed_showinfo_0.*? pts:\s*(-?\d+)', err2)]
        pts_kept = set(int(m.group(1)) for m in re.finditer(r'Parsed_showinfo_2.*? pts:\s*(-?\d+)', err2))
        n = len(pts_all); kept = len(pts_kept)
        dropped_idx = [i for i, p in enumerate(pts_all) if p not in pts_kept]
        odd = sum(i % 2 for i in dropped_idx)
        parity = max(odd, len(dropped_idx) - odd) / len(dropped_idx) if dropped_idx else 0.0
        row.update(frames=n, video_dur=round(n / row['fps'], 4) if n else 0, unique_frames=kept,
                   dup_frac=round(1 - kept / n, 3) if n else 0,
                   # 1.0 = every dropped frame has the same index parity (classic 2:1 padding); ~0.5 = random
                   dup_parity=round(parity, 3),
                   eff_fps=round(kept / (n / row['fps']), 2) if n else 0)
        if as_ and 'audio_dur' in row:
            row['av_diff'] = round(row['audio_dur'] - row['video_dur'], 4)
    return row, lines, perr


def main():
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < args.min_free_gb:
        sys.exit(f'only {free:.0f} GB free in {ROOT}; need {args.min_free_gb} GB')
    items = []
    for lang in LANGS:
        for root, _, files in os.walk(f'{args.src}/{lang}'):
            for f in files:
                if f.endswith('.rpl'):
                    items.append((lang, os.path.relpath(os.path.join(root, f), f'{args.src}/{lang}')[:-4]))
    items.sort()
    if args.only:
        items = [i for i in items if args.only in i[1]]
    print(f'{len(items)} files, {free:.0f} GB free, {args.jobs} jobs', flush=True)
    with Pool(args.jobs) as p:
        res = p.map(job, items, chunksize=1)
    os.makedirs(ANALYSIS, exist_ok=True)
    cols = ['lang', 'name', 'bytes', 'sha256', 'vcodec', 'width', 'height', 'pix_fmt', 'fps', 'frames', 'video_dur',
            'unique_frames', 'dup_frac', 'dup_parity', 'eff_fps', 'acodec', 'arate', 'ach', 'audio_samples', 'audio_dur',
            'av_diff', 'decode_rc', 'warnings', 'benign_warnings']
    with open(f'{RECIPES}/{args.out}', 'w', newline='') as f:
        w = csv.DictWriter(f, cols); w.writeheader()
        for row, _, _ in res:
            w.writerow({c: row.get(c, '') for c in cols})
    with open(f'{ANALYSIS}/decode_log.txt', 'w') as f:
        for row, lines, perr in res:
            if lines or perr or row['decode_rc']:
                f.write(f"== {row['lang']}/{row['name']} rc={row['decode_rc']}\n")
                for l in ([perr] if perr else []) + lines:
                    f.write(f'   {l}\n')
    bad = [r for r, _, _ in res if r['decode_rc'] or r['warnings']]
    print(f'decode failures/warnings: {len(bad)} files; benign-only: '
          f"{sum(1 for r, _, _ in res if r['benign_warnings'] and not r['warnings'])}")
    for r in bad[:20]:
        print(f"   {r['lang']}/{r['name']}: rc={r['decode_rc']} warnings={r['warnings']}")


if __name__ == '__main__':
    main()
