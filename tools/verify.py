#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Checks of this folder (README.md, "Checking").

  verify.py files [--update]       every file against SHA256SUMS.txt (--update rewrites the list instead), except README.md
  verify.py rpl                    your .rpl files (paths.RPL) against stage_a/manifest.json: the same file, or, where
                                   the bytes differ, the same decoded video and audio
  verify.py rebuild [--scratch <dir>] [--reuse]
                                   rebuild stage (b) from stage (a), the recipes and the tools in a scratch copy of this
                                   folder, then compare every file with stage_b/manifest.json.
                                   --reuse compares an earlier rebuild in <dir> again.
                                   For a .mkv that differs, it reports whether its packets or frames are the same
                                   (only the container differs, or only the FFV1 data); for a .flac, how large the
                                   sample differences are.
"""
import concurrent.futures, glob, hashlib, json, os, shutil, subprocess, sys, tempfile
from paths import ROOT, STAGE_A, STAGE_B, RPL, FF

SUMS = f'{ROOT}/SHA256SUMS.txt'
MADE_HERE = ('rpl/', 'stage_b/preview/', 'stage_c/', 'stage_d/', 'analysis/fp/', 'analysis/stock_frames.pkl', 'analysis/alignment.csv',
             'analysis/alignment_detail.txt', 'analysis/audio_analysis.csv', 'analysis/keyframes.csv',
             'analysis/decode_log.txt')   # made by the tools
UNLISTED = ('README.md',)
PACKET_HASH = ['-map', '0', '-c', 'copy', '-f', 'hash', '-hash', 'sha256', '-']
FRAME_HASH = ['-map', '0:v', '-f', 'hash', '-hash', 'sha256', '-']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def stream_hash(path, opts):
    out = subprocess.run([FF, '-v', 'error', '-i', path] + opts, capture_output=True, text=True, check=True).stdout
    return out.strip().split('=', 1)[1]


def files(update):
    have = sorted(rel for p in glob.glob(f'{ROOT}/**/*', recursive=True)
                  if os.path.isfile(p) and (rel := os.path.relpath(p, ROOT)) != 'SHA256SUMS.txt'
                  and '__pycache__' not in rel and not rel.startswith(MADE_HERE) and rel not in UNLISTED)
    if update:
        with open(SUMS, 'w') as f:
            f.writelines(f'{sha256(f"{ROOT}/{rel}")}  {rel}\n' for rel in have)
        print(f'{len(have)} files -> SHA256SUMS.txt')
        return
    listed = dict(reversed(line.rstrip('\n').split('  ', 1)) for line in open(SUMS))
    bad = [rel for rel in have if rel in listed and sha256(f'{ROOT}/{rel}') != listed[rel]]
    extra, missing = sorted(set(have) - set(listed)), sorted(set(listed) - set(have))
    for kind, rels in (('changed', bad), ('not in SHA256SUMS.txt', extra), ('missing', missing)):
        for rel in rels:
            print(f'  {kind}: {rel}')
    print(f'{len(listed)} files listed: {len(bad)} changed, {len(extra)} not listed, {len(missing)} missing')
    sys.exit(1 if bad or extra or missing else 0)


def rpl():
    man = json.load(open(f'{STAGE_A}/manifest.json'))
    cmd = man['decoded_sha256']

    def one(item):
        key, e = item; path = f'{RPL}/{key}'
        if not os.path.exists(path):
            return key, 'not found'
        if sha256(path) == e['sha256']:
            return key, 'same file'
        for kind in ('video', 'audio'):
            if kind in e and stream_hash(path, cmd[kind].split(' ')[3:]) != e[kind]['decoded_sha256']:
                return key, f'different {kind}'
        return key, 'same video and audio (the file bytes differ)'

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        res = dict(ex.map(one, man['sources'].items()))
    count = {}
    for key, r in sorted(res.items()):
        count[r] = count.get(r, 0) + 1
        if r != 'same file':
            print(f'  {key}: {r}')
    print(f'{len(res)} .rpl files in stage_a/manifest.json, looked for in {RPL}:', count)
    sys.exit(1 if any(r.startswith('different') for r in res.values()) else 0)


def rebuild(scratch, reuse):
    """The scratch copy holds tools/ and recipes/, links stage_a/ (read only), and starts with an empty stage_b/."""
    ref = json.load(open(f'{STAGE_B}/manifest.json'))
    S = f'{scratch}/pack'
    if reuse:
        if not os.path.exists(f'{S}/stage_b/manifest.json'):
            sys.exit(f'--reuse: no finished rebuild in {S}')
    elif os.path.exists(S):
        sys.exit(f'{S} exists: give an empty --scratch (or --reuse)')
    else:
        os.makedirs(f'{S}/stage_b')
        os.symlink(STAGE_A, f'{S}/stage_a')
        for d in ('tools', 'recipes'):
            shutil.copytree(f'{ROOT}/{d}', f'{S}/{d}', ignore=shutil.ignore_patterns('__pycache__'))
        for step in ('video', 'langvideo', 'audio', 'edits', 'check'):
            print(f'== build_masters.py {step}', flush=True)
            if subprocess.run([sys.executable, f'{S}/tools/build_masters.py', step]).returncode:
                sys.exit(f'build_masters.py {step} failed')
    got = json.load(open(f'{S}/stage_b/manifest.json'))
    diff = {k: 'recipe differs' for k in ref['recipes'] if ref['recipes'][k] != got['recipes'].get(k)}
    for k in sorted(set(ref['files']) | set(got['files'])):
        if ref['files'].get(k) == got['files'].get(k):
            continue
        if k not in got['files'] or k not in ref['files']:
            diff[k] = 'missing' if k not in got['files'] else 'extra'
        else:
            diff[k] = 'differs'
            if k.endswith('.mkv'):
                a, b = f'{STAGE_B}/{k}', f'{S}/stage_b/{k}'
                if stream_hash(a, PACKET_HASH) == stream_hash(b, PACKET_HASH):
                    diff[k] += ' (the same packets: only the container differs)'
                elif stream_hash(a, FRAME_HASH) == stream_hash(b, FRAME_HASH):
                    diff[k] += ' (the same frames: the FFV1 data differs)'
                else:
                    diff[k] += ' (the frames differ)'
            elif k.endswith('.flac'):
                import numpy as np
                x, y = (np.frombuffer(subprocess.run([FF, '-v', 'error', '-i', p, '-f', 's32le', '-'], capture_output=True,
                                                     check=True).stdout, '<i4') >> 8 for p in (f'{STAGE_B}/{k}', f'{S}/stage_b/{k}'))
                diff[k] += (f' (length {len(x)} vs {len(y)})' if len(x) != len(y) else
                            f' (largest difference {np.abs(x - y).max()} in 24-bit, {np.count_nonzero(x != y)} samples)')
    print(f"{len(ref['files'])} stage (b) files; rebuilt with {got['ffmpeg']}")
    print(f"{len(ref['files']) - len(diff)} identical files")
    for k, why in diff.items():
        print(f'  {k}: {why}')
    print('stage (b) rebuilt identically' if not diff else f'{len(diff)} differences')
    sys.exit(1 if diff else 0)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['files']:
        files('--update' in a)
    elif a[:1] == ['rpl']:
        rpl()
    elif a[:1] == ['rebuild']:
        rebuild(a[a.index('--scratch') + 1] if '--scratch' in a else tempfile.mkdtemp(prefix='wzseq_rebuild_'), '--reuse' in a)
    else:
        sys.exit(__doc__)
