#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Stage (c): 2x upscaled 10-bit masters of the stage (b) videos, one folder per method.

  build_stage_c.py standard [<video> ...] [--force]   spline36 to 2x, then a light deband
                                                      -> stage_c/standard/video/<video>.mkv
  build_stage_c.py ml [<video> ...] [--force]         machine-learning upscale: Real-ESRGAN realesr-general-x4v3
                                                      (denoise 0.25, run with ncnn) to 4x, then area-downscaled to 2x
                                                      -> stage_c/ml/video/<video>.mkv
  build_stage_c.py check <method>                     frame counts, frame rate, size, pixel format and tags against
                                                      stage (b); exit code 1 on a problem
<video>: a stage (b) video name, ex. cam1/c001 or cam2/cam2_2n2.fre (default: all of them).
Every output starts with the levels mapping (settings.LEVELS) and is FFV1 10-bit 4:2:0, tagged BT.601 TV range.
Existing outputs are kept unless --force is given.
Each video is written to a .tmp.mkv file and renamed when complete, so an interrupted run continues where it stopped.
The sources must match stage_b/manifest.json.
The ml method needs the Python packages in tools/requirements.txt (ncnn, NumPy).
It runs one video at a time on all CPU threads.
"""
import hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import settings as S
from paths import STAGE_B, STAGE_C, MODELS, FF, FP

METHODS = ('standard', 'ml')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def all_videos():
    d = f'{STAGE_B}/video'
    return sorted(os.path.relpath(os.path.join(r, f), d)[:-4] for r, _, fs in os.walk(d) for f in fs if f.endswith('.mkv'))


def probe(path):
    out = subprocess.run([FP, '-v', 'error', '-select_streams', 'v', '-count_packets', '-show_entries',
                          'stream=width,height,pix_fmt,r_frame_rate,nb_read_packets,color_range,color_space,'
                          'color_primaries,color_transfer', '-of', 'json', path], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)['streams'][0]


def out_path(method, name):
    return f'{STAGE_C}/{method}/video/{name}.mkv'


# ------------------------------------------------------------------------------------------------------------- standard
def standard_one(name):
    src, out = f'{STAGE_B}/video/{name}.mkv', out_path('standard', name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    s = probe(src); w, h = s['width'], s['height']
    vf = f"{S.LEVELS},{S.STANDARD_VF.format(w2=2 * w, h2=2 * h)},format=yuv420p10le,{S.TAGS}"
    p = subprocess.run([FF, '-v', 'error', '-y', '-i', src, '-vf', vf, *S.FFV1, '-threads', '2', '-an',
                        out[:-4] + '.tmp.mkv'], capture_output=True, text=True)
    if p.returncode:
        return f'{name}: FAILED {p.stderr[-400:]}'
    os.replace(out[:-4] + '.tmp.mkv', out)
    return f'{name}: {w}x{h} -> {2 * w}x{2 * h}'


# ------------------------------------------------------------------------------------------------------------------- ml
class Upscaler:
    """The ncnn model, loaded once. upscale() streams a whole video through it (never the whole clip in memory)."""
    def __init__(self):
        import ncnn
        self.ncnn = ncnn
        m = ncnn.Net(); o = m.opt
        o.use_vulkan_compute = False; o.num_threads = os.cpu_count(); o.use_winograd_convolution = True
        o.use_fp16_packed = o.use_fp16_storage = o.use_fp16_arithmetic = False; o.use_bf16_storage = False
        m.load_param(f'{MODELS}/{S.ML_MODEL}.ncnn.param'); m.load_model(f'{MODELS}/{S.ML_MODEL}.ncnn.bin')
        self.net = m

    def infer(self, f):
        import numpy as np
        # ncnn.Mat wraps the NumPy buffer without copying: both stay alive until the extraction is done
        chw = np.ascontiguousarray(f.transpose(2, 0, 1)); mat = self.ncnn.Mat(chw)
        ex = self.net.create_extractor(); ex.input('in0', mat); _, out = ex.extract('out0')
        y = np.array(out)
        del ex, mat, chw
        return np.clip(y, 0, 1).transpose(1, 2, 0)

    def upscale(self, src, out, half):
        """src -> out: the levels mapping, RGB (BT.601 TV range, bicubic chroma), the model (4x),
        an area downscale to 2x, and FFV1 10-bit 4:2:0.
        half: take one pixel of each 2x2 block first (pixel-doubled content). Returns the frame count."""
        import numpy as np
        s = probe(src); w, h = s['width'], s['height']
        dw, dh = (w // 2, h // 2) if half else (w, h)
        dec = subprocess.Popen([FF, '-v', 'error', '-i', src, '-vf',
                                f'{S.LEVELS},' + (f'scale={dw}:{dh}:flags=neighbor,' if half else '') +
                                'scale=iw:ih:flags=bicubic+full_chroma_int+full_chroma_inp:in_range=tv:in_color_matrix=bt601,'
                                'format=rgb48le', '-f', 'rawvideo', '-'], stdout=subprocess.PIPE)
        enc = subprocess.Popen([FF, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', f'{4 * dw}x{4 * dh}',
                                '-r', s['r_frame_rate'], '-i', '-', '-vf',
                                f'scale={2 * w}:{2 * h}:flags=area,scale=out_range=tv:out_color_matrix=bt601,'
                                f'format=yuv420p10le,{S.TAGS}', *S.FFV1, out], stdin=subprocess.PIPE)
        size = dw * dh * 3 * 2; n = 0
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            f = np.frombuffer(buf, np.uint16).reshape(dh, dw, 3)
            y = self.infer(f.astype(np.float32) / 65535)
            enc.stdin.write((y * 65535 + 0.5).astype(np.uint16).tobytes())
            n += 1
        enc.stdin.close()
        if dec.wait() != 0 or enc.wait() != 0:
            raise RuntimeError(f'ffmpeg failed for {src}')
        return n


def ml(names):
    up = Upscaler()
    for k, name in enumerate(names, 1):
        src, out = f'{STAGE_B}/video/{name}.mkv', out_path('ml', name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        half = name in S.PIXEL_DOUBLED
        t0 = time.time()
        try:
            n = up.upscale(src, out[:-4] + '.tmp.mkv', half)
        except BaseException:
            if os.path.exists(out[:-4] + '.tmp.mkv'):
                os.remove(out[:-4] + '.tmp.mkv')
            raise
        want = int(probe(src)['nb_read_packets'])
        if n != want:
            os.remove(out[:-4] + '.tmp.mkv')
            sys.exit(f'{name}: {n} frames written, stage (b) has {want}')
        os.replace(out[:-4] + '.tmp.mkv', out)
        dt = time.time() - t0
        print(f"[{k}/{len(names)}] {name}{' (from 160x120)' if half else ''}: {n} frames in {dt:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------------- build
def build(method, names, force):
    names = names or all_videos()
    man = json.load(open(f'{STAGE_B}/manifest.json'))['files']
    bad = [n for n in names if man.get(f'video/{n}.mkv') != sha256(f'{STAGE_B}/video/{n}.mkv')]
    if bad:
        sys.exit(f'stage (b) videos that differ from stage_b/manifest.json: {bad}')
    todo = [n for n in names if force or not os.path.exists(out_path(method, n))]
    print(f'{method}: {len(todo)} of {len(names)} videos to make', flush=True)
    if method == 'ml':
        ml(todo)
        return
    with ThreadPoolExecutor(max(1, (os.cpu_count() or 2) // 2)) as ex:       # each FFV1 encode uses 2 threads
        for k, r in enumerate(ex.map(standard_one, todo), 1):
            print(f'[{k}/{len(todo)}] {r}', flush=True)


def check(method):
    names = all_videos(); problems = []; missing = []
    def one(n):
        if not os.path.exists(out_path(method, n)):
            return n, None
        a, b = probe(f'{STAGE_B}/video/{n}.mkv'), probe(out_path(method, n))
        want = dict(width=2 * a['width'], height=2 * a['height'], pix_fmt='yuv420p10le', r_frame_rate=a['r_frame_rate'],
                    nb_read_packets=a['nb_read_packets'], color_range='tv', color_space='smpte170m',
                    color_primaries='smpte170m', color_transfer='smpte170m')
        return n, {k: (b.get(k), v) for k, v in want.items() if b.get(k) != v}
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        for n, diff in ex.map(one, names):
            if diff is None:
                missing.append(n)
            elif diff:
                problems.append(f'{n}: {diff}')
    print(f'{method}: {len(names)} videos, {len(missing)} missing, {len(problems)} with problems')
    for p in problems:
        print('  ' + p)
    sys.exit(1 if problems or missing else 0)


if __name__ == '__main__':
    a = sys.argv[1:]; force = '--force' in a; a = [x for x in a if x != '--force']
    if len(a) == 2 and a[0] == 'check' and a[1] in METHODS:
        check(a[1])
    elif a and a[0] in METHODS:
        build(a[0], a[1:], force)
    else:
        sys.exit(__doc__)
