#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Recreate the ncnn model of the ml method (models/<settings.ML_MODEL>.ncnn.param and .bin)
from the Real-ESRGAN weights.

  convert_model.py [--weights <dir>] [<out dir>]     default out dir: models/ (replaces the included files)
Downloads realesr-general-x4v3.pth and realesr-general-wdn-x4v3.pth (settings.WEIGHTS), or takes them from --weights,
and checks their SHA-256.
Blends them (settings.ML_DENOISE), traces the network with PyTorch, and converts it with pnnx (fp16=0: fp32 weights) in
a temporary folder. Only the .ncnn.param and .ncnn.bin files are kept (pnnx's other files hold local paths).
Needs the Python packages in tools/requirements-convert.txt (PyTorch, pnnx).
NOTE: PyTorch warns that torch.jit.trace isn't supported on Python 3.14 and later (it works with torch 2.14.0).
"""
import hashlib, os, shutil, subprocess, sys, tempfile, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F
import settings as S
from paths import MODELS


class SRVGGNetCompact(nn.Module):
    """The Real-ESRGAN compact network (as in basicsr and realesrgan), general-x4v3 configuration."""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body += [nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1)]
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for m in self.body:
            out = m(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode='nearest')


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main(weights, out):
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, (url, digest) in S.WEIGHTS.items():
            p = os.path.join(weights or tmp, name)
            if not os.path.exists(p):
                print(f'downloading {url}', flush=True)
                urllib.request.urlretrieve(url, p)
            if sha256(p) != digest:
                sys.exit(f'{p}: SHA-256 differs from settings.WEIGHTS')
            paths.append(p)
        sds = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
        sds = [sd.get('params', sd.get('params_ema', sd)) for sd in sds]
        dn = S.ML_DENOISE
        net = SRVGGNetCompact()
        net.load_state_dict({k: dn * sds[0][k] + (1 - dn) * sds[1][k] for k in sds[0]}); net.eval()
        torch.jit.trace(net, torch.rand(1, 3, 240, 320)).save(f'{tmp}/model.pt')
        pnnx = shutil.which('pnnx') or os.path.join(os.path.dirname(sys.executable), 'pnnx')
        subprocess.run([pnnx, 'model.pt', 'inputshape=[1,3,240,320]', 'fp16=0'], cwd=tmp, check=True,
                       capture_output=True)
        os.makedirs(out, exist_ok=True)
        for ext in ('ncnn.param', 'ncnn.bin'):
            shutil.copyfile(f'{tmp}/model.{ext}', f'{out}/{S.ML_MODEL}.{ext}')
            print(f'{out}/{S.ML_MODEL}.{ext}: SHA-256 {sha256(f"{out}/{S.ML_MODEL}.{ext}")}')


if __name__ == '__main__':
    a = sys.argv[1:]; weights = None
    if a[:1] == ['--weights']:
        weights, a = a[1], a[2:]
    if len(a) > 1 or (a and a[0].startswith('-')):
        sys.exit(__doc__)
    main(weights, a[0] if a else MODELS)
