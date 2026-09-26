# SPDX-License-Identifier: GPL-2.0-or-later
"""Where everything is (README.md, "Layout"): every tool takes its paths from here, relative to this folder.

The .rpl originals and FFmpeg are found through two environment variables:
  WZSEQ_RPL      the folder with the .rpl files as <lang>/<name>.rpl (default: rpl/ in this folder)
  WZSEQ_FFMPEG   the folder with ffmpeg and ffprobe, version 8.1 or later (default: the ones on the PATH)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_A = f'{ROOT}/stage_a'        # lossless decodes of the .rpl files
STAGE_B = f'{ROOT}/stage_b'        # edited masters
STAGE_C = f'{ROOT}/stage_c'        # 2x upscaled 10-bit masters, one folder per method (made by the tools)
STAGE_D = f'{ROOT}/stage_d'        # WebM encodes and .wz packs, one folder per tier (made by the tools)
MODELS = f'{ROOT}/models'          # the converted ncnn model
RECIPES = f'{ROOT}/recipes'        # the edit decisions and the other tables the builds read
ANALYSIS = f'{ROOT}/analysis'      # the alignment result (and the other analysis files, when the tools make them)
RPL = os.environ.get('WZSEQ_RPL', f'{ROOT}/rpl')
_FFBIN = os.environ.get('WZSEQ_FFMPEG', '')
FF = os.path.join(_FFBIN, 'ffmpeg') if _FFBIN else 'ffmpeg'
FP = os.path.join(_FFBIN, 'ffprobe') if _FFBIN else 'ffprobe'
