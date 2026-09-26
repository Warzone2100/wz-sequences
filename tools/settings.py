# SPDX-License-Identifier: GPL-2.0-or-later
"""Settings of stage (c) (the upscaled masters) and stage (d) (the WebM encodes and packs)."""

# ------------------------------------------------------------------------------------------------------------ stage (c)
# The decoder's full-range values mapped to TV range, in 10-bit (all 64 source levels stay distinct).
LEVELS = 'format=yuv420p10le,scale=in_range=pc:out_range=tv,setparams=range=tv'
TAGS = 'setparams=range=tv:colorspace=smpte170m:color_primaries=smpte170m:color_trc=smpte170m'
# standard: spline36 to 2x, then a light deband ({w2}x{h2} = 2x the source size)
STANDARD_VF = ('format=yuv420p10le,zscale=w={w2}:h={h2}:filter=spline36,'
               'deband=1thr=0.010:2thr=0.012:3thr=0.012:range=8:blur=1')
# ml: Real-ESRGAN realesr-general-x4v3 at denoise 0.25, run with ncnn (models/<ML_MODEL>.ncnn.param and .bin)
ML_MODEL = 'realesr-general-x4v3-dn025'
ML_DENOISE = 0.25       # the model's weights: 0.25 x general-x4v3 + 0.75 x general-wdn-x4v3
WEIGHTS = {             # the Real-ESRGAN v0.2.5.0 release: file -> (URL, SHA-256)
    'realesr-general-x4v3.pth': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        '8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292'),
    'realesr-general-wdn-x4v3.pth': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
        '1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca'),
}
# 160x120 content shown as 320x240: the ml method starts from one pixel of each 2x2 block, so its 4x lands on 2x
PIXEL_DOUBLED = {'cam2/cam2_2n', 'cam2/cam2_2n2', 'cam3/c3ad2_12', 'cam3/c3ad2_14', 'cam3/c3ad2_16', 'cam3/cam34fmv',
                 'cam3/cam3_1bn'}
# No random IDs, dates or FFmpeg version in the files FFmpeg writes: the same input always gives the same file.
BITEXACT = ['-fflags', '+bitexact', '-flags', '+bitexact']
FFV1 = ['-c:v', 'ffv1', '-level', '3', '-g', '1', '-slices', '4', '-slicecrc', '1', *BITEXACT]

# ------------------------------------------------------------------------------------------------------------ stage (d)
# 2-pass constant quality (-crf per tier). One thread per encode: the thread count changes libvpx's output.
VP9 = ['-c:v', 'libvpx-vp9', '-b:v', '0', '-deadline', 'good', '-cpu-used', '2', '-row-mt', '1', '-tile-columns', '0',
       '-auto-alt-ref', '1', '-lag-in-frames', '25', '-g', '250',
       '-color_range', 'tv', '-colorspace', 'smpte170m', '-color_primaries', 'smpte170m', '-color_trc', 'smpte170m']
VF = 'zscale=dither=error_diffusion,format=yuv420p'        # 10-bit master -> 8-bit
# CELT only (restricted low delay), 12 kHz audio bandwidth
OPUS = ['-c:a', 'libopus', '-vbr', 'on', '-compression_level', '10', '-frame_duration', '20', '-application', 'lowdelay',
        '-cutoff', '12000']
AF = 'aresample=48000:resampler=soxr:precision=28'
PEAK_LIMIT = -0.1       # dBFS, the highest decoded Opus peak allowed
# rc: per-video rate control in place of '-b:v 0', for the videos dominated by full-screen static.
# '-b:v' alone caps the average bitrate. '-b:v' with '-maxrate' sets a 2-pass target and limits each section.
TIERS = {
    'standard': dict(src='standard', crf=26, mono=48, stereo=80, static_cap=3000,
                     rc={'cam3/cam3bg': ['-b:v', '3000k'], 'cam1/c001end': ['-b:v', '3226k', '-maxrate', '3000k'],
                         'cam2/cam2int1': ['-b:v', '2132k', '-maxrate', '3000k']}),
    'enhanced': dict(src='ml', crf=26, mono=48, stereo=80, static_cap=3000,
                     rc={'cam3/cam3bg': ['-b:v', '3000k'], 'cam1/c001end': ['-b:v', '3383k', '-maxrate', '3000k'],
                         'cam2/cam2int1': ['-b:v', '2318k', '-maxrate', '3000k']}),
    'low': dict(src='ml', crf=48, mono=40, stereo=64, static_cap=2000,
                rc={'cam3/cam3bg': ['-b:v', '2000k']}),
}
LANGS = {'eng': 'English', 'fre': 'Français', 'ger': 'Deutsch'}
# The top-level c001 is kept as a master but not packed: nothing in the game refers to it (the intro is cam1/c001).
PACK_EXCLUDE = {'c001'}
