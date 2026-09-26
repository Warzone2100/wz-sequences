# Warzone 2100 campaign videos - lossless sources and build chain

This folder holds the source files and scripts behind the WebM sequence video
packs (Low, Standard and Enhanced quality): the original videos decoded to
lossless files, the edited masters made from them, the edit decisions, and the
tools that turn the originals into the finished packs. With it you can see how
each edited master was made, rebuild the masters exactly, change an edit and
rebuild, and make the upscaled masters and the packs.

```
.rpl originals  ->  stage (a): lossless decodes      stage_a/
(not included)  ->  stage (b): edited masters        stage_b/
                ->  stage (c): upscaled masters      stage_c/ (made by the tools)
                ->  stage (d): WebM encodes, packs   stage_d/ (made by the tools)
```


## Layout

| Path | Contents |
| --- | --- |
| `stage_a/` | Stage (a), see below. |
| `stage_b/` | Stage (b), see below. |
| `recipes/` | The tables the builds read: |
| `recipes/sequence_edits.csv` | the edit decisions made in review (see below) |
| `recipes/sequence_edits_auto.csv` | the automatic plans, for every track without a decision in `sequence_edits.csv` |
| `recipes/inventory.csv` | every .rpl file: size, SHA-256, stream details, frame and sample counts, decode warnings |
| `recipes/audio_gains.csv` | the loudness target of every audio track |
| `recipes/keyframes.json` | the frames the edit lists jump to |
| `analysis/alignment_segments.json` | how every French and German file maps onto the English one (the input of the automatic plans) |
| `models/` | The ml upscaling model (see "The ml model"). |
| `tools/` | The tools: Python 3, and one shell script. They find everything through `tools/paths.py`, relative to this folder. The stage (c) and (d) settings are in `tools/settings.py`. |
| `README.md` | This file. |
| `COPYING`, `COPYING.README` | The license (see the end of this file). |
| `SHA256SUMS.txt` | A checksum for every file in this folder, except this README. |

Not included, but used or made by the tools:
- `rpl/` (the default place for your .rpl files)
- `stage_c/` and `stage_d/`
- more files in `analysis/` (see "How the edits were decided")
- `stage_b/preview/` (each language's edit list applied to the picture, for
  checking)


## Setup

The tools need Python 3 and FFmpeg 8.1 or later (ffmpeg and ffprobe, with
libvpx, libopus, zimg and soxr). The Python packages, at the versions that made
the files here:

```sh
pip install -r tools/requirements.txt           # NumPy, and ncnn for the ml method
pip install -r tools/requirements-convert.txt   # PyTorch and pnnx, only for tools/convert_model.py
```

The files here were made with Python 3.14.4 and FFmpeg n8.1.3 (BtbN's static
build, with libvpx 1.16.0 and soxr 0.1.3). Stage (b) was made with NumPy 2.3.5,
and rebuilds identically with the 2.5.3 in `requirements.txt`.

Two environment variables tell the tools where things are:

| Variable | Meaning |
| --- | --- |
| `WZSEQ_FFMPEG` | the folder with ffmpeg and ffprobe (default: the ones on the PATH) |
| `WZSEQ_RPL` | the folder with your .rpl files as `<lang>/<name>.rpl`, ex. `eng/cam1/c001.rpl` (default: `rpl/` in this folder). Only the tools that read the originals need it. |

The tools run from any directory, ex. `python3 tools/build_masters.py check`.


## Stage (a): lossless decodes (`stage_a/`)

- **`video/eng/**.mkv`**: The English video of every sequence: FFV1, the decoded
  frames unchanged (4:2:0, native size and frame rate).
- **`audio/<lang>/**.flac`**: The audio of every sequence in English, French and
  German: FLAC, the decoded samples unchanged (16-bit, native rate and
  channels). Sequences without sound have no file. There is no French version of
  the top-level c001 (cam1/c001, the campaign intro, has all three).
- **`manifest.json`**: Ties every file to its .rpl original: the .rpl's size and
  SHA-256, and a SHA-256 of its decoded video and audio streams, which match a
  direct decode of the .rpl (the command for each is in the manifest).

The French and German video is not included. Stage (b) uses the English picture
only, so nothing here needs it. To look at the localized picture, decode your
own .rpl files.

The .rpl files use Escape 130 video and 4-bit ADPCM audio. Decode the audio with
FFmpeg 8.1 or later. Earlier versions use another ADPCM decoder, which makes
these files about 16 dB too quiet, with a large DC drift.

Levels: the decoded pictures use the full 0-255 range (black is 0), but the
files are tagged as TV range (16-235), as the decoder reports them. A player
that follows the tag shows dark detail as flat black and the brightest areas as
flat white. The pixel values themselves are unchanged. (The WebM packs map them
to TV range before encoding.)


## Stage (b): edited masters (`stage_b/`)

- **`video/<seq>.mkv`**: The shared video of each sequence, played by all three
  languages: the English picture with the picture edits.
  - 8 sequences lose a black, silent lead-in (0.5-2 s) that the French and
    German don't have.
  - 11 have their ending changed. Most continue with matching English footage
    (6 of them fading to black), one end freeze is replaced, and one video is
    swapped for a cleaner English copy of the same loop.
  - One gets 6 more frames of its opening cycle.

  FFV1, native size and frame rate, tagged BT.601 TV range (the levels note
  above applies here too).
- **`video/<seq>.<lang>.mkv`**: A separate video for a language whose picture
  differs from the English ("loc" = French and German). These are made from
  English footage too: frame ranges matched to the localized picture
  (`tools/build_masters.py`, `PER_LANG`).
- **`audio/<seq>.<lang>.flac`**: One conformed audio track per sequence and
  language: a 10 Hz high-pass (removes the DC offset and drift of the ADPCM
  decode), then the edits (padding, trims, inserted or removed gaps), then one
  static gain for the loudness target (true peak at most -1 dBTP). FLAC 24-bit,
  native rate and channels. French or German tracks that are identical to the
  English are not written (the game falls back to the English track).
- **`edits/<seq>.edits.json`**: The edit list of each sequence where a language
  doesn't simply play the video from start to end: how the picture is played
  (play ranges, holds, loops, black, fades) so it fits that language's audio.
  The game applies it during playback. The format is described at the top of
  `tools/build_masters.py`.
- **`audio_build.csv`**: Per track: the conformed length, the planned length and
  the gain applied.
- **`manifest.json`**: The SHA-256 of every file above and of every recipe (the
  files in `recipes/` and the build tools), the FFmpeg version, and the result
  of the build's checks.


## The edit decisions (`recipes/sequence_edits.csv`)

One row per edit. The rows here win over the automatic plans for the same
sequence and language.

| Column | Meaning |
| --- | --- |
| `sequence` | the sequence, ex. cam1/c001 |
| `applies_to` | the track: a language, its audio or edit list, or the shared video |
| `op` | the kind of edit |
| `params` | the edit itself (read by `tools/edit_model.py`) |
| `source` | **manual review**: decided by comparing the English and localized videos side by side<br>**review of a flagged automatic plan**: an automatic plan that needed a look, approved or changed<br>**consistency check**: found by `tools/consistency_check.py`<br>**manual review, corrected in the consistency check** (one row) |
| `status` | confirmed, with the date |
| `note` | what the edit is based on |

`sequence_edits_auto.csv` has the same columns, for every other track (its
source names the automatic planner's method, and its status is auto).


## How the edits were decided

The French and German releases are recuts of the English videos: most are at
half the frame rate, and many run longer or shorter (loops that repeat more
often, held frames, different cuts) to fit the translated speech. The steps,
in the order they were run:

1. `tools/decode_rpl.py` decodes every .rpl file, checks it, and writes
   stage (a) and `recipes/inventory.csv`.
2. `tools/fingerprint.sh` (run it with bash) makes a small fingerprint of every
   frame of every .rpl file, in all three languages (`analysis/fp/`).
3. `tools/detect_stock.py` finds the stock animations whose length follows the
   speech (the rotating atoms, the NEXUS logo) in the fingerprints
   (`analysis/stock_frames.pkl`).
4. `tools/align_languages.py` maps every French and German file onto the
   English one, frame by frame, from the fingerprints
   (`analysis/alignment_segments.json`, and readable forms of it in
   `analysis/alignment.csv` and `alignment_detail.txt`).
5. The files that don't map cleanly were reviewed by comparing the English and
   localized videos side by side. The decisions went into
   `recipes/sequence_edits.csv` (source "manual review").
6. `tools/auto_plans.py` plans every other track
   (`recipes/sequence_edits_auto.csv`) and flags the plans that need a look.
   Those were reviewed the same way, and the decisions went into
   `recipes/sequence_edits.csv` (source "review of a flagged automatic plan").
7. `tools/consistency_check.py` checks that every track has exactly one plan and
   that the planned picture and audio lengths agree.
8. `tools/audio_analysis.py` measures every track
   (`analysis/audio_analysis.csv`), and `tools/loudness_gains.py` sets the
   loudness targets (`recipes/audio_gains.csv`).
9. `tools/keyframes.py` lists the frames the edit lists jump to
   (`recipes/keyframes.json`, and `analysis/keyframes.csv` with the reasons).
10. `tools/build_masters.py` builds stage (b) from stage (a) and `recipes/`, and
    checks it.

Steps 1-4 need the .rpl files. Steps 6-10 run from this folder alone and give
the same files as the ones included (the automatic plans, the loudness targets,
the keyframes and stage (b)). The tools that made the review files are not
included.


## Stage (c): upscaled masters (`stage_c/`, made by the tools)

Every stage (b) video, upscaled to 2x (ex. 320x240 -> 640x480) in 10-bit, one
folder per method:

| Method | Upscale |
| --- | --- |
| `standard` | spline36 to 2x, then a light deband (FFmpeg) |
| `ml` | a machine-learning upscale to 4x (Real-ESRGAN realesr-general-x4v3 at denoise 0.25, run with ncnn), then an area downscale to 2x. The seven videos made from 160x120 content start from that resolution (`settings.PIXEL_DOUBLED`). |

Both start by mapping the levels to TV range (see "Levels" above), and write
FFV1 10-bit 4:2:0, tagged BT.601 TV range, to `stage_c/<method>/video/`.

```sh
python3 tools/build_stage_c.py standard
python3 tools/build_stage_c.py ml
python3 tools/build_stage_c.py check <method>
```

The ml method runs one video at a time on all CPU threads. An interrupted run
continues where it stopped. The finished masters take about 13 GB (standard)
and 14 GB (ml).


## Stage (d): WebM encodes and packs (`stage_d/`, made by the tools)

Three tiers (`tools/settings.py`, `TIERS`):

| Tier | Master | VP9 CRF | Opus mono/stereo | Pack size |
| --- | --- | --- | --- | --- |
| standard | standard | 26 | 48/80 kbps | 485 MB |
| enhanced | ml | 26 | 48/80 kbps | 521 MB |
| low | ml | 48 | 40/64 kbps | 167 MB |

Three videos dominated by full-screen static (cam3bg, c001end, cam2int1) have
their bitrate capped (Low caps only cam3bg). The top-level c001 isn't packed:
nothing in the game refers to it (the intro is cam1/c001).

```sh
python3 tools/build_stage_d.py video <tier>       # VP9 encodes
python3 tools/build_stage_d.py audiogain <tier>   # Opus gains (see below)
python3 tools/build_stage_d.py mux <tier>         # Opus audio, muxing, edit lists
python3 tools/build_stage_d.py check <tier>       # checks every video
python3 tools/build_stage_d.py pack <tier>        # the .wz pack
```

The pack is `stage_d/<tier>-quality-sequences.wz`, with a README.txt and the
license files inside, and `stage_d/<tier>-quality-encoding.txt` next to it (the
settings, and the pack's size and SHA-256). The video and mux commands also take
video names.

The audiogain command encodes every audio track, measures how much Opus changes
its loudness, and stores a small gain per track that undoes it, with the decoded
peaks kept at or below -0.1 dBFS (`stage_d/<tier>/opus_gains.json`). The mux
command applies those gains.

Exactness: on the same machine and versions, the tools give the same masters
and the same encodes (checked on a sample of videos). On another machine the
results may differ in the last bits, because the upscaling filters, ncnn and
libopus use floating point. Any difference in a master changes its whole VP9
encode.


## The ml model (`models/`)

`realesr-general-x4v3-dn025.ncnn.param` and `.bin` are the model the ml method
runs: the two Real-ESRGAN networks below, blended 0.25 / 0.75, converted to ncnn
with fp32 weights. They come from the Real-ESRGAN v0.2.5.0 release
(BSD-3-Clause license, `models/LICENSE-Real-ESRGAN.txt`):

- <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth>  
  SHA-256 `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`
- <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth>  
  SHA-256 `1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca`

```sh
python3 tools/convert_model.py
```

downloads both, checks them, and makes the model again, identical to the
included files. It needs the packages in `tools/requirements-convert.txt`.


## Checking

```sh
python3 tools/verify.py files
```

checks every file against `SHA256SUMS.txt` (all but this README).

```sh
python3 tools/verify.py rpl
```

checks your .rpl files against `stage_a/manifest.json`. Where a file's bytes
differ, it compares the decoded streams (a file with other bytes can still hold
the same video and audio).

```sh
python3 tools/verify.py rebuild
```

rebuilds all of stage (b) from stage (a), `recipes/` and `tools/` in a temporary
copy of this folder (about 2.5 GB, `--scratch <folder>` picks the place), and
compares every file with `stage_b/manifest.json`.
- The video is rebuilt byte for byte (the tools write their Matroska files
  without random IDs or dates). For a video that differs, the check says
  whether its packets or its frames are still the same.
- The audio goes through floating-point steps (filtering, loudness measurement,
  gain). It rebuilds bit for bit on the machine that made it, but another
  FFmpeg, NumPy or processor may change the last bit of some samples. The check
  then reports how large the differences are.


## Changing an edit

Change the row in `recipes/sequence_edits.csv`. A change to the shared picture
also goes into `tools/shared_video.py` (`SPECS`), and a change to a per-language
video into `tools/build_masters.py` (`PER_LANG`). If the frames the edit lists
jump to change, run `tools/keyframes.py`. Then rebuild:

```sh
python3 tools/build_masters.py video       # shared videos
python3 tools/build_masters.py langvideo   # per-language videos
python3 tools/build_masters.py audio       # conformed audio
python3 tools/build_masters.py edits       # edit lists
python3 tools/build_masters.py check       # checks, and a new manifest.json
```

The video, audio, edits and check commands also take sequence names (ex.
cam1/c001) to work on those only.


## License

The videos and audio are part of Warzone 2100. See `COPYING.README` (the
Warzone 2100 source and data license) and `COPYING` (the GNU General Public
License, version 2).

The tools, the recipes, the analysis file, the manifests and this README are
licensed under the GNU General Public License, version 2 or (at your option) any
later version (SPDX: GPL-2.0-or-later, text in `COPYING`).

The ml model is under the BSD 3-Clause license of Real-ESRGAN
(`models/LICENSE-Real-ESRGAN.txt`).
