# Auditable multi-codec signal extraction

The standalone C tool decodes the input with FFmpeg's selected decoder, requests
motion-vector side data, and writes motion records plus a diagnostic residual.
It does not require transcoding the input, vendor a modified FFmpeg build, or
claim to recover native transform residuals for every codec.

This release includes the C source, portable build entry point, and five-codec
validator promised during review. Portability and safety fixes in this release
are not a claim that this exact source revision produced historical model
benchmark numbers.

## Build and run

Requirements: a C11 compiler, `make`, `pkg-config`, and installed FFmpeg development
libraries for `libavformat`, `libavcodec`, `libavutil`, and `libswscale`. The smoke
test additionally requires a compatible FFmpeg command-line program and encoders
`libx264`, `libx265`, `libvpx-vp9`, `libaom-av1`, and `mpeg4`. Missing encoders cause
a test failure, not a silent skip. Python 3 is sufficient for the validator;
NumPy is needed for the optional array reader.

```bash
bash tools/codec/build.sh
tools/codec/extract_codec_signals input.mp4 extraction 0 16 8
python3 tools/codec/read_signals.py extraction --frame 2
python3 tools/codec/validate_codecs.py
python3 -m unittest discover -s tests -p 'test_codec*.py'
```

`extraction` must not exist; its parent directory must exist. Arguments after
the output path are maximum frames (`0`: all), block size (default `16`), and
integer search radius (default `8`). The optional `CC`, `CFLAGS`, `CPPFLAGS`,
`LDFLAGS`, and `PKG_CONFIG` build variables follow normal Make conventions.
Set `PKG_CONFIG_PATH` if FFmpeg development files are outside the default search
path. Only a local executable is built; do not commit the resulting binary.

## What the output means

| Output | Shape / representation | Meaning |
| --- | --- | --- |
| `summary.csv` | One row per decoded display-order frame | Dimensions, type, signal provenance, native MV count, fallback reason, timestamp and stream time base |
| `motion_vectors/frame_XXXXXX.csv` | Variable number of block records | Block geometry, source direction, integer motion numerator, motion scale, flags and origin |
| `residuals/frame_XXXXXX.s16le` | `[H, W]`, signed 16-bit little-endian | Diagnostic residual in decoded 8-bit grayscale units |
| `residuals/frame_XXXXXX.pgm` | `[H, W]`, unsigned 8-bit | `clip(residual + 128, 0, 255)` visualization; not lossless |

Frame filenames are **one-based** in display order. The first frame has no prior
displayed frame: its residual is zero, and usually no motion records are present.
Timestamps retain FFmpeg's missing-timestamp sentinel if unavailable. Pixels are
converted using `libswscale` to GRAY8; high-bit-depth input is reduced to 8 bits.
The tool rejects midstream dimension changes instead of resizing pixels while
leaving native vector coordinates inconsistent.

### Native vectors versus estimates

The sequence is `avformat_open_input` → `avformat_find_stream_info` →
`av_find_best_stream` → `avcodec_parameters_to_context` → `avcodec_open2` →
`avcodec_send_packet` / `avcodec_receive_frame`. Decoder option `+export_mvs`
requests `AV_FRAME_DATA_MOTION_VECTORS`; availability depends on the selected
decoder and FFmpeg build, not just the codec name.

* `mv_origin=bitstream`: records came from decoder-exported motion-vector side
  data. In the tested build, H.264 and MPEG-4 Part 2 export these records.
* `mv_origin=estimated`: no usable side data was available, or at least one
  exported vector referenced a future frame. The entire field is estimated
  against the previous **displayed** frame. Tested HEVC, VP9 and AV1 clips use
  this path. This is not a claim of native HEVC/VP9/AV1 MV extraction.
* `mv_origin=none`: no field is available, normally the first frame.

The fallback searches integer displacements within the configured radius for
each non-overlapping block. It minimizes sum of absolute differences sampled at
two-pixel strides. Ties are resolved in a fixed scan order. This implementation
is deterministic and intentionally simple; it is not a learned optical-flow
estimator or an optimized codec motion search.

`native_mv_count` preserves how many native records were seen even when the
field is replaced. `fallback_reason` is `no_side_data`, `future_reference`, or
`none`. The native `source` field indicates past/future direction; it is not a
sufficient reference-picture index for reconstructing the decoder's predictor.

### Residual definition and important limits

For both motion origins, the residual here is:

```text
R[t] = gray8(frame[t]) - integer_coordinate_warp(gray8(frame[t-1]), blocks[t])
```

This is a **diagnostic decoded-luma surrogate**, not the exact codec prediction
error. In particular:

* A native past-directed vector can reference an older frame, not necessarily
  the previous displayed frame. This tool does not retain the decoder's full
  reference-picture buffer.
* Native fractional motion values are preserved in CSV, but the diagnostic
  warp uses integer `src_x` / `src_y` block coordinates, without codec-specific
  subpixel interpolation, weighted prediction, or bidirectional prediction.
* Future-reference fields are replaced coherently, rather than combining
  incompatible past and future reference directions in one approximate warp.
* Uncovered pixels retain the previous frame's value in the predictor.
* Raw transform coefficients, dequantized residuals, and decoder internals are
  not exported. This implementation performs decoding; it is not a substitute
  for a decode-free throughput claim.

Nonzero residual is asserted only for the validator's moving synthetic clips.
A static real clip may legitimately have a zero residual.

## Array reader and relationship to training

`read_signals.load_frame(directory, frame_number)` returns metadata, the original
block records, a dense `[H, W, 2]` `float32` motion map, and an `[H, W]` `int16`
residual. Dense displacement is `motion_x / motion_scale` and
`motion_y / motion_scale`: **source minus destination**, in pixels. Overlapping
blocks use last-record-wins rasterization; retain the CSV if block structure or
multiple prediction records are important. Fractional values are not rounded
by this reader.

The historical Python `compressed_backend="coviar"` path is separate: it uses
CoViAR's MPEG-4 Part 2 / fixed-GOP interface. Its raw MV array and reference
accumulation semantics differ from this tool. The standalone reader is **not
silently substituted into that training loader**. Doing so would change signal
semantics and requires a deliberate configuration and new evaluation.

Image-like model inputs are a further representation change: signed MV values
may be clipped/scaled into x/y/magnitude uint8 channels; residuals may be clipped
and repeated across channels, then spatially resized and normalized. Neither
the resulting three-channel image nor this diagnostic residual is a lossless
replacement for raw codec data. Always retain provenance alongside caches.

## Validation performed for this release

On macOS with Apple Clang and GCC **15.2.0**, linked to FFmpeg **8.0.1**, compiled with
`-std=c11 -Wall -Wextra -Werror`:

* Independently encoded H.264, HEVC, VP9, AV1 and MPEG-4 Part 2 clips each decoded
  all 10 frames at 160×96. All 9 subsequent frames had motion records and nonzero
  residual for this moving fixture. Each raw residual was 30,720 bytes.
* H.264/MPEG-4 used bitstream vectors; HEVC/VP9/AV1 used estimated vectors.
* An additional H.264 B-frame clip exercised future-reference fallback.
* Repeating extraction on the same AV1 file produced byte-identical CSV, PGM
  and signed residual files.
* Invalid integers, absent inputs, audio-only input and reused output directories
  failed explicitly. Existing output files were unchanged.

The validator prints its installed FFmpeg version and exact output location.
These are extraction smoke tests, **not reruns of paper accuracy or latency
benchmarks**, and not a blanket guarantee for all codecs, profiles or builds.

## Licensing

`extract_codec_signals.c` retains its MIT SPDX designation; see
[`LICENSE-MIT`](../tools/codec/LICENSE-MIT), which also retains the FFmpeg example
notices. Other newly added helper files follow the repository license. FFmpeg
and optional CoViAR are external dependencies with their own licenses; their
source, libraries and binaries are not redistributed in this release.
