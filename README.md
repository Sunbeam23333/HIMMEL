<p align="center">
  <img src="assets/himmel-hero.png" alt="HIMMEL — Hierarchical Interleaved Multi-stream Motion Encoding for Long Video Understanding" width="100%">
</p>

# HIMMEL

**Hierarchical Interleaved Multi-stream Motion Encoding for Long Video Understanding**

NeurIPS 2026 · [Conference page](https://neurips.cc/virtual/2026/poster/151828) ·
[Earlier technical report](https://arxiv.org/abs/2605.08158)

HIMMEL combines sparse semantic anchors with a lightweight motion path for long
videos. The motion adapter encodes motion vectors, residual maps and appearance
context, then interleaves projected motion tokens with the host model's visual
tokens. Training uses motion-space alignment followed by instruction tuning
with LoRA.

This is a compact **core-code release**: four host adapter snapshots, the tri-stream
adapter, data loaders, two-stage training and evaluation entry points, and a
standalone multi-codec C extractor. It contains no datasets, checkpoints, private
training logs, credentials or internal infrastructure scripts. Full reproduction
of every paper/rebuttal table still requires the artifacts listed in
[reproduction notes](docs/reproduction.md).

The released appearance-context path is implemented for Qwen2.5-VL. The other
three host snapshots support narrower motion-adapter modes; unsupported
appearance settings fail explicitly rather than claiming four-host parity.

## What is included

| Component | Location |
| --- | --- |
| Tri-stream encoders and staged fusion | [`code/models/compressed_motion_adapter.py`](code/models/compressed_motion_adapter.py) |
| Qwen2.5-VL, Qwen3-VL, InternVL3 and LLaVA-OneVision wrappers | [`code/models/`](code/models/) |
| Compressed-domain and RGB-difference data loading | [`code/data/`](code/data/) |
| Two-stage trainer and archived example configurations | [`code/train/`](code/train/) |
| Video/image inference and prediction export | [`code/eval/eval_himmel.py`](code/eval/eval_himmel.py) |
| Portable multi-codec extraction and validation | [`tools/codec/`](tools/codec/) |
| Exact counts and paired error-case analysis | [`tools/analyze_scored_predictions.py`](tools/analyze_scored_predictions.py) |

## Multi-codec extraction

Build against your installed FFmpeg development libraries; no FFmpeg binaries
or modified full source tree are redistributed.

```bash
make -C tools/codec
tools/codec/extract_codec_signals --help
```

The tool uses libavformat to open the input, selects its video decoder, and
requests exported motion-vector side data. **Native vectors and estimated
vectors are explicitly distinguished.** Decoders without exported vectors use
deterministic decoded-luma block matching. Future-reference/B-frame handling is
also recorded in the output metadata.

The signed residual is a diagnostic difference against the warped **previous
display frame**, not a universal extraction of the codec's original prediction
residual. The C tool is standalone: the archived compressed training loader uses
CoViAR/MPEG-4 and does not silently switch to this tool or to RGB proxies.
See [formats, limitations, build and test instructions](docs/codec_extraction.md).

## Training and evaluation

Use a CUDA environment and the host model's compatible dependencies:

```bash
python -m pip install -r code/requirements.txt
# Install a CUDA-compatible DeepSpeed build for distributed training.
```

The two JSON files under `code/train/configs/` are sanitized **archived examples**.
Set model, dataset, output and Stage-1 checkpoint paths before launching:

```bash
deepspeed --num_gpus=8 code/train/train_himmel.py \
  --config code/train/configs/stage1_qwen_compressed_tristream.json
deepspeed --num_gpus=8 code/train/train_himmel.py \
  --config code/train/configs/stage2_qwen_compressed_tristream.json
```

Do not assume these example hyperparameters identify the paper's final
checkpoint: configuration/checkpoint mapping and the paper's simplified fusion
equation differ from this archived implementation. The precise differences,
data schema, configuration options and evaluation commands are documented in
[reproduction notes](docs/reproduction.md). GPU training and paper benchmark
scores have not been rerun as part of packaging this release.

For MathVista/GSM8K analysis, first score predictions using the declared benchmark
protocol, then run [exact-count and paired-case analysis](docs/scored_predictions.md).
The aggregator validates denominators and matching question IDs; it is not an
inference runner or a substitute for official answer extraction.

## Tests

```bash
python -m pip install numpy
python -m unittest discover -s tests -v
python tools/codec/validate_codecs.py
```

Codec tests require a compiler, `pkg-config`, FFmpeg CLI and development libraries.
The tests use synthetic clips, not private videos or benchmark data. See the
reproduction and codec notes for test dependencies and the distinction between
CPU smoke tests and model-level validation.

## Citation

For the accepted conference paper:

```bibtex
@inproceedings{jin2026himmel,
  title     = {HIMMEL: Hierarchical Interleaved Multi-stream Motion Encoding for Long Video Understanding},
  author    = {Jin, Haopeng and Zhao, Wenlong and Yu, Tao and Yang, Tiankun and Guan, Zhenyu and Dong, ShiQuan and Yang, Yujia and Chen, Chubin and Yi, Hongzhu and Luo, Jinwen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Accepted; proceedings metadata forthcoming},
  url       = {https://neurips.cc/virtual/2026/poster/151828}
}
```

The linked arXiv v1 is an earlier technical report with a different author list,
not an identical copy of the conference camera-ready. Cite the version actually
used. Contact: [Haopeng Jin](mailto:king.ham.roc@gmail.com).

## Licenses and acknowledgments

The repository retains its [Apache-2.0 license](LICENSE). The standalone C
extractor retains its separate [MIT license](tools/codec/LICENSE-MIT). FFmpeg,
CoViAR, host model weights and datasets retain their respective licenses and
must be obtained separately. [Banner artwork notes](assets/README.md) distinguish
decorative fan art from the code and from scientific results.

This work was supported by Tencent through funding and computational resources.
