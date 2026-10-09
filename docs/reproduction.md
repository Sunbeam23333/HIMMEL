# Reproduction scope and experiment recipes

This is a **sanitized core implementation release**, not a checkpoint-exact,
end-to-end reproduction bundle for every paper or rebuttal table. No model
training or benchmark inference was rerun while preparing this release. The
authors' measured results are not substituted with newly generated estimates.

## Read before comparing numbers

| Item | Released implementation | Reproduction consequence |
| --- | --- | --- |
| LoRA | Archived Qwen Stage-2 example uses `r=64`, `alpha=128` | The manuscript default is `r=32`; do not identify this example with that checkpoint. |
| Motion budget | Archived examples use four decoded motion observations per anchor interval | This is not an implementation of a global `K_m=64` cap; changing the number of observations is not sufficient evidence of reproducing the paper's token budget. |
| Fusion | Sequential two-stream gated fusion, described below | More expressive than the manuscript's simplified two-input gate equation; retain the actual trained architecture when loading a matching checkpoint. |
| Python compressed loader | Optional upstream CoViAR; MPEG-4 Part 2 with compatible fixed GOP | Not a generic H.264/PyAV residual reader. |
| Multi-codec C tool | Standalone extraction/diagnostic utility under `tools/codec` | Separate from the legacy Python training loader; see [codec documentation](codec_extraction.md). |
| Checkpoints | Not included | Published checkpoint-to-config hashes still need to be supplied. |

The decoder tool does **not** make the Python training path automatically
multi-codec, and its reconstructed luma prediction error is not the codec's
native transform residual. Never silently substitute these representations in
a checkpoint comparison.

## Included code

- `code/models/compressed_motion_adapter.py`: separate motion-vector and
  residual branches; optional appearance context or independent appearance
  encoder; hierarchical fusion and alignment supervision.
- `code/models/delta_components.py`: ResNet/MotionSSM/SigLIP/shared-ViT
  alternatives, motion projector and contrastive alignment. The optional,
  later LaVIT-VQ exploratory extension was excluded to avoid unrelated code.
- `code/models/{qwen_delta,qwen3_delta,internvl_delta,llava_delta}.py`: four
  host adapter snapshots. Only the Qwen2.5 wrapper in this archive forwards
  the complete appearance-context branch; the other three support their
  archived RGB-diff/MV+residual paths and explicitly reject unsupported
  appearance-branch flags. This is not full tri-stream parity across all
  four hosts. Their compatibility with any newer Transformers release
  requires an actual model-load test; syntax checks alone do not establish it.
- `code/data`: legacy compressed loading, explicit RGB-difference alternative,
  training dataset and host-specific collators.
- `code/train/train_himmel.py`: Stage-1 alignment and Stage-2 SFT/LoRA.
- `code/eval/eval_himmel.py`: baseline/adapter prediction runner for prepared
  Video-MME, MVBench, MLVU, MathVista and MathVision manifests.

### Actual gate implemented in this snapshot

For aligned temporal streams `p,s`, the code constructs
`z = concat(LN(p), LN(s), LN(p)-LN(s), LN(p)*LN(s))` (dimension `4d`),
then `g = sigmoid(Wg z + b)` and
`y = LN(g*p + (1-g)*s + MLP(z))`.
The motion/residual gate is followed by an optional appearance-context gate.
Available controls are `gated`, `concat_mlp`, and `sum`; a simultaneous
two-logit-softmax fusion ablation is **not** included. This actual formula
differs from the manuscript's simplified `2d` gate equation. The public release
does not rewrite the trained architecture to conceal that difference.

## Environment

Use a Linux CUDA machine for full training. Install a mutually compatible
PyTorch, Torchvision, CUDA and DeepSpeed stack, then:

```bash
python -m pip install -r code/requirements.txt
```

The dependency ranges are installation guidance, not a recovered historical
lockfile. `flash-attn` is an optional separately installed native package;
some host-loading paths request `flash_attention_2`, so match those settings
to the machine. InternVL uses `trust_remote_code=True`: review and pin the
upstream revision before executing it. Load only trusted checkpoints.

Model and dataset downloads are intentionally not automated through private
proxy settings. Obtain assets from their official distributors and comply
with their respective licenses. Set `HIMMEL_RESNET18_CHECKPOINT` to the
ImageNet ResNet-18 weights when using that initialization; without the file,
the legacy implementation logs a random-initialization warning. Such a run is
not equivalent to a pretrained-encoder experiment.

## Training

The two JSON files in `code/train/configs` are **archived implementation
recipes** with portable placeholder paths. Set paths in a copy of the file,
or use the environment overrides below. The expected training JSON is a
list of records with the required `video` path relative to `video_root`,
plus `conversations` containing fields `from`/`value`.
Inspect `DeltaVideoDataset` before adapting another dataset format. Do not
publish videos, participant information or private annotations in this repo.

```bash
export HIMMEL_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export HIMMEL_DATA_PATH_OVERRIDE=/path/to/stage1_caption.json
export HIMMEL_VIDEO_ROOT_OVERRIDE=/path/to/compatible_mpeg4_videos
export HIMMEL_OUTPUT_DIR_OVERRIDE=/path/to/checkpoints/stage1
deepspeed --num_gpus 8 code/train/train_himmel.py \
  --config code/train/configs/stage1_qwen_compressed_tristream.json

export HIMMEL_DATA_PATH_OVERRIDE=/path/to/stage2_sft.json
export HIMMEL_OUTPUT_DIR_OVERRIDE=/path/to/checkpoints/stage2
export HIMMEL_STAGE1_CHECKPOINT=/path/to/checkpoints/stage1/final/motion_module.pt
deepspeed --num_gpus 8 code/train/train_himmel.py \
  --config code/train/configs/stage2_qwen_compressed_tristream.json
```

The GPU count is an example, not a fresh benchmark. Effective batch size is
`per_device_batch_size × number_of_GPUs × gradient_accumulation_steps`.
The default cache can be redirected with `HIMMEL_CACHE_DIR`.

For a **new**, explicitly labeled rank-controlled experiment:

```bash
python code/train/make_ablation_config.py \
  code/train/configs/stage2_qwen_compressed_tristream.json stage2_r32.json \
  --set lora_r=32 --set lora_alpha=64
```

The generator refuses to overwrite an existing file. Other meaningful
controls include `use_motion_vectors`, `use_residuals`,
`compressed_use_iframe_context`, `compressed_fusion_mode`,
`compressed_*_alignment_weight`, `alignment_motion_pool`,
`max_keyframes`, `num_diffs_per_interval`, and the two accumulation flags.
Only vary one factor when making a controlled comparison. Record seeds,
data revisions, checkpoint hashes, precision and decoding settings yourself.
No unseen historical config or manifest is reconstructed by this tool.

## Prediction runner and scoring

Prepare benchmark JSON **without changing the official split**. The runner
looks below `HIMMEL_BENCHMARK_DIR` for:

```text
Video-MME/video_mme_eval.json  and Video-MME/videos/
MVBench/mvbench_eval.json     and MVBench/videos/
MLVU/mlvu_eval.json           and MLVU/videos/
MathVista/mathvista_eval.json and MathVista/images/
MathVision/mathvision_eval.json and MathVision/images/
```

Video rows use `video`, `question`, `options`, `answer` and a task/category
field. Image rows use `image`, `question`, `options` (if multiple choice),
`answer`, `question_type`, and `answer_type`. Preserve original sample IDs
in your source manifest and retain a stable mapping to runner `idx`.

Example for a checkpoint actually trained with the archived context branch:

```bash
export HIMMEL_BENCHMARK_DIR=/path/to/benchmarks
python code/eval/eval_himmel.py --model-type qwen \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --delta-checkpoint /path/to/checkpoints/stage2/final \
  --benchmark video_mme --motion-input-mode compressed \
  --compressed-backend coviar --visual-stream-mode iframe \
  --motion-encoder-type motionssm --compressed-use-iframe-context 1 \
  --compressed-fusion-mode gated --alignment-motion-pool linear \
  --accumulate-motion-vectors 1 --accumulate-residuals 1 \
  --max-keyframes 32 --output-dir /path/to/eval_outputs
```

Omit `--delta-checkpoint` for the original host baseline. Do not use the
compressed settings for a model trained on a different representation.
Evaluation records include raw responses for downstream rescoring. The
runner's printed number is a **local diagnostic micro accuracy**:

- For MLVU, use its official task subset and M-Avg computation, not a pooled
  micro accuracy across arbitrary tasks.
- For MathVista and MathVision, use the official extraction/normalization
  and scorer. The local free-form comparison is deliberately not advertised
  as the official metric.
- MathVista `testmini` has 1,000 examples; GSM8K test has 1,319. A GSM8K
  inference runner and the historical math-rehearsal mixture manifest are
  not included in this snapshot. An exact-count aggregation script cannot
  substitute for inference or official answer extraction.
- Wall-clock statistics here are diagnostics, not a synchronized CUDA TTFT
  or end-to-end efficiency reproduction protocol.

The experimental `dopamine_hierarchical` selector uses RGB differences and
question heuristics. It is **not** the paper's codec-energy selector, nor an
implementation of AKS, QuoTA or BOLT. Local approximate training-free
baselines were excluded; use their official implementations.

## Still required for a complete rebuttal artifact release

The frozen Stage-1 interval manifest, source-level counts and historical
motion-balanced sampling rule promised during discussion have not been
located in the material used for this release. Also pending are the exact
checkpoint/config mapping, historical optical-flow control runs,
codec-energy selector/baseline implementations, complete math-rehearsal
data provenance, and the synchronized efficiency runner. These omissions
are listed explicitly rather than replaced with newly invented artifacts.

## Release hardening and verification

Only an allowlisted core was copied; logs, video data, weights, operational
download/sync scripts, private paths and credentials were excluded. Relative
model internals were preserved. Changes from the internal implementation:

- portable cache and benchmark roots, with configuration/environment inputs;
- training defaults to **no RGB proxy fallback**; evaluation fails on
  incompatible compressed inputs, decode failure or inference failure;
- training also fails on extraction errors or empty decoded-frame outputs
  instead of silently substituting an empty sample;
- SigLIP load failures stop the run instead of substituting a random CNN;
- invalid multiple-choice output no longer defaults to option A;
- the local diagnostic free-form matcher no longer accepts arbitrary
  substring matches or a broad relative-error tolerance;
- evaluation exposes the appearance-context/encoder flags;
- non-Qwen2.5 wrappers no longer reference an undefined context flag, and
  explicitly reject branches whose inputs they do not forward;
- fusion truncates residual branches consistently for unequal sequence
  lengths (without changing equal-length behavior).

Verification during packaging: Python syntax/AST checks, JSON config checks,
NumPy encoding/sampling tests and answer-parser tests. No CUDA training,
host-model loading or official benchmark evaluation was performed. Run:

```bash
python -m unittest discover -s tests -p 'test_core*.py' -v
python -m compileall -q code
```
