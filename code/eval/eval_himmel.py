#!/usr/bin/env python3
"""
Himmel-VLM: 统一评测脚本
============================
支持四种基线模型和对应的 Delta (双流) 训练后 checkpoint:
  - Qwen2.5-VL-7B-Instruct
  - Qwen3-VL-8B-Instruct
  - LLaVA-OneVision-7B
  - InternVL3-8B

评测 benchmark:
  - Video-MME (长/中/短视频综合理解)
  - MVBench (20 种时序理解任务)
  - MLVU (多任务长视频理解)
  - MathVista (视觉数学推理, 图像)
  - MathVision (多模态数学竞赛题, 图像)

支持多卡并行评测和 CPU/GPU 异步 pipeline:
  - 数据自动分片到各 GPU, 评测后聚合结果
  - CPU 预处理线程与 GPU 推理异步执行, 消除 GPU 空闲气泡

用法:
    # 单卡评测 (兼容旧方式)
    python eval/eval_himmel.py --model-type qwen --model-path /path/to/model --benchmark video_mme

    # 多卡评测 (推荐: 8 卡并行)
    torchrun --nproc_per_node=8 eval/eval_himmel.py --model-type qwen --model-path /path/to/model --benchmark all

    # Delta 评测
    torchrun --nproc_per_node=8 eval/eval_himmel.py --model-type qwen --model-path /path/to/model \
        --delta-checkpoint /path/to/checkpoints/qwen_delta_stage2/final --benchmark video_mme

    # 只评测图像类 benchmark
    torchrun --nproc_per_node=4 eval/eval_himmel.py --model-type qwen --model-path /path/to/model --benchmark all_image

    # 控制预取队列深度 (默认 2)
    torchrun --nproc_per_node=8 eval/eval_himmel.py --model-type qwen --model-path /path/to/model \
        --benchmark all --prefetch-depth 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_ROOT))

from video_sampling import extract_video_observations, build_sampling_prompt_prefix
def run_training_free(**kwargs):
    raise NotImplementedError("Third-party baseline reimplementations are not part of this release; use their official repositories.")

logger = logging.getLogger(__name__)

# ============================================================================
# 路径配置
# ============================================================================
BENCHMARK_DIR = os.environ.get("HIMMEL_BENCHMARK_DIR", str(PROJECT_ROOT.parent / "benchmarks"))

BENCHMARK_CONFIGS = {
    # ---- 视频类 benchmark ----
    "video_mme": {
        "eval_json": f"{BENCHMARK_DIR}/Video-MME/video_mme_eval.json",
        "video_dir": f"{BENCHMARK_DIR}/Video-MME/videos",
        "type": "video",
    },
    "mvbench": {
        "eval_json": f"{BENCHMARK_DIR}/MVBench/mvbench_eval.json",
        "video_dir": f"{BENCHMARK_DIR}/MVBench/videos",
        "type": "video",
    },
    "mlvu": {
        "eval_json": f"{BENCHMARK_DIR}/MLVU/mlvu_eval.json",
        "video_dir": f"{BENCHMARK_DIR}/MLVU/videos",
        "type": "video",
    },
    # ---- 图像类 benchmark ----
    "mathvista": {
        "eval_json": f"{BENCHMARK_DIR}/MathVista/mathvista_eval.json",
        "image_dir": f"{BENCHMARK_DIR}/MathVista/images",
        "type": "image",
    },
    "mathvision": {
        "eval_json": f"{BENCHMARK_DIR}/MathVision/mathvision_eval.json",
        "image_dir": f"{BENCHMARK_DIR}/MathVision/images",
        "type": "image",
    },
}

VIDEO_BENCHMARKS = [k for k, v in BENCHMARK_CONFIGS.items() if v["type"] == "video"]
IMAGE_BENCHMARKS = [k for k, v in BENCHMARK_CONFIGS.items() if v["type"] == "image"]


# ============================================================================
# 分布式工具
# ============================================================================

def _is_dist() -> bool:
    """检查是否在分布式环境中。"""
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _shard_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将数据按 rank 分片, 尽量均匀。"""
    rank = _get_rank()
    world = _get_world_size()
    return [item for i, item in enumerate(data) if i % world == rank]


def _gather_results(local_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    在所有 rank 上收集评测结果 (通过 all_gather_object)。
    返回按原始 idx 排序的完整结果列表。
    """
    if not _is_dist():
        return sorted(local_results, key=lambda r: r["idx"])

    gathered: List[Optional[List[Dict[str, Any]]]] = [None] * _get_world_size()
    dist.all_gather_object(gathered, local_results)
    all_results: List[Dict[str, Any]] = []
    for part in gathered:
        if part is not None:
            all_results.extend(part)
    all_results.sort(key=lambda r: r["idx"])
    return all_results


# ============================================================================
# 异步预取 Pipeline (CPU 预处理 + GPU 推理)
# ============================================================================

_SENTINEL = None  # 预取线程结束哨兵


def _get_prefetched_item(prefetcher):
    """Do not hang indefinitely if a decoding worker exits unexpectedly."""
    while True:
        try:
            return prefetcher._queue.get(timeout=1.0)
        except queue.Empty:
            if not prefetcher._thread.is_alive():
                raise RuntimeError("Prefetch worker exited without a completion sentinel; inspect the decode error.")

class _VideoPrefetcher:
    """
    视频预处理异步预取器。
    后台线程负责视频解码 + 差分图计算 (CPU 密集), 结果放入 queue 供主线程 GPU 推理。
    """
    def __init__(
        self,
        items: List[Dict[str, Any]],
        max_keyframes: int = 32,
        prefetch_depth: int = 2,
        sampling_config: Optional[Dict[str, Any]] = None,
    ):
        self._items = items
        self._max_keyframes = max_keyframes
        self._sampling_config = dict(sampling_config or {})
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=prefetch_depth)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        for item in self._items:
            video_name = item.get("video", "")
            video_dir = item["_video_dir"]
            video_path = os.path.join(video_dir, video_name)
            if not os.path.isfile(video_path):
                for ext in (".mp4", ".mkv", ".webm", ".avi"):
                    candidate = video_path + ext
                    if os.path.isfile(candidate):
                        video_path = candidate
                        break
                if not os.path.isfile(video_path):
                    found = list(Path(video_dir).rglob(f"{video_name}*"))
                    if found:
                        video_path = str(found[0])

            t0 = time.time()
            keyframes, diff_frames, motion_vectors, residuals, kf_indices, sampling_meta = extract_keyframes_and_diffs_for_eval(
                video_path,
                max_keyframes=self._max_keyframes,
                question=item.get("question", ""),
                options=item.get("options", []),
                **self._sampling_config,
            )
            preprocess_time = time.time() - t0

            sampling_meta = dict(sampling_meta or {})
            sampling_meta["keyframe_indices"] = kf_indices
            sampling_meta["video_path"] = video_path
            self._queue.put((item, keyframes, diff_frames, motion_vectors, residuals, preprocess_time, sampling_meta))
        self._queue.put(_SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        result = _get_prefetched_item(self)
        if result is _SENTINEL:
            raise StopIteration
        return result


class _ImagePrefetcher:
    """
    图像预处理异步预取器。
    后台线程负责图像读取 + PIL 解码 (IO 密集), 结果放入 queue 供主线程 GPU 推理。
    """
    def __init__(self, items: List[Dict[str, Any]], prefetch_depth: int = 4):
        self._items = items
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=prefetch_depth)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        for item in self._items:
            image_name = item.get("image", "")
            image_dir = item["_image_dir"]
            image = None
            if image_name:
                img_path = os.path.join(image_dir, image_name)
                if os.path.isfile(img_path):
                    try:
                        image = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        logger.warning("图像加载失败 %s: %s", img_path, e)
            self._queue.put((item, image))
        self._queue.put(_SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        result = _get_prefetched_item(self)
        if result is _SENTINEL:
            raise StopIteration
        return result
# ============================================================================

def extract_keyframes_and_diffs_for_eval(
    video_path: str,
    keyframe_fps: float = 1.0,
    num_diffs_per_interval: int = 4,
    max_keyframes: int = 32,
    question: str = "",
    options: Optional[List[str]] = None,
    sampling_strategy: str = "uniform",
    memory_keyframes: int = 12,
    dense_local_frames: int = 3,
    question_aware_sampling: bool = True,
    memory_diffs_per_interval: int = 1,
    dense_diffs_per_interval: int = 4,
    diff_mode: str = "abs",
    motion_input_mode: str = "rgb_diff",
    compressed_backend: str = "coviar",
    visual_stream_mode: str = "iframe",
    compressed_gop_size: int = 12,
    use_motion_vectors: bool = True,
    use_residuals: bool = True,
    accumulate_motion_vectors: bool = True,
    accumulate_residuals: bool = False,
    motion_vector_clip: float = 20.0,
) -> Tuple[
    List[Image.Image],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    List[int],
    Dict[str, Any],
]:
    """
    提取关键帧和运动输入 (评测用)。

    Returns:
        keyframes: List[PIL.Image]
        diff_frames: Tensor[T_diff, 3, 224, 224] or None
        motion_vectors: Tensor[T_mv, 3, 224, 224] or None
        residuals: Tensor[T_res, 3, 224, 224] or None
        keyframe_indices: List[int]
        sampling_meta: Dict[str, Any]
    """
    try:
        return extract_video_observations(
            video_path=video_path,
            keyframe_fps=keyframe_fps,
            num_diffs_per_interval=num_diffs_per_interval,
            max_keyframes=max_keyframes,
            question=question,
            options=options,
            sampling_strategy=sampling_strategy,
            memory_keyframes=memory_keyframes,
            dense_local_frames=dense_local_frames,
            question_aware_sampling=question_aware_sampling,
            memory_diffs_per_interval=memory_diffs_per_interval,
            dense_diffs_per_interval=dense_diffs_per_interval,
            diff_mode=diff_mode,
            motion_input_mode=motion_input_mode,
            compressed_backend=compressed_backend,
            visual_stream_mode=visual_stream_mode,
            compressed_gop_size=compressed_gop_size,
            use_motion_vectors=use_motion_vectors,
            use_residuals=use_residuals,
            accumulate_motion_vectors=accumulate_motion_vectors,
            accumulate_residuals=accumulate_residuals,
            motion_vector_clip=motion_vector_clip,
        )
    except Exception as e:
        logger.warning("视频处理失败 %s: %s", video_path, e)
        return (
            [],
            None,
            None,
            None,
            [],
            {
                "strategy": sampling_strategy,
                "error": str(e),
                "diff_mode": diff_mode,
                "motion_input_mode": motion_input_mode,
            },
        )


def _get_motion_token_count(
    diff_frames: Optional[torch.Tensor] = None,
    motion_vectors: Optional[torch.Tensor] = None,
    residuals: Optional[torch.Tensor] = None,
) -> int:
    for tensor in (motion_vectors, residuals, diff_frames):
        if tensor is not None:
            return int(tensor.shape[0])
    return 0


# ============================================================================
# 多选题答案提取
# ============================================================================

def extract_answer_letter(response: str, num_options: int = 4) -> str:
    """
    从模型回复中提取选项字母 (A/B/C/D/...)。

    策略 (优先级从高到低):
    1. 精确匹配 "The answer is X" / "答案是 X"
    2. 匹配 "(X)" 模式
    3. 回复的第一个字母如果是合法选项
    4. 默认返回 "A"
    """
    response = response.strip()
    valid_letters = [chr(ord('A') + i) for i in range(num_options)]

    # 策略 1: "The answer is X" 类模式
    patterns = [
        r"(?:the\s+)?answer\s+is\s*[:：]?\s*\(?([A-Z])\)?",
        r"答案\s*[:：是]\s*\(?([A-Z])\)?",
        r"选\s*\(?([A-Z])\)?",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m and m.group(1).upper() in valid_letters:
            return m.group(1).upper()

    # 策略 2: 匹配 "(X)" 或 "X." 或 "X)"
    m = re.search(r"\(([A-Z])\)", response)
    if m and m.group(1) in valid_letters:
        return m.group(1)

    m = re.search(r"\b([A-Z])\.", response)
    if m and m.group(1) in valid_letters:
        return m.group(1)

    # 策略 3: 回复的第一个大写字母
    for ch in response:
        if ch.upper() in valid_letters:
            return ch.upper()

    return ""  # Unparseable output is not a guessed option A.


def extract_free_form_answer(response: str) -> str:
    """
    从模型回复中提取自由格式答案 (用于 MathVista/MathVision 的 free_form 题)。

    策略:
    1. 查找 "The answer is ..." 模式
    2. 查找最后出现的数字 (含小数)
    3. 返回原始回复的最后一行
    """
    response = response.strip()

    # 策略 1: "the answer is ..." 模式
    patterns = [
        r"(?:the\s+)?answer\s+is\s*[:：]?\s*(.+?)[\.\n]",
        r"答案\s*[:：是]\s*(.+?)[\.\n]",
        r"=\s*(.+?)$",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # 策略 2: 查找最后出现的数字
    numbers = re.findall(r"-?\d+\.?\d*", response)
    if numbers:
        return numbers[-1]

    # 策略 3: 最后一行
    lines = response.strip().split("\n")
    return lines[-1].strip() if lines else response


# ============================================================================
# 模型加载
# ============================================================================

def load_model_baseline(model_type: str, model_path: str, device: str = "cuda"):
    """加载基线模型 (无 motion module)。"""

    if model_type == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=128 * 28 * 28, max_pixels=768 * 28 * 28,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            device_map=device,
        )
        return model, processor, processor.tokenizer

    elif model_type == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=128 * 32 * 32, max_pixels=768 * 32 * 32,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            device_map=device,
        )
        return model, processor, processor.tokenizer

    elif model_type == "llava":
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            device_map=device,
        )
        return model, processor, processor.tokenizer

    elif model_type == "internvl3":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            low_cpu_mem_usage=False,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, None, tokenizer

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def _infer_motion_encoder_type(delta_checkpoint: Optional[str]) -> str:
    ckpt_text = (delta_checkpoint or "").lower()
    for encoder_type in ("shared_vit", "siglip", "motionssm", "resnet"):
        if encoder_type in ckpt_text:
            return encoder_type
    return "resnet"


def _build_delta_hf_config(
    model_type: str,
    model_path: str,
    delta_model_config: Optional[Dict[str, Any]] = None,
):
    delta_model_config = dict(delta_model_config or {})
    if not delta_model_config or model_type == "internvl3":
        return None

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config_overrides = {
        "motion_encoder_type": delta_model_config.get("motion_encoder_type", "resnet"),
        "siglip_model_path": delta_model_config.get("siglip_model_path", "google/siglip-base-patch16-224"),
        "freeze_motion_encoder": delta_model_config.get("freeze_motion_encoder", False),
        "motion_use_cross_attention": delta_model_config.get("motion_use_cross_attention", False),
        "motion_input_mode": delta_model_config.get("motion_input_mode", "rgb_diff"),
        "use_motion_vectors": delta_model_config.get("use_motion_vectors", True),
        "use_residuals": delta_model_config.get("use_residuals", True),
        "compressed_fusion_mode": delta_model_config.get("compressed_fusion_mode", "gated"),
        "compressed_use_iframe_context": delta_model_config.get("compressed_use_iframe_context", False),
        "compressed_use_iframe_encoder": delta_model_config.get("compressed_use_iframe_encoder", False),
        "alignment_motion_pool": delta_model_config.get("alignment_motion_pool", "mean"),
    }
    for key, value in config_overrides.items():
        setattr(hf_config, key, value)

    if "motion_vit_hidden_size" in delta_model_config:
        hf_config.motion_vit_hidden_size = delta_model_config["motion_vit_hidden_size"]
    elif model_type in {"qwen", "qwen3"}:
        hf_config.motion_vit_hidden_size = 1280
    elif model_type == "llava":
        hf_config.motion_vit_hidden_size = 1152

    return hf_config


def _build_delta_model_kwargs(
    model_type: str,
    model_path: str,
    delta_model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hf_config = _build_delta_hf_config(model_type, model_path, delta_model_config)
    return {"config": hf_config} if hf_config is not None else {}


def _normalized_tensor_to_images(
    frames: Optional[torch.Tensor],
    diff_mode: str = "abs",
) -> List[Image.Image]:
    if frames is None:
        return []

    tensor = frames
    if tensor.ndim == 5:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        return []

    frames_cpu = tensor.detach().cpu().float()
    if diff_mode in {"signed", "signed_accum", "luma_motion_edge"}:
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=frames_cpu.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=frames_cpu.dtype).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=frames_cpu.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=frames_cpu.dtype).view(1, 3, 1, 1)

    frames_cpu = (frames_cpu * std + mean).clamp(0, 1)
    return [
        Image.fromarray((frame.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
        for frame in frames_cpu
    ]


def _convert_deepspeed_checkpoint(
    ckpt_dir: str, tag: str = "latest", output_dir: str | None = None,
) -> None:
    """
    从 DeepSpeed 原生 checkpoint 中恢复完整 state_dict，
    然后提取并保存 motion_module 权重，同时保存完整 merged state_dict
    用于后续直接 load_state_dict。

    转换后会在 output_dir (默认=ckpt_dir) 下生成:
      - motion_module.pt              (motion_module 参数, 已去前缀)
      - merged_state_dict.pt          (完整模型 state_dict, 已合并 LoRA)

    注意: 在多卡 (torchrun) 环境下, 只有 rank 0 执行转换, 其他 rank 等待。

    Args:
        ckpt_dir: 包含 global_step* 子目录的 checkpoint 目录 (如 best/)
        tag: 使用哪个 checkpoint tag, 默认 "latest"
        output_dir: 转换产物输出目录, 默认与 ckpt_dir 相同。
                    当指定特定 global_step 时, 可将产物放到该 step 目录下,
                    避免不同 step 之间互相覆盖。
    """
    if output_dir is None:
        output_dir = ckpt_dir

    rank = _get_rank()

    # ---- 只让 rank 0 执行转换, 其他 rank 跳过并等待 barrier ----
    if rank != 0:
        logger.info("[rank %d] 等待 rank 0 完成 DeepSpeed checkpoint 转换...", rank)
        if _is_dist():
            dist.barrier()
        return

    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    logger.info("=" * 60)
    logger.info("[rank 0] 检测到 DeepSpeed 原生 checkpoint, 自动转换中...")
    logger.info("目录: %s, tag: %s, 输出: %s", ckpt_dir, tag, output_dir)
    logger.info("=" * 60)

    # 获取完整 FP32 state dict (这一步非常耗时, 需读取所有分片并合并)
    logger.info("[rank 0] 正在读取并合并 %s 的分片 checkpoint, 这可能需要几分钟...", tag)
    t0 = time.time()
    state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir, tag=tag)
    logger.info(
        "[rank 0] 成功提取 state_dict, 共 %d 个 key, 耗时 %.1f 秒",
        len(state_dict), time.time() - t0,
    )

    # --- 1) 分离并保存 motion_module 权重 ---
    motion_keys = [k for k in state_dict if "motion_module." in k]
    if not motion_keys:
        raise RuntimeError(
            f"在 DeepSpeed checkpoint 中未找到 motion_module 相关参数。"
            f"前 10 个 key: {list(state_dict.keys())[:10]}"
        )

    motion_sd = {}
    for k in motion_keys:
        # key 格式: "motion_module.xxx" 或 "base_model.model.motion_module.xxx" (PeftModel)
        idx = k.index("motion_module.") + len("motion_module.")
        motion_sd[k[idx:]] = state_dict[k]

    os.makedirs(output_dir, exist_ok=True)
    motion_pt_path = os.path.join(output_dir, "motion_module.pt")
    torch.save(motion_sd, motion_pt_path)
    logger.info("已保存 motion_module.pt (%d 参数) -> %s", len(motion_sd), motion_pt_path)

    # --- 2) 保存完整 merged state_dict ---
    # 如果含 LoRA 参数 (PeftModel 包裹), 做 LoRA merge:
    #   base_model.model.xxx.base_layer.weight + lora_A + lora_B -> xxx.weight
    has_lora = any("lora_" in k for k in state_dict)

    if has_lora:
        logger.info("检测到 LoRA 参数, 正在合并 LoRA 到 base 权重...")
        merged_sd = _merge_lora_state_dict(state_dict)
    else:
        # Full SFT 或无 LoRA: 直接清理前缀
        merged_sd = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            merged_sd[new_key] = v

    merged_path = os.path.join(output_dir, "merged_state_dict.pt")
    torch.save(merged_sd, merged_path)
    logger.info("已保存 merged_state_dict.pt (%d 参数) -> %s", len(merged_sd), merged_path)

    del state_dict, merged_sd
    logger.info("[rank 0] DeepSpeed checkpoint 转换完成! 总耗时 %.1f 秒", time.time() - t0)

    # rank 0 转换完毕, 释放 barrier 让其他 rank 继续
    if _is_dist():
        dist.barrier()


def _merge_lora_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 PeftModel 包裹的 state_dict 中的 LoRA 权重合并到 base 权重中，
    返回干净的 (无 LoRA / 无 base_model.model 前缀) state_dict。

    PeftModel state_dict key 格式示例:
      base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
      base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
      base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
      base_model.model.motion_module.encoder.features.0.weight

    合并后:
      model.layers.0.self_attn.q_proj.weight  (= base + B @ A * scaling)
      motion_module.encoder.features.0.weight
    """
    prefix = "base_model.model."

    # 收集所有 LoRA A/B 权重, 按 module 分组
    lora_a: Dict[str, Any] = {}  # module_key -> tensor
    lora_b: Dict[str, Any] = {}
    base_weights: Dict[str, Any] = {}  # module_key -> tensor (base_layer.weight)
    other: Dict[str, Any] = {}  # 非 LoRA 相关的参数

    for k, v in state_dict.items():
        # 去掉 PeftModel 前缀
        clean_k = k[len(prefix):] if k.startswith(prefix) else k

        if ".lora_A." in k:
            # extract module key: xxx.lora_A.default.weight -> xxx
            module_key = clean_k.split(".lora_A.")[0]
            lora_a[module_key] = v
        elif ".lora_B." in k:
            module_key = clean_k.split(".lora_B.")[0]
            lora_b[module_key] = v
        elif ".base_layer." in k:
            # xxx.base_layer.weight -> xxx.weight
            module_key = clean_k.replace(".base_layer.", ".")
            base_weights[module_key] = v
        else:
            other[clean_k] = v

    # 合并 LoRA: W_merged = W_base + B @ A * scaling
    # scaling = lora_alpha / lora_rank (默认 peft 通常 alpha=rank, 所以 scaling=1)
    # DeepSpeed checkpoint 中 LoRA A/B 已经包含了 scaling, 直接 B @ A 即可
    merged = dict(other)
    for module_key, base_w in base_weights.items():
        if module_key.endswith(".weight") and module_key.rsplit(".weight", 1)[0] in lora_a:
            prefix_key = module_key.rsplit(".weight", 1)[0]
            a = lora_a[prefix_key]
            b = lora_b.get(prefix_key)
            if b is not None:
                # LoRA merge: W = W_base + B @ A
                delta = b.float() @ a.float()
                merged[module_key] = (base_w.float() + delta).to(base_w.dtype)
                continue
        merged[module_key] = base_w

    # 处理有 lora_A/B 但没有 base_layer 的情况 (不太常见)
    for module_key in lora_a:
        weight_key = module_key + ".weight"
        if weight_key not in merged and weight_key in other:
            a = lora_a[module_key]
            b = lora_b.get(module_key)
            if b is not None:
                delta = b.float() @ a.float()
                merged[weight_key] = (other[weight_key].float() + delta).to(other[weight_key].dtype)

    logger.info(
        "LoRA merge 完成: %d base weights, %d LoRA modules, %d other params",
        len(base_weights), len(lora_a), len(other),
    )
    return merged


def _resolve_delta_paths(
    delta_checkpoint: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    从 delta_checkpoint 路径解析出各类权重路径。

    支持的目录格式:
      1. final/ 格式:  final/motion_module.pt + final/lora_weights/ (LoRA 模式)
                       final/motion_module.pt + final/full_model/   (Full 模式)
      2. checkpoint-N/ 格式: 同上结构
      3. best/ 格式: 同上结构
      4. DeepSpeed 原生 checkpoint (global_step*/): 自动转换后加载 merged_state_dict

    Returns:
        (motion_module_path, lora_dir_or_none, full_model_dir_or_none, merged_sd_path_or_none)
        - merged_sd_path: 从 DeepSpeed 原生 checkpoint 转换得到的完整合并 state_dict (.pt 文件)
          当此路径非 None 时, 优先使用 merged_sd 加载方式 (无需 lora_dir / full_model_dir)
    """
    if not os.path.isdir(delta_checkpoint):
        # 直接指向一个 .pt 文件
        return delta_checkpoint, None, None, None

    def _try_resolve(
        path: str,
    ) -> Optional[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]]:
        """尝试在指定目录中解析 delta 权重。"""
        motion_pt = os.path.join(path, "motion_module.pt")
        if os.path.isfile(motion_pt):
            lora_dir = os.path.join(path, "lora_weights")
            full_model_dir = os.path.join(path, "full_model")
            merged_sd = os.path.join(path, "merged_state_dict.pt")
            lora = lora_dir if os.path.isdir(lora_dir) else None
            full = full_model_dir if os.path.isdir(full_model_dir) else None
            merged = merged_sd if os.path.isfile(merged_sd) else None
            return motion_pt, lora, full, merged
        return None

    # 1) 直接在 delta_checkpoint 目录中查找
    result = _try_resolve(delta_checkpoint)
    if result is not None:
        return result

    # 2) 如果用户指向了 global_step* 子目录 (如 best/global_step199)
    dirname = os.path.basename(delta_checkpoint.rstrip("/"))
    if dirname.startswith("global_step"):
        parent = os.path.dirname(delta_checkpoint.rstrip("/"))

        # 2a) 先检查该 global_step 目录本身是否已有转换产物
        result = _try_resolve(delta_checkpoint)
        if result is not None:
            logger.info(
                "路径 '%s' 是 DeepSpeed global_step 目录, 已找到该 step 的导出权重",
                delta_checkpoint,
            )
            return result

        # 2b) 再检查上级目录是否有转换产物 (兼容旧逻辑)
        result = _try_resolve(parent)
        if result is not None:
            logger.info(
                "路径 '%s' 是 DeepSpeed global_step 目录, 自动在上级目录 '%s' 找到导出权重",
                delta_checkpoint, parent,
            )
            return result

        # 2c) 都没有: 使用该 step 的 tag 进行转换, 产物放在该 step 目录下
        logger.info(
            "指定 global_step '%s', 对上级目录 '%s' 使用 tag='%s' 进行 DeepSpeed 转换...",
            dirname, parent, dirname,
        )
        _convert_deepspeed_checkpoint(
            parent, tag=dirname,
            output_dir=delta_checkpoint.rstrip("/"),
        )

        result = _try_resolve(delta_checkpoint)
        if result is not None:
            logger.info("DeepSpeed checkpoint 转换成功 (step=%s)", dirname)
            return result

        raise FileNotFoundError(
            f"DeepSpeed checkpoint 自动转换完成, 但仍未找到 motion_module.pt: {delta_checkpoint}"
        )

    # 3) 目录内包含 global_step* 子目录但没有 motion_module.pt — DeepSpeed 原生 checkpoint
    #    自动转换为标准格式
    gs_dirs = sorted(
        [d for d in os.listdir(delta_checkpoint) if d.startswith("global_step")],
        key=lambda x: int(x.replace("global_step", "")) if x.replace("global_step", "").isdigit() else 0,
    )
    if gs_dirs:
        logger.info(
            "Delta checkpoint '%s' 是 DeepSpeed 原生格式 (找到 %s), 尝试自动转换...",
            delta_checkpoint, gs_dirs,
        )
        # 自动转换: 使用 "latest" tag (即最新的 global_step)
        _convert_deepspeed_checkpoint(delta_checkpoint, tag="latest")

        # 转换后重新解析
        result = _try_resolve(delta_checkpoint)
        if result is not None:
            logger.info("DeepSpeed checkpoint 转换成功, 已解析 delta 权重")
            return result

        # 转换完成但仍找不到 (不太可能, 但兜底)
        raise FileNotFoundError(
            f"DeepSpeed checkpoint 自动转换完成, 但仍未找到 motion_module.pt: {delta_checkpoint}"
        )

    raise FileNotFoundError(
        f"Delta checkpoint 目录 '{delta_checkpoint}' 中找不到 motion_module.pt"
    )


def load_model_delta(
    model_type: str,
    model_path: str,
    delta_checkpoint: str,
    device: str = "cuda",
    delta_model_config: Optional[Dict[str, Any]] = None,
):
    """
    加载 Delta 模型 (motion_module + LoRA 或 motion_module + Full Model)。

    支持的 delta_checkpoint 格式:
        final/                          # 训练结束导出
        ├── motion_module.pt
        ├── lora_weights/               # LoRA 模式
        └── full_model/                 # Full SFT 模式

        checkpoint-1000/                # 训练中间 checkpoint
        ├── motion_module.pt
        ├── lora_weights/  或  full_model/
        └── global_step1000/            # DeepSpeed 原生文件 (不使用)

        best/                           # Early Stopping 最优 checkpoint
        ├── motion_module.pt
        └── lora_weights/  或  full_model/

        best/                           # 老版本: 只有 DeepSpeed 原生 checkpoint
        ├── global_step*/               # 自动转换 → motion_module.pt + merged_state_dict.pt
        └── latest
    """
    delta_model_config = dict(delta_model_config or {})
    motion_pt, lora_dir, full_model_dir, merged_sd_path = _resolve_delta_paths(delta_checkpoint)

    # --- 优先级 1: merged_state_dict (从 DeepSpeed 原生 checkpoint 自动转换) ---
    if merged_sd_path is not None:
        logger.info("使用 merged_state_dict 加载: %s", merged_sd_path)
        return _load_model_from_merged_sd(
            model_type,
            model_path,
            motion_pt,
            merged_sd_path,
            device,
            delta_model_config=delta_model_config,
        )

    # --- 优先级 2: Full SFT 模式 (full_model/ 目录) ---
    if full_model_dir is not None:
        logger.info("检测到 Full SFT checkpoint, 从 %s 加载完整模型", full_model_dir)
        return _load_model_full(
            model_type, full_model_dir, motion_pt, device,
            delta_model_config=delta_model_config,
        )

    # --- 优先级 3: LoRA 模式 (baseline + motion_module + lora_weights) ---
    if model_type == "qwen":
        from transformers import AutoProcessor
        from models.qwen_delta import QwenDeltaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=128 * 28 * 28, max_pixels=768 * 28 * 28,
        )
        model = QwenDeltaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            **_build_delta_model_kwargs("qwen", model_path, delta_model_config),
        )
        model.motion_module.reload_pretrained_weights()
        model.setup_motion_token_ids(processor.tokenizer)
        tokenizer = processor.tokenizer

    elif model_type == "qwen3":
        from transformers import AutoProcessor
        from models.qwen3_delta import Qwen3DeltaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=128 * 32 * 32, max_pixels=768 * 32 * 32,
        )
        model = Qwen3DeltaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            **_build_delta_model_kwargs("qwen3", model_path, delta_model_config),
        )
        model.motion_module.reload_pretrained_weights()
        model.setup_motion_token_ids(processor.tokenizer)
        tokenizer = processor.tokenizer

    elif model_type == "llava":
        from transformers import AutoProcessor
        from models.llava_delta import LlavaDeltaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = LlavaDeltaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            **_build_delta_model_kwargs("llava", model_path, delta_model_config),
        )
        model.motion_module.reload_pretrained_weights()
        model.setup_motion_token_ids(processor.tokenizer)
        tokenizer = processor.tokenizer

    elif model_type == "internvl3":
        from transformers import AutoTokenizer
        from models.internvl_delta import load_internvl3_delta

        model = load_internvl3_delta(model_path, dtype=torch.bfloat16, **delta_model_config)
        model.motion_module.reload_pretrained_weights()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.setup_motion_token_ids(tokenizer)
        processor = None

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 加载 motion_module 权重
    if motion_pt and os.path.isfile(motion_pt):
        motion_sd = torch.load(motion_pt, map_location="cpu", weights_only=True)
        missing, unexpected = model.motion_module.load_state_dict(motion_sd, strict=False)
        if missing:
            logger.warning("motion_module 缺失 key: %s", missing)
        logger.info("加载 motion_module: %s (%d 参数)", motion_pt, len(motion_sd))
    else:
        logger.warning("motion_module.pt 不存在: %s", motion_pt)

    # 统一 motion_module 的 dtype 为 bfloat16
    # reload_pretrained_weights 用 assign=True 加载 float32 的 ImageNet 权重,
    # 导致 ResNet-18 的 parameters 和 BN buffers 变为 float32
    model.motion_module.to(dtype=torch.bfloat16)

    # 加载 LoRA 权重
    if lora_dir is not None:
        from peft import PeftModel
        if model_type == "internvl3":
            model.base_model.language_model = PeftModel.from_pretrained(
                model.base_model.language_model, lora_dir,
            )
            model.base_model.language_model = model.base_model.language_model.merge_and_unload()
        else:
            model = PeftModel.from_pretrained(model, lora_dir)
            model = model.merge_and_unload()
        logger.info("加载并合并 LoRA: %s", lora_dir)

    model = model.to(device).eval()
    return model, processor, tokenizer


def _load_model_full(
    model_type: str,
    full_model_dir: str,
    motion_pt: Optional[str],
    device: str = "cuda",
    delta_model_config: Optional[Dict[str, Any]] = None,
):
    """
    加载 Full SFT 模型: 从 full_model/ 目录直接加载完整 Delta 模型权重。
    不需要加载 baseline 后再合并, 因为完整权重已包含所有参数。
    """
    delta_model_config = dict(delta_model_config or {})

    if model_type == "qwen":
        from transformers import AutoProcessor
        from models.qwen_delta import QwenDeltaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            full_model_dir, trust_remote_code=True,
            min_pixels=128 * 28 * 28, max_pixels=768 * 28 * 28,
        )
        model = QwenDeltaForConditionalGeneration.from_pretrained(
            full_model_dir, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            **_build_delta_model_kwargs("qwen", full_model_dir, delta_model_config),
        )
        tokenizer = processor.tokenizer

    elif model_type == "qwen3":
        from transformers import AutoProcessor
        from models.qwen3_delta import Qwen3DeltaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            full_model_dir, trust_remote_code=True,
            min_pixels=128 * 32 * 32, max_pixels=768 * 32 * 32,
        )
        model = Qwen3DeltaForConditionalGeneration.from_pretrained(
            full_model_dir, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            **_build_delta_model_kwargs("qwen3", full_model_dir, delta_model_config),
        )
        tokenizer = processor.tokenizer

    elif model_type == "llava":
        from transformers import AutoProcessor
        from models.llava_delta import LlavaDeltaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(full_model_dir, trust_remote_code=True)
        model = LlavaDeltaForConditionalGeneration.from_pretrained(
            full_model_dir, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            **_build_delta_model_kwargs("llava", full_model_dir, delta_model_config),
        )
        tokenizer = processor.tokenizer

    elif model_type == "internvl3":
        from transformers import AutoTokenizer
        from models.internvl_delta import load_internvl3_delta

        model = load_internvl3_delta(full_model_dir, dtype=torch.bfloat16, **delta_model_config)
        tokenizer = AutoTokenizer.from_pretrained(full_model_dir, trust_remote_code=True)
        processor = None

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 加载训练后的 motion_module 权重 (覆盖 from_pretrained 的 ImageNet 初始化)
    if motion_pt and os.path.isfile(motion_pt):
        motion_sd = torch.load(motion_pt, map_location="cpu", weights_only=True)
        missing, unexpected = model.motion_module.load_state_dict(motion_sd, strict=False)
        if missing:
            logger.warning("motion_module 缺失 key: %s", missing)
        logger.info("加载 motion_module (full mode): %s (%d 参数)", motion_pt, len(motion_sd))

    # setup motion token ids
    if model_type != "internvl3":
        model.setup_motion_token_ids(tokenizer)
    else:
        model.setup_motion_token_ids(tokenizer)

    model = model.to(device).eval()
    logger.info("Full SFT 模型加载完成: %s", full_model_dir)
    return model, processor, tokenizer


def _load_model_from_merged_sd(
    model_type: str,
    model_path: str,
    motion_pt: Optional[str],
    merged_sd_path: str,
    device: str = "cuda",
    delta_model_config: Optional[Dict[str, Any]] = None,
):
    """
    从 merged_state_dict.pt 加载模型。

    流程:
      1. 从 baseline (model_path) 用 from_pretrained 创建 Delta 模型结构
      2. 用 merged_state_dict 直接 load_state_dict (覆盖所有参数)
      3. 单独加载 motion_module.pt (确保 motion_module 正确)
      4. setup_motion_token_ids

    这是从 DeepSpeed 原生 checkpoint 自动转换后的加载路径。
    merged_state_dict 已经包含 LoRA 合并后的完整权重, 可直接覆盖 baseline 参数。
    """
    logger.info("从 baseline '%s' 创建模型结构, 然后加载 merged_state_dict '%s'", model_path, merged_sd_path)

    # 1) 创建 Delta 模型结构 (从 baseline)
    if model_type == "qwen":
        from transformers import AutoProcessor
        from models.qwen_delta import QwenDeltaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=128 * 28 * 28, max_pixels=768 * 28 * 28,
        )
        model = QwenDeltaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            **_build_delta_model_kwargs("qwen", model_path, delta_model_config),
        )
        tokenizer = processor.tokenizer

    elif model_type == "qwen3":
        from transformers import AutoProcessor
        from models.qwen3_delta import Qwen3DeltaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=128 * 32 * 32, max_pixels=768 * 32 * 32,
        )
        model = Qwen3DeltaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            **_build_delta_model_kwargs("qwen3", model_path, delta_model_config),
        )
        tokenizer = processor.tokenizer

    elif model_type == "llava":
        from transformers import AutoProcessor
        from models.llava_delta import LlavaDeltaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = LlavaDeltaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            **_build_delta_model_kwargs("llava", model_path, delta_model_config),
        )
        tokenizer = processor.tokenizer

    elif model_type == "internvl3":
        from transformers import AutoTokenizer
        from models.internvl_delta import load_internvl3_delta

        internvl_kwargs = dict(delta_model_config or {})
        model = load_internvl3_delta(model_path, dtype=torch.bfloat16, **internvl_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = None

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 2) 加载 merged state_dict (覆盖 baseline 参数)
    merged_sd = torch.load(merged_sd_path, map_location="cpu", weights_only=True)

    # 处理 vocab size 不匹配: 训练时 add_special_tokens + resize_token_embeddings
    # 导致 checkpoint 的 embed/lm_head shape 与 from_pretrained 默认的不同。
    # 先检测 checkpoint vocab size, 必要时先 resize 模型再 load。
    _ckpt_vocab_size = None
    for _emb_key in ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight",
                      "language_model.embed_tokens.weight", "embed_tokens.weight"):
        if _emb_key in merged_sd:
            _ckpt_vocab_size = merged_sd[_emb_key].shape[0]
            break
    if _ckpt_vocab_size is not None:
        _model_vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else None
        if _model_vocab_size is None and hasattr(model.config, "text_config"):
            _model_vocab_size = getattr(model.config.text_config, "vocab_size", None)
        if _model_vocab_size is not None and _ckpt_vocab_size != _model_vocab_size:
            logger.info(
                "Vocab size 不匹配: 模型=%d, checkpoint=%d, 先 resize 模型 embedding",
                _model_vocab_size, _ckpt_vocab_size,
            )
            model.resize_token_embeddings(_ckpt_vocab_size)

    missing, unexpected = model.load_state_dict(merged_sd, strict=False)
    if missing:
        logger.warning("merged_state_dict 缺失 %d 个 key (前 10 个): %s", len(missing), missing[:10])
    if unexpected:
        logger.warning("merged_state_dict 多出 %d 个 key (前 10 个): %s", len(unexpected), unexpected[:10])
    logger.info("已加载 merged_state_dict (%d 参数)", len(merged_sd))
    del merged_sd

    # 3) 加载 motion_module 权重 (确保 motion_module 正确, 覆盖 merged_sd 中可能不完整的部分)
    if motion_pt and os.path.isfile(motion_pt):
        motion_sd = torch.load(motion_pt, map_location="cpu", weights_only=True)
        m_missing, _ = model.motion_module.load_state_dict(motion_sd, strict=False)
        if m_missing:
            logger.warning("motion_module 缺失 key: %s", m_missing)
        logger.info("加载 motion_module (merged mode): %s (%d 参数)", motion_pt, len(motion_sd))

    # 4) setup motion token ids
    model.setup_motion_token_ids(tokenizer)

    model = model.to(device).eval()
    logger.info("Merged state_dict 模型加载完成")
    return model, processor, tokenizer


def load_model_lora_only(model_type: str, model_path: str, delta_checkpoint: str, device: str = "cuda"):
    """
    加载基线模型 + 仅 LoRA (不加载 motion_module)。
    用于图像类 benchmark 评测, 避免浪费显存加载不需要的 motion_module。

    Full SFT 模式: 直接从 full_model/ 加载完整模型 (无需 motion_module)。
    Merged SD 模式: 从 merged_state_dict 加载 (跳过 motion_module 部分)。
    """
    _, lora_dir, full_model_dir, merged_sd_path = _resolve_delta_paths(delta_checkpoint)

    # Merged state_dict 模式: 从 baseline 结构 + merged_sd 加载 (无 motion_module)
    if merged_sd_path is not None:
        logger.info("图像评测 merged SD 模式: 从 %s 加载 (跳过 motion_module)", merged_sd_path)
        model, processor, tokenizer = load_model_baseline(model_type, model_path, device)
        # 加载 merged state_dict 中非 motion_module 部分
        merged_sd = torch.load(merged_sd_path, map_location="cpu", weights_only=True)
        non_motion_sd = {k: v for k, v in merged_sd.items() if "motion_module." not in k}

        # 处理 vocab size 不匹配 (同 _load_model_from_merged_sd)
        _ckpt_vocab_size = None
        for _emb_key in ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight",
                          "language_model.embed_tokens.weight", "embed_tokens.weight"):
            if _emb_key in non_motion_sd:
                _ckpt_vocab_size = non_motion_sd[_emb_key].shape[0]
                break
        if _ckpt_vocab_size is not None:
            _model_vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else None
            if _model_vocab_size is None and hasattr(model.config, "text_config"):
                _model_vocab_size = getattr(model.config.text_config, "vocab_size", None)
            if _model_vocab_size is not None and _ckpt_vocab_size != _model_vocab_size:
                logger.info(
                    "图像评测 vocab size 不匹配: 模型=%d, checkpoint=%d, resize",
                    _model_vocab_size, _ckpt_vocab_size,
                )
                model.resize_token_embeddings(_ckpt_vocab_size)

        model.load_state_dict(non_motion_sd, strict=False)
        logger.info("已加载 merged_state_dict (跳过 motion_module, %d 参数)", len(non_motion_sd))
        del merged_sd, non_motion_sd
        return model, processor, tokenizer

    # Full SFT 模式: 直接从 full_model/ 加载 (但不加载 motion_module)
    if full_model_dir is not None:
        logger.info("Full SFT 图像评测: 从 %s 加载完整模型 (跳过 motion_module)", full_model_dir)
        model, processor, tokenizer = load_model_baseline(model_type, full_model_dir, device)
        return model, processor, tokenizer

    # LoRA 模式: 加载基线模型 + LoRA
    model, processor, tokenizer = load_model_baseline(model_type, model_path, device)

    # 加载 LoRA
    if lora_dir is not None:
        from peft import PeftModel
        if model_type == "internvl3":
            model.language_model = PeftModel.from_pretrained(model.language_model, lora_dir)
            model.language_model = model.language_model.merge_and_unload()
        else:
            model = PeftModel.from_pretrained(model, lora_dir)
            model = model.merge_and_unload()
        logger.info("加载并合并 LoRA (无 motion_module): %s", lora_dir)
    else:
        logger.warning("图像评测 delta 模式: 未找到 LoRA 权重, 等同于 baseline")

    return model, processor, tokenizer


def load_model_stage1(
    model_type: str,
    model_path: str,
    delta_checkpoint: str,
    device: str = "cuda",
    delta_model_config: Optional[Dict[str, Any]] = None,
):
    """
    加载 Stage 1 模型 (baseline + 仅 motion_module, 无 LoRA/Full SFT)。

    Stage 1 checkpoint 只训练了 MotionEncoder + Projector, LLM 保持原始 baseline 参数。
    因此只需加载 baseline Delta 模型 + 覆盖 motion_module 权重即可。

    对比 load_model_delta: 跳过 LoRA 合并和 merged_state_dict 路径,
    直接从 DeepSpeed checkpoint 提取 motion_module.pt 后加载, 节省大量时间和内存。

    支持的 checkpoint 格式:
      1. 直接 motion_module.pt 文件
      2. 含 motion_module.pt 的目录 (final/, checkpoint-N/)
      3. DeepSpeed 原生 checkpoint (global_step*/) — 只提取 motion_module 部分

    用法:
        torchrun --nproc_per_node=8 eval/eval_himmel.py \\
            --model-type qwen --model-path /path/to/model \\
            --delta-checkpoint /path/to/stage1_checkpoint --stage1 \\
            --benchmark all_video
    """
    delta_model_config = dict(delta_model_config or {})

    # 解析 motion_module.pt 路径
    motion_pt = _resolve_stage1_motion_module(delta_checkpoint)

    if motion_pt is None:
        logger.warning("Stage 1 checkpoint 中未找到 motion_module 权重: %s", delta_checkpoint)
        logger.warning("回退到 baseline 模型 (无 motion_module)")
        return load_model_baseline(model_type, model_path, device)

    # 加载 Delta 模型 (baseline + 空 motion_module)
    if model_type == "qwen":
        from transformers import AutoProcessor
        from models.qwen_delta import QwenDeltaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=128 * 28 * 28, max_pixels=768 * 28 * 28,
        )
        model = QwenDeltaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            **_build_delta_model_kwargs("qwen", model_path, delta_model_config),
        )
        model.motion_module.reload_pretrained_weights()
        model.setup_motion_token_ids(processor.tokenizer)
        tokenizer = processor.tokenizer

    elif model_type == "qwen3":
        from transformers import AutoProcessor
        from models.qwen3_delta import Qwen3DeltaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=128 * 32 * 32, max_pixels=768 * 32 * 32,
        )
        model = Qwen3DeltaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            **_build_delta_model_kwargs("qwen3", model_path, delta_model_config),
        )
        model.motion_module.reload_pretrained_weights()
        model.setup_motion_token_ids(processor.tokenizer)
        tokenizer = processor.tokenizer

    elif model_type == "llava":
        from transformers import AutoProcessor
        from models.llava_delta import LlavaDeltaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = LlavaDeltaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
            **_build_delta_model_kwargs("llava", model_path, delta_model_config),
        )
        model.motion_module.reload_pretrained_weights()
        model.setup_motion_token_ids(processor.tokenizer)
        tokenizer = processor.tokenizer

    elif model_type == "internvl3":
        from transformers import AutoTokenizer
        from models.internvl_delta import load_internvl3_delta

        model = load_internvl3_delta(model_path, dtype=torch.bfloat16, **delta_model_config)
        model.motion_module.reload_pretrained_weights()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.setup_motion_token_ids(tokenizer)
        processor = None

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 加载训练后的 motion_module 权重
    motion_sd = torch.load(motion_pt, map_location="cpu", weights_only=True)
    missing, unexpected = model.motion_module.load_state_dict(motion_sd, strict=False)
    if missing:
        logger.warning("Stage 1 motion_module 缺失 key: %s", missing)
    if unexpected:
        logger.warning("Stage 1 motion_module 多余 key: %s", unexpected)
    logger.info(
        "Stage 1 motion_module 加载完成: %s (%d 参数, missing=%d, unexpected=%d)",
        motion_pt, len(motion_sd), len(missing), len(unexpected),
    )

    # 统一 motion_module 的 dtype 为 bfloat16
    # reload_pretrained_weights 用 assign=True 加载 float32 的 ImageNet 权重,
    # 导致 ResNet-18 的 parameters 和 BN buffers 变为 float32,
    # 而模型其余部分是 bfloat16, 前向传播时矩阵乘法会报 dtype 不匹配错误。
    model.motion_module.to(dtype=torch.bfloat16)
    logger.info("motion_module dtype 统一为 bfloat16")

    model = model.to(device).eval()
    logger.info("Stage 1 模型加载完成 (baseline LLM + 训练后 motion_module)")
    return model, processor, tokenizer


def _resolve_stage1_motion_module(delta_checkpoint: str) -> Optional[str]:
    """
    从 Stage 1 checkpoint 路径解析出 motion_module.pt。

    与 _resolve_delta_paths 不同, 这里只提取 motion_module, 不做 LoRA merge / merged_sd。
    对于 DeepSpeed 原生 checkpoint, 直接从 mp_rank_00_model_states.pt 中提取。

    支持的格式:
      1. 直接 .pt 文件
      2. 含 motion_module.pt 的目录
      3. DeepSpeed 原生 checkpoint (global_step*/) — 从 mp_rank_00_model_states.pt 提取

    Returns:
        motion_module.pt 的路径, 或 None
    """
    # 情况 1: 直接指向 .pt 文件
    if os.path.isfile(delta_checkpoint) and delta_checkpoint.endswith(".pt"):
        return delta_checkpoint

    if not os.path.isdir(delta_checkpoint):
        logger.warning("Stage 1 checkpoint 路径不存在: %s", delta_checkpoint)
        return None

    # 情况 2: 目录中有 motion_module.pt
    direct_pt = os.path.join(delta_checkpoint, "motion_module.pt")
    if os.path.isfile(direct_pt):
        return direct_pt

    # 情况 3: DeepSpeed 原生 checkpoint — 需要提取 motion_module
    # 查找 global_step* 子目录
    gs_dirs = sorted(
        [d for d in os.listdir(delta_checkpoint) if d.startswith("global_step")],
        key=lambda x: int(x.replace("global_step", "")) if x.replace("global_step", "").isdigit() else 0,
        reverse=True,  # 取最大 step
    )

    if not gs_dirs:
        logger.warning("Stage 1 checkpoint 目录中无 motion_module.pt 也无 global_step: %s", delta_checkpoint)
        return None

    # 尝试从 DeepSpeed checkpoint 提取 motion_module
    for gs_dir in gs_dirs:
        gs_path = os.path.join(delta_checkpoint, gs_dir)

        # 先检查是否已有提取结果 (在 gs_path 或 delta_checkpoint 下)
        for check_dir in [gs_path, delta_checkpoint]:
            cached_pt = os.path.join(check_dir, "motion_module.pt")
            if os.path.isfile(cached_pt):
                logger.info("发现已提取的 Stage 1 motion_module: %s", cached_pt)
                return cached_pt

        # 查找 mp_rank_00_model_states.pt
        model_states_pt = os.path.join(gs_path, "mp_rank_00_model_states.pt")
        if not os.path.isfile(model_states_pt):
            continue

        rank = _get_rank()
        output_pt = os.path.join(delta_checkpoint, "motion_module.pt")

        if rank == 0:
            logger.info("从 DeepSpeed checkpoint 提取 Stage 1 motion_module: %s", model_states_pt)
            file_size_gb = os.path.getsize(model_states_pt) / (1024 ** 3)
            logger.info("  文件大小: %.1f GB, 加载中...", file_size_gb)
            try:
                t0 = time.time()
                full_state = torch.load(model_states_pt, map_location="cpu", weights_only=False)

                # DeepSpeed 存储格式: {"module": {full_model_state_dict}, ...}
                if "module" in full_state:
                    full_state = full_state["module"]

                # 提取 motion_module.* 前缀的 key
                prefix = "motion_module."
                motion_sd = {}
                for k, v in full_state.items():
                    if k.startswith(prefix):
                        motion_sd[k[len(prefix):]] = v

                del full_state  # 释放大 state_dict 内存

                if not motion_sd:
                    logger.warning("DeepSpeed checkpoint 中未找到 motion_module.* 权重")
                    continue

                # 保存提取结果到 checkpoint 根目录
                torch.save(motion_sd, output_pt)
                logger.info(
                    "Stage 1 motion_module 提取完成: %d 参数 -> %s (耗时 %.1f 秒)",
                    len(motion_sd), output_pt, time.time() - t0,
                )
            except Exception as e:
                logger.warning("提取 Stage 1 motion_module 失败: %s", e)
                continue

        # 多卡同步: 等 rank 0 提取完毕
        if _is_dist():
            dist.barrier()

        if os.path.isfile(output_pt):
            return output_pt

    return None


# ============================================================================
# 推理函数 (各模型类型)
# ============================================================================

@torch.no_grad()
def generate_qwen(
    model, processor, tokenizer,
    keyframes: List[Image.Image],
    diff_frames: Optional[torch.Tensor],
    motion_vectors: Optional[torch.Tensor],
    residuals: Optional[torch.Tensor],
    question: str,
    options: List[str],
    is_delta: bool = False,
    max_new_tokens: int = 128,
    sampling_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Qwen2.5-VL / Qwen3-VL 推理。"""
    # 构造选项文本
    option_text = "\n".join(
        f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
    )
    prompt_prefix = build_sampling_prompt_prefix(sampling_meta)
    prompt_text = (
        f"{prompt_prefix}{question}\n{option_text}\n"
        "Answer with the option letter only."
    )

    # 构造 motion 占位符 (仅 delta 模式)
    motion_placeholder = ""
    num_motion_tokens = _get_motion_token_count(diff_frames, motion_vectors, residuals)
    if is_delta and num_motion_tokens > 0:
        motion_placeholder = (
            "<|motion_start|>"
            + "<|motion_pad|>" * num_motion_tokens
            + "<|motion_end|>\n"
        )

    if keyframes:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": keyframes},
                    {"type": "text", "text": motion_placeholder + prompt_text},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], videos=[keyframes],
            return_tensors="pt", padding=True,
        )
    else:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors="pt", padding=True)

    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )

    if is_delta:
        motion_encoder_type = getattr(getattr(model, "motion_module", None), "motion_encoder_type", "resnet")
        if diff_frames is not None:
            gen_kwargs["diff_frames"] = diff_frames.unsqueeze(0).to(model.device)
        if motion_vectors is not None:
            gen_kwargs["motion_vectors"] = motion_vectors.unsqueeze(0).to(model.device)
        if residuals is not None:
            gen_kwargs["residuals"] = residuals.unsqueeze(0).to(model.device)

        if motion_encoder_type == "shared_vit" and diff_frames is not None:
            diff_mode = (sampling_meta or {}).get("diff_mode", "abs")
            diff_images = _normalized_tensor_to_images(diff_frames, diff_mode=diff_mode)
            if diff_images:
                diff_inputs = processor(
                    videos=[diff_images],
                    return_tensors="pt",
                    padding=True,
                )
                if "pixel_values_videos" in diff_inputs:
                    gen_kwargs["diff_pixel_values_videos"] = diff_inputs["pixel_values_videos"].to(model.device)
                if "video_grid_thw" in diff_inputs:
                    gen_kwargs["diff_video_grid_thw"] = diff_inputs["video_grid_thw"].to(model.device)

    output_ids = model.generate(**inputs, **gen_kwargs)
    # 去掉 input 部分
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_llava(
    model, processor, tokenizer,
    keyframes: List[Image.Image],
    diff_frames: Optional[torch.Tensor],
    motion_vectors: Optional[torch.Tensor],
    residuals: Optional[torch.Tensor],
    question: str,
    options: List[str],
    is_delta: bool = False,
    max_new_tokens: int = 128,
    sampling_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """LLaVA-OneVision 推理。"""
    option_text = "\n".join(
        f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
    )
    prompt_prefix = build_sampling_prompt_prefix(sampling_meta)
    prompt_text = (
        f"{prompt_prefix}{question}\n{option_text}\n"
        "Answer with the option letter only."
    )

    motion_placeholder = ""
    num_motion_tokens = _get_motion_token_count(diff_frames, motion_vectors, residuals)
    if is_delta and num_motion_tokens > 0:
        motion_placeholder = (
            "<|motion_start|>"
            + "<|motion_pad|>" * num_motion_tokens
            + "<|motion_end|>\n"
        )

    if keyframes:
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": motion_placeholder + prompt_text},
            ]},
        ]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=text, images=keyframes,
            return_tensors="pt", padding=True,
        )
    else:
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
        ]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt", padding=True)

    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )

    if is_delta:
        if diff_frames is not None:
            gen_kwargs["diff_frames"] = diff_frames.unsqueeze(0).to(model.device)
        if motion_vectors is not None:
            gen_kwargs["motion_vectors"] = motion_vectors.unsqueeze(0).to(model.device)
        if residuals is not None:
            gen_kwargs["residuals"] = residuals.unsqueeze(0).to(model.device)

    output_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_internvl3(
    model, tokenizer,
    keyframes: List[Image.Image],
    diff_frames: Optional[torch.Tensor],
    motion_vectors: Optional[torch.Tensor],
    residuals: Optional[torch.Tensor],
    question: str,
    options: List[str],
    is_delta: bool = False,
    max_new_tokens: int = 128,
    sampling_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """InternVL3 推理。"""
    from torchvision import transforms as T

    option_text = "\n".join(
        f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
    )
    prompt_prefix = build_sampling_prompt_prefix(sampling_meta)
    prompt_text = (
        f"{prompt_prefix}{question}\n{option_text}\n"
        "Answer with the option letter only."
    )

    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    NUM_IMG_TOKENS = 256

    motion_placeholder = ""
    num_motion_tokens = _get_motion_token_count(diff_frames, motion_vectors, residuals)
    if is_delta and num_motion_tokens > 0:
        motion_placeholder = (
            "<|motion_start|>"
            + "<|motion_pad|>" * num_motion_tokens
            + "<|motion_end|>\n"
        )

    # 处理关键帧
    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if keyframes:
        pixel_values = torch.stack([transform(img) for img in keyframes]).to(
            model.device, dtype=torch.bfloat16
        )
        img_placeholder = "<img>" + IMG_CONTEXT_TOKEN * NUM_IMG_TOKENS + "</img>"
        visual_text = "\n".join([img_placeholder] * len(keyframes))
    else:
        pixel_values = None
        visual_text = ""

    text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{visual_text}\n{motion_placeholder}{prompt_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    tokens = tokenizer(text, return_tensors="pt", padding=False)
    input_ids = tokens["input_ids"].to(model.device)
    attention_mask = tokens["attention_mask"].to(model.device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )

    if is_delta:
        # Delta model: InternVLDeltaChatModel
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
        if diff_frames is not None:
            gen_kwargs["diff_frames"] = diff_frames.unsqueeze(0).to(model.device)
        if motion_vectors is not None:
            gen_kwargs["motion_vectors"] = motion_vectors.unsqueeze(0).to(model.device)
        if residuals is not None:
            gen_kwargs["residuals"] = residuals.unsqueeze(0).to(model.device)
        output_ids = model.generate(**gen_kwargs)
    else:
        # Baseline: InternVLChatModel - 需要手动处理 vision embedding
        if pixel_values is not None:
            # 对于 baseline InternVLChatModel, 使用 chat 接口或手动处理
            # 这里采用手动方式与 delta 保持一致
            language_model = model.language_model
            inputs_embeds = language_model.get_input_embeddings()(input_ids)

            vit_dtype = next(model.vision_model.parameters()).dtype
            vit_embeds = model.extract_feature(pixel_values.to(vit_dtype))

            B, S, D = inputs_embeds.shape
            input_ids_flat = input_ids.reshape(-1)
            img_ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            if img_ctx_id != tokenizer.unk_token_id:
                selected = (input_ids_flat == img_ctx_id)
                if selected.sum() > 0:
                    vit_flat = vit_embeds.reshape(-1, vit_embeds.shape[-1])
                    n_select = min(selected.sum().item(), vit_flat.shape[0])
                    inputs_embeds_flat = inputs_embeds.reshape(-1, D)
                    inputs_embeds_flat[selected.nonzero(as_tuple=True)[0][:n_select]] = \
                        vit_flat[:n_select].to(inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds_flat.reshape(B, S, D)

            output_ids = language_model.generate(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        else:
            output_ids = model.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    input_len = input_ids.shape[1]
    generated = output_ids[0, input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ============================================================================
# 图像推理函数 (MathVista / MathVision)
# ============================================================================

@torch.no_grad()
def generate_image_qwen(
    model, processor, tokenizer,
    image: Image.Image,
    question: str,
    options: List[str],
    is_multi_choice: bool = True,
    max_new_tokens: int = 256,
) -> str:
    """Qwen2.5-VL / Qwen3-VL 图像推理。"""
    if is_multi_choice and options:
        option_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        prompt_text = f"{question}\n{option_text}\nAnswer with the option letter only."
    else:
        prompt_text = f"{question}\nPlease provide the answer directly."

    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if image is not None:
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True)

    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_image_llava(
    model, processor, tokenizer,
    image: Image.Image,
    question: str,
    options: List[str],
    is_multi_choice: bool = True,
    max_new_tokens: int = 256,
) -> str:
    """LLaVA-OneVision 图像推理。"""
    if is_multi_choice and options:
        option_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        prompt_text = f"{question}\n{option_text}\nAnswer with the option letter only."
    else:
        prompt_text = f"{question}\nPlease provide the answer directly."

    content = []
    if image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt_text})

    conversation = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    if image is not None:
        inputs = processor(text=text, images=[image], return_tensors="pt", padding=True)
    else:
        inputs = processor(text=text, return_tensors="pt", padding=True)

    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_image_internvl3(
    model, tokenizer,
    image: Image.Image,
    question: str,
    options: List[str],
    is_multi_choice: bool = True,
    max_new_tokens: int = 256,
) -> str:
    """InternVL3 图像推理。"""
    from torchvision import transforms as T

    if is_multi_choice and options:
        option_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        prompt_text = f"{question}\n{option_text}\nAnswer with the option letter only."
    else:
        prompt_text = f"{question}\nPlease provide the answer directly."

    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    NUM_IMG_TOKENS = 256

    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if image is not None:
        pixel_values = transform(image).unsqueeze(0).to(model.device, dtype=torch.bfloat16)
        img_placeholder = "<img>" + IMG_CONTEXT_TOKEN * NUM_IMG_TOKENS + "</img>"
    else:
        pixel_values = None
        img_placeholder = ""

    text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{img_placeholder}\n{prompt_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    tokens = tokenizer(text, return_tensors="pt", padding=False)
    input_ids = tokens["input_ids"].to(model.device)
    attention_mask = tokens["attention_mask"].to(model.device)

    if pixel_values is not None:
        language_model = model.language_model
        inputs_embeds = language_model.get_input_embeddings()(input_ids)
        vit_dtype = next(model.vision_model.parameters()).dtype
        vit_embeds = model.extract_feature(pixel_values.to(vit_dtype))
        B, S, D = inputs_embeds.shape
        input_ids_flat = input_ids.reshape(-1)
        img_ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        if img_ctx_id != tokenizer.unk_token_id:
            selected = (input_ids_flat == img_ctx_id)
            if selected.sum() > 0:
                vit_flat = vit_embeds.reshape(-1, vit_embeds.shape[-1])
                n_select = min(selected.sum().item(), vit_flat.shape[0])
                inputs_embeds_flat = inputs_embeds.reshape(-1, D)
                inputs_embeds_flat[selected.nonzero(as_tuple=True)[0][:n_select]] = \
                    vit_flat[:n_select].to(inputs_embeds.dtype)
                inputs_embeds = inputs_embeds_flat.reshape(B, S, D)
        output_ids = language_model.generate(
            input_ids=None, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=False,
        )
    else:
        output_ids = model.language_model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
        )

    input_len = input_ids.shape[1]
    generated = output_ids[0, input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ============================================================================
# 评测主循环
# ============================================================================

def evaluate_benchmark(
    model,
    processor,
    tokenizer,
    model_type: str,
    benchmark_name: str,
    eval_json_path: str,
    video_dir: str,
    is_delta: bool = False,
    max_keyframes: int = 32,
    max_samples: int = 0,
    prefetch_depth: int = 2,
    sampling_strategy: str = "uniform",
    keyframe_fps: float = 1.0,
    memory_keyframes: int = 12,
    dense_local_frames: int = 3,
    question_aware_sampling: bool = True,
    memory_diffs_per_interval: int = 1,
    dense_diffs_per_interval: int = 4,
    diff_mode: str = "abs",
    motion_input_mode: str = "rgb_diff",
    compressed_backend: str = "coviar",
    visual_stream_mode: str = "iframe",
    compressed_gop_size: int = 12,
    use_motion_vectors: bool = True,
    use_residuals: bool = True,
    accumulate_motion_vectors: bool = True,
    accumulate_residuals: bool = False,
    motion_vector_clip: float = 20.0,
    tf_method: str = "none",
    tf_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    在指定视频 benchmark 上评测。
    支持多卡分布式 (自动分片) + CPU/GPU 异步 pipeline (后台预取)。

    Returns:
        dict: {
            "accuracy": float,
            "total": int,
            "correct": int,
            "per_category": {category: {"accuracy": float, "total": int, "correct": int}},
            "results": [{"video": ..., "question": ..., "pred": ..., "answer": ..., "correct": bool}]
        }
    """
    rank = _get_rank()
    world = _get_world_size()

    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    if max_samples > 0:
        eval_data = eval_data[:max_samples]

    # 为每条数据记录原始全局 idx 和 video_dir
    for i, item in enumerate(eval_data):
        item["_global_idx"] = i
        item["_video_dir"] = video_dir

    total_samples = len(eval_data)

    # 分片
    local_data = _shard_data(eval_data)
    logger.info("评测 %s: 总样本 %d, rank %d/%d 分得 %d 条",
                benchmark_name, total_samples, rank, world, len(local_data))

    results = []
    correct = 0
    total = 0
    per_category = {}
    inference_times = []
    preprocess_times = []
    num_keyframes_list = []
    num_diffs_list = []
    long_term_keyframes_list = []
    focus_keyframes_list = []

    # 使用异步预取 pipeline
    prefetcher = _VideoPrefetcher(
        local_data,
        max_keyframes=max_keyframes,
        prefetch_depth=prefetch_depth,
        sampling_config={
            "sampling_strategy": sampling_strategy,
            "keyframe_fps": keyframe_fps,
            "memory_keyframes": memory_keyframes,
            "dense_local_frames": dense_local_frames,
            "question_aware_sampling": question_aware_sampling,
            "memory_diffs_per_interval": memory_diffs_per_interval,
            "dense_diffs_per_interval": dense_diffs_per_interval,
            "diff_mode": diff_mode,
            "motion_input_mode": motion_input_mode,
            "compressed_backend": compressed_backend,
            "visual_stream_mode": visual_stream_mode,
            "compressed_gop_size": compressed_gop_size,
            "use_motion_vectors": use_motion_vectors,
            "use_residuals": use_residuals,
            "accumulate_motion_vectors": accumulate_motion_vectors,
            "accumulate_residuals": accumulate_residuals,
            "motion_vector_clip": motion_vector_clip,
        },
    )
    show_progress = (rank == 0)
    pbar = tqdm(total=len(local_data), desc=f"Eval {benchmark_name}", disable=not show_progress)

    for item, keyframes, diff_frames, motion_vectors, residuals, preprocess_time, sampling_meta in prefetcher:
        if sampling_meta.get("error") or not keyframes:
            raise RuntimeError(f"Invalid video sample {item.get('_global_idx')}: {sampling_meta.get('error', 'no decoded frames')}")
        global_idx = item["_global_idx"]
        question = item.get("question", "")
        options = item.get("options", [])
        answer = item.get("answer", "")
        video_name = item.get("video", "")
        category = item.get("task_type", item.get("sub_category", item.get("domain", "unknown")))

        preprocess_times.append(preprocess_time)
        num_kf = len(keyframes)
        num_df = diff_frames.shape[0] if diff_frames is not None else 0
        num_mv = motion_vectors.shape[0] if motion_vectors is not None else 0
        num_res = residuals.shape[0] if residuals is not None else 0
        sample_motion_input_mode = sampling_meta.get("motion_input_mode", motion_input_mode)
        num_keyframes_list.append(num_kf)
        num_diffs_list.append(num_df)
        long_term_keyframes_list.append(int(sampling_meta.get("long_term_keyframes", num_kf)))
        focus_keyframes_list.append(int(sampling_meta.get("focus_keyframes", 0)))

        # 推理
        tf_meta: Dict[str, Any] = {}
        t_infer = time.time()
        try:
            # Training-free 方法: 仅对 Qwen baseline (非 delta) 生效
            if tf_method != "none" and model_type in ("qwen", "qwen3") and not is_delta:
                response, tf_meta = run_training_free(
                    method=tf_method,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    keyframes=keyframes,
                    video_path=sampling_meta.get("video_path", ""),
                    question=question,
                    options=options,
                    config=tf_config or {},
                    max_new_tokens=128,
                )
            elif model_type in ("qwen", "qwen3"):
                response = generate_qwen(
                    model, processor, tokenizer,
                    keyframes, diff_frames, motion_vectors, residuals, question, options,
                    is_delta=is_delta,
                    sampling_meta=sampling_meta,
                )
            elif model_type == "llava":
                response = generate_llava(
                    model, processor, tokenizer,
                    keyframes, diff_frames, motion_vectors, residuals, question, options,
                    is_delta=is_delta,
                    sampling_meta=sampling_meta,
                )
            elif model_type == "internvl3":
                response = generate_internvl3(
                    model, tokenizer,
                    keyframes, diff_frames, motion_vectors, residuals, question, options,
                    is_delta=is_delta,
                    sampling_meta=sampling_meta,
                )
            else:
                response = ""
        except Exception as e:
            raise RuntimeError(f"Inference failed for video sample {global_idx}; no guessed answer was scored.") from e
        infer_time = time.time() - t_infer
        inference_times.append(infer_time)

        pred_letter = extract_answer_letter(response, len(options))
        is_correct = _check_mc_answer(pred_letter, answer, options)

        if is_correct:
            correct += 1
        total += 1

        if category not in per_category:
            per_category[category] = {"correct": 0, "total": 0}
        per_category[category]["total"] += 1
        if is_correct:
            per_category[category]["correct"] += 1

        results.append({
            "idx": global_idx,
            "video": video_name,
            "question": question,
            "options": options,
            "answer": answer,
            "prediction": pred_letter,
            "response": response,
            "correct": is_correct,
            "category": category,
            "num_keyframes": num_kf,
            "num_diffs": num_df,
            "num_motion_vectors": num_mv,
            "num_residuals": num_res,
            "motion_input_mode": sampling_meta.get("motion_input_mode", motion_input_mode),
            "sampling_strategy": sampling_meta.get("strategy", sampling_strategy),
            "long_term_keyframes": int(sampling_meta.get("long_term_keyframes", num_kf)),
            "focus_keyframes": int(sampling_meta.get("focus_keyframes", 0)),
            "dense_intervals": int(sampling_meta.get("dense_intervals", 0)),
            "memory_diff_tokens": int(sampling_meta.get("memory_diff_tokens", num_df)),
            "dense_diff_tokens": int(sampling_meta.get("dense_diff_tokens", 0)),
            "tf_method": tf_meta.get("tf_method", tf_method),
            "tf_meta": tf_meta,
            "preprocess_time": round(preprocess_time, 3),
            "inference_time": round(infer_time, 3),
        })

        pbar.update(1)
        if total % 50 == 0 and total > 0:
            acc = correct / total * 100
            avg_infer = sum(inference_times) / len(inference_times)
            avg_kf = sum(num_keyframes_list) / len(num_keyframes_list)
            logger.info("[rank %d] [%d/%d] Acc: %.1f%% (%d/%d) | AvgInfer: %.2fs | AvgKF: %.1f",
                       rank, total, len(local_data), acc, correct, total, avg_infer, avg_kf)

    pbar.close()

    # 多卡聚合结果
    all_results = _gather_results(results)

    # 只在 rank 0 计算最终统计
    if rank == 0:
        all_correct = sum(1 for r in all_results if r["correct"])
        all_total = len(all_results)
        all_per_category: Dict[str, Dict[str, Any]] = {}
        all_inference_times = []
        all_preprocess_times = []
        all_kf_list = []
        all_df_list = []
        all_long_term_kf_list = []
        all_focus_kf_list = []

        for r in all_results:
            cat = r["category"]
            if cat not in all_per_category:
                all_per_category[cat] = {"correct": 0, "total": 0}
            all_per_category[cat]["total"] += 1
            if r["correct"]:
                all_per_category[cat]["correct"] += 1
            all_inference_times.append(r["inference_time"])
            all_preprocess_times.append(r["preprocess_time"])
            all_kf_list.append(r["num_keyframes"])
            all_df_list.append(r["num_diffs"])
            all_long_term_kf_list.append(r.get("long_term_keyframes", r["num_keyframes"]))
            all_focus_kf_list.append(r.get("focus_keyframes", 0))

        for cat in all_per_category:
            c = all_per_category[cat]
            c["accuracy"] = c["correct"] / c["total"] * 100 if c["total"] > 0 else 0.0

        overall_acc = all_correct / all_total * 100 if all_total > 0 else 0.0

        timing_stats = {}
        if all_inference_times:
            timing_stats = {
                "avg_inference_time": round(sum(all_inference_times) / len(all_inference_times), 3),
                "median_inference_time": round(sorted(all_inference_times)[len(all_inference_times) // 2], 3),
                "avg_preprocess_time": round(sum(all_preprocess_times) / len(all_preprocess_times), 3),
                "total_inference_time": round(sum(all_inference_times), 1),
                "total_preprocess_time": round(sum(all_preprocess_times), 1),
                "avg_keyframes": round(sum(all_kf_list) / len(all_kf_list), 1),
                "avg_diffs": round(sum(all_df_list) / len(all_df_list), 1),
                "avg_long_term_keyframes": round(sum(all_long_term_kf_list) / len(all_long_term_kf_list), 1),
                "avg_focus_keyframes": round(sum(all_focus_kf_list) / len(all_focus_kf_list), 1),
                "max_keyframes": max(all_kf_list) if all_kf_list else 0,
                "max_diffs": max(all_df_list) if all_df_list else 0,
                "sampling_strategy": sampling_strategy,
                "tf_method": tf_method,
                "num_gpus": world,
            }

        return {
            "benchmark": benchmark_name,
            "accuracy": overall_acc,
            "total": all_total,
            "correct": all_correct,
            "per_category": all_per_category,
            "timing": timing_stats,
            "results": all_results,
        }
    else:
        # 非 rank 0 返回空壳 (由 rank 0 负责输出)
        return {
            "benchmark": benchmark_name,
            "accuracy": 0.0,
            "total": 0,
            "correct": 0,
            "per_category": {},
            "timing": {},
            "results": [],
        }


def _check_mc_answer(pred_letter: str, answer: str, options: List[str]) -> bool:
    """检查多选题答案正确性。"""
    answer = answer.strip()
    # answer 是字母
    if answer.upper() in [chr(65 + i) for i in range(len(options))]:
        return pred_letter == answer.upper()
    # answer 是选项内容文本
    if options:
        for i, opt in enumerate(options):
            if opt.strip() == answer:
                return pred_letter == chr(65 + i)
    return False


def _check_free_form_answer(pred: str, answer: str, answer_type: str = "text",
                            precision: float = 1e-3) -> bool:
    """Local diagnostic comparison; NOT the official MathVista/MathVision scorer."""
    pred = pred.strip().lower()
    answer = answer.strip().lower()

    if not pred or not answer:
        return False

    # 精确匹配
    if pred == answer:
        return True

    # 数值比较
    try:
        pred_num = float(pred.replace(",", ""))
        ans_num = float(answer.replace(",", ""))
        if abs(pred_num - ans_num) < precision:
            return True
    except (ValueError, ZeroDivisionError):
        pass

    return False


def evaluate_image_benchmark(
    model,
    processor,
    tokenizer,
    model_type: str,
    benchmark_name: str,
    eval_json_path: str,
    image_dir: str,
    max_samples: int = 0,
    prefetch_depth: int = 4,
) -> Dict[str, Any]:
    """
    在图像类 benchmark (MathVista/MathVision) 上评测。
    支持多卡分布式 (自动分片) + CPU/GPU 异步 pipeline (后台预取图像)。

    图像类 benchmark 不使用 motion module, 仅评测基线能力。
    支持 multi_choice 和 free_form 两种题型。
    """
    rank = _get_rank()
    world = _get_world_size()

    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    if max_samples > 0:
        eval_data = eval_data[:max_samples]

    for i, item in enumerate(eval_data):
        item["_global_idx"] = i
        item["_image_dir"] = image_dir

    total_samples = len(eval_data)
    local_data = _shard_data(eval_data)
    logger.info("评测 %s (图像): 总样本 %d, rank %d/%d 分得 %d 条",
                benchmark_name, total_samples, rank, world, len(local_data))

    results = []
    correct = 0
    total = 0
    per_category = {}
    inference_times = []

    prefetcher = _ImagePrefetcher(local_data, prefetch_depth=prefetch_depth)
    show_progress = (rank == 0)
    pbar = tqdm(total=len(local_data), desc=f"Eval {benchmark_name}", disable=not show_progress)

    for item, image in prefetcher:
        if image is None:
            raise RuntimeError(f"Image load failed for sample {item.get('_global_idx')}")
        global_idx = item["_global_idx"]
        question = item.get("question", "")
        options = item.get("options", [])
        answer = item.get("answer", "")
        question_type = item.get("question_type", "multi_choice" if options else "free_form")
        answer_type = item.get("answer_type", "text")
        category = item.get("task_type", item.get("subject", "unknown"))
        image_name = item.get("image", "")

        is_multi_choice = (question_type == "multi_choice" and len(options) > 0)

        t_infer = time.time()
        try:
            if model_type in ("qwen", "qwen3"):
                response = generate_image_qwen(
                    model, processor, tokenizer, image, question, options,
                    is_multi_choice=is_multi_choice,
                )
            elif model_type == "llava":
                response = generate_image_llava(
                    model, processor, tokenizer, image, question, options,
                    is_multi_choice=is_multi_choice,
                )
            elif model_type == "internvl3":
                response = generate_image_internvl3(
                    model, tokenizer, image, question, options,
                    is_multi_choice=is_multi_choice,
                )
            else:
                response = ""
        except Exception as e:
            raise RuntimeError(f"Inference failed for image sample {global_idx}; no guessed answer was scored.") from e
        infer_time = time.time() - t_infer
        inference_times.append(infer_time)

        if is_multi_choice:
            pred = extract_answer_letter(response, len(options))
            is_correct = _check_mc_answer(pred, answer, options)
        else:
            pred = extract_free_form_answer(response)
            is_correct = _check_free_form_answer(pred, answer, answer_type)

        if is_correct:
            correct += 1
        total += 1

        if category not in per_category:
            per_category[category] = {"correct": 0, "total": 0}
        per_category[category]["total"] += 1
        if is_correct:
            per_category[category]["correct"] += 1

        results.append({
            "idx": global_idx,
            "image": image_name,
            "question": question,
            "options": options,
            "answer": answer,
            "prediction": pred,
            "response": response,
            "correct": is_correct,
            "category": category,
            "question_type": question_type,
            "inference_time": round(infer_time, 3),
        })

        pbar.update(1)
        if total % 50 == 0 and total > 0:
            acc = correct / total * 100
            avg_infer = sum(inference_times) / len(inference_times)
            logger.info("[rank %d] [%d/%d] Acc: %.1f%% (%d/%d) | AvgInfer: %.2fs",
                       rank, total, len(local_data), acc, correct, total, avg_infer)

    pbar.close()

    # 多卡聚合结果
    all_results = _gather_results(results)

    if rank == 0:
        all_correct = sum(1 for r in all_results if r["correct"])
        all_total = len(all_results)
        all_per_category: Dict[str, Dict[str, Any]] = {}
        all_inference_times = []

        for r in all_results:
            cat = r["category"]
            if cat not in all_per_category:
                all_per_category[cat] = {"correct": 0, "total": 0}
            all_per_category[cat]["total"] += 1
            if r["correct"]:
                all_per_category[cat]["correct"] += 1
            all_inference_times.append(r["inference_time"])

        for cat in all_per_category:
            c = all_per_category[cat]
            c["accuracy"] = c["correct"] / c["total"] * 100 if c["total"] > 0 else 0.0

        overall_acc = all_correct / all_total * 100 if all_total > 0 else 0.0

        timing_stats = {}
        if all_inference_times:
            timing_stats = {
                "avg_inference_time": round(sum(all_inference_times) / len(all_inference_times), 3),
                "median_inference_time": round(sorted(all_inference_times)[len(all_inference_times) // 2], 3),
                "total_inference_time": round(sum(all_inference_times), 1),
                "num_gpus": world,
            }

        return {
            "benchmark": benchmark_name,
            "accuracy": overall_acc,
            "total": all_total,
            "correct": all_correct,
            "per_category": all_per_category,
            "timing": timing_stats,
            "results": all_results,
        }
    else:
        return {
            "benchmark": benchmark_name,
            "accuracy": 0.0,
            "total": 0,
            "correct": 0,
            "per_category": {},
            "timing": {},
            "results": [],
        }


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Himmel-VLM 统一评测脚本 (支持多卡并行 + 异步 pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["qwen", "qwen3", "llava", "internvl3"],
                        help="模型类型")
    parser.add_argument("--model-path", type=str, required=True,
                        help="基线模型路径")
    parser.add_argument("--delta-checkpoint", type=str, default=None,
                        help="Delta checkpoint 路径 (包含 motion_module.pt 和 lora_weights/)")
    parser.add_argument("--stage1", action="store_true", default=False,
                        help="Stage 1 评测模式: 只加载 motion_module (无 LoRA), "
                             "需同时指定 --delta-checkpoint 指向 Stage 1 checkpoint")
    parser.add_argument("--benchmark", type=str, default="all",
                        choices=["video_mme", "mvbench", "mlvu",
                                 "mathvista", "mathvision",
                                 "all", "all_video", "all_image"],
                        help="评测 benchmark (all=全部, all_video=视频类, all_image=图像类)")
    parser.add_argument("--max-keyframes", type=int, default=32,
                        help="最大关键帧数 (仅视频类)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="最大评测样本数 (0=全部, 用于调试)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备 (多卡时自动设为 cuda:local_rank)")
    parser.add_argument("--prefetch-depth", type=int, default=2,
                        help="预取队列深度 (CPU 预处理异步预取的缓冲数量, 默认 2)")
    parser.add_argument("--sampling-strategy", type=str, default="uniform",
                        choices=["uniform", "dopamine_hierarchical"],
                        help="training-free 视频采样策略")
    parser.add_argument("--keyframe-fps", type=float, default=1.0,
                        help="基础关键帧采样率 (默认 1fps)")
    parser.add_argument("--memory-keyframes", type=int, default=12,
                        help="分层采样时的长期记忆锚点预算")
    parser.add_argument("--dense-local-frames", type=int, default=3,
                        help="每个高显著区间追加的局部加密帧数")
    parser.add_argument("--memory-diffs-per-interval", type=int, default=1,
                        help="非显著长期记忆区间保留的 diff 数")
    parser.add_argument("--dense-diffs-per-interval", type=int, default=4,
                        help="高显著聚焦区间保留的 diff 数")
    parser.add_argument("--diff-mode", type=str, default="abs",
                        choices=["abs", "signed", "signed_accum", "luma_motion_edge"],
                        help="motion diff 表示模式，需与训练时保持一致")
    parser.add_argument("--question-aware-sampling", type=int, default=1, choices=[0, 1],
                        help="是否启用基于问题语义的时间采样偏置 (1=启用)")
    parser.add_argument("--motion-encoder-type", type=str, default="auto",
                        choices=["auto", "resnet", "motionssm", "siglip", "shared_vit"],
                        help="Delta motion encoder 类型；auto 时根据 checkpoint 路径名推断")
    parser.add_argument("--siglip-model-path", type=str, default="google/siglip-base-patch16-224",
                        help="SigLIP motion encoder 使用的模型路径")
    parser.add_argument("--freeze-motion-encoder", type=int, default=0, choices=[0, 1],
                        help="是否按训练配置冻结 motion encoder (主要用于保持构造参数一致)")
    parser.add_argument("--motion-use-cross-attention", type=int, default=0, choices=[0, 1],
                        help="shared_vit 是否启用 keyframe->motion cross attention")
    parser.add_argument("--motion-input-mode", type=str, default="rgb_diff",
                        choices=["rgb_diff", "compressed"],
                        help="运动输入模式，需与训练时保持一致")
    parser.add_argument("--compressed-backend", type=str, default="coviar",
                        help="compressed 模式后端")
    parser.add_argument("--visual-stream-mode", type=str, default="iframe",
                        choices=["iframe", "keyframe"],
                        help="compressed 模式视觉流类型")
    parser.add_argument("--compressed-gop-size", type=int, default=12,
                        help="compressed 模式 GOP 大小")
    parser.add_argument("--use-motion-vectors", type=int, default=1, choices=[0, 1],
                        help="compressed 模式是否使用 motion vectors")
    parser.add_argument("--use-residuals", type=int, default=1, choices=[0, 1],
                        help="compressed 模式是否使用 residuals")
    parser.add_argument("--accumulate-motion-vectors", type=int, default=1, choices=[0, 1],
                        help="compressed 模式是否读取 accumulated motion vectors")
    parser.add_argument("--accumulate-residuals", type=int, default=0, choices=[0, 1],
                        help="compressed 模式是否读取 accumulated residuals")
    parser.add_argument("--motion-vector-clip", type=float, default=20.0,
                        help="compressed 模式 motion vector 裁剪阈值")
    parser.add_argument("--compressed-fusion-mode", type=str, default="gated",
                        choices=["gated", "concat_mlp", "sum"],
                        help="compressed 双流融合方式，需与训练时保持一致")
    parser.add_argument("--compressed-use-iframe-context", type=int, default=0, choices=[0, 1])
    parser.add_argument("--compressed-use-iframe-encoder", type=int, default=0, choices=[0, 1])
    parser.add_argument("--alignment-motion-pool", type=str, default="mean",
                        choices=["mean", "last", "linear"],
                        help="Stage 1 alignment 的 motion pooling 方式，需与训练时保持一致")
    # ---- Training-Free 方法 ----
    parser.add_argument("--tf-method", type=str, default="none",
                        choices=["none"],
                        help="Qwen training-free 长视频方法 (仅对 baseline Qwen/Qwen3 生效)")
    parser.add_argument("--tf-config", type=str, default="{}",
                        help="Training-free 方法的 JSON 配置, 如 '{\"panel_rows\":3,\"panel_cols\":3,\"max_panels\":8}'")
    args = parser.parse_args()

    # ---- 分布式初始化 ----
    # 如果通过 torchrun 启动, 环境变量 RANK/LOCAL_RANK/WORLD_SIZE 会被自动设置
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        rank = 0
        world = 1

    # 日志: 只在 rank 0 显示 INFO, 其他 rank 只显示 WARNING
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[%(asctime)s] [Rank {rank}] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    is_delta = args.delta_checkpoint is not None
    is_stage1 = args.stage1
    if is_stage1 and not is_delta:
        logger.warning("--stage1 需要同时指定 --delta-checkpoint, 回退到 baseline 模式")
        is_stage1 = False
    mode_str = "stage1" if is_stage1 else ("delta" if is_delta else "baseline")

    # 解析 training-free 配置
    try:
        tf_config = json.loads(args.tf_config) if args.tf_config else {}
    except json.JSONDecodeError:
        logger.warning("--tf-config JSON 解析失败, 使用空配置: %s", args.tf_config)
        tf_config = {}

    resolved_motion_encoder_type = (
        _infer_motion_encoder_type(args.delta_checkpoint)
        if args.motion_encoder_type == "auto"
        else args.motion_encoder_type
    )
    delta_model_config = {
        "motion_encoder_type": resolved_motion_encoder_type,
        "siglip_model_path": args.siglip_model_path,
        "freeze_motion_encoder": bool(args.freeze_motion_encoder),
        "motion_use_cross_attention": bool(args.motion_use_cross_attention),
        "motion_input_mode": args.motion_input_mode,
        "use_motion_vectors": bool(args.use_motion_vectors),
        "use_residuals": bool(args.use_residuals),
        "compressed_fusion_mode": args.compressed_fusion_mode,
        "compressed_use_iframe_context": bool(args.compressed_use_iframe_context),
        "compressed_use_iframe_encoder": bool(args.compressed_use_iframe_encoder),
        "alignment_motion_pool": args.alignment_motion_pool,
    }

    logger.info("=" * 60)
    logger.info("Himmel-VLM 评测")
    logger.info("  模型类型: %s", args.model_type)
    logger.info("  模型路径: %s", args.model_path)
    logger.info("  模式: %s", mode_str)
    logger.info("  GPU 数量: %d", world)
    logger.info("  预取深度: %d", args.prefetch_depth)
    logger.info("  采样策略: %s", args.sampling_strategy)
    logger.info("  Diff 模式: %s", args.diff_mode)
    logger.info("  Motion 输入: %s | Encoder: %s", args.motion_input_mode, resolved_motion_encoder_type)
    if args.motion_input_mode == "compressed":
        logger.info(
            "  Compressed 配置: backend=%s visual=%s gop=%d mv=%d residual=%d fusion=%s",
            args.compressed_backend,
            args.visual_stream_mode,
            args.compressed_gop_size,
            args.use_motion_vectors,
            args.use_residuals,
            args.compressed_fusion_mode,
        )
    if args.sampling_strategy != "uniform":
        logger.info(
            "  分层采样配置: keyframe_fps=%.2f, memory_kf=%d, dense_local=%d, memory_diffs=%d, dense_diffs=%d, question_aware=%d",
            args.keyframe_fps,
            args.memory_keyframes,
            args.dense_local_frames,
            args.memory_diffs_per_interval,
            args.dense_diffs_per_interval,
            args.question_aware_sampling,
        )
    if is_delta:
        logger.info("  Delta checkpoint: %s", args.delta_checkpoint)
    if args.tf_method != "none":
        logger.info("  Training-Free 方法: %s", args.tf_method)
        logger.info("  TF 配置: %s", tf_config)
    if is_stage1:
        logger.info("  Stage 1 模式: 只加载 motion_module (无 LoRA)")
    logger.info("  Benchmark: %s", args.benchmark)
    logger.info("=" * 60)

    # 确定评测 benchmark
    if args.benchmark == "all":
        benchmarks = list(BENCHMARK_CONFIGS.keys())
    elif args.benchmark == "all_video":
        benchmarks = VIDEO_BENCHMARKS
    elif args.benchmark == "all_image":
        benchmarks = IMAGE_BENCHMARKS
    else:
        benchmarks = [args.benchmark]

    # 加载模型 (每张卡独立加载到自己的 device)
    has_video_bench = any(
        BENCHMARK_CONFIGS.get(b, {}).get("type") == "video"
        for b in benchmarks
    )
    only_image_bench = not has_video_bench

    t0 = time.time()
    if is_stage1:
        # Stage 1 模式: 只加载 baseline + motion_module (无 LoRA)
        logger.info("Stage 1 评测模式: baseline LLM + 训练后 motion_module")
        model, processor, tokenizer = load_model_stage1(
            args.model_type, args.model_path, args.delta_checkpoint, args.device,
            delta_model_config=delta_model_config,
        )
    elif is_delta:
        if only_image_bench:
            logger.info("纯图像 benchmark, 加载 baseline + LoRA (跳过 motion_module)")
            model, processor, tokenizer = load_model_lora_only(
                args.model_type, args.model_path, args.delta_checkpoint, args.device,
            )
        else:
            model, processor, tokenizer = load_model_delta(
                args.model_type,
                args.model_path,
                args.delta_checkpoint,
                args.device,
                delta_model_config=delta_model_config,
            )
    else:
        model, processor, tokenizer = load_model_baseline(
            args.model_type, args.model_path, args.device,
        )
    logger.info("模型加载完成, 耗时 %.1f 秒", time.time() - t0)

    # 多卡同步: 等所有 rank 加载模型完毕
    if _is_dist():
        dist.barrier()

    # 确定输出目录 (只 rank 0 创建)
    if args.output_dir is None:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        args.output_dir = os.path.join(
            PROJECT_ROOT, "eval_results", f"{model_name}_{mode_str}"
        )
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # 评测
    all_results = {}
    for bench_name in benchmarks:
        cfg = BENCHMARK_CONFIGS[bench_name]
        eval_json = cfg["eval_json"]

        if not os.path.isfile(eval_json):
            if rank == 0:
                logger.warning("跳过 %s: 评测 JSON 不存在 (%s)", bench_name, eval_json)
                logger.warning("请先运行: python prepare_data.py --benchmarks")
            continue

        t0 = time.time()

        if cfg["type"] == "image":
            result = evaluate_image_benchmark(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                model_type=args.model_type,
                benchmark_name=bench_name,
                eval_json_path=eval_json,
                image_dir=cfg["image_dir"],
                max_samples=args.max_samples,
                prefetch_depth=args.prefetch_depth * 2,  # 图像预取更快, 加大队列
            )
        else:
            result = evaluate_benchmark(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                model_type=args.model_type,
                benchmark_name=bench_name,
                eval_json_path=eval_json,
                video_dir=cfg["video_dir"],
                is_delta=is_delta or is_stage1,
                max_keyframes=args.max_keyframes,
                max_samples=args.max_samples,
                prefetch_depth=args.prefetch_depth,
                sampling_strategy=args.sampling_strategy,
                keyframe_fps=args.keyframe_fps,
                memory_keyframes=args.memory_keyframes,
                dense_local_frames=args.dense_local_frames,
                question_aware_sampling=bool(args.question_aware_sampling),
                memory_diffs_per_interval=args.memory_diffs_per_interval,
                dense_diffs_per_interval=args.dense_diffs_per_interval,
                diff_mode=args.diff_mode,
                motion_input_mode=args.motion_input_mode,
                compressed_backend=args.compressed_backend,
                visual_stream_mode=args.visual_stream_mode,
                compressed_gop_size=args.compressed_gop_size,
                use_motion_vectors=bool(args.use_motion_vectors),
                use_residuals=bool(args.use_residuals),
                accumulate_motion_vectors=bool(args.accumulate_motion_vectors),
                accumulate_residuals=bool(args.accumulate_residuals),
                motion_vector_clip=args.motion_vector_clip,
                tf_method=args.tf_method,
                tf_config=tf_config,
            )
        elapsed = time.time() - t0

        result["metric_name"] = (
            "local_diagnostic_freeform_accuracy_not_official_benchmark_score"
            if cfg["type"] == "image" else "micro_accuracy"
        )
        if bench_name == "mlvu":
            result["metric_name"] = "micro_accuracy_not_official_MLVU_M_Avg"

        all_results[bench_name] = result

        # 只在 rank 0 打印和保存
        if rank == 0:
            logger.info("\n" + "=" * 60)
            logger.info("评测结果: %s", bench_name)
            logger.info("=" * 60)
            logger.info("  总体准确率: %.2f%% (%d/%d)", result["accuracy"], result["correct"], result["total"])
            logger.info("  耗时: %.1f 分钟 (wall-clock, %d GPUs)", elapsed / 60, world)
            timing = result.get("timing", {})
            if timing:
                logger.info("  推理统计:")
                logger.info("    平均推理时间: %.3fs | 中位数: %.3fs",
                           timing.get("avg_inference_time", 0), timing.get("median_inference_time", 0))
                if "avg_keyframes" in timing:
                    logger.info("    平均关键帧: %.1f | 平均差分帧: %.1f",
                               timing.get("avg_keyframes", 0), timing.get("avg_diffs", 0))
                    logger.info("    平均长期锚点: %.1f | 平均聚焦帧: %.1f | 采样策略: %s",
                               timing.get("avg_long_term_keyframes", 0),
                               timing.get("avg_focus_keyframes", 0),
                               timing.get("sampling_strategy", args.sampling_strategy))
                if "avg_preprocess_time" in timing:
                    logger.info("    平均预处理时间: %.3fs", timing.get("avg_preprocess_time", 0))
            logger.info("  各类别:")
            for cat, stats in sorted(result["per_category"].items(), key=lambda x: -x[1]["accuracy"]):
                logger.info("    %s: %.1f%% (%d/%d)", cat, stats["accuracy"], stats["correct"], stats["total"])
            logger.info("=" * 60)

            result_file = os.path.join(args.output_dir, f"{bench_name}_results.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info("结果已保存: %s", result_file)

    # 输出总结 (rank 0)
    if rank == 0:
        logger.info("\n" + "=" * 60)
        logger.info("总结")
        logger.info("=" * 60)
        for bench_name, result in all_results.items():
            timing = result.get("timing", {})
            timing_str = ""
            if timing:
                timing_str = f" | AvgInfer: {timing.get('avg_inference_time', 0):.3f}s"
                if "avg_keyframes" in timing:
                    timing_str += f" | AvgKF: {timing.get('avg_keyframes', 0):.1f}"
            logger.info("  %s: %.2f%% (%d/%d)%s",
                        bench_name, result["accuracy"], result["correct"], result["total"], timing_str)

        summary = {
            "model_type": args.model_type,
            "model_path": args.model_path,
            "mode": mode_str,
            "delta_checkpoint": args.delta_checkpoint,
            "max_keyframes": args.max_keyframes,
            "sampling_strategy": args.sampling_strategy,
            "keyframe_fps": args.keyframe_fps,
            "memory_keyframes": args.memory_keyframes,
            "dense_local_frames": args.dense_local_frames,
            "memory_diffs_per_interval": args.memory_diffs_per_interval,
            "dense_diffs_per_interval": args.dense_diffs_per_interval,
            "question_aware_sampling": bool(args.question_aware_sampling),
            "tf_method": args.tf_method,
            "tf_config": tf_config,
            "num_gpus": world,
            "prefetch_depth": args.prefetch_depth,
            "results": {
                name: {
                    "accuracy": r["accuracy"],
                    "total": r["total"],
                    "correct": r["correct"],
                    "timing": r.get("timing", {}),
                }
                for name, r in all_results.items()
            },
        }
        summary_file = os.path.join(args.output_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("总结已保存: %s", summary_file)
        logger.info("=" * 60)

    # 等所有 rank 完成
    if _is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
