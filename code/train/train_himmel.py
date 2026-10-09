#!/usr/bin/env python3
"""
Himmel-VLM: 训练入口脚本
============================
支持四种模型:
- Qwen2.5-VL-7B + Delta (旗舰)
- Qwen3-VL-8B + Delta (新旗舰, Interleaved-MRoPE + DeepStack)
- LLaVA-OneVision-7B + Delta (对照)
- InternVL3-8B + Delta (InternVL3, ViT-MLP-LLM)

支持两个训练阶段:
- Stage 1: Alignment (冻结 LLM+ViT, 只训练 MotionEncoder+Projector)
- Stage 2: SFT (冻结 ViT, 训练 MotionEncoder+Projector + LLM)
    - sft_mode="lora": 使用 LoRA 微调 LLM
    - sft_mode="full": 全量微调 LLM (需要 ZeRO-3)

启动方式:
    deepspeed --num_gpus 8 train/train_himmel.py --config train/configs/stage1_qwen.json
    deepspeed --num_gpus 8 train/train_himmel.py --config train/configs/stage2_qwen_full.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
import deepspeed

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.delta_components import get_trainable_params_info, freeze_module, unfreeze_module
from data.delta_video_loader import DeltaVideoDataset

logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

DEFAULT_CONFIG = {
    # 模型
    "model_type": "qwen",              # "qwen" 或 "llava"
    "model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "stage": 1,                         # 1=Alignment, 2=SFT

    # Motion Encoder (可插拔)
    "motion_encoder_type": "resnet",    # "resnet", "motionssm", "siglip", "shared_vit"
    "siglip_model_path": "google/siglip-base-patch16-224",
    "freeze_motion_encoder": False,      # 是否冻结 motion encoder backbone (SigLIP 预训练模式常设为 true)
    "siglip_unfreeze_last_n_layers": 0,  # SigLIP 冻结主体时，额外解冻最后 N 个 vision layers
    "motion_use_cross_attention": False, # shared_vit 模式是否使用 cross-attention
    "shared_vit_input_mode": "video",  # "video" | "images"，shared_vit 对 diff 的编码粒度

    # LoRA (Stage 2, sft_mode="lora")
    "sft_mode": "lora",                 # "lora" 或 "full" (Stage 2 专用)
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # 数据
    "data_path": "",                    # JSON 数据文件
    "video_root": "",                   # 视频根目录
    "keyframe_fps": 1.0,
    "num_diffs_per_interval": 4,
    "motion_input_mode": "rgb_diff",   # "rgb_diff" | "compressed"
    "diff_mode": "abs",                # RGB diff 模式: "abs" | "signed" | "signed_accum" | "luma_motion_edge"
    "compressed_backend": "coviar",
    "visual_stream_mode": "iframe",    # "iframe" | "keyframe"
    "compressed_gop_size": 12,
    "use_motion_vectors": True,
    "use_residuals": True,
    "accumulate_motion_vectors": True,
    "accumulate_residuals": False,
    "motion_vector_clip": 20.0,
    "compressed_fusion_mode": "gated", # "gated" | "concat_mlp" | "sum"
    "compressed_use_iframe_context": False,
    "compressed_use_iframe_encoder": False,
    "compressed_fused_alignment_weight": 1.0,
    "compressed_iframe_alignment_weight": 0.25,
    "compressed_mv_alignment_weight": 0.25,
    "compressed_residual_alignment_weight": 0.25,
    "alignment_motion_pool": "mean",  # "mean" | "last" | "linear"
    "alignment_interval_diff_weight": 1.0,
    "alignment_decoded_frame_weight": 1.0,
    "alignment_gather_global_negatives": True,
    "max_keyframes": 32,
    "max_length": 2048,
    "dataloader_num_workers": 8,
    "dataloader_prefetch_factor": 4,
    "dataloader_persistent_workers": True,
    "decord_num_threads": 4,
    "allow_compressed_proxy_fallback": False,

    # 训练参数
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 16,  # 有效 batch = 1*8*16 = 128
    "num_epochs": 1,
    "learning_rate": 1e-4,              # Stage1: 1e-4, Stage2: 2e-5
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "freeze_spatial_encoder_steps": 0,  # Stage1 前 N 个 optimizer steps 冻结 MotionSSM spatial_encoder
    "bootstrap_projector_from_alignment": True,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "gradient_checkpointing": True,

    # DeepSpeed
    "deepspeed_stage": 2,              # ZeRO Stage 2

    # 保存
    "output_dir": "",
    "save_steps": 500,
    "logging_steps": 10,

    # Eval & Early Stopping (Stage 2)
    "val_ratio": 0.05,              # 验证集比例 (5%)
    "max_val_samples": 500,         # 验证集最大样本数 (避免 eval 耗时过长)
    "eval_steps": 200,              # 每 N 个 optimizer step 做一次 eval
    "early_stopping_patience": 3,   # val loss 连续 N 次不降则停止
    "focal_loss_gamma": 2.0,        # Focal Loss gamma, 0=标准CE
}


def load_config(config_path: str) -> dict:
    """加载配置, 合并默认值，并支持少量环境变量覆盖。"""
    config = DEFAULT_CONFIG.copy()
    if not config_path:
        raise ValueError("必须指定 --config 配置文件路径")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r") as f:
        user_config = json.load(f)
    config.update(user_config)

    env_override_map = {
        "HIMMEL_MODEL_PATH": "model_name_or_path",
        "HIMMEL_STAGE1_CHECKPOINT": "stage1_checkpoint",
        "HIMMEL_DATA_PATH_OVERRIDE": "data_path",
        "HIMMEL_VIDEO_ROOT_OVERRIDE": "video_root",
        "HIMMEL_OUTPUT_DIR_OVERRIDE": "output_dir",
    }
    for env_name, config_key in env_override_map.items():
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            logger.info("应用环境变量覆盖: %s -> %s=%s", env_name, config_key, env_value)
            config[config_key] = env_value

    logger.info("已加载配置: %s (model_type=%s, model_path=%s)",
                config_path, config.get("model_type"), config.get("model_name_or_path"))
    return config


def _should_freeze_stage1_spatial_encoder(config: dict) -> bool:
    return bool(
        config.get("stage") == 1
        and config.get("alignment_loss", True)
        and config.get("motion_encoder_type") == "motionssm"
        and int(config.get("freeze_spatial_encoder_steps", 0)) > 0
    )



def _build_model_parameters_for_deepspeed(model: nn.Module, config: dict):
    """构造传给 DeepSpeed 的参数列表。

    Stage 1 两段式解冻时，需要把暂时冻结的 spatial_encoder 也提前注册进 optimizer，
    后续解冻后无需重建 DeepSpeed engine。
    """
    use_alignment = (config.get("stage") == 1 and config.get("alignment_loss", True))
    freeze_stage1_spatial = _should_freeze_stage1_spatial_encoder(config)

    selected_params = []
    seen = set()
    for name, param in model.named_parameters():
        include = param.requires_grad
        if (
            not include
            and use_alignment
            and freeze_stage1_spatial
            and "motion_module" in name
            and ".encoder.spatial_encoder." in name
        ):
            include = True

        if include and id(param) not in seen:
            selected_params.append(param)
            seen.add(id(param))

    logger.info(
        "DeepSpeed 参数构造: selected=%d, stage1_spatial_warmup=%s",
        len(selected_params),
        freeze_stage1_spatial,
    )
    return selected_params



def _copy_file_parallel(src_path: str, dst_path: str,
                        chunk_size: int = 64 * 1024 * 1024,
                        max_workers: int = 64) -> int:
    """
    多线程分块并行拷贝单个文件 (参考 fast_sync.py)。
    使用 pread/pwrite 线程安全读写, 临时文件 + rename 保证原子性。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    file_size = os.path.getsize(src_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if file_size == 0:
        open(dst_path, "wb").close()
        return 0

    tmp_path = dst_path + ".synctmp"
    try:
        if file_size <= chunk_size:
            src_fd = os.open(src_path, os.O_RDONLY)
            dst_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                buf = os.pread(src_fd, file_size, 0)
                os.pwrite(dst_fd, buf, 0)
            finally:
                os.close(src_fd)
                os.close(dst_fd)
        else:
            src_fd = os.open(src_path, os.O_RDONLY)
            dst_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.ftruncate(dst_fd, file_size)
                chunks = []
                offset = 0
                while offset < file_size:
                    length = min(chunk_size, file_size - offset)
                    chunks.append((offset, length))
                    offset += length

                actual_workers = min(max_workers, len(chunks))
                with ThreadPoolExecutor(max_workers=actual_workers) as pool:
                    futures = [
                        pool.submit(os.pread, src_fd, ln, off)
                        for off, ln in chunks
                    ]
                    # 按提交顺序收集结果并写入
                    for fut, (off, ln) in zip(futures, chunks):
                        buf = fut.result()
                        os.pwrite(dst_fd, buf, off)
            finally:
                os.close(src_fd)
                os.close(dst_fd)

        os.rename(tmp_path, dst_path)
        # 保留源文件时间戳
        st = os.stat(src_path)
        os.utime(dst_path, (st.st_atime, st.st_mtime))
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise

    return file_size


def _copy_to_local(src_path: str) -> str:
    """
    将 group 盘文件复制到 /tmp 本地盘以加速读取。
    使用多线程分块并行拷贝 (pread/pwrite), 大幅加速大文件传输。
    如果文件已在本地盘或本地缓存已存在且大小一致, 直接返回本地路径。

    多 rank 安全: 使用文件锁确保只有一个进程执行复制, 其他进程等待完成后直接使用。
    """
    import fcntl
    import time as _time

    LOCAL_TMP = os.path.join(os.environ.get("HIMMEL_CACHE_DIR", os.path.expanduser("~/.cache/himmel")), "checkpoints")

    # 已经在本地盘, 无需复制
    if src_path.startswith("/tmp/"):
        return src_path

    # 构造本地缓存路径: 保留 group 盘的相对结构
    basename = os.path.basename(src_path)
    parent = os.path.dirname(src_path)
    parts = parent.rstrip("/").split("/")
    sub_key = "/".join(parts[-3:]) if len(parts) >= 3 else "/".join(parts[-2:])
    local_path = os.path.join(LOCAL_TMP, sub_key, basename)
    src_size = os.path.getsize(src_path)

    # 检查本地缓存是否已存在且大小一致
    if os.path.isfile(local_path) and os.path.getsize(local_path) == src_size:
        logger.info("本地缓存已存在, 跳过复制: %s", local_path)
        return local_path

    # 用文件锁确保多 rank 只有一个进程执行复制
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    lock_path = local_path + ".lock"
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # 拿到锁后再检查一次 (可能其他 rank 已经复制完了)
        if os.path.isfile(local_path) and os.path.getsize(local_path) == src_size:
            logger.info("本地缓存已由其他进程复制完成: %s", local_path)
            return local_path

        src_size_gb = src_size / (1024 ** 3)
        logger.info("多线程并行复制 checkpoint 到本地盘: %.1f GB (64 workers, 64MB chunks)", src_size_gb)
        logger.info("  src:  %s", src_path)
        logger.info("  dest: %s", local_path)

        t0 = _time.time()
        _copy_file_parallel(src_path, local_path, chunk_size=64 * 1024 * 1024, max_workers=64)
        elapsed = _time.time() - t0
        logger.info("复制完成, 耗时 %.1f 秒 (%.1f GB/s)", elapsed, src_size_gb / max(elapsed, 0.1))
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        # 清理锁文件 (best effort)
        try:
            os.unlink(lock_path)
        except OSError:
            pass

    return local_path


def _load_motion_module_from_checkpoint(ckpt_path: str) -> dict | None:
    """
    从 Stage 1 checkpoint 加载 motion_module 权重。
    支持两种格式:
      1. 直接的 motion_module.pt 文件 (final/ 目录下)
      2. DeepSpeed checkpoint 目录 (checkpoint-XXXX/), 从
         mp_rank_00_model_states.pt 中提取 motion_module.* 前缀的权重

    对于 group 盘上的大文件, 会先复制到 /tmp 本地盘再加载。

    参数:
        ckpt_path: 可以是 .pt 文件路径, 也可以是 DeepSpeed checkpoint 目录路径

    返回:
        motion_module 的 state_dict, 或 None (加载失败)
    """
    # 情况 1: 直接指向 .pt 文件
    if os.path.isfile(ckpt_path) and ckpt_path.endswith(".pt"):
        local_path = _copy_to_local(ckpt_path)
        logger.info("加载 Stage 1 motion_module 权重 (直接文件): %s", local_path)
        return torch.load(local_path, map_location="cpu", weights_only=True)

    # 情况 2: DeepSpeed checkpoint 目录
    if os.path.isdir(ckpt_path):
        # DeepSpeed checkpoint 结构: checkpoint-XXXX/global_stepYYYY/mp_rank_00_model_states.pt
        model_file = None

        # 先在目录内直接查找
        direct = os.path.join(ckpt_path, "mp_rank_00_model_states.pt")
        if os.path.isfile(direct):
            model_file = direct
        else:
            # 查找 global_step* 子目录
            subdirs = sorted(
                [d for d in os.listdir(ckpt_path)
                 if d.startswith("global_step") and os.path.isdir(os.path.join(ckpt_path, d))],
                key=lambda x: int(x.replace("global_step", "")) if x.replace("global_step", "").isdigit() else 0,
                reverse=True,  # 取最大 step
            )
            for subdir in subdirs:
                candidate = os.path.join(ckpt_path, subdir, "mp_rank_00_model_states.pt")
                if os.path.isfile(candidate):
                    model_file = candidate
                    break

        if model_file is None:
            logger.warning("DeepSpeed checkpoint 目录中未找到 mp_rank_00_model_states.pt: %s", ckpt_path)
            return None

        # 检查是否已有缓存的提取结果 (小文件, ~100MB)
        # 优先检查本地盘缓存, 再检查 group 盘缓存
        local_cache = os.path.join(os.environ.get("HIMMEL_CACHE_DIR", os.path.expanduser("~/.cache/himmel")), "checkpoints", "motion_module_extracted")
        ckpt_id = os.path.basename(os.path.dirname(os.path.dirname(model_file)))  # e.g. checkpoint-500
        local_cache_file = os.path.join(local_cache, f"{ckpt_id}_motion_module.pt")
        group_cache_file = os.path.join(os.path.dirname(model_file), "motion_module_extracted.pt")

        if os.path.isfile(local_cache_file):
            logger.info("发现本地缓存的 motion_module 权重: %s", local_cache_file)
            return torch.load(local_cache_file, map_location="cpu", weights_only=True)
        if os.path.isfile(group_cache_file):
            logger.info("发现 group 盘缓存的 motion_module 权重, 复制到本地...")
            local_path = _copy_to_local(group_cache_file)
            return torch.load(local_path, map_location="cpu", weights_only=True)

        # 先复制大文件到本地盘, 再从本地盘加载 (避免 group 盘慢速随机读)
        local_model_file = _copy_to_local(model_file)

        file_size_gb = os.path.getsize(local_model_file) / (1024 ** 3)
        logger.info("从本地 DeepSpeed checkpoint 提取 motion_module: %s (%.1f GB)",
                    local_model_file, file_size_gb)
        full_state = torch.load(local_model_file, map_location="cpu", weights_only=False)
        logger.info("checkpoint 加载完成, 共 %d 个 key, 开始提取 motion_module...",
                    len(full_state.get("module", full_state)))

        # DeepSpeed 存储格式: {"module": {full_model_state_dict}, ...}
        if "module" in full_state:
            full_state = full_state["module"]

        # 提取 motion_module.* 前缀的 key, 并去掉前缀
        prefix = "motion_module."
        motion_sd = {}
        for k, v in full_state.items():
            if k.startswith(prefix):
                motion_sd[k[len(prefix):]] = v

        # 释放大 state_dict 内存
        del full_state

        if not motion_sd:
            logger.warning("DeepSpeed checkpoint 中未找到 motion_module.* 权重")
            return None

        logger.info("从 DeepSpeed checkpoint 提取了 %d 个 motion_module 参数", len(motion_sd))

        # 缓存到本地盘 (~100MB, 后续直接加载)
        try:
            os.makedirs(os.path.dirname(local_cache_file), exist_ok=True)
            torch.save(motion_sd, local_cache_file)
            logger.info("已缓存 motion_module 权重到本地: %s", local_cache_file)
        except Exception as e:
            logger.warning("缓存 motion_module 权重失败: %s", e)

        # 也缓存到 group 盘 (供其他节点使用)
        try:
            torch.save(motion_sd, group_cache_file)
            logger.info("已缓存 motion_module 权重到 group 盘: %s", group_cache_file)
        except Exception as e:
            logger.warning("缓存到 group 盘失败 (非关键): %s", e)

        # 清理本地盘上的大文件副本, 释放 /tmp 空间
        try:
            if local_model_file != model_file and os.path.isfile(local_model_file):
                os.remove(local_model_file)
                logger.info("已清理本地大文件副本: %s", local_model_file)
        except Exception:
            pass

        return motion_sd

    logger.warning("Stage 1 checkpoint 路径不存在: %s", ckpt_path)
    return None


# ============================================================================
# 模型加载
# ============================================================================

def load_model_and_processor(config: dict):
    """根据配置加载模型和 processor。"""
    model_type = config["model_type"]
    model_path = config["model_name_or_path"]
    stage = config["stage"]

    # 提取 motion encoder / compressed-domain 配置 (传递给各 delta model 的 __init__)
    motion_encoder_type = config.get("motion_encoder_type", "resnet")
    siglip_model_path = config.get("siglip_model_path", "google/siglip-base-patch16-224")
    freeze_motion_encoder = config.get("freeze_motion_encoder", False)
    siglip_unfreeze_last_n_layers = int(config.get("siglip_unfreeze_last_n_layers", 0))
    motion_use_cross_attention = config.get("motion_use_cross_attention", False)
    motion_input_mode = config.get("motion_input_mode", "rgb_diff")
    use_motion_vectors = config.get("use_motion_vectors", True)
    use_residuals = config.get("use_residuals", True)
    compressed_fusion_mode = config.get("compressed_fusion_mode", "gated")
    compressed_use_iframe_context = bool(config.get("compressed_use_iframe_context", False))
    compressed_use_iframe_encoder = bool(config.get("compressed_use_iframe_encoder", False))
    alignment_gather_global_negatives = bool(config.get("alignment_gather_global_negatives", True))
    alignment_interval_diff_weight = float(config.get("alignment_interval_diff_weight", 1.0))
    alignment_decoded_frame_weight = float(config.get("alignment_decoded_frame_weight", 1.0))
    compressed_fused_alignment_weight = float(config.get("compressed_fused_alignment_weight", 1.0))
    compressed_iframe_alignment_weight = float(config.get("compressed_iframe_alignment_weight", 0.25))
    compressed_mv_alignment_weight = float(config.get("compressed_mv_alignment_weight", 0.25))
    compressed_residual_alignment_weight = float(config.get("compressed_residual_alignment_weight", 0.25))

    if model_type == "qwen":
        from transformers import AutoProcessor, AutoConfig
        from models.qwen_delta import QwenDeltaForConditionalGeneration

        logger.info("加载 Qwen2.5-VL Delta 模型: %s (motion_encoder=%s)", model_path, motion_encoder_type)
        min_pixels = config.get("min_pixels", 128 * 28 * 28)
        max_pixels = config.get("max_pixels", 768 * 28 * 28)
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
        # 注入 motion encoder 配置到 HuggingFace config
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        hf_config.motion_encoder_type = motion_encoder_type
        hf_config.siglip_model_path = siglip_model_path
        hf_config.freeze_motion_encoder = freeze_motion_encoder
        hf_config.motion_vit_hidden_size = 3584  # Qwen2.5-VL ViT 经 PatchMerger 后维度
        hf_config.motion_use_cross_attention = motion_use_cross_attention
        hf_config.motion_input_mode = motion_input_mode
        hf_config.use_motion_vectors = use_motion_vectors
        hf_config.use_residuals = use_residuals
        hf_config.compressed_fusion_mode = compressed_fusion_mode
        hf_config.compressed_use_iframe_context = compressed_use_iframe_context
        hf_config.compressed_use_iframe_encoder = compressed_use_iframe_encoder
        hf_config.siglip_unfreeze_last_n_layers = siglip_unfreeze_last_n_layers
        hf_config.alignment_gather_global_negatives = alignment_gather_global_negatives
        hf_config.alignment_interval_diff_weight = alignment_interval_diff_weight
        hf_config.alignment_decoded_frame_weight = alignment_decoded_frame_weight
        hf_config.compressed_fused_alignment_weight = compressed_fused_alignment_weight
        hf_config.compressed_iframe_alignment_weight = compressed_iframe_alignment_weight
        hf_config.compressed_mv_alignment_weight = compressed_mv_alignment_weight
        hf_config.compressed_residual_alignment_weight = compressed_residual_alignment_weight

        model = QwenDeltaForConditionalGeneration.from_pretrained(
            model_path,
            config=hf_config,
            torch_dtype=torch.bfloat16 if config["bf16"] else torch.float32,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        # 修复: from_pretrained 的 meta device 初始化会导致 ResNet-18 权重丢失
        # 必须在 from_pretrained 完成后重新加载
        model.motion_module.reload_pretrained_weights()
        # 注册特殊 token
        model.setup_motion_token_ids(processor.tokenizer)
        tokenizer = processor.tokenizer

    elif model_type == "qwen3":
        from transformers import AutoProcessor, AutoConfig
        from models.qwen3_delta import Qwen3DeltaForConditionalGeneration

        logger.info("加载 Qwen3-VL Delta 模型: %s (motion_encoder=%s)", model_path, motion_encoder_type)
        # Qwen3-VL: 每 32x32 patch = 1 vision token, 空间压缩32, 时间压缩2
        min_pixels = config.get("min_pixels", 128 * 32 * 32)
        max_pixels = config.get("max_pixels", 768 * 32 * 32)
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
        # 注入 motion encoder 配置到 HuggingFace config
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        hf_config.motion_encoder_type = motion_encoder_type
        hf_config.siglip_model_path = siglip_model_path
        hf_config.freeze_motion_encoder = freeze_motion_encoder
        hf_config.motion_vit_hidden_size = 4096  # Qwen3-VL ViT 经 PatchMerger 后维度
        hf_config.motion_use_cross_attention = motion_use_cross_attention
        hf_config.motion_input_mode = motion_input_mode
        hf_config.use_motion_vectors = use_motion_vectors
        hf_config.use_residuals = use_residuals
        hf_config.compressed_fusion_mode = compressed_fusion_mode
        hf_config.compressed_use_iframe_context = compressed_use_iframe_context
        hf_config.compressed_use_iframe_encoder = compressed_use_iframe_encoder
        hf_config.siglip_unfreeze_last_n_layers = siglip_unfreeze_last_n_layers
        hf_config.alignment_gather_global_negatives = alignment_gather_global_negatives
        hf_config.alignment_interval_diff_weight = alignment_interval_diff_weight
        hf_config.alignment_decoded_frame_weight = alignment_decoded_frame_weight
        hf_config.compressed_fused_alignment_weight = compressed_fused_alignment_weight
        hf_config.compressed_iframe_alignment_weight = compressed_iframe_alignment_weight
        hf_config.compressed_mv_alignment_weight = compressed_mv_alignment_weight
        hf_config.compressed_residual_alignment_weight = compressed_residual_alignment_weight

        model = Qwen3DeltaForConditionalGeneration.from_pretrained(
            model_path,
            config=hf_config,
            torch_dtype=torch.bfloat16 if config["bf16"] else torch.float32,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        # 修复: from_pretrained 的 meta device 初始化会导致 ResNet-18 权重丢失
        model.motion_module.reload_pretrained_weights()
        model.setup_motion_token_ids(processor.tokenizer)
        tokenizer = processor.tokenizer

    elif model_type == "internvl3":
        from transformers import AutoTokenizer
        from models.internvl_delta import load_internvl3_delta

        logger.info("加载 InternVL3 Delta 模型: %s (motion_encoder=%s)", model_path, motion_encoder_type)
        model = load_internvl3_delta(
            model_path,
            dtype=torch.bfloat16 if config["bf16"] else torch.float32,
            motion_encoder_type=motion_encoder_type,
            siglip_model_path=siglip_model_path,
            freeze_motion_encoder=freeze_motion_encoder,
            motion_vit_hidden_size=3584,  # InternViT 经 MLP projector 后与 LLM hidden_size 对齐
            motion_use_cross_attention=motion_use_cross_attention,
            motion_input_mode=motion_input_mode,
            use_motion_vectors=use_motion_vectors,
            use_residuals=use_residuals,
            compressed_fusion_mode=compressed_fusion_mode,
            compressed_use_iframe_context=compressed_use_iframe_context,
            compressed_use_iframe_encoder=compressed_use_iframe_encoder,
        )
        # InternVL3 用组合模式, motion_module 不在 from_pretrained 路径中,
        # 但为安全起见仍然检查并重新加载
        model.motion_module.reload_pretrained_weights()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.setup_motion_token_ids(tokenizer)
        # InternVL3 没有统一的 processor, 用 tokenizer 代替
        processor = None

    elif model_type == "llava":
        from transformers import AutoProcessor, AutoConfig
        from models.llava_delta import LlavaDeltaForConditionalGeneration

        logger.info("加载 LLaVA-OneVision Delta 模型: %s (motion_encoder=%s)", model_path, motion_encoder_type)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # 注入 motion encoder 配置到 HuggingFace config
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        hf_config.motion_encoder_type = motion_encoder_type
        hf_config.siglip_model_path = siglip_model_path
        hf_config.freeze_motion_encoder = freeze_motion_encoder
        hf_config.motion_vit_hidden_size = 1152  # LLaVA SigLIP ViT
        hf_config.motion_use_cross_attention = motion_use_cross_attention
        hf_config.motion_input_mode = motion_input_mode
        hf_config.use_motion_vectors = use_motion_vectors
        hf_config.use_residuals = use_residuals
        hf_config.compressed_fusion_mode = compressed_fusion_mode
        hf_config.compressed_use_iframe_context = compressed_use_iframe_context
        hf_config.compressed_use_iframe_encoder = compressed_use_iframe_encoder
        hf_config.siglip_unfreeze_last_n_layers = siglip_unfreeze_last_n_layers
        hf_config.alignment_gather_global_negatives = alignment_gather_global_negatives
        hf_config.alignment_interval_diff_weight = alignment_interval_diff_weight
        hf_config.alignment_decoded_frame_weight = alignment_decoded_frame_weight
        hf_config.compressed_fused_alignment_weight = compressed_fused_alignment_weight
        hf_config.compressed_iframe_alignment_weight = compressed_iframe_alignment_weight
        hf_config.compressed_mv_alignment_weight = compressed_mv_alignment_weight
        hf_config.compressed_residual_alignment_weight = compressed_residual_alignment_weight

        model = LlavaDeltaForConditionalGeneration.from_pretrained(
            model_path,
            config=hf_config,
            torch_dtype=torch.bfloat16 if config["bf16"] else torch.float32,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        # 修复: from_pretrained 的 meta device 初始化会导致 ResNet-18 权重丢失
        # 必须在 from_pretrained 完成后重新加载
        model.motion_module.reload_pretrained_weights()
        model.setup_motion_token_ids(processor.tokenizer)
        tokenizer = processor.tokenizer

    else:
        raise ValueError(f"不支持的模型类型: {model_type} (支持: qwen, qwen3, llava, internvl3)")

    # 注入运行期 delta 配置，确保模型 forward 读取到和训练配置一致的参数
    if hasattr(model, "config") and model.config is not None:
        setattr(model.config, "diff_mode", config.get("diff_mode", "abs"))
        setattr(model.config, "motion_input_mode", config.get("motion_input_mode", "rgb_diff"))
        setattr(model.config, "use_motion_vectors", config.get("use_motion_vectors", True))
        setattr(model.config, "use_residuals", config.get("use_residuals", True))
        setattr(model.config, "compressed_fusion_mode", config.get("compressed_fusion_mode", "gated"))
        setattr(model.config, "compressed_use_iframe_context", config.get("compressed_use_iframe_context", False))
        setattr(model.config, "compressed_use_iframe_encoder", config.get("compressed_use_iframe_encoder", False))
        setattr(model.config, "alignment_motion_pool", config.get("alignment_motion_pool", "mean"))
        setattr(model.config, "siglip_unfreeze_last_n_layers", siglip_unfreeze_last_n_layers)
        setattr(model.config, "alignment_gather_global_negatives", alignment_gather_global_negatives)
        setattr(model.config, "alignment_interval_diff_weight", alignment_interval_diff_weight)
        setattr(model.config, "alignment_decoded_frame_weight", alignment_decoded_frame_weight)
        setattr(model.config, "compressed_fused_alignment_weight", compressed_fused_alignment_weight)
        setattr(model.config, "compressed_iframe_alignment_weight", compressed_iframe_alignment_weight)
        setattr(model.config, "compressed_mv_alignment_weight", compressed_mv_alignment_weight)
        setattr(model.config, "compressed_residual_alignment_weight", compressed_residual_alignment_weight)
        setattr(model.config, "shared_vit_input_mode", config.get("shared_vit_input_mode", "video"))

    if hasattr(model, "motion_module"):
        if hasattr(model.motion_module, "set_alignment_options"):
            model.motion_module.set_alignment_options(
                gather_global_negatives=alignment_gather_global_negatives,
                fused_weight=compressed_fused_alignment_weight,
                iframe_weight=compressed_iframe_alignment_weight,
                mv_weight=compressed_mv_alignment_weight,
                residual_weight=compressed_residual_alignment_weight,
            )
        elif hasattr(model.motion_module, "set_alignment_gather_global_negatives"):
            model.motion_module.set_alignment_gather_global_negatives(alignment_gather_global_negatives)

    # 配置训练阶段 (冻结策略)
    sft_mode = config.get("sft_mode", "lora")
    freeze_spatial_encoder = _should_freeze_stage1_spatial_encoder(config)
    model.configure_training(
        stage=stage,
        sft_mode=sft_mode,
        freeze_spatial_encoder=freeze_spatial_encoder,
    )

    # Stage 2: 加载 Stage 1 训练好的 motion_module 权重
    if stage == 2:
        stage1_ckpt = config.get("stage1_checkpoint")
        if stage1_ckpt:
            motion_sd = _load_motion_module_from_checkpoint(stage1_ckpt)
            if motion_sd is not None:
                missing, unexpected = model.motion_module.load_state_dict(motion_sd, strict=False)
                if missing:
                    logger.warning("Stage 1 checkpoint 缺失 key: %s", missing)
                if unexpected:
                    logger.warning("Stage 1 checkpoint 多余 key: %s", unexpected)
                if config.get("bootstrap_projector_from_alignment", True) and hasattr(model.motion_module, "bootstrap_projector_from_alignment"):
                    model.motion_module.bootstrap_projector_from_alignment()
                logger.info("Stage 1 motion_module 权重加载完成 (%d 个参数)", len(motion_sd))
            else:
                logger.warning("Stage 1 checkpoint 无法加载: %s, motion_module 使用 ImageNet 初始化", stage1_ckpt)
        else:
            logger.warning("未指定 stage1_checkpoint, motion_module 使用 ImageNet 初始化 (建议先跑 Stage 1)")

    # Stage 2 + LoRA 模式: 注入 LoRA
    if stage == 2 and sft_mode == "lora":
        from peft import LoraConfig, get_peft_model

        # InternVL3 的 LoRA 需要注入到 base_model.language_model
        if model_type == "internvl3":
            lora_target_model = model.base_model.language_model
        else:
            lora_target_model = model

        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        if model_type == "internvl3":
            # InternVL3: 只给 language_model 加 LoRA
            model.base_model.language_model = get_peft_model(
                model.base_model.language_model, lora_config
            )
            # 确保 motion_module 可训练
            unfreeze_module(model.motion_module)
            model.base_model.language_model.print_trainable_parameters()
        else:
            model = get_peft_model(model, lora_config)
            # LoRA 注入后, 确保 motion_module 仍然可训练
            if hasattr(model, "base_model"):
                unfreeze_module(model.base_model.model.motion_module)
            else:
                unfreeze_module(model.motion_module)
            model.print_trainable_parameters()

    elif stage == 2 and sft_mode == "full":
        logger.info("=== Full SFT 模式: 不注入 LoRA, LLM 全量微调 ===")
        info = get_trainable_params_info(model)
        logger.info("可训练参数: %s", info)

    # Gradient Checkpointing
    # Stage 1 alignment 模式不走 LLM forward, 不需要 gradient checkpointing
    # 启用它反而会导致 "None of the inputs have requires_grad=True" 警告
    use_alignment = (stage == 1 and config.get("alignment_loss", True))
    if config["gradient_checkpointing"] and not use_alignment:
        model.gradient_checkpointing_enable()
    elif use_alignment:
        logger.info("Stage 1 alignment 模式: 跳过 gradient_checkpointing (不走 LLM forward)")

    return model, processor, tokenizer


# ============================================================================
# 数据加载
# ============================================================================

def create_dataloader(config: dict, processor, tokenizer):
    """创建训练/验证数据加载器。Stage 2 时按 val_ratio 做 train/val split。"""
    motion_input_mode = config.get("motion_input_mode", "rgb_diff")
    model_type = config["model_type"]
    alignment_mode = (config["stage"] == 1 and config.get("alignment_loss", True))

    requested_num_workers = int(config.get(
        "dataloader_num_workers",
        0 if motion_input_mode == "compressed" else 4,
    ))
    requested_prefetch_factor = int(config.get("dataloader_prefetch_factor", 2))
    requested_persistent_workers = bool(
        config.get("dataloader_persistent_workers", requested_num_workers > 0)
    )
    dataloader_pin_memory = bool(
        config.get("dataloader_pin_memory", motion_input_mode != "compressed")
    )
    decord_num_threads = int(config.get("decord_num_threads", 2))

    dataloader_num_workers = requested_num_workers
    dataloader_prefetch_factor = requested_prefetch_factor
    dataloader_persistent_workers = requested_persistent_workers

    risky_compressed_loader = (
        motion_input_mode == "compressed"
        and alignment_mode
        and (
            bool(config.get("compressed_use_iframe_encoder", False))
            or float(config.get("alignment_decoded_frame_weight", 0.0)) > 0.0
        )
    )
    allow_unsafe_compressed_loader = (
        os.environ.get("VIDEO_HORIZON_UNSAFE_COMPRESSED_DATALOADER", "0") == "1"
    )

    if risky_compressed_loader and requested_num_workers > 0 and not allow_unsafe_compressed_loader:
        logger.warning(
            "检测到高风险 compressed Stage1 DataLoader 组合（workers=%d, prefetch=%d, persistent=%s, iframe_encoder=%s, decoded_frame_weight=%.2f）；"
            "为避免 worker 被系统 OOM-kill，自动切换为保守模式：num_workers=0, persistent_workers=False, pin_memory=False。"
            "如需恢复旧行为，可显式设置环境变量 VIDEO_HORIZON_UNSAFE_COMPRESSED_DATALOADER=1。",
            requested_num_workers,
            requested_prefetch_factor,
            requested_persistent_workers,
            bool(config.get("compressed_use_iframe_encoder", False)),
            float(config.get("alignment_decoded_frame_weight", 0.0)),
        )
        dataloader_num_workers = 0
        dataloader_prefetch_factor = 0
        dataloader_persistent_workers = False
        dataloader_pin_memory = False
    elif motion_input_mode == "compressed" and dataloader_num_workers > 0:
        logger.warning(
            "compressed-domain + coviar 在多进程 DataLoader 下稳定性较差，当前 num_workers=%d。若仍遇到 worker segfault / Killed，建议改为 0。",
            dataloader_num_workers,
        )
    if motion_input_mode == "rgb_diff" and dataloader_num_workers == 0:
        logger.warning(
            "rgb_diff 且 dataloader_num_workers=0：视频解码、差分生成和 processor 预处理都会串行执行，GPU 利用率通常会偏低。"
        )

    # shared_vit 需要将 diff 图像保留为 PIL Image 列表，以便 collator 通过
    # Qwen Processor 编码为 diff_pixel_values_videos / diff_video_grid_thw
    need_diff_images = (
        config.get("motion_encoder_type", "resnet") == "shared_vit"
        or config.get("return_diff_images", False)
    )
    need_distill_target_frames = (
        alignment_mode
        and config.get("motion_input_mode", "rgb_diff") == "compressed"
        and model_type in ("qwen", "qwen3")
        and float(config.get("alignment_decoded_frame_weight", 1.0)) > 0.0
    )

    dataset = DeltaVideoDataset(
        data_path=config["data_path"],
        video_root=config["video_root"],
        keyframe_fps=config["keyframe_fps"],
        num_diffs_per_interval=config["num_diffs_per_interval"],
        diff_mode=config.get("diff_mode", "abs"),
        max_keyframes=config["max_keyframes"],
        max_length=config["max_length"],
        motion_input_mode=config.get("motion_input_mode", "rgb_diff"),
        compressed_backend=config.get("compressed_backend", "coviar"),
        visual_stream_mode=config.get("visual_stream_mode", "iframe"),
        compressed_gop_size=config.get("compressed_gop_size", 12),
        use_motion_vectors=config.get("use_motion_vectors", True),
        use_residuals=config.get("use_residuals", True),
        use_iframe_encoder=config.get("compressed_use_iframe_encoder", False),
        accumulate_motion_vectors=config.get("accumulate_motion_vectors", True),
        accumulate_residuals=config.get("accumulate_residuals", False),
        motion_vector_clip=config.get("motion_vector_clip", 20.0),
        decord_num_threads=decord_num_threads,
        return_diff_images=need_diff_images,
        return_distill_target_frames=need_distill_target_frames,
        allow_compressed_proxy_fallback=config.get("allow_compressed_proxy_fallback", False),
    )

    # 根据模型类型选择 collator
    if model_type in ("qwen", "qwen3"):
        from data.delta_video_loader import QwenDeltaCollator
        collator = QwenDeltaCollator(
            processor,
            tokenizer,
            config["max_length"],
            alignment_mode=alignment_mode,
            shared_vit_input_mode=config.get("shared_vit_input_mode", "video"),
        )
    elif model_type == "internvl3":
        from data.delta_video_loader import InternVLDeltaCollator
        collator = InternVLDeltaCollator(tokenizer, config["max_length"],
                                         alignment_mode=alignment_mode)
    else:
        from data.delta_video_loader import LlavaDeltaCollator
        collator = LlavaDeltaCollator(processor, tokenizer, config["max_length"],
                                      alignment_mode=alignment_mode)

    train_loader_kwargs = {
        "batch_size": config["per_device_batch_size"],
        "collate_fn": collator,
        "num_workers": dataloader_num_workers,
        "pin_memory": dataloader_pin_memory,
    }
    val_loader_kwargs = {
        "batch_size": config["per_device_batch_size"],
        "collate_fn": collator,
        "num_workers": dataloader_num_workers,
        "pin_memory": dataloader_pin_memory,
    }
    if dataloader_num_workers > 0:
        train_loader_kwargs["persistent_workers"] = dataloader_persistent_workers
        val_loader_kwargs["persistent_workers"] = dataloader_persistent_workers
        train_loader_kwargs["prefetch_factor"] = dataloader_prefetch_factor
        val_loader_kwargs["prefetch_factor"] = dataloader_prefetch_factor

    # Stage 2: train/val split
    val_ratio = config.get("val_ratio", 0.05) if config["stage"] == 2 else 0.0
    max_val_samples = config.get("max_val_samples", 500)
    val_dataloader = None

    if val_ratio > 0 and len(dataset) > 100:
        import random
        total = len(dataset)
        indices = list(range(total))
        random.seed(42)  # 固定种子确保可复现
        random.shuffle(indices)
        val_size = int(total * val_ratio)
        # 限制验证集最大样本数, 避免 eval 耗时过长
        if max_val_samples > 0:
            val_size = min(val_size, max_val_samples)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None

        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            drop_last=True,
            **train_loader_kwargs,
        )
        val_dataloader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            shuffle=False,
            drop_last=False,
            **val_loader_kwargs,
        )
        logger.info("Train/Val split: %d train, %d val (ratio=%.2f, max_val=%d)",
                     len(train_dataset), len(val_dataset), val_ratio, max_val_samples)
    else:
        train_sampler = DistributedSampler(dataset) if dist.is_initialized() else None
        train_dataloader = DataLoader(
            dataset,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            drop_last=True,
            **train_loader_kwargs,
        )

    logger.info(
        "create_dataloader: motion_input_mode=%s, workers=%d(requested=%d), prefetch=%s(requested=%s), persistent_workers=%s(requested=%s), pin_memory=%s, decord_num_threads=%d, dataset_size=%d",
        motion_input_mode,
        dataloader_num_workers,
        requested_num_workers,
        dataloader_prefetch_factor if dataloader_num_workers > 0 else "n/a",
        requested_prefetch_factor if requested_num_workers > 0 else "n/a",
        dataloader_persistent_workers if dataloader_num_workers > 0 else False,
        requested_persistent_workers if requested_num_workers > 0 else False,
        dataloader_pin_memory,
        decord_num_threads,
        len(dataset),
    )
    return train_dataloader, val_dataloader, dataset


# ============================================================================
# DeepSpeed 配置
# ============================================================================

def get_deepspeed_config(config: dict, total_steps: int) -> dict:
    """生成 DeepSpeed 配置。

    Args:
        config: 训练配置字典
        total_steps: 优化器总步数 (已除以 gradient_accumulation_steps)
    """
    warmup_steps = max(2, int(total_steps * config.get("warmup_ratio", 0.03)))

    # Stage 1 alignment 只训练 motion_module (~11M 参数), 不需要 ZeRO 分片。
    # 使用 ZeRO-2 会导致 save_checkpoint 时跨 rank gather 全量 7B 参数,
    # 触发 NCCL 通信, GPU 显存不足时直接 OOM 崩溃。
    # 因此 Stage 1 alignment 强制使用 ZeRO-0 (不分片)。
    use_alignment = (config["stage"] == 1 and config.get("alignment_loss", True))
    if use_alignment:
        zero_stage = 0
    else:
        zero_stage = config["deepspeed_stage"]

    zero_config = {
        "stage": zero_stage,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "contiguous_gradients": True,
        "overlap_comm": True,
    }

    # ZeRO-3 全量微调: 添加额外配置
    if zero_stage == 3:
        zero_config.update({
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        })

    ds_config = {
        "train_batch_size": (
            config["per_device_batch_size"]
            * config["gradient_accumulation_steps"]
            * max(1, torch.cuda.device_count())
        ),
        "train_micro_batch_size_per_gpu": config["per_device_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config["learning_rate"],
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": config["weight_decay"],
            },
        },

        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config["learning_rate"],
                "warmup_num_steps": warmup_steps,
                "total_num_steps": total_steps,
            },
        },

        "bf16": {"enabled": config["bf16"]},

        "zero_optimization": zero_config,

        "gradient_clipping": 1.0,
        "steps_per_print": config["logging_steps"],
        "wall_clock_breakdown": False,
    }

    return ds_config


# ============================================================================
# Focal Loss
# ============================================================================

def focal_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Focal Loss for causal LM (支持 shift-by-1)。

    当 gamma=0 时退化为标准 Cross-Entropy。
    gamma>0 时降低 easy sample 的权重, 让模型更关注 hard sample。

    Args:
        logits: [B, S, V] 模型输出 logits
        labels: [B, S] 目标 token ID, -100 为忽略
        gamma:  focusing parameter
        ignore_index: 忽略的 label 值
    Returns:
        scalar loss
    """
    # Shift: predict next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    B_S, V = shift_logits.view(-1, shift_logits.size(-1)).shape
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)

    # 标准 CE (per-token, 不 reduce)
    ce_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, reduction="none")

    # 只对有效位置求均值
    valid_mask = flat_labels != ignore_index

    if gamma > 0:
        # 复用 ce_loss 推导 pt, 避免对 150K vocab 重复做 log_softmax:
        # ce_loss = -log(pt), 所以 pt = exp(-ce_loss)
        pt = (-ce_loss).exp()
        # Focal weight: (1 - p_t)^gamma, 仅在有效位置计算
        focal_weight = (1.0 - pt).pow(gamma)
        ce_loss = ce_loss * focal_weight

    if valid_mask.sum() == 0:
        return ce_loss.sum() * 0.0  # 保持计算图
    return ce_loss[valid_mask].mean()


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(model_engine, val_dataloader, local_rank, use_alignment=False, gamma=0.0):
    """
    在验证集上计算 loss (标准 CE, 不用 focal)。
    返回 (avg_val_loss, task_losses_dict)。
    """
    model_engine.eval()
    total_loss = 0.0
    total_tokens = 0
    task_losses = {}  # {task_type: [sum_loss, count]}

    for batch in val_dataloader:
        if batch.get("is_dummy", False):
            continue

        task_types = batch.pop("task_types", ["open_qa"])

        batch = {
            k: v.to(local_rank) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        try:
            if use_alignment:
                break  # Stage 1 不做 eval

            outputs = model_engine(**batch)
            logits = outputs.logits
            labels = batch["labels"]

            # 标准 CE (eval 不用 focal)
            loss = focal_cross_entropy_loss(logits, labels, gamma=0.0)
            loss_val = loss.item()

            # 统计有效 token 数
            shift_labels = labels[..., 1:].contiguous()
            n_tokens = (shift_labels != -100).sum().item()

            total_loss += loss_val * max(n_tokens, 1)
            total_tokens += max(n_tokens, 1)

            # 分任务统计
            for tt in task_types:
                if tt not in task_losses:
                    task_losses[tt] = [0.0, 0]
                task_losses[tt][0] += loss_val
                task_losses[tt][1] += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise

    model_engine.train()

    avg_loss = total_loss / max(total_tokens, 1)

    # 多卡同步 val loss
    if dist.is_initialized():
        loss_tensor = torch.tensor([total_loss, float(total_tokens)], device=local_rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor[0] / max(loss_tensor[1], 1)).item()

    task_avg = {tt: v[0] / max(v[1], 1) for tt, v in task_losses.items()}
    return avg_loss, task_avg


# ============================================================================
# Delta 权重导出 (用于评测直接加载)
# ============================================================================

def _export_delta_weights(model_engine, config: dict, export_dir: str, local_rank: int):
    """
    从 DeepSpeed model_engine 导出 delta 权重。
    导出轻量文件方便评测直接加载。

    - sft_mode="lora" (ZeRO-2): rank 0 导出 motion_module.pt + lora_weights/
    - sft_mode="full" (ZeRO-3): 所有 rank 参与导出 motion_module.pt + full_model/
      (ZeRO-3 参数分片, 需要 gather 后才能保存)
    """
    sft_mode = config.get("sft_mode", "lora")
    zero_stage = config.get("deepspeed_stage", 2)

    # ZeRO-3 full 模式: 使用 save_16bit_model (所有 rank 参与 gather, 仅 rank 0 写入)
    if sft_mode == "full" and zero_stage == 3:
        if local_rank == 0:
            os.makedirs(export_dir, exist_ok=True)

        # 导出 motion_module (ZeRO-3 下 GatheredParameters 需要所有 rank 参与)
        import deepspeed as _ds
        with _ds.zero.GatheredParameters(
            list(model_engine.module.motion_module.parameters()), modifier_rank=0
        ):
            if local_rank == 0:
                torch.save(
                    model_engine.module.motion_module.state_dict(),
                    os.path.join(export_dir, "motion_module.pt"),
                )

        # 导出完整模型 (save_16bit_model 内部处理 gather + 仅 rank 0 写入)
        if config["stage"] == 2:
            full_model_dir = os.path.join(export_dir, "full_model")
            if local_rank == 0:
                os.makedirs(full_model_dir, exist_ok=True)
            # save_16bit_model 所有 rank 必须调用
            model_engine.save_16bit_model(full_model_dir)
            if local_rank == 0:
                logger.info("Delta 权重已导出 (Full/ZeRO-3): %s", export_dir)
        return

    # 非 ZeRO-3: 仅 rank 0 操作
    if local_rank != 0:
        return

    os.makedirs(export_dir, exist_ok=True)
    base_model = model_engine.module

    # 解包 PeftModel (仅 lora 模式会有)
    try:
        from peft import PeftModel
        is_peft = isinstance(base_model, PeftModel)
        if is_peft:
            underlying = base_model.base_model.model
        else:
            underlying = base_model
    except ImportError:
        is_peft = False
        underlying = base_model

    # 导出 motion_module (总是需要)
    torch.save(
        underlying.motion_module.state_dict(),
        os.path.join(export_dir, "motion_module.pt"),
    )

    if config["stage"] == 2:
        if sft_mode == "lora" and is_peft:
            base_model.save_pretrained(os.path.join(export_dir, "lora_weights"))
            logger.info("Delta 权重已导出 (LoRA): %s", export_dir)
        elif sft_mode == "full":
            # ZeRO-2 下的 full 模式 (不太常见但兼容)
            full_model_dir = os.path.join(export_dir, "full_model")
            os.makedirs(full_model_dir, exist_ok=True)
            underlying.save_pretrained(full_model_dir)
            logger.info("Delta 权重已导出 (Full): %s", export_dir)
        else:
            logger.info("Delta 权重已导出 (motion_module only): %s", export_dir)
    else:
        logger.info("Delta 权重已导出 (Stage 1): %s", export_dir)


def _get_dist_timeout_seconds(config: dict) -> int:
    """获取分布式 process group 超时时间（秒）。"""
    raw_value = config.get(
        "dist_timeout_seconds",
        os.environ.get("TORCH_DIST_TIMEOUT_SECONDS", os.environ.get("NCCL_TIMEOUT", 1800)),
    )
    try:
        return max(int(raw_value), 60)
    except (TypeError, ValueError):
        return 1800


def _sync_step_skip(local_skip: bool, local_rank: int) -> bool:
    """把当前 micro step 的 skip 决策在所有 rank 上同步，避免 collective 对不齐。"""
    if not (dist.is_available() and dist.is_initialized()):
        return bool(local_skip)

    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    skip_tensor = torch.tensor([1 if local_skip else 0], device=device, dtype=torch.int32)
    dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
    return bool(skip_tensor.item())


# ============================================================================
# 训练循环
# ============================================================================

def train(config: dict):
    """主训练循环。"""
    # 初始化分布式
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist_timeout_seconds = _get_dist_timeout_seconds(config)
    dist_timeout_applied = True
    try:
        deepspeed.init_distributed(timeout=timedelta(seconds=dist_timeout_seconds))
    except TypeError:
        dist_timeout_applied = False
        deepspeed.init_distributed()

    logging.basicConfig(
        level=logging.INFO if local_rank == 0 else logging.WARNING,
        format=f"[Rank {local_rank}] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if local_rank == 0:
        logger.info("=" * 60)
        logger.info("Himmel-VLM Training")
        logger.info("Model: %s (%s)", config["model_name_or_path"], config["model_type"])
        logger.info("Stage: %d", config["stage"])
        logger.info(
            "Dist timeout: %ds (%s)",
            dist_timeout_seconds,
            "explicit PG timeout" if dist_timeout_applied else "DeepSpeed fallback",
        )
        if config["stage"] == 2:
            logger.info("SFT Mode: %s", config.get("sft_mode", "lora"))
        logger.info("=" * 60)

    # 加载模型
    model, processor, tokenizer = load_model_and_processor(config)

    # 加载数据 (Stage 2 自动做 train/val split)
    dataloader, val_dataloader, dataset = create_dataloader(config, processor, tokenizer)

    total_steps = (
        len(dataloader) * config["num_epochs"]
        // config["gradient_accumulation_steps"]
    )

    if local_rank == 0:
        logger.info("数据集: %d 样本", len(dataset))
        logger.info("Dataloader 步数/epoch: %d", len(dataloader))
        logger.info("梯度累积: %d", config["gradient_accumulation_steps"])
        logger.info("优化器总步数: %d", total_steps)
        if _should_freeze_stage1_spatial_encoder(config) and int(config.get("freeze_spatial_encoder_steps", 0)) >= max(total_steps, 1):
            logger.warning(
                "freeze_spatial_encoder_steps=%d 已大于等于总 optimizer steps=%d，spatial_encoder 可能整轮都不会解冻。",
                int(config.get("freeze_spatial_encoder_steps", 0)),
                total_steps,
            )
        info = get_trainable_params_info(model)
        logger.info("可训练参数: %s", info)

    # DeepSpeed 初始化
    ds_config = get_deepspeed_config(config, total_steps)
    if local_rank == 0:
        logger.info("DeepSpeed ZeRO stage: %d", ds_config["zero_optimization"]["stage"])
    model_parameters = _build_model_parameters_for_deepspeed(model, config)
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config=ds_config,
    )

    # 训练
    os.makedirs(config["output_dir"], exist_ok=True)
    global_step = 0
    nccl_timeout_seconds = int(os.environ.get("NCCL_TIMEOUT", 1800))  # 默认30分钟

    # Stage 1 Alignment 模式: 不走 LLM next-token prediction,
    # 而是直接对齐 motion tokens 与 ViT visual tokens
    use_alignment = (config["stage"] == 1 and config.get("alignment_loss", True))
    stage1_spatial_freeze_steps = int(config.get("freeze_spatial_encoder_steps", 0))
    if local_rank == 0 and use_alignment:
        logger.info("=" * 60)
        logger.info("Stage 1: 使用 InfoNCE 对比学习 Alignment Loss")
        logger.info("  Target: ViT 帧差 (vf[i+1] - vf[i]) — 语义级变化方向")
        logger.info("  Loss  : 双向 InfoNCE + 0.1 * cosine 辅助")
        logger.info("  MotionInput: %s", config.get("motion_input_mode", "rgb_diff"))
        logger.info("  Diff  : %s", config.get("diff_mode", "abs"))
        logger.info("  Visual: %s", config.get("visual_stream_mode", "iframe"))
        logger.info("  Streams: mv=%s residual=%s iframe_ctx=%s iframe_enc=%s fusion=%s",
                    config.get("use_motion_vectors", True),
                    config.get("use_residuals", True),
                    config.get("compressed_use_iframe_context", False),
                    config.get("compressed_use_iframe_encoder", False),
                    config.get("compressed_fusion_mode", "gated"))
        logger.info("  AuxW  : fused=%.2f iframe=%.2f mv=%.2f residual=%.2f",
                    float(config.get("compressed_fused_alignment_weight", 1.0)),
                    float(config.get("compressed_iframe_alignment_weight", 0.25)),
                    float(config.get("compressed_mv_alignment_weight", 0.25)),
                    float(config.get("compressed_residual_alignment_weight", 0.25)))
        logger.info("  RouteW: interval_diff=%.2f decoded_frame=%.2f",
                    float(config.get("alignment_interval_diff_weight", 1.0)),
                    float(config.get("alignment_decoded_frame_weight", 1.0)))
        logger.info("  Pool  : %s", config.get("alignment_motion_pool", "mean"))
        logger.info("  SharedViT Input: %s | Global negatives: %s",
                    config.get("shared_vit_input_mode", "video"),
                    config.get("alignment_gather_global_negatives", True))
        logger.info("  负例: 同视频仅远时距区间 + 同 batch 跨视频区间；相邻区间不参与负采样")
        if _should_freeze_stage1_spatial_encoder(config):
            logger.info("  Warmup: 前 %d 个 optimizer steps 冻结 MotionSSM spatial_encoder", stage1_spatial_freeze_steps)
        logger.info("不走 LLM forward, 直接对齐 motion tokens 与 ViT visual tokens")
        logger.info("=" * 60)

    # Focal Loss 配置
    focal_gamma = config.get("focal_loss_gamma", 2.0) if not use_alignment else 0.0
    if local_rank == 0 and focal_gamma > 0:
        logger.info("Stage 2: 使用 Focal Loss (gamma=%.1f)", focal_gamma)

    # Eval & Early Stopping 配置
    eval_steps = config.get("eval_steps", 500)
    es_patience = config.get("early_stopping_patience", 3)
    best_val_loss = float("inf")
    es_counter = 0
    early_stopped = False

    if local_rank == 0 and val_dataloader is not None:
        logger.info("Eval: 每 %d 步, Early Stopping patience=%d", eval_steps, es_patience)

    # 窗口平均 loss (替代全局累积平均, 更准确反映训练状态)
    loss_window = []
    loss_window_size = 100

    # 分任务 loss 统计窗口
    task_loss_windows = {}  # {task_type: [loss_values]}
    task_window_size = 100

    for epoch in range(config["num_epochs"]):
        if early_stopped:
            break

        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        model_engine.train()
        running_loss = 0.0
        micro_steps = 0

        for step, batch in enumerate(dataloader):
            if early_stopped:
                break

            # ---- 判断是否需要跳过 (dummy / OOM / 异常 loss) ----
            # 所有跳过原因合并为一次 all_reduce 同步, 避免每个 micro step 做多次 NCCL 通信
            local_skip_reason = None

            if bool(batch.get("is_dummy", False)):
                local_skip_reason = "dummy batch"

            # 提取 task_types (非 tensor, 不送 GPU)
            task_types = batch.pop("task_types", ["open_qa"])

            # Forward (包含 batch 移动到 GPU, 避免 OOM 时未捕获)
            loss = None
            outputs = {}
            if local_skip_reason is None:
                try:
                    # 移动到 GPU
                    batch = {
                        k: v.to(local_rank) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    if use_alignment:
                        # Stage 1 Alignment: 不走 LLM, 直接对齐
                        # 需要从 batch 中提取 alignment 所需的字段
                        align_kwargs = {
                            "pixel_values": batch.get("pixel_values"),
                            "pixel_values_videos": batch.get("pixel_values_videos"),
                            "image_grid_thw": batch.get("image_grid_thw"),
                            "video_grid_thw": batch.get("video_grid_thw"),
                            "image_sizes": batch.get("image_sizes"),
                            "image_flags": batch.get("image_flags"),
                            "diff_frames": batch.get("diff_frames"),
                            "diff_pixel_values_videos": batch.get("diff_pixel_values_videos"),
                            "diff_video_grid_thw": batch.get("diff_video_grid_thw"),
                            "diff_pixel_values": batch.get("diff_pixel_values"),
                            "diff_image_grid_thw": batch.get("diff_image_grid_thw"),
                            "diff_image_counts": batch.get("diff_image_counts"),
                            "motion_vectors": batch.get("motion_vectors"),
                            "residuals": batch.get("residuals"),
                            "iframe_frames": batch.get("iframe_frames"),
                            "target_pixel_values_videos": batch.get("target_pixel_values_videos"),
                            "target_video_grid_thw": batch.get("target_video_grid_thw"),
                            "target_interval_counts": batch.get("target_interval_counts"),
                            "num_diffs_per_keyframe": config.get("num_diffs_per_interval", 4),
                        }
                        # 诊断日志: 打印 batch 中的 key 和 shape (前 10 步)
                        if local_rank == 0 and step < 10:
                            batch_info = {
                                k: (v.shape, v.dtype) if (v is not None and isinstance(v, torch.Tensor)) else type(v).__name__
                                for k, v in align_kwargs.items()
                            }
                            logger.info("Step %d batch: %s", step, batch_info)
                        # 需要访问底层模型 (DeepSpeed 包裹, 可能还有 PeftModel 包裹)
                        base_model = model_engine.module
                        try:
                            from peft import PeftModel
                            if isinstance(base_model, PeftModel):
                                base_model = base_model.base_model.model
                        except ImportError:
                            pass
                        outputs = base_model.forward_alignment(**align_kwargs)
                        loss = outputs["loss"]
                        # 诊断: 检查 loss 的 grad_fn (前 10 步)
                        if local_rank == 0 and step < 10:
                            logger.info("Step %d: loss=%.6f, grad_fn=%s, requires_grad=%s",
                                          step, loss.item(), loss.grad_fn, loss.requires_grad)
                    else:
                        # Stage 2: 正常 forward, 但使用 Focal Loss 替代模型内置 CE
                        outputs = model_engine(**batch)
                        if focal_gamma > 0:
                            # 用 Focal Loss 替代 outputs.loss
                            loss = focal_cross_entropy_loss(
                                outputs.logits, batch["labels"], gamma=focal_gamma
                            )
                        else:
                            loss = outputs.loss

                    # 检测异常 loss (非阻塞: 用 torch.isinf/isnan 在 GPU 上做布尔运算,
                    # 合并到 skip_tensor 的 all_reduce 中一起同步, 避免额外的隐式 GPU→CPU sync)
                    if not loss.isfinite():
                        local_skip_reason = "异常 loss (nan/inf)"

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        local_skip_reason = "OOM"
                        torch.cuda.empty_cache()
                    else:
                        raise

            # 安全检查: loss 缺失或无 grad_fn 时也必须整步同步跳过
            if local_skip_reason is None and (loss is None or loss.grad_fn is None):
                local_skip_reason = "loss 缺失" if loss is None else "loss 无 grad_fn"

            if local_skip_reason is not None:
                logger.warning(
                    "Step %d: 本 rank 触发 step skip，原因=%s",
                    step + 1,
                    local_skip_reason,
                )

            # 单次 all_reduce 同步所有 skip 原因 (dummy / OOM / nan / inf / 无 grad_fn)
            if _sync_step_skip(local_skip_reason is not None, local_rank):
                if local_rank == 0:
                    logger.warning(
                        "Step %d: 检测到至少一个 rank 需要跳过 (dummy/OOM/异常loss/无grad_fn)，整步跳过以保持分布式同步",
                        step + 1,
                    )
                continue

            if loss is None:
                raise RuntimeError(f"Step {step + 1}: loss unexpectedly missing after skip sync")

            model_engine.backward(loss)
            model_engine.step()

            micro_steps += 1
            loss_val = loss.item()
            running_loss += loss_val

            # 窗口平均
            loss_window.append(loss_val)
            if len(loss_window) > loss_window_size:
                loss_window.pop(0)
            window_avg = sum(loss_window) / len(loss_window)

            # 分任务 loss 统计
            for tt in task_types:
                if tt not in task_loss_windows:
                    task_loss_windows[tt] = []
                task_loss_windows[tt].append(loss_val)
                if len(task_loss_windows[tt]) > task_window_size:
                    task_loss_windows[tt].pop(0)

            # global_step 按 optimizer step 计数 (每 gradient_accumulation_steps 个 micro step 递增一次)
            # DeepSpeed 内部已处理梯度累积, 通过 is_gradient_accumulation_boundary 判断是否真正做了 optimizer step
            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1
                if (
                    use_alignment
                    and _should_freeze_stage1_spatial_encoder(config)
                    and global_step == stage1_spatial_freeze_steps
                    and hasattr(model_engine.module, "motion_module")
                    and hasattr(model_engine.module.motion_module, "set_spatial_encoder_trainable")
                ):
                    model_engine.module.motion_module.set_spatial_encoder_trainable(True)
                    if local_rank == 0:
                        info = get_trainable_params_info(model_engine.module)
                        logger.info(
                            "Stage1 warmup 结束: optimizer step=%d，已解冻 MotionSSM spatial_encoder，当前可训练参数=%s",
                            global_step,
                            info,
                        )

            # 日志 (每 logging_steps 个 micro step 打印一次, 保持高频可观测)
            if local_rank == 0 and micro_steps % config["logging_steps"] == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = running_loss / max(micro_steps, 1)
                if use_alignment:
                    cosine_sim = outputs.get("avg_cosine_sim", 0.0)
                    num_pairs = outputs.get("num_pairs", 0)
                    contrastive = outputs.get("contrastive_loss", torch.tensor(0.0))
                    cosine = outputs.get("cosine_loss", torch.tensor(0.0))
                    scale = outputs.get("logit_scale", 0.0)
                    contrastive_val = contrastive.item() if hasattr(contrastive, "item") else contrastive
                    cosine_val = cosine.item() if hasattr(cosine, "item") else cosine

                    extra_metrics = []
                    for key, prefix in (
                        ("iframe_alignment_loss", "Ifr"),
                        ("mv_alignment_loss", "MV"),
                        ("residual_alignment_loss", "Res"),
                    ):
                        loss_metric = outputs.get(key)
                        cos_metric = outputs.get(key.replace("alignment_loss", "avg_cosine_sim"))
                        if loss_metric is None:
                            continue
                        loss_val_metric = loss_metric.item() if hasattr(loss_metric, "item") else loss_metric
                        if cos_metric is None:
                            cos_val_metric = float("nan")
                        else:
                            cos_val_metric = cos_metric.item() if hasattr(cos_metric, "item") else cos_metric
                        extra_metrics.append(f"{prefix}={loss_val_metric:.4f}/{cos_val_metric:.4f}")

                    extra_suffix = f" | Aux {' '.join(extra_metrics)}" if extra_metrics else ""
                    logger.info(
                        "Epoch %d | MicroStep %d/%d | OptStep %d/%d | "
                        "Loss %.4f | InfoNCE %.4f | Cos %.4f | WinAvg %.4f | "
                        "CosSim %.4f | Scale %.1f | Pairs %d | LR %.2e%s",
                        epoch + 1, step + 1, len(dataloader), global_step, total_steps,
                        loss_val, contrastive_val, cosine_val, window_avg,
                        cosine_sim, scale, num_pairs, lr, extra_suffix,
                    )
                else:
                    # 分任务 loss 摘要
                    task_info = " | ".join(
                        f"{tt}={sum(vals)/len(vals):.4f}"
                        for tt, vals in sorted(task_loss_windows.items())
                        if vals
                    )
                    logger.info(
                        "Epoch %d | MicroStep %d/%d | OptStep %d/%d | Loss %.4f | WinAvg %.4f | AvgLoss %.4f | LR %.2e | %s",
                        epoch + 1, step + 1, len(dataloader), global_step, total_steps,
                        loss_val, window_avg, avg_loss, lr, task_info,
                    )

            # 以下操作仅在 optimizer step boundary 时触发 (而非每个 micro step)
            if not model_engine.is_gradient_accumulation_boundary():
                continue

            # 保存 checkpoint (基于 optimizer step)
            if global_step > 0 and global_step % config["save_steps"] == 0:
                save_path = os.path.join(
                    config["output_dir"], f"checkpoint-{global_step}"
                )
                # 释放 GPU 缓存, 避免 save_checkpoint gather 时 OOM
                torch.cuda.empty_cache()
                model_engine.save_checkpoint(save_path)
                # 同时导出 delta 权重 (方便评测直接加载)
                _export_delta_weights(model_engine, config, save_path, local_rank)
                if local_rank == 0:
                    logger.info("Checkpoint saved: %s (optimizer step %d)", save_path, global_step)

            # Eval & Early Stopping (基于 optimizer step)
            if (val_dataloader is not None
                    and global_step > 0
                    and global_step % eval_steps == 0
                    and not use_alignment):
                if local_rank == 0:
                    logger.info(">>> 开始 Eval (optimizer step %d, val 样本数 %d)...",
                                global_step, len(val_dataloader.dataset))
                val_loss, val_task_losses = evaluate(
                    model_engine, val_dataloader, local_rank,
                    use_alignment=use_alignment, gamma=focal_gamma,
                )
                if local_rank == 0:
                    task_info = " | ".join(
                        f"{tt}={v:.4f}" for tt, v in sorted(val_task_losses.items())
                    )
                    logger.info(
                        ">>> EVAL OptStep %d | ValLoss %.4f | Best %.4f | Patience %d/%d | %s",
                        global_step, val_loss, best_val_loss, es_counter, es_patience,
                        task_info,
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    es_counter = 0
                    # 保存 best checkpoint
                    best_path = os.path.join(config["output_dir"], "best")
                    torch.cuda.empty_cache()
                    model_engine.save_checkpoint(best_path)
                    # 同时导出 delta 权重 (方便评测直接加载)
                    _export_delta_weights(model_engine, config, best_path, local_rank)
                    if local_rank == 0:
                        logger.info(">>> New best! ValLoss=%.4f, saved to %s", val_loss, best_path)
                else:
                    es_counter += 1
                    if local_rank == 0:
                        logger.info(">>> No improvement. Patience %d/%d", es_counter, es_patience)
                    if es_counter >= es_patience:
                        if local_rank == 0:
                            logger.info("=" * 60)
                            logger.info("Early Stopping triggered at optimizer step %d!", global_step)
                            logger.info("Best ValLoss: %.4f", best_val_loss)
                            logger.info("=" * 60)
                        early_stopped = True
                        break

        # Epoch 结束日志
        if local_rank == 0:
            avg_epoch_loss = running_loss / max(micro_steps, 1)
            logger.info(
                "Epoch %d 完成 | 平均 Loss: %.4f | micro_steps: %d | optimizer_steps: %d",
                epoch + 1, avg_epoch_loss, micro_steps, global_step,
            )

    # 训练循环结束, 确保所有 rank 同步
    if dist.is_initialized():
        dist.barrier()

    # 训练结束时保存最后一份 DeepSpeed checkpoint (所有 rank 参与)
    final_ckpt_path = os.path.join(config["output_dir"], f"checkpoint-{global_step}")
    torch.cuda.empty_cache()
    model_engine.save_checkpoint(final_ckpt_path)
    if local_rank == 0:
        logger.info("训练结束 checkpoint 已保存: %s", final_ckpt_path)

    # 最终保存
    sft_mode = config.get("sft_mode", "lora")
    zero_stage = config.get("deepspeed_stage", 2)
    final_path = os.path.join(config["output_dir"], "final")

    if sft_mode == "full" and zero_stage == 3:
        # ZeRO-3 full 模式: 所有 rank 参与最终保存
        if local_rank == 0:
            os.makedirs(final_path, exist_ok=True)

        # motion_module (ZeRO-3 下 GatheredParameters 需要所有 rank 参与)
        import deepspeed as _ds
        with _ds.zero.GatheredParameters(
            list(model_engine.module.motion_module.parameters()), modifier_rank=0
        ):
            if local_rank == 0:
                torch.save(
                    model_engine.module.motion_module.state_dict(),
                    os.path.join(final_path, "motion_module.pt"),
                )
                logger.info("Motion module saved to %s", final_path)

        # full_model (所有 rank 参与)
        if config["stage"] == 2:
            full_model_dir = os.path.join(final_path, "full_model")
            if local_rank == 0:
                os.makedirs(full_model_dir, exist_ok=True)
            model_engine.save_16bit_model(full_model_dir)
            if local_rank == 0:
                logger.info("Full model (16bit) saved to %s", full_model_dir)
    elif local_rank == 0:
        # 非 ZeRO-3: 仅 rank 0
        os.makedirs(final_path, exist_ok=True)

        # 保存 motion 模块
        base_model = model_engine.module
        try:
            from peft import PeftModel
            if isinstance(base_model, PeftModel):
                base_model = base_model.base_model.model
        except ImportError:
            pass

        torch.save(
            base_model.motion_module.state_dict(),
            os.path.join(final_path, "motion_module.pt"),
        )
        logger.info("Motion module saved to %s", final_path)

        # Stage 2: 根据 sft_mode 保存不同格式的权重
        if config["stage"] == 2:
            if sft_mode == "lora" and hasattr(model_engine.module, "save_pretrained"):
                model_engine.module.save_pretrained(
                    os.path.join(final_path, "lora_weights")
                )
                logger.info("LoRA weights saved")
            elif sft_mode == "full":
                full_model_dir = os.path.join(final_path, "full_model")
                os.makedirs(full_model_dir, exist_ok=True)
                base_model.save_pretrained(full_model_dir)
                logger.info("Full model saved to %s", full_model_dir)

    if local_rank == 0:
        if early_stopped:
            logger.info("=" * 60)
            logger.info("训练因 Early Stopping 提前结束 (step %d)", global_step)
            logger.info("Best ValLoss: %.4f, 建议使用 'best' 目录的 checkpoint", best_val_loss)
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("训练完成!")
            logger.info("=" * 60)

    # 确保 rank 0 保存完毕后再退出
    if dist.is_initialized():
        dist.barrier()


# ============================================================================
# 入口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Himmel-VLM Training")
    parser.add_argument("--config", type=str, required=True, help="训练配置 JSON 文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed 自动设置")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
