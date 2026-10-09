#!/usr/bin/env python3
"""
Himmel-VLM: 双流视频数据加载器
=================================
核心功能:
1. 从视频中提取 Keyframes (稀疏, 1fps) 和 Diff-Maps (密集, 帧间差分)
2. 构造包含 <motion_pad> 占位符的文本模板
3. 同时支持 Qwen2.5-VL / Qwen3-VL / LLaVA-OneVision / InternVL3 的数据格式

数据集格式 (JSON):
[
    {
        "video": "path/to/video.mp4",
        "conversations": [
            {"from": "human", "value": "<video>\nDescribe the video."},
            {"from": "gpt", "value": "A cat is playing..."}
        ]
    },
    ...
]

Keyframe + Diff-Map 提取策略:
- 每秒取 1 个 Keyframe (RGB, 原始分辨率给 ViT)
- 每两个相邻 Keyframe 之间, 均匀取 N_diff 帧, 计算帧间差分
- Diff-Map = |Frame_t - Frame_{t-1}|, resize 到 224x224 (给 ResNet)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .compressed_video_loader import (
    decode_video_frames,
    extract_compressed_video_streams,
    validate_compressed_backend,
)

logger = logging.getLogger(__name__)

# ============================================================================
# 差分图预处理
# ============================================================================
FORMULA_DIFF_MODES = {"luma_motion_edge"}

DIFF_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet 标准化 (匹配 ResNet 预训练)
        std=[0.229, 0.224, 0.225],
    ),
])

SIGNED_DIFF_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# 把上面那条 PIL pipeline 改成纯 torch 的 batched 版本, 用于 CPU dataloader 加速:
#   1. np.stack: O(N) 一次性拷贝, 比 N 次 Image.fromarray 快 ~3x
#   2. F.interpolate(bilinear): C++ 实现, 比 PIL.Resize 快 ~3-5x
#   3. 向量化 normalize: 直接用 broadcast, 一行代替 ToTensor + Normalize
def fast_compressed_stream_transform_batch(
    arrays: list,
    target_size: tuple = (224, 224),
    mean: float = 0.5,
    std: float = 0.5,
) -> torch.Tensor:
    """把一组 [H, W, 3] uint8 数组一次性转成 [N, 3, target_h, target_w] float tensor。

    与 SIGNED_DIFF_TRANSFORM 数学上等价 (resize=bilinear/antialias=False, mean/std=0.5),
    但走 torch C++ kernel + 向量化, 跳过 PIL.Image。在压缩域 stage1 的 dataloader
    里能从 ~600ms/sample 降到 ~80ms/sample。
    """
    if not arrays:
        return torch.empty(0, 3, target_size[0], target_size[1], dtype=torch.float32)

    # 1. 统一到同一尺寸再 stack (mv_maps 各帧分辨率可能略有差异)
    # 但 coviar 输出的 mv_maps 通常是同一视频内同分辨率, 直接尝试 stack
    try:
        stacked = np.stack(arrays, axis=0)                    # [N, H, W, 3] uint8
    except ValueError:
        # 各帧分辨率不一致, 退化到单张 resize 后再 stack
        stacked_list = []
        for arr in arrays:
            t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).float()
            t = torch.nn.functional.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
            stacked_list.append(t)
        batched = torch.cat(stacked_list, dim=0)              # [N, 3, H, W]
        return (batched / 255.0 - mean) / std

    # 2. uint8 → float32 + permute 到 [N, 3, H, W]
    batched = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous().float()

    # 3. resize 到目标尺寸 (跳过当且仅当已经是目标尺寸)
    if batched.shape[-2:] != target_size:
        batched = torch.nn.functional.interpolate(
            batched, size=target_size, mode="bilinear", align_corners=False
        )

    # 4. ToTensor + Normalize 等价: x/255 -> (x - mean) / std
    batched = batched / 255.0
    batched.sub_(mean).div_(std)
    return batched


def build_diff_transform(diff_mode: str = "abs"):
    """根据差分表示选择预处理。

    - abs: 传统绝对差分, 更像边缘能量图
    - signed/signed_accum: 以 128 作为零点编码正负残差, 映射到 [-1, 1]
    - luma_motion_edge: 三通道公式特征, 统一以 128 作为零运动基线
    """
    if diff_mode in {"signed", "signed_accum", *FORMULA_DIFF_MODES}:
        return SIGNED_DIFF_TRANSFORM
    return DIFF_TRANSFORM


def build_compressed_stream_transform(stream_name: str):
    """压缩域流统一使用有符号残差风格归一化。"""
    if stream_name in {"motion_vectors", "residuals"}:
        return SIGNED_DIFF_TRANSFORM
    raise ValueError(f"不支持的压缩域流: {stream_name}")


# I-frame 像素使用 ImageNet 归一化 (和 ResNet 预训练一致)
IFRAME_PIXEL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])



def get_motion_token_count(
    diff_frames: Optional[torch.Tensor] = None,
    motion_vectors: Optional[torch.Tensor] = None,
    residuals: Optional[torch.Tensor] = None,
    iframe_frames: Optional[torch.Tensor] = None,
) -> int:
    for tensor in (motion_vectors, residuals, diff_frames, iframe_frames):
        if tensor is not None:
            return int(tensor.shape[0])
    return 0



def slice_motion_stream(
    tensor: Optional[torch.Tensor],
    indices: List[int],
    num_units_per_interval: int,
) -> Optional[torch.Tensor]:
    if tensor is None or tensor.numel() == 0:
        return tensor
    if num_units_per_interval <= 0:
        return tensor[:0]

    new_chunks = []
    for i in range(len(indices) - 1):
        start = indices[i] * num_units_per_interval
        end = start + num_units_per_interval
        if end <= tensor.shape[0]:
            new_chunks.append(tensor[start:end])
        else:
            new_chunks.append(tensor[-num_units_per_interval:])

    if new_chunks:
        return torch.cat(new_chunks, dim=0)
    return tensor[:num_units_per_interval]



def slice_motion_items(
    items: Optional[Any],
    indices: List[int],
    num_units_per_interval: int,
):
    """按 keyframe 子采样结果同步裁剪 motion 序列，兼容 Tensor 和原始图像列表。"""
    if items is None:
        return items

    if isinstance(items, list):
        if not items:
            return items
        if num_units_per_interval <= 0:
            return []

        new_chunks = []
        for i in range(len(indices) - 1):
            start = indices[i] * num_units_per_interval
            end = start + num_units_per_interval
            if end <= len(items):
                new_chunks.extend(items[start:end])
            else:
                new_chunks.extend(items[-num_units_per_interval:])

        return new_chunks if new_chunks else items[:num_units_per_interval]

    return slice_motion_stream(items, indices, num_units_per_interval)



def collect_motion_streams(sample: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    return (
        sample.get("diff_frames"),
        sample.get("motion_vectors"),
        sample.get("residuals"),
        sample.get("iframe_frames"),
    )



def _rgb_to_luma(frame: np.ndarray) -> np.ndarray:
    frame_f32 = frame.astype(np.float32)
    return 0.299 * frame_f32[..., 0] + 0.587 * frame_f32[..., 1] + 0.114 * frame_f32[..., 2]



def _edge_energy_map(gray_frame: np.ndarray) -> np.ndarray:
    dx = np.abs(np.diff(gray_frame, axis=1, prepend=gray_frame[:, :1]))
    dy = np.abs(np.diff(gray_frame, axis=0, prepend=gray_frame[:1, :]))
    return np.clip(dx + dy, 0.0, 255.0)



def _encode_zero_centered_channel(values: np.ndarray, *, symmetric: bool) -> np.ndarray:
    if symmetric:
        encoded = values + 128.0
    else:
        encoded = 128.0 + 0.5 * values
    return np.clip(encoded, 0.0, 255.0).astype(np.uint8)



def encode_diff_map(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    *,
    diff_mode: str = "abs",
    ref_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    """把 RGB 帧对编码成 motion 输入图。

    压缩域启发:
    - abs:              |I_t - I_{t-1}|，无方向，旧实现
    - signed:           (I_t - I_{t-1}) + 128，保留变化方向
    - signed_accum:     (I_t - I_ref) + 128，近似 accumulated residual
    - luma_motion_edge: 三通道公式特征
        C1 = 累计亮度差        (Y_t - Y_ref)
        C2 = 局部运动能量      |Y_t - Y_{t-1}|
        C3 = 边缘变化强度      |∇Y_t - ∇Y_ref|

    其中 128 表示“零变化/零运动”，便于后续用 0.5/0.5 归一化映射到以 0 为中心的表示。
    """
    prev_i16 = prev_frame.astype(np.int16)
    curr_i16 = curr_frame.astype(np.int16)

    if diff_mode == "abs":
        return np.abs(curr_i16 - prev_i16).astype(np.uint8)

    if diff_mode == "signed":
        signed = curr_i16 - prev_i16
    elif diff_mode == "signed_accum":
        base = ref_frame if ref_frame is not None else prev_frame
        signed = curr_i16 - base.astype(np.int16)
    elif diff_mode in FORMULA_DIFF_MODES:
        base_frame = ref_frame if ref_frame is not None else prev_frame
        prev_luma = _rgb_to_luma(prev_frame)
        curr_luma = _rgb_to_luma(curr_frame)
        base_luma = _rgb_to_luma(base_frame)

        accum_luma = np.clip(curr_luma - base_luma, -127.0, 127.0)
        motion_energy = np.clip(np.abs(curr_luma - prev_luma), 0.0, 254.0)
        edge_change = np.clip(
            np.abs(_edge_energy_map(curr_luma) - _edge_energy_map(base_luma)),
            0.0,
            254.0,
        )

        return np.stack([
            _encode_zero_centered_channel(accum_luma, symmetric=True),
            _encode_zero_centered_channel(motion_energy, symmetric=False),
            _encode_zero_centered_channel(edge_change, symmetric=False),
        ], axis=-1)
    else:
        raise ValueError(f"不支持的 diff_mode: {diff_mode}")

    return np.clip(signed + 128, 0, 255).astype(np.uint8)


def make_empty_diff_map(diff_size: Tuple[int, int], diff_mode: str = "abs") -> np.ndarray:
    """构造零运动的占位 diff map。"""
    fill = 128 if diff_mode in {"signed", "signed_accum", *FORMULA_DIFF_MODES} else 0
    return np.full((*diff_size, 3), fill, dtype=np.uint8)


def extract_keyframes_and_diffs(
    video_path: str,
    keyframe_fps: float = 1.0,
    num_diffs_per_interval: int = 4,
    max_keyframes: int = 32,
    diff_size: Tuple[int, int] = (224, 224),
    diff_mode: str = "abs",
    decord_num_threads: int = 2,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    从视频中提取关键帧和差分图。

    Args:
        video_path: 视频文件路径
        keyframe_fps: 关键帧采样率 (默认 1fps)
        num_diffs_per_interval: 每两个关键帧之间采样的差分帧数
        max_keyframes: 最大关键帧数 (防止超长视频 OOM)
        diff_size: 差分图 resize 尺寸
        decord_num_threads: 单个 VideoReader 的解码线程数

    Returns:
        keyframes: List[np.ndarray], 每个 [H, W, 3] RGB
        diff_maps: List[np.ndarray], 每个 [224, 224, 3] motion 输入图
        keyframe_indices: List[int], 关键帧在原视频中的帧索引

    diff_mode:
        - "abs":              传统绝对差分 |I_t - I_{t-1}|
        - "signed":           保留正负方向的局部残差 (I_t - I_{t-1})
        - "signed_accum":     保留区间累计方向信息 (I_t - I_ref), 近似 H.264 residual accumulation
        - "luma_motion_edge": 三通道公式运动特征
    """
    try:
        import decord
        # native bridge 返回 decord.NDArray, 通过 .asnumpy() 转换
        # 注: 某些 decord 版本不支持 "numpy" bridge
        decord.bridge.set_bridge("native")
    except ImportError:
        raise ImportError("请安装 decord: pip install decord")

    vr = decord.VideoReader(video_path, num_threads=max(1, int(decord_num_threads)))
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()

    if total_frames == 0 or video_fps <= 0:
        logger.warning("无法读取视频: %s (frames=%d, fps=%.1f)", video_path, total_frames, video_fps)
        return [], [], []

    # 计算关键帧索引 (每秒取一帧)
    keyframe_interval = max(1, int(video_fps / keyframe_fps))
    keyframe_indices = list(range(0, total_frames, keyframe_interval))

    # 限制最大关键帧数
    if len(keyframe_indices) > max_keyframes:
        # 均匀采样
        step = len(keyframe_indices) / max_keyframes
        keyframe_indices = [keyframe_indices[int(i * step)] for i in range(max_keyframes)]

    if len(keyframe_indices) < 2:
        # 视频太短, 至少取首尾两帧
        keyframe_indices = [0, total_frames - 1] if total_frames > 1 else [0]

    # 批量读取关键帧
    keyframes_np = vr.get_batch(keyframe_indices).asnumpy()  # [T_key, H, W, 3]
    keyframes = [keyframes_np[i] for i in range(len(keyframe_indices))]

    # 先收集所有需要的 prev/curr 帧索引，再一次性批量解码，避免 N 次小 get_batch 拖慢 CPU。
    pending_requests: List[Tuple[np.ndarray, int, int]] = []
    needed_frame_indices = set()
    for k in range(len(keyframe_indices) - 1):
        start_idx = keyframe_indices[k]
        end_idx = keyframe_indices[k + 1]
        ref_frame = keyframes[k]

        if end_idx - start_idx <= 1:
            for _ in range(num_diffs_per_interval):
                pending_requests.append((ref_frame, -1, -1))
            continue

        interval_indices = np.linspace(
            start_idx, end_idx, num_diffs_per_interval + 2, dtype=int
        )[1:-1]
        interval_indices = np.clip(interval_indices, 0, total_frames - 1)

        for idx in interval_indices:
            prev_idx = max(0, int(idx) - 1)
            idx = int(idx)
            pending_requests.append((ref_frame, prev_idx, idx))
            needed_frame_indices.add(prev_idx)
            needed_frame_indices.add(idx)

    decoded_frames = {}
    if needed_frame_indices:
        sorted_indices = sorted(needed_frame_indices)
        decoded_batch = vr.get_batch(sorted_indices).asnumpy()
        decoded_frames = {
            frame_idx: decoded_batch[pos]
            for pos, frame_idx in enumerate(sorted_indices)
        }

    diff_maps = []
    for ref_frame, prev_idx, curr_idx in pending_requests:
        if prev_idx < 0 or curr_idx < 0:
            diff_maps.append(make_empty_diff_map(diff_size, diff_mode))
            continue

        diff = encode_diff_map(
            decoded_frames[prev_idx],
            decoded_frames[curr_idx],
            diff_mode=diff_mode,
            ref_frame=ref_frame,
        )
        diff_maps.append(diff)

    return keyframes, diff_maps, keyframe_indices


def _is_incompatible_compressed_data_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    return isinstance(exc, ValueError) and (
        "mpeg-4" in error_text
        or "video codec=" in error_text
        or "gop" in error_text
        or "raw video" in error_text
    )



def extract_proxy_compressed_streams(
    video_path: str,
    *,
    keyframe_fps: float = 1.0,
    num_motion_units_per_interval: int = 4,
    max_keyframes: int = 32,
    visual_stream_mode: str = "iframe",
    use_motion_vectors: bool = True,
    use_residuals: bool = True,
    accumulate_motion_vectors: bool = True,
    accumulate_residuals: bool = False,
    diff_size: Tuple[int, int] = (224, 224),
    decord_num_threads: int = 2,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], Dict[str, Any]]:
    """为非 CoViAR 兼容视频构造一个 RGB-decoded proxy 三流，保持训练入口不断。"""
    try:
        import decord
        decord.bridge.set_bridge("native")
    except ImportError as exc:
        raise ImportError("compressed proxy fallback 依赖 decord: pip install decord") from exc

    vr = decord.VideoReader(video_path, num_threads=max(1, int(decord_num_threads)))
    total_frames = len(vr)
    video_fps = float(vr.get_avg_fps())

    if total_frames == 0 or video_fps <= 0:
        logger.warning("无法读取视频: %s (frames=%d, fps=%.3f)", video_path, total_frames, video_fps)
        return [], [], [], [], {
            "video_fps": video_fps,
            "total_frames": total_frames,
            "visual_stream_mode": visual_stream_mode,
            "motion_input_mode": "compressed",
            "compressed_proxy_fallback": True,
        }

    keyframe_interval = max(1, int(video_fps / max(float(keyframe_fps), 1e-6)))
    keyframe_indices = list(range(0, total_frames, keyframe_interval))
    if len(keyframe_indices) > max_keyframes:
        step = len(keyframe_indices) / max_keyframes
        keyframe_indices = [keyframe_indices[int(i * step)] for i in range(max_keyframes)]
    if len(keyframe_indices) < 2:
        keyframe_indices = [0, total_frames - 1] if total_frames > 1 else [0]

    keyframes_np = vr.get_batch(keyframe_indices).asnumpy()
    keyframes = [keyframes_np[i] for i in range(len(keyframe_indices))]

    interval_indices_list: List[np.ndarray] = []
    interval_frame_indices: List[List[int]] = []
    needed_frame_indices = set()

    for start_idx, end_idx in zip(keyframe_indices[:-1], keyframe_indices[1:]):
        if num_motion_units_per_interval <= 0:
            interval_indices = np.zeros((0,), dtype=int)
        elif end_idx - start_idx <= 1:
            fallback_idx = min(max(start_idx + 1, 0), max(total_frames - 1, 0))
            interval_indices = np.full((num_motion_units_per_interval,), fallback_idx, dtype=int)
        else:
            interval_indices = np.linspace(
                start_idx + 1,
                max(start_idx + 1, end_idx - 1),
                num_motion_units_per_interval,
                dtype=int,
            )
            interval_indices = np.clip(interval_indices, start_idx + 1, max(start_idx + 1, end_idx - 1))

        interval_indices_list.append(interval_indices)
        interval_frame_indices.append([int(start_idx)] + [int(idx) for idx in interval_indices.tolist()])

        for idx in interval_indices:
            curr_idx = int(idx)
            prev_idx = max(0, curr_idx - 1)
            needed_frame_indices.add(prev_idx)
            needed_frame_indices.add(curr_idx)

    decoded_frames: Dict[int, np.ndarray] = {}
    if needed_frame_indices:
        sorted_indices = sorted(needed_frame_indices)
        decoded_batch = vr.get_batch(sorted_indices).asnumpy()
        decoded_frames = {
            frame_idx: decoded_batch[pos]
            for pos, frame_idx in enumerate(sorted_indices)
        }

    mv_maps: List[np.ndarray] = []
    residual_maps: List[np.ndarray] = []
    residual_mode = "signed_accum" if accumulate_residuals else "signed"

    for interval_idx, interval_indices in enumerate(interval_indices_list):
        ref_frame = keyframes[interval_idx]
        for idx in interval_indices:
            curr_idx = int(idx)
            prev_idx = max(0, curr_idx - 1)
            prev_frame = decoded_frames.get(prev_idx, ref_frame)
            curr_frame = decoded_frames.get(curr_idx, prev_frame)

            if use_motion_vectors:
                mv_maps.append(
                    encode_diff_map(
                        prev_frame,
                        curr_frame,
                        diff_mode="luma_motion_edge",
                        ref_frame=ref_frame if accumulate_motion_vectors else None,
                    )
                )

            if use_residuals:
                residual_maps.append(
                    encode_diff_map(
                        prev_frame,
                        curr_frame,
                        diff_mode=residual_mode,
                        ref_frame=ref_frame if accumulate_residuals else None,
                    )
                )

    if num_motion_units_per_interval > 0 and len(keyframe_indices) >= 2 and not mv_maps and use_motion_vectors:
        mv_maps = [make_empty_diff_map(diff_size, "luma_motion_edge") for _ in range(num_motion_units_per_interval)]
    if num_motion_units_per_interval > 0 and len(keyframe_indices) >= 2 and not residual_maps and use_residuals:
        residual_maps = [make_empty_diff_map(diff_size, residual_mode) for _ in range(num_motion_units_per_interval)]

    meta = {
        "video_fps": video_fps,
        "total_frames": total_frames,
        "visual_stream_mode": visual_stream_mode,
        "motion_input_mode": "compressed",
        "compressed_backend": "rgb_proxy",
        "compressed_proxy_fallback": True,
        "compressed_proxy_mv_mode": "luma_motion_edge",
        "compressed_proxy_residual_mode": residual_mode,
        "num_visual_frames": len(keyframes),
        "num_motion_vectors": len(mv_maps),
        "num_residuals": len(residual_maps),
        "interval_frame_indices": interval_frame_indices,
        "accumulate_motion_vectors": accumulate_motion_vectors,
        "accumulate_residuals": accumulate_residuals,
    }
    return keyframes, mv_maps, residual_maps, keyframe_indices, meta


# ============================================================================
# 数据集类
# ============================================================================

class DeltaVideoDataset(Dataset):
    """
    双流 / 压缩域视频理解数据集。

    输出格式 (每条样本):
    {
        "keyframes": List[PIL.Image],                    # 视觉流输入
        "diff_frames": Optional[Tensor[T, 3, 224, 224]], # 旧 RGB diff 流
        "motion_vectors": Optional[Tensor[T, 3, 224, 224]],
        "residuals": Optional[Tensor[T, 3, 224, 224]],
        "conversations": List[Dict],
        "num_keyframes": int,
        "num_diffs_per_keyframe": int,
        "video_path": str,
    }

    下游 collator 负责:
    - 用 Qwen/LLaVA processor 处理 keyframes -> pixel_values
    - 构造包含 <motion_pad> 占位符的 input_ids
    - 将 diff/MV/residual 以 flat tensor 字段打包进 batch
    """

    def __init__(
        self,
        data_path: str,
        video_root: str,
        keyframe_fps: float = 1.0,
        num_diffs_per_interval: int = 4,
        diff_mode: str = "abs",
        max_keyframes: int = 32,
        max_length: int = 2048,
        motion_input_mode: str = "rgb_diff",
        compressed_backend: str = "coviar",
        visual_stream_mode: str = "iframe",
        compressed_gop_size: int = 12,
        use_motion_vectors: bool = True,
        use_residuals: bool = True,
        use_iframe_encoder: bool = False,
        accumulate_motion_vectors: bool = True,
        accumulate_residuals: bool = False,
        motion_vector_clip: float = 20.0,
        decord_num_threads: int = 2,
        return_diff_images: bool = False,
        return_distill_target_frames: bool = False,
        allow_compressed_proxy_fallback: bool = False,
    ):
        """
        Args:
            data_path: JSON 数据文件路径
            video_root: 视频文件根目录
            keyframe_fps: 关键帧采样率
            num_diffs_per_interval: 每个关键帧间隔的 motion 单元数
            diff_mode: RGB-diff 模式, 支持 abs / signed / signed_accum / luma_motion_edge
            max_keyframes: 最大关键帧数 (防 OOM)
            max_length: 最大文本 token 长度
            motion_input_mode: "rgb_diff" 或 "compressed"
            compressed_backend: 真实压缩域 backend, 当前支持 coviar
            visual_stream_mode: "iframe" 或 "keyframe"
            compressed_gop_size: 压缩域 GOP 大小 (CoViAR 默认 12)
            use_motion_vectors: 是否启用 MV 流
            use_residuals: 是否启用 residual 流
            use_iframe_encoder: 是否启用独立 I-frame 编码器 (CoViAR 真三流)
            accumulate_motion_vectors: 是否读取 accumulated MV
            accumulate_residuals: 是否读取 accumulated residual
            motion_vector_clip: motion vector 编码时的裁剪阈值
            decord_num_threads: RGB-diff 路径下单个 VideoReader 的解码线程数
            return_diff_images: 是否保留原始 diff 图像列表 (shared_vit 需要)
            return_distill_target_frames: 是否额外返回每个区间对应的 k+1 张解码 RGB 帧 (Stage 1 蒸馏)
            allow_compressed_proxy_fallback: 当 `coviar` 不支持当前视频编码/GOP 时，是否自动退回 RGB proxy 三流
        """
        super().__init__()
        self.video_root = video_root
        self.keyframe_fps = keyframe_fps
        self.num_diffs_per_interval = num_diffs_per_interval
        self.diff_mode = diff_mode
        self.diff_transform = build_diff_transform(diff_mode)
        self.motion_vector_transform = build_compressed_stream_transform("motion_vectors")
        self.residual_transform = build_compressed_stream_transform("residuals")
        self.iframe_pixel_transform = IFRAME_PIXEL_TRANSFORM
        self.max_keyframes = max_keyframes
        self.max_length = max_length
        self.motion_input_mode = motion_input_mode
        self.compressed_backend = compressed_backend
        self.visual_stream_mode = visual_stream_mode
        self.compressed_gop_size = compressed_gop_size
        self.use_motion_vectors = use_motion_vectors
        self.use_residuals = use_residuals
        self.use_iframe_encoder = use_iframe_encoder
        self.accumulate_motion_vectors = accumulate_motion_vectors
        self.accumulate_residuals = accumulate_residuals
        self.motion_vector_clip = motion_vector_clip
        self.decord_num_threads = max(1, int(decord_num_threads))
        self.return_diff_images = return_diff_images
        self.return_distill_target_frames = return_distill_target_frames
        self.allow_compressed_proxy_fallback = bool(allow_compressed_proxy_fallback)
        self.compressed_backend_ready = False
        self._logged_proxy_fallback = False

        if self.motion_input_mode == "compressed":
            try:
                validate_compressed_backend(self.compressed_backend)
                self.compressed_backend_ready = True
                logger.info("compressed-domain backend 就绪: %s", self.compressed_backend)
            except Exception as exc:
                if not self.allow_compressed_proxy_fallback:
                    raise
                logger.warning(
                    "compressed-domain backend 预检失败，将允许 RGB proxy fallback (backend=%s): %s",
                    self.compressed_backend,
                    exc,
                )

        # 加载数据
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        logger.info(
            "加载 %d 条训练样本 from %s (motion_input_mode=%s, diff_mode=%s, visual_stream_mode=%s)",
            len(self.data),
            data_path,
            motion_input_mode,
            diff_mode,
            visual_stream_mode,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        video_path = os.path.join(self.video_root, item["video"])

        # 自动补全视频扩展名 (LLaVA-Video-178K 部分样本缺少 .mp4 后缀)
        if not os.path.exists(video_path) and not os.path.splitext(video_path)[1]:
            for ext in (".mp4", ".mkv", ".webm", ".avi"):
                if os.path.exists(video_path + ext):
                    video_path = video_path + ext
                    break

        # 提取关键帧与运动流
        compressed_meta: Dict[str, Any] = {}
        keyframes: List[np.ndarray] = []
        kf_indices: List[int] = []
        mv_maps: List[np.ndarray] = []
        residual_maps: List[np.ndarray] = []
        motion_vectors = None
        residuals = None
        iframe_frames = None
        diff_tensors = None
        diff_images: List[Image.Image] = []
        distill_target_frames: List[List[Image.Image]] = []
        try:
            if self.motion_input_mode == "compressed":
                fallback_reason: Optional[Exception] = None
                if self.compressed_backend_ready:
                    try:
                        keyframes, mv_maps, residual_maps, kf_indices, compressed_meta = extract_compressed_video_streams(
                            video_path=video_path,
                            keyframe_fps=self.keyframe_fps,
                            num_motion_units_per_interval=self.num_diffs_per_interval,
                            max_keyframes=self.max_keyframes,
                            visual_stream_mode=self.visual_stream_mode,
                            compressed_backend=self.compressed_backend,
                            compressed_gop_size=self.compressed_gop_size,
                            use_motion_vectors=self.use_motion_vectors,
                            use_residuals=self.use_residuals,
                            accumulate_motion_vectors=self.accumulate_motion_vectors,
                            accumulate_residuals=self.accumulate_residuals,
                            motion_vector_clip=self.motion_vector_clip,
                        )
                    except Exception as exc:
                        error_text = str(exc).lower()
                        backend_error = isinstance(exc, (ImportError, AttributeError)) and "coviar" in error_text
                        if self.allow_compressed_proxy_fallback and (backend_error or _is_incompatible_compressed_data_error(exc)):
                            fallback_reason = exc
                        else:
                            raise
                else:
                    fallback_reason = RuntimeError(f"compressed backend 未就绪: {self.compressed_backend}")

                if fallback_reason is not None:
                    if not self._logged_proxy_fallback:
                        logger.warning(
                            "compressed-domain 已启用 RGB proxy fallback；后续不兼容样本将自动降级继续训练。backend=%s, video=%s, reason=%s",
                            self.compressed_backend,
                            video_path,
                            fallback_reason,
                        )
                        self._logged_proxy_fallback = True
                    keyframes, mv_maps, residual_maps, kf_indices, compressed_meta = extract_proxy_compressed_streams(
                        video_path=video_path,
                        keyframe_fps=self.keyframe_fps,
                        num_motion_units_per_interval=self.num_diffs_per_interval,
                        max_keyframes=self.max_keyframes,
                        visual_stream_mode=self.visual_stream_mode,
                        use_motion_vectors=self.use_motion_vectors,
                        use_residuals=self.use_residuals,
                        accumulate_motion_vectors=self.accumulate_motion_vectors,
                        accumulate_residuals=self.accumulate_residuals,
                        decord_num_threads=self.decord_num_threads,
                    )
                    compressed_meta["proxy_reason"] = str(fallback_reason)
                    compressed_meta["requested_backend"] = self.compressed_backend

                if mv_maps:
                    # 用 batched 纯 torch 转换替代逐张 PIL pipeline (~5-7x 提速)
                    motion_vectors = fast_compressed_stream_transform_batch(
                        mv_maps, target_size=(224, 224), mean=0.5, std=0.5,
                    )
                if residual_maps:
                    residuals = fast_compressed_stream_transform_batch(
                        residual_maps, target_size=(224, 224), mean=0.5, std=0.5,
                    )
                # CoViAR 真三流: 为每个 motion interval 生成对应的 I-frame 像素 tensor
                # 每个 interval 的 I-frame 取区间起始 keyframe，重复 num_diffs_per_interval 次
                if self.use_iframe_encoder and keyframes and len(keyframes) > 1:
                    iframe_tensors = []
                    for k_idx in range(len(keyframes) - 1):
                        kf_img = Image.fromarray(keyframes[k_idx])
                        kf_tensor = self.iframe_pixel_transform(kf_img)
                        for _ in range(self.num_diffs_per_interval):
                            iframe_tensors.append(kf_tensor)
                    if iframe_tensors:
                        iframe_frames = torch.stack(iframe_tensors)

                if self.return_distill_target_frames:
                    interval_frame_indices = compressed_meta.get("interval_frame_indices", [])
                    if interval_frame_indices:
                        flat_indices = [idx for interval in interval_frame_indices for idx in interval]
                        decoded_frames = decode_video_frames(video_path, flat_indices)
                        cursor = 0
                        for interval in interval_frame_indices:
                            interval_len = len(interval)
                            interval_frames = decoded_frames[cursor:cursor + interval_len]
                            cursor += interval_len
                            if interval_frames:
                                distill_target_frames.append([
                                    Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in interval_frames
                                ])
            else:
                keyframes, diff_maps, kf_indices = extract_keyframes_and_diffs(
                    video_path=video_path,
                    keyframe_fps=self.keyframe_fps,
                    num_diffs_per_interval=self.num_diffs_per_interval,
                    max_keyframes=self.max_keyframes,
                    diff_mode=self.diff_mode,
                    decord_num_threads=self.decord_num_threads,
                )
                if diff_maps:
                    if self.return_diff_images:
                        diff_images = [Image.fromarray(dm) for dm in diff_maps]
                    diff_tensors = torch.stack([
                        self.diff_transform(Image.fromarray(dm)) for dm in diff_maps
                    ])
        except Exception as e:
            error_text = str(e).lower()
            backend_error = (
                self.motion_input_mode == "compressed"
                and isinstance(e, (ImportError, AttributeError))
                and "coviar" in error_text
            )
            incompatible_compressed_data_error = (
                self.motion_input_mode == "compressed"
                and _is_incompatible_compressed_data_error(e)
            )
            if backend_error and not self.allow_compressed_proxy_fallback:
                raise RuntimeError(
                    "compressed-domain backend 不可用：当前配置要求 coviar，但训练环境中未正确安装或未暴露 load() 接口。"
                ) from e
            if incompatible_compressed_data_error and not self.allow_compressed_proxy_fallback:
                raise RuntimeError(
                    "compressed-domain 数据不兼容：coviar / pytorch-coviar 当前要求预先转码成固定 GOP 的 MPEG-4 视频，不能直接读取当前这批原始训练视频。请先对训练集做统一转码，或改回 motion_input_mode=rgb_diff。"
                ) from e

            raise RuntimeError(f"Video extraction failed for training sample {idx}: {video_path}") from e

        if not keyframes:
            raise RuntimeError(f"No decoded frames for training sample {idx}: {video_path}")

        # 转换关键帧为 PIL Image (后续由 processor 处理)
        keyframe_images = [Image.fromarray(kf) for kf in keyframes]

        if self.motion_input_mode == "rgb_diff":
            if diff_tensors is None:
                diff_tensors = torch.zeros(1, 3, 224, 224)
        else:
            if motion_vectors is None and self.use_motion_vectors:
                motion_vectors = torch.zeros(1, 3, 224, 224)
            if residuals is None and self.use_residuals:
                residuals = torch.zeros(1, 3, 224, 224)
            if iframe_frames is None and self.use_iframe_encoder:
                iframe_frames = torch.zeros(1, 3, 224, 224)

        # 对话数据
        conversations = item.get("conversations", [])

        # 推断任务类型: caption / open_qa / multi_choice
        task_type = self._infer_task_type(conversations)

        return {
            "keyframes": keyframe_images,
            "diff_frames": diff_tensors,
            "diff_images": diff_images,
            "motion_vectors": motion_vectors,
            "residuals": residuals,
            "iframe_frames": iframe_frames,
            "distill_target_frames": distill_target_frames,
            "motion_input_mode": self.motion_input_mode,
            "conversations": conversations,
            "num_keyframes": len(keyframes),
            "num_diffs_per_keyframe": self.num_diffs_per_interval,
            "video_path": video_path,
            "task_type": task_type,
            "compressed_meta": compressed_meta,
        }

    @staticmethod
    def _infer_task_type(conversations: List[Dict[str, Any]]) -> str:
        """
        从对话内容启发式推断任务类型。
        - multi_choice: answer 是单字母 (A/B/C/D/...) 或极短 (<=5 words)
                        且 question 中包含选项模式 (A) (B) 或 A. B.
        - caption:      question 含 describe/caption/summarize 等关键词
        - open_qa:      其余
        """
        question = ""
        answer = ""
        for conv in conversations:
            if conv["from"] == "human":
                question = conv["value"].lower()
            elif conv["from"] == "gpt":
                answer = conv["value"].strip()

        # Multi-choice: 极短回答 + 问题含选项
        answer_words = answer.split()
        has_options = any(p in question for p in ["(a)", "(b)", "(1)", "(2)", " a.", " b.", " a)", " b)"])
        if len(answer_words) <= 3 and (has_options or (len(answer) <= 2 and answer.upper() in "ABCDEFGH")):
            return "multi_choice"

        # Caption: 描述类问题或长回答
        caption_keywords = ["describe", "caption", "summarize", "summary", "what is happening",
                            "what happens", "provide a detailed", "narrate"]
        if any(kw in question for kw in caption_keywords) or len(answer_words) > 50:
            return "caption"

        return "open_qa"


# ============================================================================
# Collator: 为 Qwen2.5-VL 构造 batch
# ============================================================================

class QwenDeltaCollator:
    """
    为 QwenDelta 模型构造 batch 数据。

    职责:
    1. 用 Qwen2.5-VL processor 处理关键帧 -> pixel_values_videos + video_grid_thw
    2. 构造包含 <motion_pad> 占位符的文本序列
    3. 打包 diff_frames tensor

    文本模板:
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <|vision_start|><|video_pad|>...<|vision_end|>
    <|motion_start|><|motion_pad|>...<|motion_end|>
    {question}<|im_end|>
    <|im_start|>assistant
    {answer}<|im_end|>
    """

    def __init__(
        self,
        processor,
        tokenizer,
        max_length: int = 2048,
        alignment_mode: bool = False,
        shared_vit_input_mode: str = "video",
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.alignment_mode = alignment_mode  # Stage 1 Alignment: 不走 LLM, 不限帧数
        self.shared_vit_input_mode = shared_vit_input_mode

    def _limit_keyframes(
        self,
        keyframes: list[Image.Image],
        diff_frames,
        diff_images: list[Image.Image] | None,
        motion_vectors,
        residuals,
        iframe_frames,
        num_diffs_per_kf: int,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """
        根据 max_length 动态限制关键帧数量，并同步裁剪所有运动流。

        Qwen2.5-VL 的 vision token 数 = 每帧像素数 / (28*28) (patch size)
        通过 processor 的 min_pixels/max_pixels 已经控制了每帧 token 上限,
        这里用 max_pixels 估算最坏情况下的 token 数。

        Qwen3-VL patch size=32, 但 processor 参数同样通过 max_pixels 控制。
        """
        # 从 processor 获取实际 max_pixels 配置
        try:
            img_proc = getattr(self.processor, "image_processor", self.processor)
            max_px = getattr(img_proc, "max_pixels", 768 * 28 * 28)
            patch_size = getattr(img_proc, "patch_size", 28)
            if isinstance(patch_size, dict):
                patch_size = patch_size.get("height", 28)
            merge_size = getattr(img_proc, "merge_size", 2)
            tokens_per_frame = max_px // (patch_size * patch_size * merge_size * merge_size)
        except Exception:
            tokens_per_frame = 768

        motion_tokens = num_diffs_per_kf * (len(keyframes) - 1) if len(keyframes) > 1 else 0
        text_budget = 512 + motion_tokens
        vision_budget = self.max_length - text_budget
        max_frames = max(2, vision_budget // max(1, tokens_per_frame))
        diff_images = diff_images or []

        if len(keyframes) <= max_frames:
            return keyframes, diff_frames, diff_images, motion_vectors, residuals, iframe_frames

        logger.info(
            "关键帧过多 (%d -> %d), tokens/frame=%d, max_length=%d",
            len(keyframes), max_frames, tokens_per_frame, self.max_length,
        )

        step = len(keyframes) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        new_keyframes = [keyframes[i] for i in indices]

        diff_frames = slice_motion_items(diff_frames, indices, num_diffs_per_kf)
        diff_images = slice_motion_items(diff_images, indices, num_diffs_per_kf) or []
        motion_vectors = slice_motion_items(motion_vectors, indices, num_diffs_per_kf)
        residuals = slice_motion_items(residuals, indices, num_diffs_per_kf)
        iframe_frames = slice_motion_items(iframe_frames, indices, num_diffs_per_kf)

        return new_keyframes, diff_frames, diff_images, motion_vectors, residuals, iframe_frames

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        将 DeltaVideoDataset 的输出组装为模型输入。
        """
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_diff_frames = []
        all_motion_vectors = []
        all_residuals = []
        all_iframe_frames = []
        all_pixel_values_videos = []
        all_video_grid_thw = []
        all_second_per_grid_ts = []
        all_diff_pixel_values_videos = []
        all_diff_video_grid_thw = []
        all_diff_pixel_values = []
        all_diff_image_grid_thw = []
        all_diff_image_counts = []
        all_target_videos = []
        all_target_video_texts = []
        all_target_interval_counts = []
        all_task_types = []

        for sample in batch:
            keyframes = sample["keyframes"]
            diff_frames, motion_vectors, residuals, iframe_frames = collect_motion_streams(sample)
            diff_images: list[Image.Image] = sample.get("diff_images") or []
            distill_target_frames: list[list[Image.Image]] = (
                sample.get("distill_target_frames") or []
            ) if self.alignment_mode else []
            conversations = sample["conversations"]
            num_diffs_per_kf = sample.get("num_diffs_per_keyframe", 4)
            task_type = sample.get("task_type", "open_qa")

            # 根据 max_length 限制关键帧数量
            # alignment_mode 下不走 LLM, 不需要受 token 预算限制
            if keyframes and not self.alignment_mode:
                keyframes, diff_frames, diff_images, motion_vectors, residuals, iframe_frames = self._limit_keyframes(
                    keyframes, diff_frames, diff_images, motion_vectors, residuals, iframe_frames, num_diffs_per_kf
                )

            num_diffs = get_motion_token_count(diff_frames, motion_vectors, residuals, iframe_frames)

            # 构造 motion 占位符文本
            motion_placeholder = (
                "<|motion_start|>"
                + "<|motion_pad|>" * num_diffs
                + "<|motion_end|>"
            )

            # 从对话中提取 question 和 answer
            question = ""
            answer = ""
            for conv in conversations:
                if conv["from"] == "human":
                    question = conv["value"].replace("<video>", "").replace("<image>", "").strip()
                elif conv["from"] == "gpt":
                    answer = conv["value"]

            # 用 Qwen processor 处理视频帧
            if keyframes:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": keyframes},
                            {"type": "text", "text": motion_placeholder + "\n" + question},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    },
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                inputs = self.processor(
                    text=[text],
                    videos=[keyframes],
                    return_tensors="pt",
                    padding=True,
                )

                all_input_ids.append(inputs["input_ids"].squeeze(0))
                all_attention_mask.append(inputs["attention_mask"].squeeze(0))
                if "pixel_values_videos" in inputs:
                    all_pixel_values_videos.append(inputs["pixel_values_videos"])
                if "video_grid_thw" in inputs:
                    all_video_grid_thw.append(inputs["video_grid_thw"])
                if "second_per_grid_ts" in inputs:
                    all_second_per_grid_ts.append(inputs["second_per_grid_ts"])

                if self.shared_vit_input_mode == "images":
                    all_diff_image_counts.append(len(diff_images))

                if diff_images:
                    if self.shared_vit_input_mode == "images":
                        diff_inputs = self.processor(
                            text=["describe"] * len(diff_images),
                            images=diff_images,
                            return_tensors="pt",
                            padding=True,
                        )
                        if "pixel_values" in diff_inputs:
                            all_diff_pixel_values.append(diff_inputs["pixel_values"])
                        if "image_grid_thw" in diff_inputs:
                            all_diff_image_grid_thw.append(diff_inputs["image_grid_thw"])
                    else:
                        # Qwen Processor 要求 text 参数非空且包含视频占位符,
                        # 构造一个 dummy message 让 processor 正确处理 diff 帧
                        diff_messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "video", "video": diff_images},
                                    {"type": "text", "text": "describe"},
                                ],
                            }
                        ]
                        diff_text = self.processor.apply_chat_template(
                            diff_messages, tokenize=False, add_generation_prompt=False
                        )
                        diff_inputs = self.processor(
                            text=[diff_text],
                            videos=[diff_images],
                            return_tensors="pt",
                            padding=True,
                        )
                        if "pixel_values_videos" in diff_inputs:
                            all_diff_pixel_values_videos.append(diff_inputs["pixel_values_videos"])
                        if "video_grid_thw" in diff_inputs:
                            all_diff_video_grid_thw.append(diff_inputs["video_grid_thw"])

                # 构造 labels (只计算 answer 部分的 loss)
                input_ids = inputs["input_ids"].squeeze(0)
                labels = torch.full_like(input_ids, -100)
                # 找到 assistant 回复的起始位置
                # Qwen 格式: ...<|im_start|>assistant\n{answer}<|im_end|>
                assistant_token = self.tokenizer.encode(
                    "<|im_start|>assistant\n", add_special_tokens=False
                )
                # 简化: 从后往前找 assistant 标记
                ids_list = input_ids.tolist()
                for i in range(len(ids_list) - len(assistant_token)):
                    if ids_list[i:i + len(assistant_token)] == assistant_token:
                        labels[i + len(assistant_token):] = input_ids[i + len(assistant_token):]
                        break
                all_labels.append(labels)

                if diff_frames is not None:
                    all_diff_frames.append(diff_frames)
                if motion_vectors is not None:
                    all_motion_vectors.append(motion_vectors)
                if residuals is not None:
                    all_residuals.append(residuals)
                if iframe_frames is not None:
                    all_iframe_frames.append(iframe_frames)

                all_target_interval_counts.append(len(distill_target_frames))
                for interval_frames in distill_target_frames:
                    target_messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": interval_frames},
                                {"type": "text", "text": "describe"},
                            ],
                        }
                    ]
                    all_target_video_texts.append(
                        self.processor.apply_chat_template(
                            target_messages,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                    )
                    all_target_videos.append(interval_frames)

                all_task_types.append(task_type)
            else:
                # 视频加载失败, 跳过此样本 (不添加到 all_input_ids)
                logger.warning("跳过无关键帧样本 (Qwen collator)")

        # 所有样本都失败时, 返回 dummy batch 避免 crash
        if not all_input_ids:
            logger.warning("Batch 内所有样本视频加载失败, 返回 dummy batch")
            dummy_ids = torch.zeros(1, 1, dtype=torch.long)
            return {
                "input_ids": dummy_ids,
                "labels": torch.full((1, 1), -100, dtype=torch.long),
                "attention_mask": torch.zeros(1, 1, dtype=torch.long),
                "diff_frames": torch.zeros(1, 1, 3, 224, 224),
                "motion_vectors": torch.zeros(1, 1, 3, 224, 224),
                "residuals": torch.zeros(1, 1, 3, 224, 224),
                "iframe_frames": torch.zeros(1, 1, 3, 224, 224),
                "is_dummy": True,
            }

        # Pad to same length
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, lbl, mask in zip(all_input_ids, all_labels, all_attention_mask):
            pad_len = max_len - ids.shape[0]
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)])
            )
            padded_labels.append(
                torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype)])
            )
            padded_attention_mask.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            )

        result = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attention_mask),
            "diff_frames": torch.stack(all_diff_frames) if all_diff_frames else None,
            "motion_vectors": torch.stack(all_motion_vectors) if all_motion_vectors else None,
            "residuals": torch.stack(all_residuals) if all_residuals else None,
            "iframe_frames": torch.stack(all_iframe_frames) if all_iframe_frames else None,
            "task_types": all_task_types,
        }

        # 添加视觉输入
        if all_pixel_values_videos:
            result["pixel_values_videos"] = torch.cat(all_pixel_values_videos, dim=0)
        if all_video_grid_thw:
            result["video_grid_thw"] = torch.cat(all_video_grid_thw, dim=0)
        if all_second_per_grid_ts:
            result["second_per_grid_ts"] = torch.cat(all_second_per_grid_ts, dim=0)
        if all_diff_pixel_values_videos:
            result["diff_pixel_values_videos"] = torch.cat(all_diff_pixel_values_videos, dim=0)
        if all_diff_video_grid_thw:
            result["diff_video_grid_thw"] = torch.cat(all_diff_video_grid_thw, dim=0)
        if all_diff_pixel_values:
            result["diff_pixel_values"] = torch.cat(all_diff_pixel_values, dim=0)
        if all_diff_image_grid_thw:
            result["diff_image_grid_thw"] = torch.cat(all_diff_image_grid_thw, dim=0)
        if all_diff_image_counts:
            result["diff_image_counts"] = torch.tensor(all_diff_image_counts, dtype=torch.long)

        if all_target_videos:
            target_inputs = self.processor(
                text=all_target_video_texts,
                videos=all_target_videos,
                return_tensors="pt",
                padding=True,
            )
            if "pixel_values_videos" in target_inputs:
                result["target_pixel_values_videos"] = target_inputs["pixel_values_videos"]
            if "video_grid_thw" in target_inputs:
                result["target_video_grid_thw"] = target_inputs["video_grid_thw"]
        if all_target_interval_counts:
            result["target_interval_counts"] = torch.tensor(all_target_interval_counts, dtype=torch.long)

        return result


# ============================================================================
# Collator: 为 LLaVA-OneVision 构造 batch
# ============================================================================

class LlavaDeltaCollator:
    """
    为 LlavaDelta 模型构造 batch 数据。

    LLaVA 使用 <image> token 作为占位符,
    我们在其后添加 <motion_pad> 占位符。
    """

    def __init__(self, processor, tokenizer, max_length: int = 2048, alignment_mode: bool = False):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.alignment_mode = alignment_mode  # Stage 1 Alignment: 不走 LLM, 不限帧数

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_diff_frames = []
        all_motion_vectors = []
        all_residuals = []
        all_iframe_frames = []
        all_pixel_values = []
        all_image_sizes = []
        all_task_types = []

        for sample in batch:
            keyframes = sample["keyframes"]
            diff_frames, motion_vectors, residuals, iframe_frames = collect_motion_streams(sample)
            conversations = sample["conversations"]
            num_diffs_per_kf = sample.get("num_diffs_per_keyframe", 4)
            task_type = sample.get("task_type", "open_qa")

            # LLaVA-OneVision: 每帧约 729 vision tokens (27x27 patches)
            # 限制帧数以适配 max_length
            # alignment_mode 下不走 LLM, 不需要受 token 预算限制
            if keyframes and not self.alignment_mode:
                tokens_per_frame = 729
                motion_tokens = num_diffs_per_kf * max(0, len(keyframes) - 1)
                text_budget = 512 + motion_tokens
                vision_budget = self.max_length - text_budget
                max_frames = max(2, vision_budget // tokens_per_frame)
                if len(keyframes) > max_frames:
                    step = len(keyframes) / max_frames
                    indices = [int(i * step) for i in range(max_frames)]
                    keyframes = [keyframes[i] for i in indices]
                    diff_frames = slice_motion_stream(diff_frames, indices, num_diffs_per_kf)
                    motion_vectors = slice_motion_stream(motion_vectors, indices, num_diffs_per_kf)
                    residuals = slice_motion_stream(residuals, indices, num_diffs_per_kf)
                    iframe_frames = slice_motion_stream(iframe_frames, indices, num_diffs_per_kf)

            num_diffs = get_motion_token_count(diff_frames, motion_vectors, residuals, iframe_frames)

            motion_placeholder = (
                "<|motion_start|>"
                + "<|motion_pad|>" * num_diffs
                + "<|motion_end|>"
            )

            question = ""
            answer = ""
            for conv in conversations:
                if conv["from"] == "human":
                    question = conv["value"].replace("<video>", "").replace("<image>", "").strip()
                elif conv["from"] == "gpt":
                    answer = conv["value"]

            if keyframes:
                # LLaVA 把视频帧当作多张图片处理
                prompt = f"<image>\n{motion_placeholder}\n{question}"
                conversation = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{motion_placeholder}\n{question}"},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": answer},
                    ]},
                ]

                text = self.processor.apply_chat_template(
                    conversation, tokenize=False
                )
                # 注意: 不在 processor 中做 truncation, 否则 image token 数量
                # 与 text 中 <image> 标记不一致, 新版 transformers 会报 ValueError
                inputs = self.processor(
                    text=text,
                    images=keyframes,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )

                # 手动从尾部截断 (保留前 max_length 个 token, 截断的是 assistant 回答尾部)
                input_ids_raw = inputs["input_ids"].squeeze(0)
                attn_mask_raw = inputs["attention_mask"].squeeze(0)
                if input_ids_raw.shape[0] > self.max_length:
                    input_ids_raw = input_ids_raw[:self.max_length]
                    attn_mask_raw = attn_mask_raw[:self.max_length]

                all_input_ids.append(input_ids_raw)
                all_attention_mask.append(attn_mask_raw)
                if "pixel_values" in inputs:
                    all_pixel_values.append(inputs["pixel_values"])
                if "image_sizes" in inputs:
                    all_image_sizes.append(inputs["image_sizes"])

                # Labels: 只计算 assistant 回复
                input_ids = input_ids_raw
                labels = torch.full_like(input_ids, -100)
                # LLaVA 格式中找 assistant 回复起始
                # 修复: 使用 chat template 中的 assistant 标记来定位,
                # 而非搜索 answer 文本 (短答案如 "A"/"B" 会在 question 中误匹配)
                # LLaVA-OneVision 的 chat template 通常包含 "assistant\n" 或类似标记
                ids_list = input_ids.tolist()
                found_assistant = False

                # 方法 1: 用 chat template 生成 assistant header token 序列
                try:
                    assistant_header = self.processor.apply_chat_template(
                        [{"role": "assistant", "content": [{"type": "text", "text": ""}]}],
                        tokenize=False,
                    )
                    # 提取 assistant header (去掉空 content 后的部分)
                    # 通常格式类似 "assistant\n" 或 "<|im_start|>assistant\n"
                    for header_candidate in ["assistant\n", "ASSISTANT:"]:
                        if header_candidate in assistant_header:
                            assistant_token_ids = self.tokenizer.encode(
                                header_candidate, add_special_tokens=False
                            )
                            if len(assistant_token_ids) > 0:
                                for i in range(len(ids_list) - len(assistant_token_ids)):
                                    if ids_list[i:i + len(assistant_token_ids)] == assistant_token_ids:
                                        start = i + len(assistant_token_ids)
                                        labels[start:] = input_ids[start:]
                                        found_assistant = True
                                        break
                            if found_assistant:
                                break
                except Exception:
                    pass

                # 方法 2: 回退 — 从后往前搜索 answer 文本 (比原来从前往后搜更安全)
                if not found_assistant:
                    answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
                    if len(answer_ids) > 0:
                        match_len = min(5, len(answer_ids))
                        # 从后往前搜索, 避免在 question 中误匹配
                        for i in range(len(ids_list) - match_len, -1, -1):
                            if ids_list[i:i + match_len] == answer_ids[:match_len]:
                                labels[i:] = input_ids[i:]
                                found_assistant = True
                                break

                if not found_assistant:
                    logger.warning("LLaVA collator: 无法定位 assistant 回复位置, labels 全为 -100")

                all_labels.append(labels)

                if diff_frames is not None:
                    all_diff_frames.append(diff_frames)
                if motion_vectors is not None:
                    all_motion_vectors.append(motion_vectors)
                if residuals is not None:
                    all_residuals.append(residuals)
                if iframe_frames is not None:
                    all_iframe_frames.append(iframe_frames)
                all_task_types.append(task_type)
            else:
                logger.warning("跳过无关键帧样本 (LLaVA collator)")

        # 所有样本都失败时, 返回 dummy batch 避免 crash
        if not all_input_ids:
            logger.warning("Batch 内所有样本视频加载失败, 返回 dummy batch")
            dummy_ids = torch.zeros(1, 1, dtype=torch.long)
            return {
                "input_ids": dummy_ids,
                "labels": torch.full((1, 1), -100, dtype=torch.long),
                "attention_mask": torch.zeros(1, 1, dtype=torch.long),
                "diff_frames": torch.zeros(1, 1, 3, 224, 224),
                "motion_vectors": torch.zeros(1, 1, 3, 224, 224),
                "residuals": torch.zeros(1, 1, 3, 224, 224),
                "iframe_frames": torch.zeros(1, 1, 3, 224, 224),
                "is_dummy": True,
            }

        # Padding
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, lbl, mask in zip(all_input_ids, all_labels, all_attention_mask):
            pad_len = max_len - ids.shape[0]
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)])
            )
            padded_labels.append(
                torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype)])
            )
            padded_attention_mask.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            )

        result = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attention_mask),
            "diff_frames": torch.stack(all_diff_frames) if all_diff_frames else None,
            "motion_vectors": torch.stack(all_motion_vectors) if all_motion_vectors else None,
            "residuals": torch.stack(all_residuals) if all_residuals else None,
            "iframe_frames": torch.stack(all_iframe_frames) if all_iframe_frames else None,
            "task_types": all_task_types,
        }

        if all_pixel_values:
            result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_sizes:
            result["image_sizes"] = torch.cat(all_image_sizes, dim=0)

        return result


# ============================================================================
# Collator: 为 InternVL3 构造 batch
# ============================================================================

class InternVLDeltaCollator:
    """
    为 InternVLDelta 模型构造 batch 数据。

    InternVL3 特点:
    - 没有统一的 processor, 使用 AutoTokenizer
    - 图像通过 dynamic_preprocess 切分为 448x448 网格
    - 使用 <IMG_CONTEXT> 作为图像 token 占位符
    - 视频帧作为多张图片输入, 每张图片前后有 <img>...</img> 标记

    文本模板:
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <img><IMG_CONTEXT>...<IMG_CONTEXT></img>
    <|motion_start|><|motion_pad|>...<|motion_end|>
    {question}<|im_end|>
    <|im_start|>assistant
    {answer}<|im_end|>
    """

    # InternVL3 每张图片的视觉 token 数 (448x448 / 14 / 14 = 1024, Pixel Shuffle 4x -> 256)
    NUM_IMG_TOKENS_PER_TILE = 256
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"

    def __init__(self, tokenizer, max_length: int = 2048, alignment_mode: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.alignment_mode = alignment_mode  # Stage 1 Alignment: 不走 LLM, 不限帧数

    def _build_internvl_text(
        self,
        num_keyframes: int,
        num_diffs: int,
        question: str,
        answer: str,
    ) -> str:
        """构造 InternVL3 格式的对话文本。"""
        # 每个关键帧: <img><IMG_CONTEXT>*256</img>
        img_placeholder = (
            self.IMG_START_TOKEN
            + self.IMG_CONTEXT_TOKEN * self.NUM_IMG_TOKENS_PER_TILE
            + self.IMG_END_TOKEN
        )
        visual_text = "\n".join([img_placeholder] * num_keyframes)

        motion_placeholder = (
            "<|motion_start|>"
            + "<|motion_pad|>" * num_diffs
            + "<|motion_end|>"
        )

        text = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{visual_text}\n{motion_placeholder}\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>"
        )
        return text

    def _preprocess_keyframes(self, keyframes: List[Image.Image]) -> torch.Tensor:
        """将关键帧处理为 InternVL3 的 pixel_values。

        简化版: 每张图片 resize 到 448x448 作为单个 tile。
        InternVL3 原始实现支持 dynamic_preprocess (多 tile),
        但为了与其他模型对齐, 这里简化为单 tile。
        """
        from torchvision import transforms as T

        transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if not keyframes:
            return torch.zeros(1, 3, 448, 448)

        tensors = [transform(img) for img in keyframes]
        return torch.stack(tensors)  # [num_keyframes, 3, 448, 448]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_diff_frames = []
        all_motion_vectors = []
        all_residuals = []
        all_iframe_frames = []
        all_pixel_values = []
        all_task_types = []

        for sample in batch:
            keyframes = sample["keyframes"]
            diff_frames, motion_vectors, residuals, iframe_frames = collect_motion_streams(sample)
            conversations = sample["conversations"]
            num_diffs = get_motion_token_count(diff_frames, motion_vectors, residuals, iframe_frames)
            num_keyframes = len(keyframes)
            task_type = sample.get("task_type", "open_qa")

            # InternVL3: 每帧 256 tokens + <img></img> = ~258 tokens
            # 限制帧数以适配 max_length
            # alignment_mode 下不走 LLM, 不需要受 token 预算限制
            num_diffs_per_kf = sample.get("num_diffs_per_keyframe", 4)
            if num_keyframes > 0 and not self.alignment_mode:
                tokens_per_frame = self.NUM_IMG_TOKENS_PER_TILE + 2  # <img> + </img>
                motion_tokens = num_diffs_per_kf * max(0, num_keyframes - 1)
                text_budget = 512 + motion_tokens
                vision_budget = self.max_length - text_budget
                max_frames = max(2, vision_budget // max(1, tokens_per_frame))
                if num_keyframes > max_frames:
                    step = num_keyframes / max_frames
                    indices = [int(i * step) for i in range(max_frames)]
                    keyframes = [keyframes[i] for i in indices]
                    diff_frames = slice_motion_stream(diff_frames, indices, num_diffs_per_kf)
                    motion_vectors = slice_motion_stream(motion_vectors, indices, num_diffs_per_kf)
                    residuals = slice_motion_stream(residuals, indices, num_diffs_per_kf)
                    iframe_frames = slice_motion_stream(iframe_frames, indices, num_diffs_per_kf)
                    num_keyframes = len(keyframes)
                    num_diffs = get_motion_token_count(diff_frames, motion_vectors, residuals, iframe_frames)

            question = ""
            answer = ""
            for conv in conversations:
                if conv["from"] == "human":
                    question = conv["value"].replace("<video>", "").replace("<image>", "").strip()
                elif conv["from"] == "gpt":
                    answer = conv["value"]

            # 构造文本
            text = self._build_internvl_text(
                num_keyframes=num_keyframes,
                num_diffs=num_diffs,
                question=question,
                answer=answer,
            )

            # Tokenize
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = tokens["input_ids"].squeeze(0)

            # Labels: 只计算 assistant 回复部分
            labels = torch.full_like(input_ids, -100)
            assistant_token = self.tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
            ids_list = input_ids.tolist()
            for i in range(len(ids_list) - len(assistant_token)):
                if ids_list[i:i + len(assistant_token)] == assistant_token:
                    labels[i + len(assistant_token):] = input_ids[i + len(assistant_token):]
                    break

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(tokens["attention_mask"].squeeze(0))
            if diff_frames is not None:
                all_diff_frames.append(diff_frames)
            if motion_vectors is not None:
                all_motion_vectors.append(motion_vectors)
            if residuals is not None:
                all_residuals.append(residuals)
            if iframe_frames is not None:
                all_iframe_frames.append(iframe_frames)
            all_task_types.append(task_type)

            # 处理关键帧为 pixel_values
            pixel_values = self._preprocess_keyframes(keyframes)
            all_pixel_values.append(pixel_values)

        # Padding
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, lbl, mask in zip(all_input_ids, all_labels, all_attention_mask):
            pad_len = max_len - ids.shape[0]
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)])
            )
            padded_labels.append(
                torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype)])
            )
            padded_attention_mask.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            )

        result = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attention_mask),
            "diff_frames": torch.stack(all_diff_frames) if all_diff_frames else None,
            "motion_vectors": torch.stack(all_motion_vectors) if all_motion_vectors else None,
            "residuals": torch.stack(all_residuals) if all_residuals else None,
            "iframe_frames": torch.stack(all_iframe_frames) if all_iframe_frames else None,
            "task_types": all_task_types,
        }

        if all_pixel_values:
            result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            # InternVL3 需要 image_flags 来标记哪些是真实图片
            result["image_flags"] = torch.ones(
                result["pixel_values"].shape[0], dtype=torch.long
            )

        return result
