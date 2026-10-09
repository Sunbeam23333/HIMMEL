#!/usr/bin/env python3
"""
Himmel-VLM: 真实压缩域视频读取工具
===================================

目标:
1. 直接从压缩视频中读取 I-frame / motion vector / residual；
2. 与现有 Delta 数据管线兼容，产出仍然是 image-like numpy 数组；
3. 默认遵循 CoViAR 风格的 GOP 假设与 accumulate 接口；
4. 当压缩域 backend 不可用时，显式报错，而不是静默退回 RGB diff。
"""
from __future__ import annotations

import functools
import importlib
import logging
import os
import site
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

REPRESENTATION_TO_INDEX = {
    "iframe": 0,
    "mv": 1,
    "residual": 2,
}


def clip_and_scale(arr: np.ndarray, clip_value: float) -> np.ndarray:
    """把有符号压缩域信号压到近似 [-127, 127]。"""
    if clip_value <= 0:
        return arr.astype(np.float32)
    scaled = np.clip(arr.astype(np.float32), -clip_value, clip_value)
    return scaled * (127.0 / float(clip_value))



def _sample_evenly(indices: List[int], target_count: int) -> List[int]:
    if target_count <= 0 or not indices:
        return []
    if len(indices) <= target_count:
        return list(indices)
    if target_count == 1:
        return [indices[len(indices) // 2]]

    step = (len(indices) - 1) / max(1, target_count - 1)
    sampled = [indices[int(round(i * step))] for i in range(target_count)]

    deduped: List[int] = []
    seen = set()
    for idx in sampled:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)

    if len(deduped) < target_count:
        for idx in indices:
            if idx not in seen:
                deduped.append(idx)
                seen.add(idx)
            if len(deduped) >= target_count:
                break

    return sorted(deduped[:target_count])



@functools.lru_cache(maxsize=8192)
def _get_video_stats(video_path: str) -> Tuple[int, float]:
    try:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path, num_threads=1)
        return len(vr), float(vr.get_avg_fps())
    except Exception as exc:
        logger.warning("decord 读取视频元信息失败 %s: %s，尝试 PyAV", video_path, exc)

    import av

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        total_frames = int(stream.frames or 0)
        if total_frames <= 0:
            total_frames = sum(1 for _ in container.decode(video=0))
    return total_frames, fps



def _get_video_codec_name(video_path: str) -> str:
    import av

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        codec_context = getattr(stream, "codec_context", None)
        codec_name = str(getattr(codec_context, "name", "") or "")
        if not codec_name:
            codec = getattr(stream, "codec", None)
            codec_name = str(getattr(codec, "name", "") or "")
    return codec_name.lower()



@functools.lru_cache(maxsize=4096)
def _probe_coviar_video_profile(video_path: str, expected_gop_size: int) -> Tuple[str, int, int]:
    """用 PyAV 做一次保守探测，尽量在进入 coviar 前识别明显不兼容的视频。"""
    import av

    keyframe_indices: List[int] = []
    decoded_frames = 0
    max_probe_frames = max(48, int(expected_gop_size) * 4)

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        codec_context = getattr(stream, "codec_context", None)
        codec_name = str(getattr(codec_context, "name", "") or "")
        if not codec_name:
            codec = getattr(stream, "codec", None)
            codec_name = str(getattr(codec, "name", "") or "")
        codec_name = codec_name.lower()

        for frame in container.decode(video=0):
            if getattr(frame, "key_frame", False):
                keyframe_indices.append(decoded_frames)
            decoded_frames += 1
            if len(keyframe_indices) >= 3 or decoded_frames >= max_probe_frames:
                break

    if decoded_frames <= 0:
        raise ValueError(f"视频无法解码出任何帧: {video_path}")

    inferred_gop_size = 0
    if len(keyframe_indices) >= 2:
        intervals = [
            next_idx - current_idx
            for current_idx, next_idx in zip(keyframe_indices, keyframe_indices[1:])
            if next_idx > current_idx
        ]
        if intervals:
            inferred_gop_size = int(round(sum(intervals) / len(intervals)))

    return codec_name, inferred_gop_size, decoded_frames



def _validate_coviar_video_compatibility(video_path: str, expected_gop_size: int) -> None:
    """CoViAR 当前只支持预先转码成固定 GOP 的 MPEG-4 视频。"""
    codec_name, inferred_gop_size, decoded_frames = _probe_coviar_video_profile(
        video_path,
        expected_gop_size,
    )
    if codec_name and codec_name != "mpeg4":
        raise ValueError(
            f"当前视频不满足 coviar / pytorch-coviar 的输入前提：video codec={codec_name}。"
            "官方 data loader 目前只支持预先转码成 MPEG-4 的 raw videos；像 LLaVA-Video-178K / YouTube 原始 mp4 这类混合编码视频不能直接拿来跑 motion_input_mode=compressed。"
        )

    if inferred_gop_size > 0 and abs(inferred_gop_size - int(expected_gop_size)) > 1:
        raise ValueError(
            f"当前视频不满足 coviar / pytorch-coviar 的固定 GOP 假设：inferred_gop_size≈{inferred_gop_size}, expected_gop_size={expected_gop_size}。"
            "官方 data loader 依赖与论文一致的固定 GOP 结构（通常为 12）；直接拿原始训练视频进入 coviar 会导致 frame→gop 映射错误，严重时触发 worker segfault。"
        )

    if inferred_gop_size <= 0 and decoded_frames >= max(24, int(expected_gop_size) * 2):
        raise ValueError(
            f"当前视频在前期探测中未呈现出稳定的固定 GOP 结构，decoded_frames={decoded_frames}, expected_gop_size={expected_gop_size}。"
            "这通常说明它不是按 coviar 假设预先转码过的 MPEG-4 raw video。"
        )



def _get_coviar_load_fn():
    try:
        coviar = importlib.import_module("coviar")
    except ImportError as exc:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str) and user_site and os.path.isdir(user_site) and user_site not in sys.path:
            sys.path.insert(0, user_site)
            try:
                coviar = importlib.import_module("coviar")
            except ImportError:
                coviar = None
        else:
            coviar = None

        if coviar is None:
            error_message = (
                f"真实压缩域读取需要安装 coviar，或当前进程无法加载其依赖动态库。原始错误: {exc}。"
                "如果你已经成功安装过 coviar，请优先检查训练/评测启动前是否导出了 "
                "LD_LIBRARY_PATH=$FFMPEG_ROOT/lib，并确认用户 site-packages 已加入 PYTHONPATH。"
            )
            raise ImportError(error_message) from exc

    load_fn = getattr(coviar, "load", None)
    if load_fn is None:
        raise AttributeError("coviar 模块存在，但未找到 load() 接口")
    return load_fn



def validate_compressed_backend(compressed_backend: str = "coviar") -> None:
    """启动前检查 compressed-domain backend 是否可用，避免训练中途退化为空样本。"""
    if compressed_backend != "coviar":
        raise ValueError(f"暂不支持的 compressed_backend: {compressed_backend}")
    _get_coviar_load_fn()



def _frame_to_gop_position(frame_idx: int, representation: str, gop_size: int) -> Tuple[int, int]:
    if frame_idx < 0:
        frame_idx = 0

    gop_index = frame_idx // gop_size
    gop_pos = frame_idx % gop_size

    if representation in {"mv", "residual"}:
        if gop_pos == 0:
            gop_index = max(0, gop_index - 1)
            gop_pos = gop_size - 1
    else:
        gop_pos = 0

    return gop_index, gop_pos



def _to_hwc_float(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim == 3 and x.shape[0] in {1, 2, 3} and x.shape[-1] not in {1, 2, 3}:
        x = np.moveaxis(x, 0, -1)
    return x.astype(np.float32)



def _normalize_iframe(arr: np.ndarray) -> np.ndarray:
    x = _to_hwc_float(arr)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    if x.shape[-1] > 3:
        x = x[..., :3]
    # CoViAR / OpenCV 风格 I-frame 往往是 BGR，这里统一转为 RGB
    if x.shape[-1] == 3:
        x = x[..., ::-1]
    return np.clip(x, 0, 255).astype(np.uint8)



def encode_motion_vector_map(arr: np.ndarray, clip_value: float = 20.0) -> np.ndarray:
    """把真实 motion vector 编码成 3 通道图：x / y / magnitude。"""
    x = _to_hwc_float(arr)
    if x.shape[-1] == 1:
        x = np.repeat(x, 2, axis=-1)
    if x.shape[-1] > 2:
        x = x[..., :2]

    mv_x = clip_and_scale(x[..., 0], clip_value) + 128.0
    mv_y = clip_and_scale(x[..., 1], clip_value) + 128.0

    max_mag = max(1e-6, float(clip_value) * np.sqrt(2.0))
    mv_mag = np.linalg.norm(x[..., :2], axis=-1)
    mv_mag = np.clip(mv_mag, 0.0, max_mag)
    mv_mag = mv_mag * (127.0 / max_mag) + 128.0

    out = np.stack([mv_x, mv_y, mv_mag], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)



def encode_residual_map(arr: np.ndarray) -> np.ndarray:
    """把真实 residual 编码成 3 通道有符号残差图。"""
    x = _to_hwc_float(arr)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    if x.shape[-1] > 3:
        x = x[..., :3]

    if np.issubdtype(np.asarray(arr).dtype, np.uint8) and x.min() >= 0 and x.max() <= 255:
        return x.astype(np.uint8)

    return np.clip(x + 128.0, 0, 255).astype(np.uint8)



def make_empty_motion_vector_map(size: Tuple[int, int]) -> np.ndarray:
    return np.full((*size, 3), 128, dtype=np.uint8)



def make_empty_residual_map(size: Tuple[int, int]) -> np.ndarray:
    return np.full((*size, 3), 128, dtype=np.uint8)



def _load_coviar_representation(
    load_fn,
    video_path: str,
    frame_idx: int,
    representation: str,
    *,
    gop_size: int,
    accumulate: bool,
) -> np.ndarray:
    rep_idx = REPRESENTATION_TO_INDEX[representation]
    gop_index, gop_pos = _frame_to_gop_position(frame_idx, representation, gop_size)
    return load_fn(video_path, gop_index, gop_pos, rep_idx, accumulate)



def _extract_uniform_rgb_keyframes(
    video_path: str,
    *,
    total_frames: int,
    video_fps: float,
    keyframe_fps: float,
    max_keyframes: int,
) -> Tuple[List[np.ndarray], List[int]]:
    import decord

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path, num_threads=1)
    keyframe_interval = max(1, int(video_fps / max(keyframe_fps, 1e-6)))
    keyframe_indices = list(range(0, total_frames, keyframe_interval))
    if len(keyframe_indices) > max_keyframes:
        keyframe_indices = _sample_evenly(keyframe_indices, max_keyframes)
    if len(keyframe_indices) < 2 and total_frames > 1:
        keyframe_indices = sorted(set(keyframe_indices + [0, total_frames - 1]))
    if not keyframe_indices:
        keyframe_indices = [0]
    frames = vr.get_batch(keyframe_indices).asnumpy()
    return [frames[i] for i in range(len(keyframe_indices))], keyframe_indices



def decode_video_frames(video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
    """按给定帧索引解码 RGB 帧，保持输入顺序并支持重复索引。"""
    if not frame_indices:
        return []

    import decord

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path, num_threads=1)
    total_frames = len(vr)
    clipped_indices = [min(max(int(idx), 0), max(total_frames - 1, 0)) for idx in frame_indices]
    unique_indices = sorted(set(clipped_indices))
    decoded = vr.get_batch(unique_indices).asnumpy()
    frame_map = {idx: decoded[pos] for pos, idx in enumerate(unique_indices)}
    return [frame_map[idx] for idx in clipped_indices]



def extract_compressed_video_streams(
    video_path: str,
    *,
    keyframe_fps: float = 1.0,
    num_motion_units_per_interval: int = 4,
    max_keyframes: int = 32,
    visual_stream_mode: str = "iframe",
    compressed_backend: str = "coviar",
    compressed_gop_size: int = 12,
    use_motion_vectors: bool = True,
    use_residuals: bool = True,
    accumulate_motion_vectors: bool = True,
    accumulate_residuals: bool = False,
    motion_vector_clip: float = 20.0,
    diff_size: Tuple[int, int] = (224, 224),
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], Dict[str, Any]]:
    """提取真实压缩域三流：visual(I-frame/可选 keyframe) + MV + residual。"""
    if compressed_backend != "coviar":
        raise ValueError(f"暂不支持的 compressed_backend: {compressed_backend}")
    if not use_motion_vectors and not use_residuals:
        raise ValueError("compressed 模式至少需要启用 motion_vectors 或 residuals 之一")

    _validate_coviar_video_compatibility(video_path, compressed_gop_size)

    total_frames, video_fps = _get_video_stats(video_path)
    if total_frames <= 0 or video_fps <= 0:
        logger.warning("无法读取视频: %s (frames=%d, fps=%.3f)", video_path, total_frames, video_fps)
        return [], [], [], [], {
            "video_fps": video_fps,
            "total_frames": total_frames,
            "visual_stream_mode": visual_stream_mode,
            "motion_input_mode": "compressed",
        }

    load_fn = _get_coviar_load_fn()

    iframe_indices = list(range(0, total_frames, max(1, int(compressed_gop_size))))
    if not iframe_indices:
        iframe_indices = [0]

    if visual_stream_mode == "iframe":
        visual_indices = _sample_evenly(iframe_indices, min(max_keyframes, len(iframe_indices)))
        keyframes = [
            _normalize_iframe(
                _load_coviar_representation(
                    load_fn,
                    video_path,
                    idx,
                    "iframe",
                    gop_size=compressed_gop_size,
                    accumulate=False,
                )
            )
            for idx in visual_indices
        ]
    elif visual_stream_mode == "keyframe":
        keyframes, visual_indices = _extract_uniform_rgb_keyframes(
            video_path,
            total_frames=total_frames,
            video_fps=video_fps,
            keyframe_fps=keyframe_fps,
            max_keyframes=max_keyframes,
        )
    else:
        raise ValueError(f"不支持的 visual_stream_mode: {visual_stream_mode}")

    mv_maps: List[np.ndarray] = []
    residual_maps: List[np.ndarray] = []
    interval_frame_indices: List[List[int]] = []

    for start_idx, end_idx in zip(visual_indices[:-1], visual_indices[1:]):
        if num_motion_units_per_interval <= 0:
            continue

        if end_idx - start_idx <= 1:
            fallback_idx = min(max(start_idx + 1, 0), max(total_frames - 1, 0))
            interval_indices = np.full((num_motion_units_per_interval,), fallback_idx, dtype=int)
            for _ in range(num_motion_units_per_interval):
                if use_motion_vectors:
                    mv_maps.append(make_empty_motion_vector_map(diff_size))
                if use_residuals:
                    residual_maps.append(make_empty_residual_map(diff_size))
        else:
            interval_indices = np.linspace(
                start_idx + 1,
                max(start_idx + 1, end_idx - 1),
                num_motion_units_per_interval,
                dtype=int,
            )
            interval_indices = np.clip(interval_indices, start_idx + 1, max(start_idx + 1, end_idx - 1))

            for frame_idx in interval_indices:
                if use_motion_vectors:
                    mv = _load_coviar_representation(
                        load_fn,
                        video_path,
                        int(frame_idx),
                        "mv",
                        gop_size=compressed_gop_size,
                        accumulate=accumulate_motion_vectors,
                    )
                    mv_maps.append(encode_motion_vector_map(mv, clip_value=motion_vector_clip))

                if use_residuals:
                    residual = _load_coviar_representation(
                        load_fn,
                        video_path,
                        int(frame_idx),
                        "residual",
                        gop_size=compressed_gop_size,
                        accumulate=accumulate_residuals,
                    )
                    residual_maps.append(encode_residual_map(residual))

        interval_frame_indices.append([int(start_idx)] + [int(idx) for idx in interval_indices.tolist()])

    meta = {
        "video_fps": video_fps,
        "total_frames": total_frames,
        "visual_stream_mode": visual_stream_mode,
        "motion_input_mode": "compressed",
        "compressed_backend": compressed_backend,
        "compressed_gop_size": compressed_gop_size,
        "num_iframes_available": len(iframe_indices),
        "num_visual_frames": len(keyframes),
        "num_motion_vectors": len(mv_maps),
        "num_residuals": len(residual_maps),
        "interval_frame_indices": interval_frame_indices,
        "accumulate_motion_vectors": accumulate_motion_vectors,
        "accumulate_residuals": accumulate_residuals,
    }
    return keyframes, mv_maps, residual_maps, visual_indices, meta
