#!/usr/bin/env python3
"""
Himmel-VLM: training-free 视频采样与层次记忆工具
====================================================

目标:
1. 保持现有训练/模型权重不变，只在评测时改采样策略；
2. 将固定 uniform 采样升级为 "dopamine + hierarchical memory"；
3. 在相同或略高的上下文预算下，同时保留:
   - 长期记忆: 全局稀疏锚点，覆盖整段视频；
   - 工作记忆: 高显著事件段的局部加密帧；
   - 运动记忆: 对显著段保留更密集 diff，非显著段压缩 diff。

设计思想:
- Visual Token 压缩/采样: 用分层关键帧替代纯均匀采样；
- KV / 记忆稀疏化: 用 variable diff budget 模拟“长期稀疏、局部稠密”的记忆分配；
- Dopamine: 用运动显著性 + 时间问题启发式，提升高价值片段的采样密度。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_TEMPORAL_KEYWORDS = (
    "before", "after", "then", "next", "previous", "earlier", "later",
    "order", "sequence", "first", "second", "third", "finally",
    "initially", "subsequently", "following", "temporal", "chronological",
    "先", "后", "然后", "之前", "之后", "顺序", "最先", "最后",
)
_START_KEYWORDS = (
    "beginning", "start", "initial", "initially", "first", "early",
    "开头", "开始", "最初", "起初", "一开始", "前面",
)
_END_KEYWORDS = (
    "end", "ending", "finally", "last", "final", "later",
    "结尾", "最后", "最终", "后面", "末尾",
)
_COUNT_KEYWORDS = (
    "how many", "number of", "times", "count", "frequency", "twice", "three times",
    "多少次", "几次", "多少", "次数", "频率", "总共",
)
_REPETITION_KEYWORDS = (
    "repeat", "repeated", "throughout", "entire", "whole video", "always",
    "反复", "重复", "全程", "整个视频", "一直", "持续",
)


def _sample_evenly(indices: List[int], target_count: int) -> List[int]:
    """从有序索引中均匀采样固定数量，尽量保留首尾。"""
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


def _build_query_profile(question: str, options: Optional[List[str]] = None) -> Dict[str, bool]:
    text = (question or "")
    if options:
        text += "\n" + "\n".join(options)
    text = text.lower()

    def _has_any(keywords: Tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

    return {
        "temporal": _has_any(_TEMPORAL_KEYWORDS),
        "start": _has_any(_START_KEYWORDS),
        "end": _has_any(_END_KEYWORDS),
        "counting": _has_any(_COUNT_KEYWORDS),
        "repetition": _has_any(_REPETITION_KEYWORDS),
    }


def _compute_interval_motion_scores(vr, coarse_indices: List[int], probes_per_interval: int = 2) -> List[float]:
    """在 coarse interval 上估计运动显著性。"""
    if len(coarse_indices) < 2:
        return []

    total_frames = len(vr)
    scores: List[float] = []
    for start_idx, end_idx in zip(coarse_indices[:-1], coarse_indices[1:]):
        if end_idx - start_idx <= 1:
            scores.append(0.0)
            continue

        probe_indices = np.linspace(start_idx + 1, end_idx, probes_per_interval, dtype=int)
        probe_indices = np.clip(probe_indices, 1, total_frames - 1)
        interval_scores = []
        for idx in probe_indices:
            prev_idx = max(0, int(idx) - 1)
            pair = vr.get_batch([prev_idx, int(idx)]).asnumpy()
            diff = np.abs(pair[1].astype(np.int16) - pair[0].astype(np.int16)).astype(np.float32)
            diff = diff[::8, ::8]
            interval_scores.append(float(diff.mean()))

        scores.append(float(np.mean(interval_scores)) if interval_scores else 0.0)

    return scores


def _boost_interval_score(base_score: float, interval_idx: int, num_intervals: int,
                          query_profile: Dict[str, bool], question_aware_sampling: bool) -> float:
    if not question_aware_sampling:
        return base_score

    rel_pos = interval_idx / max(1, num_intervals - 1)
    boost = 1.0

    if query_profile.get("start") and rel_pos <= 0.25:
        boost += 0.35
    if query_profile.get("end") and rel_pos >= 0.75:
        boost += 0.35
    if query_profile.get("temporal"):
        boost += 0.15
    if query_profile.get("counting") or query_profile.get("repetition"):
        boost += 0.10

    return base_score * boost + 1e-4 * boost


def _build_hierarchical_indices(
    vr,
    total_frames: int,
    video_fps: float,
    keyframe_fps: float,
    max_keyframes: int,
    question: str,
    options: Optional[List[str]],
    memory_keyframes: int,
    dense_local_frames: int,
    question_aware_sampling: bool,
) -> Tuple[List[int], List[Tuple[int, int]], Dict[str, Any]]:
    """构造全局锚点 + 局部高显著段加密的关键帧索引。"""
    query_profile = _build_query_profile(question, options)

    keyframe_interval = max(1, int(video_fps / max(keyframe_fps, 1e-6)))
    coarse_indices = list(range(0, total_frames, keyframe_interval))
    if not coarse_indices:
        coarse_indices = [0]
    if coarse_indices[-1] != total_frames - 1:
        coarse_indices.append(total_frames - 1)
    if len(coarse_indices) < 2 and total_frames > 1:
        coarse_indices = [0, total_frames - 1]

    # 长期记忆预算: 计数/重复问题更强调全局覆盖
    default_memory = max(8, int(round(max_keyframes * 0.375)))
    memory_budget = memory_keyframes if memory_keyframes > 0 else default_memory
    if query_profile.get("counting") or query_profile.get("repetition"):
        memory_budget = max(memory_budget, int(round(max_keyframes * 0.50)))
    elif query_profile.get("temporal"):
        memory_budget = max(memory_budget, int(round(max_keyframes * 0.40)))
    memory_budget = max(2, min(memory_budget, len(coarse_indices), max_keyframes))

    long_term_indices = _sample_evenly(coarse_indices, memory_budget)
    selected = set(long_term_indices)
    selected.add(0)
    selected.add(total_frames - 1)

    remaining = max(0, max_keyframes - len(selected))
    dense_ranges: List[Tuple[int, int]] = []

    intervals = list(zip(coarse_indices[:-1], coarse_indices[1:]))
    motion_scores = _compute_interval_motion_scores(vr, coarse_indices)
    scored_intervals = []
    for i, ((start_idx, end_idx), score) in enumerate(zip(intervals, motion_scores)):
        boosted = _boost_interval_score(score, i, len(intervals), query_profile, question_aware_sampling)
        scored_intervals.append((boosted, score, start_idx, end_idx))
    scored_intervals.sort(key=lambda x: x[0], reverse=True)

    # 计数/重复类问题: 分散加密, 少量局部峰值；一般问题: 更密集地围绕显著段加帧
    local_dense_count = 1 if (query_profile.get("counting") or query_profile.get("repetition")) else max(1, dense_local_frames)
    max_dense_intervals = max(1, remaining // max(1, local_dense_count)) if remaining > 0 else 0
    if query_profile.get("counting") or query_profile.get("repetition"):
        max_dense_intervals = min(max_dense_intervals, 4)

    added_dense = 0
    used_dense_intervals = 0
    for _, raw_score, start_idx, end_idx in scored_intervals:
        if added_dense >= remaining or used_dense_intervals >= max_dense_intervals:
            break
        if end_idx - start_idx <= 1:
            continue

        local_budget = min(local_dense_count, remaining - added_dense)
        local_candidates = np.linspace(start_idx, end_idx, local_budget + 2, dtype=int)[1:-1]
        interval_added = 0
        for candidate in local_candidates:
            candidate = int(np.clip(candidate, 0, total_frames - 1))
            if candidate not in selected:
                selected.add(candidate)
                interval_added += 1
                added_dense += 1
                if added_dense >= remaining:
                    break

        if interval_added > 0 and raw_score > 0:
            dense_ranges.append((int(start_idx), int(end_idx)))
            used_dense_intervals += 1

    # 若还有预算，用 finer uniform candidates 补齐，提升整体采样密度
    if len(selected) < max_keyframes:
        fine_step = max(1, keyframe_interval // 2)
        fine_candidates = list(range(0, total_frames, fine_step))
        if fine_candidates[-1] != total_frames - 1:
            fine_candidates.append(total_frames - 1)
        unused_fine = [idx for idx in fine_candidates if idx not in selected]
        fill_needed = max_keyframes - len(selected)
        for idx in _sample_evenly(unused_fine, fill_needed):
            selected.add(idx)

    selected_indices = sorted(selected)
    if len(selected_indices) > max_keyframes:
        selected_indices = _sample_evenly(selected_indices, max_keyframes)
    if len(selected_indices) < 2 and total_frames > 1:
        selected_indices = [0, total_frames - 1]

    meta = {
        "strategy": "dopamine_hierarchical",
        "query_profile": query_profile,
        "long_term_keyframes": len(long_term_indices),
        "focus_keyframes": max(0, len(selected_indices) - len(long_term_indices)),
        "dense_intervals": len(dense_ranges),
        "coarse_candidates": len(coarse_indices),
    }
    return selected_indices, dense_ranges, meta


def _is_dense_interval(start_idx: int, end_idx: int, dense_ranges: List[Tuple[int, int]]) -> bool:
    if not dense_ranges:
        return False
    midpoint = 0.5 * (start_idx + end_idx)
    for dense_start, dense_end in dense_ranges:
        if dense_start <= midpoint <= dense_end:
            return True
    return False


def _extract_with_selected_indices(
    vr,
    keyframe_indices: List[int],
    dense_ranges: List[Tuple[int, int]],
    memory_diffs_per_interval: int,
    dense_diffs_per_interval: int,
    diff_size: Tuple[int, int] = (224, 224),
    diff_mode: str = "abs",
    *,
    encode_diff_map,
    make_empty_diff_map,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    """根据选定关键帧提取 RGB keyframes 和 variable-diff maps。"""
    keyframes_np = vr.get_batch(keyframe_indices).asnumpy()
    keyframes = [keyframes_np[i] for i in range(len(keyframe_indices))]

    diff_maps: List[np.ndarray] = []
    dense_diff_tokens = 0
    memory_diff_tokens = 0

    for interval_idx, (start_idx, end_idx) in enumerate(zip(keyframe_indices[:-1], keyframe_indices[1:])):
        dense_interval = _is_dense_interval(start_idx, end_idx, dense_ranges)
        num_diffs = dense_diffs_per_interval if dense_interval else memory_diffs_per_interval
        num_diffs = max(0, int(num_diffs))
        if num_diffs == 0:
            continue

        ref_frame = keyframes[interval_idx]

        if end_idx - start_idx <= 1:
            for _ in range(num_diffs):
                diff_maps.append(make_empty_diff_map(diff_size, diff_mode))
            if dense_interval:
                dense_diff_tokens += num_diffs
            else:
                memory_diff_tokens += num_diffs
            continue

        interval_indices = np.linspace(start_idx, end_idx, num_diffs + 2, dtype=int)[1:-1]
        interval_indices = np.clip(interval_indices, 1, len(vr) - 1)
        for idx in interval_indices:
            prev_idx = max(0, int(idx) - 1)
            pair = vr.get_batch([prev_idx, int(idx)]).asnumpy()
            diff = encode_diff_map(
                pair[0],
                pair[1],
                diff_mode=diff_mode,
                ref_frame=ref_frame,
            )
            diff_maps.append(diff)

        if dense_interval:
            dense_diff_tokens += len(interval_indices)
        else:
            memory_diff_tokens += len(interval_indices)

    meta = {
        "dense_diff_tokens": dense_diff_tokens,
        "memory_diff_tokens": memory_diff_tokens,
        "total_diff_tokens": len(diff_maps),
    }
    return keyframes, diff_maps, meta


def extract_video_observations(
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
    """统一视频观测提取入口，兼容 RGB diff 与真实压缩域双流。"""
    from data.delta_video_loader import (
        _is_incompatible_compressed_data_error,
        build_compressed_stream_transform,
        build_diff_transform,
        encode_diff_map,
        extract_keyframes_and_diffs,
        extract_proxy_compressed_streams,
        make_empty_diff_map,
    )
    from data.compressed_video_loader import (
        extract_compressed_video_streams,
        validate_compressed_backend,
    )

    options = options or []

    if motion_input_mode == "compressed":
        if sampling_strategy != "uniform":
            logger.warning(
                "compressed 模式暂不支持采样策略 %s，自动回退到 uniform",
                sampling_strategy,
            )
        try:
            validate_compressed_backend(compressed_backend)
            keyframes_np, mv_maps, residual_maps, kf_indices, meta = extract_compressed_video_streams(
                video_path=video_path,
                keyframe_fps=keyframe_fps,
                num_motion_units_per_interval=num_diffs_per_interval,
                max_keyframes=max_keyframes,
                visual_stream_mode=visual_stream_mode,
                compressed_backend=compressed_backend,
                compressed_gop_size=compressed_gop_size,
                use_motion_vectors=use_motion_vectors,
                use_residuals=use_residuals,
                accumulate_motion_vectors=accumulate_motion_vectors,
                accumulate_residuals=accumulate_residuals,
                motion_vector_clip=motion_vector_clip,
            )
        except Exception as exc:
            # Published compressed-domain runs must never silently become RGB-proxy runs.
            raise RuntimeError(
                f"Compressed extraction failed ({compressed_backend}); prepare compatible "
                "input or explicitly select another documented representation."
            ) from exc
        mv_transform = build_compressed_stream_transform("motion_vectors")
        residual_transform = build_compressed_stream_transform("residuals")
        keyframe_images = [Image.fromarray(kf) for kf in keyframes_np]
        motion_vectors = torch.stack([
            mv_transform(Image.fromarray(dm)) for dm in mv_maps
        ]) if mv_maps else None
        residuals = torch.stack([
            residual_transform(Image.fromarray(dm)) for dm in residual_maps
        ]) if residual_maps else None
        meta.update({
            "strategy": "uniform",
            "query_profile": _build_query_profile(question, options),
            "long_term_keyframes": len(kf_indices),
            "focus_keyframes": 0,
            "dense_intervals": 0,
            "dense_diff_tokens": 0,
            "memory_diff_tokens": len(mv_maps) if mv_maps else len(residual_maps),
            "total_diff_tokens": max(len(mv_maps), len(residual_maps)),
            "diff_mode": diff_mode,
            "motion_input_mode": motion_input_mode,
            "compressed_backend": meta.get("compressed_backend", compressed_backend),
            "visual_stream_mode": visual_stream_mode,
        })
        return keyframe_images, None, motion_vectors, residuals, kf_indices, meta

    diff_transform = build_diff_transform(diff_mode)

    if sampling_strategy == "uniform":
        keyframes_np, diff_maps, kf_indices = extract_keyframes_and_diffs(
            video_path=video_path,
            keyframe_fps=keyframe_fps,
            num_diffs_per_interval=num_diffs_per_interval,
            max_keyframes=max_keyframes,
            diff_mode=diff_mode,
        )
        keyframe_images = [Image.fromarray(kf) for kf in keyframes_np]
        diff_tensors = torch.stack([
            diff_transform(Image.fromarray(dm)) for dm in diff_maps
        ]) if diff_maps else None
        meta = {
            "strategy": "uniform",
            "query_profile": _build_query_profile(question, options),
            "long_term_keyframes": len(kf_indices),
            "focus_keyframes": 0,
            "dense_intervals": 0,
            "dense_diff_tokens": 0,
            "memory_diff_tokens": len(diff_maps),
            "total_diff_tokens": len(diff_maps),
            "diff_mode": diff_mode,
            "motion_input_mode": motion_input_mode,
        }
        return keyframe_images, diff_tensors, None, None, kf_indices, meta

    try:
        import decord
        decord.bridge.set_bridge("native")
    except ImportError as exc:
        raise ImportError("请安装 decord: pip install decord") from exc

    vr = decord.VideoReader(video_path, num_threads=2)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    if total_frames == 0 or video_fps <= 0:
        logger.warning("无法读取视频: %s (frames=%d, fps=%.1f)", video_path, total_frames, video_fps)
        return [], None, None, None, [], {
            "strategy": sampling_strategy,
            "error": "invalid_video",
            "diff_mode": diff_mode,
            "motion_input_mode": motion_input_mode,
        }

    keyframe_indices, dense_ranges, meta = _build_hierarchical_indices(
        vr=vr,
        total_frames=total_frames,
        video_fps=video_fps,
        keyframe_fps=keyframe_fps,
        max_keyframes=max_keyframes,
        question=question,
        options=options,
        memory_keyframes=memory_keyframes,
        dense_local_frames=dense_local_frames,
        question_aware_sampling=question_aware_sampling,
    )

    keyframes_np, diff_maps, diff_meta = _extract_with_selected_indices(
        vr=vr,
        keyframe_indices=keyframe_indices,
        dense_ranges=dense_ranges,
        memory_diffs_per_interval=memory_diffs_per_interval,
        dense_diffs_per_interval=dense_diffs_per_interval,
        diff_mode=diff_mode,
        encode_diff_map=encode_diff_map,
        make_empty_diff_map=make_empty_diff_map,
    )

    keyframe_images = [Image.fromarray(kf) for kf in keyframes_np]
    diff_tensors = torch.stack([
        diff_transform(Image.fromarray(dm)) for dm in diff_maps
    ]) if diff_maps else None

    meta.update(diff_meta)
    meta.update({
        "memory_keyframes_config": memory_keyframes,
        "dense_local_frames_config": dense_local_frames,
        "memory_diffs_per_interval": memory_diffs_per_interval,
        "dense_diffs_per_interval": dense_diffs_per_interval,
        "keyframe_fps": keyframe_fps,
        "diff_mode": diff_mode,
        "motion_input_mode": motion_input_mode,
    })
    return keyframe_images, diff_tensors, None, None, keyframe_indices, meta


def build_sampling_prompt_prefix(sampling_meta: Optional[Dict[str, Any]]) -> str:
    """给模型一个很短的时序记忆提示，不引入额外训练。"""
    if not sampling_meta or sampling_meta.get("strategy") in (None, "uniform"):
        return ""

    query_profile = sampling_meta.get("query_profile", {})
    hints = [
        "The sampled frames are ordered chronologically from early to late.",
        "They use hierarchical memory: sparse global anchors preserve full-video context, while denser local frames focus on salient motion changes.",
        "Use the whole video timeline instead of relying only on the most recent or most salient moment.",
    ]

    if query_profile.get("temporal"):
        hints.append("Pay attention to temporal order across the beginning, middle, and end of the video.")
    if query_profile.get("counting") or query_profile.get("repetition"):
        hints.append("Track repeated events across the full video rather than a single local segment.")
    if query_profile.get("start"):
        hints.append("Check the beginning carefully.")
    if query_profile.get("end"):
        hints.append("Check the ending carefully.")

    return " ".join(hints) + "\n\n"
