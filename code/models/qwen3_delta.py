#!/usr/bin/env python3
"""Archived Qwen3-VL motion-adapter snapshot.

Requires a Transformers build exposing Qwen3VLForConditionalGeneration.
See code/requirements.txt and docs/reproduction.md; pin and validate the exact
model/library revisions before use. Model-load and GPU compatibility were not
tested during release packaging. The complete appearance-context branch is
available in the Qwen2.5 implementation, not this archived wrapper.
"""
from __future__ import annotations

import logging
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn

from .delta_components import (
    MotionEncoderWithProjector,
    freeze_module,
    unfreeze_module,
    get_trainable_params_info,
    pool_motion_tokens,
)
from .compressed_motion_adapter import CompressedMotionAdapter

logger = logging.getLogger(__name__)

# 特殊 Token (与 Qwen2.5-VL 版本共享同一套)
MOTION_START_TOKEN = "<|motion_start|>"
MOTION_END_TOKEN = "<|motion_end|>"
MOTION_PAD_TOKEN = "<|motion_pad|>"


# ============================================================================
# 延迟导入 Qwen3VL 基类
# ============================================================================
_Qwen3VLBase = None
_Qwen3VLConfig = None
_RealDeltaClass = None


def _ensure_base():
    global _Qwen3VLBase, _Qwen3VLConfig
    if _Qwen3VLBase is not None:
        return
    try:
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
        _Qwen3VLBase = Qwen3VLForConditionalGeneration
        _Qwen3VLConfig = Qwen3VLConfig
    except ImportError:
        logger.warning(
            "transformers 中未找到 Qwen3VLForConditionalGeneration, "
            "请使用提供该模型类的兼容版本，参见 code/requirements.txt 和 docs/reproduction.md"
        )
        raise


def _get_real_class():
    """延迟构建真正的 Delta 子类 (继承 Qwen3VLForConditionalGeneration)。"""
    global _RealDeltaClass
    if _RealDeltaClass is not None:
        return _RealDeltaClass

    _ensure_base()

    class _Qwen3DeltaImpl(_Qwen3VLBase):
        """
        Qwen3-VL + 双流运动编码 (真正的实现类)。

        架构与 QwenDelta (Qwen2.5-VL) 完全对称, 核心区别:
        - 基类: Qwen3VLForConditionalGeneration
        - hidden_size: 4096 (vs Qwen2.5-VL 的 3584)
        - ViT: Qwen3-ViT (带 DeepStack 多级特征融合)

        ┌─────────────┐    ┌──────────────┐
        │ Keyframes   │    │  Diff-Maps   │
        │ (稀疏, 1fps)│    │ (密集, 4x)   │
        └──────┬──────┘    └──────┬───────┘
               │                  │
          Qwen3-ViT (冻结)    ResNet-18 (训练)
          + DeepStack             │
               │            MLP Projector (训练)
               │                  │
          RGB Tokens         Motion Tokens
               └────────┬─────────┘
                        │
              交错拼接 + Interleaved-MRoPE
                        │
                  Qwen3 LLM (LoRA)
                        │
                      Output
        """

        def __init__(self, config):
            super().__init__(config)
            self._himmel_init(config)

        def _himmel_init(self, config):
            """初始化 Himmel 新增模块。"""
            # 兼容新旧 transformers composite config
            if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
                llm_hidden_size = config.text_config.hidden_size
            else:
                llm_hidden_size = config.hidden_size

            # 从 config 中读取可插拔 motion encoder / compressed-domain 配置
            motion_encoder_type = getattr(config, "motion_encoder_type", "resnet")
            siglip_model_path = getattr(config, "siglip_model_path", "google/siglip-base-patch16-224")
            freeze_motion_encoder = getattr(config, "freeze_motion_encoder", False)
            motion_input_mode = getattr(config, "motion_input_mode", "rgb_diff")
            use_motion_vectors = getattr(config, "use_motion_vectors", True)
            use_residuals = getattr(config, "use_residuals", True)
            compressed_fusion_mode = getattr(config, "compressed_fusion_mode", "gated")
            compressed_use_iframe_context = getattr(config, "compressed_use_iframe_context", False)
            if compressed_use_iframe_context or getattr(config, "compressed_use_iframe_encoder", False):
                raise NotImplementedError("This archived Qwen3 wrapper does not forward appearance-branch inputs; use the Qwen2.5 tri-stream implementation.")
            # Qwen3-VL ViT hidden_size (用于 shared_vit)
            vit_hidden_size = getattr(config, "motion_vit_hidden_size", 1280)
            use_cross_attention = getattr(config, "motion_use_cross_attention", False)

            if motion_input_mode == "compressed":
                self.motion_module = CompressedMotionAdapter(
                    hidden_size=llm_hidden_size,
                    motion_encoder_type=motion_encoder_type,
                    siglip_model_path=siglip_model_path,
                    freeze_motion_encoder=freeze_motion_encoder,
                    vit_hidden_size=vit_hidden_size,
                    use_cross_attention=use_cross_attention,
                    use_motion_vectors=use_motion_vectors,
                    use_residuals=use_residuals,
                    use_iframe_context=compressed_use_iframe_context,
                    fusion_mode=compressed_fusion_mode,
                )
            else:
                self.motion_module = MotionEncoderWithProjector(
                    hidden_size=llm_hidden_size,
                    motion_encoder_type=motion_encoder_type,
                    siglip_model_path=siglip_model_path,
                    freeze_motion_encoder=freeze_motion_encoder,
                    vit_hidden_size=vit_hidden_size,
                    use_cross_attention=use_cross_attention,
                )
            self.num_motion_tokens_per_keyframe = 4

            self.motion_start_token_id: Optional[int] = None
            self.motion_end_token_id: Optional[int] = None
            self.motion_pad_token_id: Optional[int] = None

        def setup_motion_token_ids(self, tokenizer):
            """注册特殊 motion tokens 到 tokenizer。"""
            special_tokens = {
                "additional_special_tokens": [
                    MOTION_START_TOKEN, MOTION_END_TOKEN, MOTION_PAD_TOKEN,
                ]
            }
            num_added = tokenizer.add_special_tokens(special_tokens)
            if num_added > 0:
                self.resize_token_embeddings(len(tokenizer))
                logger.info("添加 %d 个特殊 token, 词表: %d", num_added, len(tokenizer))

            self.motion_start_token_id = tokenizer.convert_tokens_to_ids(MOTION_START_TOKEN)
            self.motion_end_token_id = tokenizer.convert_tokens_to_ids(MOTION_END_TOKEN)
            self.motion_pad_token_id = tokenizer.convert_tokens_to_ids(MOTION_PAD_TOKEN)
            logger.info(
                "Motion token IDs - start: %d, end: %d, pad: %d",
                self.motion_start_token_id, self.motion_end_token_id, self.motion_pad_token_id,
            )

        def encode_motion(
            self,
            diff_frames=None,
            motion_vectors=None,
            residuals=None,
        ):
            """编码 RGB diff 或真实压缩域双流为 Motion Tokens。"""
            return self.motion_module(
                diff_frames=diff_frames,
                motion_vectors=motion_vectors,
                residuals=residuals,
            )

        def _inject_motion_tokens(
            self,
            input_ids: torch.Tensor,
            inputs_embeds: torch.Tensor,
            motion_tokens: torch.Tensor,
        ) -> torch.Tensor:
            """
            将 Motion Tokens 可微地替换到 inputs_embeds 中 motion_pad 占位符位置。
            使用 one-hot @ mt 矩阵乘法 + mask 混合, 兼容 gradient checkpointing。
            """
            B, S, D = inputs_embeds.shape
            device = inputs_embeds.device

            if motion_tokens is None or self.motion_pad_token_id is None:
                return inputs_embeds

            new_embeds_list = []
            for b in range(B):
                ids = input_ids[b]
                embeds = inputs_embeds[b]

                motion_mask = (ids == self.motion_pad_token_id)
                motion_indices = motion_mask.nonzero(as_tuple=True)[0]

                if len(motion_indices) == 0:
                    new_embeds_list.append(embeds)
                    continue

                mt = motion_tokens[b] if motion_tokens.dim() == 3 else motion_tokens
                num_motion = min(len(motion_indices), mt.shape[0])

                embeds_detached = embeds.detach()
                padded_mt = mt[:num_motion].to(dtype=embeds.dtype)
                one_hot = torch.zeros(S, num_motion, device=device, dtype=embeds.dtype)
                one_hot[motion_indices[:num_motion], torch.arange(num_motion, device=device)] = 1.0
                motion_contrib = one_hot @ padded_mt
                mask_f = motion_mask.unsqueeze(-1).to(embeds.dtype)
                embeds = embeds_detached * (1.0 - mask_f) + motion_contrib
                new_embeds_list.append(embeds)

            return torch.stack(new_embeds_list, dim=0)

        def _get_visual_encoder(self):
            """兼容不同 transformers/Qwen3 实现, 查找视觉编码器入口。"""
            candidates = [
                ("self.visual", getattr(self, "visual", None)),
                ("self.vision_model", getattr(self, "vision_model", None)),
                ("self.model.visual", getattr(getattr(self, "model", None), "visual", None)),
                ("self.model.vision_model", getattr(getattr(self, "model", None), "vision_model", None)),
                (
                    "self.model.model.visual",
                    getattr(getattr(getattr(self, "model", None), "model", None), "visual", None),
                ),
                (
                    "self.model.model.vision_model",
                    getattr(getattr(getattr(self, "model", None), "model", None), "vision_model", None),
                ),
            ]
            for _, module in candidates:
                if module is not None:
                    return module

            self_children = [name for name, _ in self.named_children()]
            model_children = []
            if getattr(self, "model", None) is not None:
                model_children = [name for name, _ in self.model.named_children()]
            logger.warning(
                "无法找到视觉编码器 (已检查 self/self.model/self.model.model 的 visual/vision_model). "
                "self_children=%s, model_children=%s",
                self_children,
                model_children,
            )
            return None

        def _normalize_visual_output(self, out):
            """将不同视觉塔返回统一成 [N_vis, D] 的 Tensor。"""
            raw_type = type(out).__name__ if out is not None else "NoneType"

            if isinstance(out, (tuple, list)):
                out = out[0] if out else None

            if out is None:
                return None

            if not torch.is_tensor(out):
                last_hidden_state = getattr(out, "last_hidden_state", None)
                if torch.is_tensor(last_hidden_state):
                    out = last_hidden_state
                else:
                    hidden_states = getattr(out, "hidden_states", None)
                    if isinstance(hidden_states, (tuple, list)) and hidden_states:
                        out = hidden_states[-1]
                    else:
                        pooler_output = getattr(out, "pooler_output", None)
                        if torch.is_tensor(pooler_output):
                            out = pooler_output

            if out is None or not torch.is_tensor(out):
                logger.warning("[Visual] 无法将视觉输出转换为 Tensor: raw_type=%s", raw_type)
                return None

            if out.dim() == 1:
                out = out.unsqueeze(0)
            elif out.dim() > 2:
                out = out.reshape(-1, out.shape[-1])

            if out.dim() != 2:
                logger.warning(
                    "[Visual] 视觉输出归一化后维度异常: raw_type=%s, shape=%s",
                    raw_type,
                    tuple(out.shape),
                )
                return None

            return out

        def _get_visual_features(self, pixel_values, pixel_values_videos,
                                  image_grid_thw, video_grid_thw):
            """调用 ViT 编码视觉特征。

            Qwen3-VL DeepStack 差异:
            - Qwen2.5-VL 的 self.visual() 返回 Tensor [N_tokens, D]
            - Qwen3-VL 的 self.visual() 返回 tuple (hidden_states, deepstack_features)
              其中 hidden_states 是最终层特征 [N_tokens, D], deepstack_features
              是各层特征列表 (用于 DeepStack 在 LLM 解码器中的深层融合)
            - 对于 alignment, 我们只需要 hidden_states (第一个元素)
            """
            if pixel_values is None and pixel_values_videos is None:
                return None

            visual_encoder = self._get_visual_encoder()
            if visual_encoder is None:
                return None

            if pixel_values_videos is not None:
                out = visual_encoder(pixel_values_videos, grid_thw=video_grid_thw)
            elif pixel_values is not None:
                out = visual_encoder(pixel_values, grid_thw=image_grid_thw)
            else:
                return None

            return self._normalize_visual_output(out)

        def _get_embed_tokens(self):
            """获取 text embedding 层, 兼容不同 transformers 版本。"""
            inner = self.model
            if hasattr(inner, "embed_tokens"):
                return inner.embed_tokens
            if hasattr(inner, "language_model") and hasattr(inner.language_model, "embed_tokens"):
                return inner.language_model.embed_tokens
            if hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
                return inner.model.embed_tokens
            raise AttributeError(
                f"无法找到 embed_tokens。self.model type={type(inner).__name__}, "
                f"children={[n for n, _ in inner.named_children()]}"
            )

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            rope_deltas=None,
            second_per_grid_ts=None,
            # ---- Himmel 新增 ----
            diff_frames=None,
            **kwargs,
        ):
            """
            扩展的 forward, 新增 diff_frames 参数。

            策略 (兼容 gradient checkpointing):
            手动做 ViT 编码 + embedding + motion 注入, 然后传 inputs_embeds 给父类。
            这样 motion tokens 在计算图的显式路径上, 不受 checkpointing 影响。
            """
            # Step 1: 编码差分图
            motion_tokens = None
            if diff_frames is not None:
                motion_tokens = self.encode_motion(diff_frames)

            # 无 motion tokens -> 原始逻辑
            if motion_tokens is None:
                return _Qwen3VLBase.forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    rope_deltas=rope_deltas,
                    second_per_grid_ts=second_per_grid_ts,
                    **kwargs,
                )

            # ---- 有 Motion Tokens: 手动做 embedding + 注入 ----
            # Step 2: ViT 编码
            visual_features = self._get_visual_features(
                pixel_values, pixel_values_videos,
                image_grid_thw, video_grid_thw,
            )

            # Step 3: Text embedding
            embed_fn = self._get_embed_tokens()
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = embed_fn(input_ids)

            # Step 4: 替换视觉 tokens
            if visual_features is not None and input_ids is not None:
                image_token_id = getattr(self.config, "image_token_id", None)
                video_token_id = getattr(self.config, "video_token_id", None)

                if video_token_id is not None and pixel_values_videos is not None:
                    video_mask = input_ids == video_token_id
                    inputs_embeds[video_mask] = visual_features.to(inputs_embeds.dtype)
                elif image_token_id is not None and pixel_values is not None:
                    image_mask = input_ids == image_token_id
                    inputs_embeds[image_mask] = visual_features.to(inputs_embeds.dtype)

            # Step 5: 注入 Motion Tokens
            inputs_embeds = self._inject_motion_tokens(input_ids, inputs_embeds, motion_tokens)

            # Step 6: 调用父类 (传 inputs_embeds, 不传 input_ids/pixel_values)
            return _Qwen3VLBase.forward(
                self,
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                second_per_grid_ts=second_per_grid_ts,
                **kwargs,
            )

        def forward_alignment(
            self,
            input_ids=None,
            attention_mask=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            diff_frames=None,
            motion_vectors=None,
            residuals=None,
            num_diffs_per_keyframe: int = 4,
            **kwargs,
        ) -> dict:
            """
            Stage 1 专用: Contrastive/Regression Alignment Forward。

            不走 LLM 的 next-token prediction,
            直接将 motion tokens 与 ViT visual tokens 做 feature alignment。

            与 Qwen2.5-VL 版本完全对称, 核心差异:
            - hidden_size=4096 (自动适配, 无需额外处理)
            - Qwen3-ViT 含 DeepStack, 但输出格式相同 [N_vis, D]
            - spatial_merge_size / temporal_patch_size 与 Qwen2.5-VL 一致 (均为 2)

            Returns:
                dict with keys: "loss", "avg_cosine_sim", "num_pairs"
            """
            device = next(self.motion_module.parameters()).device

            if diff_frames is None and motion_vectors is None and residuals is None:
                logger.warning("[Alignment] no motion inputs provided")
                return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

            # Step 1: ViT 编码 visual features (冻结, detach)
            with torch.no_grad():
                visual_features = self._get_visual_features(
                    pixel_values, pixel_values_videos,
                    image_grid_thw, video_grid_thw,
                )

            if visual_features is None:
                logger.warning("[Alignment] visual_features is None")
                return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

            visual_features = visual_features.detach()
            D = visual_features.shape[-1]

            # Step 2: 编码 motion tokens (Stage 1 alignment 用 align_proj 对齐到 ViT 空间)
            motion_tokens = self.motion_module.encode_motion_for_alignment(
                diff_frames=diff_frames,
                motion_vectors=motion_vectors,
                residuals=residuals,
            )
            B, T_motion = motion_tokens.shape[:2]

            logger.info("[Alignment] visual_features=%s, motion_tokens=%s, video_grid_thw=%s",
                        visual_features.shape, motion_tokens.shape,
                        video_grid_thw if video_grid_thw is not None else "None")

            # Step 3: 将 visual tokens 按帧分组并 mean pool
            # Qwen3-VL 与 Qwen2.5-VL 的 merge 机制相同:
            #   temporal_patch_size=2, spatial_merge_size=2
            spatial_merge_size = getattr(self.config, "vision_config", None)
            if spatial_merge_size is not None:
                spatial_merge_size = getattr(spatial_merge_size, "spatial_merge_size", 2)
            else:
                spatial_merge_size = 2

            if video_grid_thw is not None:
                thw = video_grid_thw
                per_frame_targets = []
                offset = 0
                for v in range(thw.shape[0]):
                    t, h, w = thw[v].tolist()
                    tokens_per_frame = int(h * w) // (spatial_merge_size ** 2)
                    num_temporal = int(t)
                    for f in range(num_temporal):
                        start = offset + f * tokens_per_frame
                        end = start + tokens_per_frame
                        if end <= visual_features.shape[0]:
                            frame_feat = visual_features[start:end].mean(dim=0)
                            per_frame_targets.append(frame_feat)
                    offset += num_temporal * tokens_per_frame

                if not per_frame_targets:
                    logger.warning("[Alignment] per_frame_targets is empty (spatial_merge=%d, thw=%s)",
                                  spatial_merge_size, thw)
                    return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

                per_frame_targets = torch.stack(per_frame_targets)
            elif image_grid_thw is not None:
                thw = image_grid_thw
                per_frame_targets = []
                offset = 0
                for v in range(thw.shape[0]):
                    t, h, w = thw[v].tolist()
                    n_tokens = int(t * h * w) // (spatial_merge_size ** 2)
                    frame_feat = visual_features[offset:offset + n_tokens].mean(dim=0)
                    per_frame_targets.append(frame_feat)
                    offset += n_tokens
                per_frame_targets = torch.stack(per_frame_targets)
            else:
                logger.warning("[Alignment] both video_grid_thw and image_grid_thw are None")
                return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

            n_visual_frames = per_frame_targets.shape[0]
            logger.info("[Alignment] n_visual_frames=%d, B=%d, T_motion=%d, num_diffs_per_keyframe=%d",
                        n_visual_frames, B, T_motion, num_diffs_per_keyframe)

            # Step 4: 构造对齐 pairs (帧差 target, InfoNCE 对比学习)
            # target = vf[i+1] - vf[i] (ViT 特征空间的"语义级变化")
            alignment_pool = getattr(self.config, "alignment_motion_pool", "mean")
            all_motion_queries = []
            all_visual_targets = []

            for b in range(B):
                mt = motion_tokens[b]
                n_vf = n_visual_frames // max(B, 1)
                start_vf = b * n_vf
                vf_targets = per_frame_targets[start_vf:start_vf + n_vf]

                if n_vf < 2 or T_motion == 0:
                    if n_vf >= 1 and T_motion > 0:
                        for i in range(T_motion):
                            all_motion_queries.append(mt[i])
                            all_visual_targets.append(vf_targets[0])
                    continue

                n_motion_intervals = T_motion // max(num_diffs_per_keyframe, 1)
                if n_motion_intervals == 0:
                    n_motion_intervals = 1
                n_visual_intervals = n_vf - 1

                for mi in range(n_motion_intervals):
                    mt_start = mi * num_diffs_per_keyframe
                    mt_end = min(mt_start + num_diffs_per_keyframe, T_motion)
                    if mt_start >= T_motion:
                        break
                    vi = min(int(mi * n_visual_intervals / n_motion_intervals), n_visual_intervals - 1)
                    # ★ 核心改动: target = 帧差 (而非帧均值)
                    target = vf_targets[vi + 1] - vf_targets[vi]
                    interval_mt = mt[mt_start:mt_end]
                    pooled_mt = pool_motion_tokens(interval_mt, mode=alignment_pool)
                    all_motion_queries.append(pooled_mt)
                    all_visual_targets.append(target)

            if not all_motion_queries:
                logger.warning("[Alignment] no alignment pairs constructed")
                return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

            motion_queries = torch.stack(all_motion_queries)
            visual_targets = torch.stack(all_visual_targets).detach()
            compute_dtype = motion_queries.dtype
            visual_targets = visual_targets.to(dtype=compute_dtype)

            logger.info("[Alignment] num_pairs=%d, dtype=%s", len(all_motion_queries), compute_dtype)

            # Step 5: 计算 InfoNCE 对比学习 alignment loss
            align_result = self.motion_module.compute_alignment_loss(motion_queries, visual_targets)
            align_result["num_pairs"] = len(all_motion_queries)
            return align_result

        def configure_training(self, stage: int = 1, sft_mode: str = "lora", freeze_spatial_encoder: bool = False):
            """
            配置训练阶段。

            Stage 1 (Alignment, ~3h):
                训练: MotionEncoder + MotionProjector
                冻结: Qwen3-ViT (含 DeepStack) + LLM

            Stage 2 (SFT):
                sft_mode="lora": 训练 MotionEncoder + MotionProjector + LLM LoRA
                sft_mode="full": 训练 MotionEncoder + MotionProjector + LLM (全量)
                冻结: Qwen3-ViT
            """
            freeze_module(self)

            if stage == 1:
                self.motion_module.configure_training_state(
                    train_alignment_head=True,
                    train_projector=False,
                    train_alignment_projector=True,
                    freeze_spatial_encoder=freeze_spatial_encoder,
                )
                logger.info("=== Qwen3-VL Stage 1: Alignment ===")
                logger.info("训练: MotionSSM + Alignment MLP + AlignmentHead")
                logger.info("冻结: Qwen3-ViT (DeepStack) + LLM")
                if freeze_spatial_encoder:
                    logger.info("Stage 1 warmup: 先冻结 MotionSSM spatial_encoder，仅训练 temporal + alignment 分支")
            elif stage == 2:
                self.motion_module.configure_training_state(
                    train_alignment_head=False,
                    train_projector=True,
                    train_alignment_projector=False,
                )
                if sft_mode == "full":
                    unfreeze_module(self.model)
                    visual_encoder = self._get_visual_encoder()
                    if visual_encoder is not None:
                        freeze_module(visual_encoder)
                    logger.info("=== Qwen3-VL Stage 2: Full SFT ===")
                    logger.info("训练: MotionEncoder + MotionProjector + LLM (全量)")
                    logger.info("冻结: Qwen3-ViT")
                else:
                    logger.info("=== Qwen3-VL Stage 2: SFT with LoRA ===")
                    logger.info("训练: MotionEncoder + MotionProjector + LLM LoRA")
                    logger.info("冻结: Qwen3-ViT")
            else:
                raise ValueError(f"未知训练阶段: {stage}")

            info = get_trainable_params_info(self)
            logger.info("参数统计: %s", info)
            return self

        @torch.no_grad()
        def generate(
            self,
            input_ids=None,
            attention_mask=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            diff_frames=None,
            **generate_kwargs,
        ):
            """Generate with motion tokens injected into inputs_embeds."""
            if diff_frames is None and motion_vectors is None and residuals is None:
                return _Qwen3VLBase.generate(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    **generate_kwargs,
                )

            motion_tokens = self.encode_motion(
                diff_frames=diff_frames,
                motion_vectors=motion_vectors,
                residuals=residuals,
            )

            visual_features = self._get_visual_features(
                pixel_values, pixel_values_videos,
                image_grid_thw, video_grid_thw,
            )

            embed_fn = self._get_embed_tokens()
            inputs_embeds = embed_fn(input_ids)

            if visual_features is not None and input_ids is not None:
                image_token_id = getattr(self.config, "image_token_id", None)
                video_token_id = getattr(self.config, "video_token_id", None)
                if video_token_id is not None and pixel_values_videos is not None:
                    video_mask = input_ids == video_token_id
                    inputs_embeds[video_mask] = visual_features.to(inputs_embeds.dtype)
                elif image_token_id is not None and pixel_values is not None:
                    image_mask = input_ids == image_token_id
                    inputs_embeds[image_mask] = visual_features.to(inputs_embeds.dtype)

            inputs_embeds = self._inject_motion_tokens(input_ids, inputs_embeds, motion_tokens)

            return _Qwen3VLBase.generate(
                self,
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                **generate_kwargs,
            )

    _RealDeltaClass = _Qwen3DeltaImpl
    return _RealDeltaClass


# ============================================================================
# 公开的代理类: import 时不会触发 transformers 导入
# ============================================================================

class Qwen3DeltaForConditionalGeneration:
    """代理类, 将 from_pretrained / 实例化委托给延迟构建的真实子类。

    这样 `from models.qwen3_delta import Qwen3DeltaForConditionalGeneration`
    在 transformers 缺少 Qwen3VL 支持时也不会崩溃, 只有真正调用时才报错。
    """

    def __class_getitem__(cls, item):
        return _get_real_class().__class_getitem__(item)

    def __new__(cls, *args, **kwargs):
        real_cls = _get_real_class()
        return real_cls(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        real_cls = _get_real_class()
        return real_cls.from_pretrained(*args, **kwargs)


# ============================================================================
# 工厂函数: 简化加载流程
# ============================================================================

def load_qwen3_delta(model_path: str, dtype=torch.bfloat16, **kwargs):
    """
    便捷加载函数。

    用法:
        model = load_qwen3_delta("/path/to/Qwen3-VL-8B-Instruct")
    """
    model = Qwen3DeltaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        **kwargs,
    )
    return model
