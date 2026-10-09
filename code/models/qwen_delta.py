#!/usr/bin/env python3
"""
Himmel-VLM: Qwen2.5-VL 双流魔改
=====================================
核心改动:
1. 在 Qwen2.5-VLForConditionalGeneration 基础上新增 MotionEncoder + Projector
2. Forward 中将 RGB Tokens (来自 Qwen-ViT) 和 Motion Tokens (来自 ResNet) 交错拼接
3. M-RoPE (3D 位置编码) 处理: Motion Tokens 共享前一个 Keyframe 的时间 ID,
   但分配独立的空间位置 ID (类似 Qwen 对图像 patch 的处理方式)

Qwen2.5-VL 的 M-RoPE 说明:
- position_ids 形状: [3, seq_len], 分别是 (temporal, height, width)
- 图像 token: temporal 相同, height/width 按 patch 网格编排
- 文本 token: 三个维度共享同一个递增 ID
- 我们的 Motion Tokens: temporal 共享 keyframe, height/width 从 0 开始独立编号
"""
from __future__ import annotations

import logging
from typing import Optional, List, Tuple, Union

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .delta_components import (
    MotionEncoderWithProjector,
    freeze_module,
    unfreeze_module,
    get_trainable_params_info,
    pool_motion_tokens,
)
from .compressed_motion_adapter import CompressedMotionAdapter

logger = logging.getLogger(__name__)


# ============================================================================
# 特殊 Token ID (需要在 tokenizer 中注册)
# ============================================================================
MOTION_START_TOKEN = "<|motion_start|>"
MOTION_END_TOKEN = "<|motion_end|>"
MOTION_PAD_TOKEN = "<|motion_pad|>"  # 占位符, 后续被 motion embedding 替换


class QwenDeltaForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    Qwen2.5-VL + 双流运动编码。

    架构:
    ┌─────────────┐    ┌──────────────┐
    │ Keyframes   │    │  Diff-Maps   │
    │ (稀疏, 1fps)│    │ (密集, 4x)   │
    └──────┬──────┘    └──────┬───────┘
           │                  │
      Qwen-ViT (冻结)    ResNet-18 (训练)
           │                  │
      RGB Tokens          Motion Feat
           │                  │
           │            MLP Projector (训练)
           │                  │
           │            Motion Tokens
           └────────┬─────────┘
                    │
              交错拼接 + M-RoPE
                    │
              Qwen2.5 LLM (LoRA)
                    │
                  Output
    """

    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)

        # 新增: 运动编码模块
        # 兼容新旧 transformers: 新版用 composite config (text_config.hidden_size),
        # 旧版直接在顶层 (config.hidden_size)
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            llm_hidden_size = config.text_config.hidden_size
        else:
            llm_hidden_size = config.hidden_size

        # 从 config 中读取可插拔 motion encoder / compressed-domain 配置
        # 这些字段由 train_himmel.py 在创建模型前注入到 config 中
        motion_encoder_type = getattr(config, "motion_encoder_type", "resnet")
        siglip_model_path = getattr(config, "siglip_model_path", "google/siglip-base-patch16-224")
        freeze_motion_encoder = getattr(config, "freeze_motion_encoder", False)
        motion_input_mode = getattr(config, "motion_input_mode", "rgb_diff")
        use_motion_vectors = getattr(config, "use_motion_vectors", True)
        use_residuals = getattr(config, "use_residuals", True)
        compressed_fusion_mode = getattr(config, "compressed_fusion_mode", "gated")
        siglip_unfreeze_last_n_layers = getattr(config, "siglip_unfreeze_last_n_layers", 0)
        alignment_gather_global_negatives = getattr(config, "alignment_gather_global_negatives", True)
        compressed_fused_alignment_weight = getattr(config, "compressed_fused_alignment_weight", 1.0)
        compressed_mv_alignment_weight = getattr(config, "compressed_mv_alignment_weight", 0.25)
        compressed_residual_alignment_weight = getattr(config, "compressed_residual_alignment_weight", 0.25)
        compressed_use_iframe_context = getattr(config, "compressed_use_iframe_context", False)
        compressed_use_iframe_encoder = getattr(config, "compressed_use_iframe_encoder", False)
        compressed_iframe_alignment_weight = getattr(config, "compressed_iframe_alignment_weight", 0.25)
        # Qwen2.5-VL ViT hidden_size: 1280 (用于 shared_vit)
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
                use_iframe_encoder=compressed_use_iframe_encoder,
                fusion_mode=compressed_fusion_mode,
                siglip_unfreeze_last_n_layers=siglip_unfreeze_last_n_layers,
                fused_alignment_weight=compressed_fused_alignment_weight,
                mv_alignment_weight=compressed_mv_alignment_weight,
                residual_alignment_weight=compressed_residual_alignment_weight,
                iframe_alignment_weight=compressed_iframe_alignment_weight,
            )
        else:
            self.motion_module = MotionEncoderWithProjector(
                hidden_size=llm_hidden_size,  # Qwen2.5-VL-7B: 3584
                motion_encoder_type=motion_encoder_type,
                siglip_model_path=siglip_model_path,
                freeze_motion_encoder=freeze_motion_encoder,
                vit_hidden_size=vit_hidden_size,
                use_cross_attention=use_cross_attention,
                siglip_unfreeze_last_n_layers=siglip_unfreeze_last_n_layers,
            )

        if hasattr(self.motion_module, "set_alignment_options"):
            self.motion_module.set_alignment_options(
                gather_global_negatives=alignment_gather_global_negatives,
                fused_weight=compressed_fused_alignment_weight,
                mv_weight=compressed_mv_alignment_weight,
                residual_weight=compressed_residual_alignment_weight,
                iframe_weight=compressed_iframe_alignment_weight,
            )
        elif hasattr(self.motion_module, "set_alignment_gather_global_negatives"):
            self.motion_module.set_alignment_gather_global_negatives(alignment_gather_global_negatives)

        # 每个 keyframe 后面跟几个 motion token
        self.num_motion_tokens_per_keyframe = 4

        # 特殊 token 的 ID, 需要在外部设置
        self.motion_start_token_id: Optional[int] = None
        self.motion_end_token_id: Optional[int] = None
        self.motion_pad_token_id: Optional[int] = None

    def setup_motion_token_ids(self, tokenizer):
        """从 tokenizer 中获取特殊 token 的 ID。"""
        special_tokens = {
            "additional_special_tokens": [
                MOTION_START_TOKEN,
                MOTION_END_TOKEN,
                MOTION_PAD_TOKEN,
            ]
        }
        num_added = tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.resize_token_embeddings(len(tokenizer))
            logger.info("添加了 %d 个特殊 token, 词表大小: %d", num_added, len(tokenizer))

        self.motion_start_token_id = tokenizer.convert_tokens_to_ids(MOTION_START_TOKEN)
        self.motion_end_token_id = tokenizer.convert_tokens_to_ids(MOTION_END_TOKEN)
        self.motion_pad_token_id = tokenizer.convert_tokens_to_ids(MOTION_PAD_TOKEN)
        logger.info(
            "Motion token IDs - start: %d, end: %d, pad: %d",
            self.motion_start_token_id,
            self.motion_end_token_id,
            self.motion_pad_token_id,
        )

    def encode_motion(
        self,
        diff_frames: Optional[torch.Tensor] = None,
        motion_vectors: Optional[torch.Tensor] = None,
        residuals: Optional[torch.Tensor] = None,
        iframe_context: Optional[torch.Tensor] = None,
        iframe_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """编码 RGB diff 或真实压缩域三流为 Motion Tokens。"""
        return self.motion_module(
            diff_frames=diff_frames,
            motion_vectors=motion_vectors,
            residuals=residuals,
            iframe_context=iframe_context,
            iframe_frames=iframe_frames,
        )

    def _inject_motion_tokens(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        motion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 Motion Tokens 可微地替换到 inputs_embeds 中 motion_pad 占位符位置。

        使用 one-hot @ mt 矩阵乘法 + mask 混合, 确保梯度能从 loss 回传到
        MotionEncoder, 同时兼容 gradient checkpointing。

        Args:
            input_ids: [B, S] 原始 token ids (用于定位 motion_pad 位置)
            inputs_embeds: [B, S, D] 已包含 text + visual embedding
            motion_tokens: [B, T_motion, D] 编码后的 motion tokens
        Returns:
            inputs_embeds: [B, S, D] 注入了 motion tokens 的 embedding
        """
        B, S, D = inputs_embeds.shape
        device = inputs_embeds.device

        if motion_tokens is None or self.motion_pad_token_id is None:
            return inputs_embeds

        new_embeds_list = []
        for b in range(B):
            ids = input_ids[b]          # [S]
            embeds = inputs_embeds[b]    # [S, D]

            motion_mask = (ids == self.motion_pad_token_id)
            motion_indices = motion_mask.nonzero(as_tuple=True)[0]

            if len(motion_indices) == 0:
                new_embeds_list.append(embeds)
                continue

            mt = motion_tokens[b] if motion_tokens.dim() == 3 else motion_tokens
            num_motion = min(len(motion_indices), mt.shape[0])

            # embeds 来自冻结的 LLM embedding 层, detach 以确保
            # 梯度只流经 motion_contrib 路径
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
        """兼容不同 transformers/Qwen 实现, 查找视觉编码器入口。"""
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
        """
        调用 ViT 编码视觉特征。兼容新版 transformers 中 visual 属性的不同命名。

        Returns:
            visual_features: Tensor or None
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

    def _get_spatial_merge_size(self) -> int:
        vision_config = getattr(self.config, "vision_config", None)
        if vision_config is not None:
            return getattr(vision_config, "spatial_merge_size", 2)
        return 2

    def _split_video_features_per_sample(
        self,
        visual_features: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.LongTensor],
    ) -> List[torch.Tensor]:
        if visual_features is None or video_grid_thw is None:
            return []

        spatial_merge_size = self._get_spatial_merge_size()
        per_sample_features = []
        offset = 0
        for v in range(video_grid_thw.shape[0]):
            t, h, w = video_grid_thw[v].tolist()
            tokens_per_frame = int(h * w) // (spatial_merge_size ** 2)
            num_temporal = int(t)
            sample_frames = []
            for f in range(num_temporal):
                start = offset + f * tokens_per_frame
                end = start + tokens_per_frame
                if end <= visual_features.shape[0]:
                    sample_frames.append(visual_features[start:end].mean(dim=0))
            offset += num_temporal * tokens_per_frame

            if sample_frames:
                per_sample_features.append(torch.stack(sample_frames))
            else:
                per_sample_features.append(visual_features.new_zeros((0, visual_features.shape[-1])))

        return per_sample_features

    def _split_image_features_by_counts(
        self,
        visual_features: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.LongTensor],
        image_counts: Optional[torch.LongTensor],
    ) -> List[torch.Tensor]:
        if visual_features is None or image_grid_thw is None or image_counts is None:
            return []

        spatial_merge_size = self._get_spatial_merge_size()
        per_sample_features = []
        feat_offset = 0
        grid_offset = 0
        hidden_size = visual_features.shape[-1]

        for count in image_counts.tolist():
            sample_frames = []
            for _ in range(int(count)):
                if grid_offset >= image_grid_thw.shape[0]:
                    break
                t, h, w = image_grid_thw[grid_offset].tolist()
                num_tokens = int(t * h * w) // (spatial_merge_size ** 2)
                end = feat_offset + num_tokens
                if end <= visual_features.shape[0] and num_tokens > 0:
                    sample_frames.append(visual_features[feat_offset:end].mean(dim=0))
                feat_offset = end
                grid_offset += 1

            if sample_frames:
                per_sample_features.append(torch.stack(sample_frames))
            else:
                per_sample_features.append(visual_features.new_zeros((0, hidden_size)))

        return per_sample_features

    @staticmethod
    def _pad_feature_sequences(feature_list: List[torch.Tensor]):
        if not feature_list:
            return None, None

        max_len = max((feat.shape[0] for feat in feature_list), default=0)
        if max_len == 0:
            dim = feature_list[0].shape[-1]
            empty = feature_list[0].new_zeros((len(feature_list), 0, dim))
            mask = torch.zeros(len(feature_list), 0, dtype=torch.bool, device=feature_list[0].device)
            return empty, mask

        dim = feature_list[0].shape[-1]
        padded = feature_list[0].new_zeros((len(feature_list), max_len, dim))
        mask = torch.zeros(len(feature_list), max_len, dtype=torch.bool, device=feature_list[0].device)
        for idx, feat in enumerate(feature_list):
            cur_len = feat.shape[0]
            if cur_len == 0:
                continue
            padded[idx, :cur_len] = feat
            mask[idx, :cur_len] = True
        return padded, mask

    def _prepare_shared_vit_motion_inputs(
        self,
        diff_pixel_values_videos: Optional[torch.Tensor],
        diff_video_grid_thw: Optional[torch.LongTensor],
        *,
        diff_pixel_values: Optional[torch.Tensor] = None,
        diff_image_grid_thw: Optional[torch.LongTensor] = None,
        diff_image_counts: Optional[torch.LongTensor] = None,
        keyframe_visual_features: Optional[torch.Tensor] = None,
        keyframe_video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        if diff_pixel_values_videos is None and diff_pixel_values is None:
            return None, {}

        with torch.no_grad():
            diff_visual_features = self._get_visual_features(
                pixel_values=diff_pixel_values,
                pixel_values_videos=diff_pixel_values_videos,
                image_grid_thw=diff_image_grid_thw,
                video_grid_thw=diff_video_grid_thw,
            )

        if diff_visual_features is None:
            return None, {}

        if diff_pixel_values_videos is not None and diff_video_grid_thw is not None:
            diff_per_sample = self._split_video_features_per_sample(
                diff_visual_features.detach(),
                diff_video_grid_thw,
            )
        elif diff_pixel_values is not None and diff_image_grid_thw is not None and diff_image_counts is not None:
            diff_per_sample = self._split_image_features_by_counts(
                diff_visual_features.detach(),
                diff_image_grid_thw,
                diff_image_counts,
            )
        else:
            raise ValueError("shared_vit 的 diff 输入缺少对应的 grid/count 元信息")

        motion_inputs, motion_attention_mask = self._pad_feature_sequences(diff_per_sample)
        if motion_inputs is None:
            return None, {}

        shared_kwargs = {}
        if motion_attention_mask is not None:
            shared_kwargs["motion_attention_mask"] = motion_attention_mask

        if keyframe_visual_features is not None and keyframe_video_grid_thw is not None:
            keyframe_per_sample = self._split_video_features_per_sample(
                keyframe_visual_features.detach(),
                keyframe_video_grid_thw,
            )
            keyframe_inputs, keyframe_attention_mask = self._pad_feature_sequences(keyframe_per_sample)
            if keyframe_inputs is not None:
                shared_kwargs["keyframe_features"] = keyframe_inputs.to(dtype=motion_inputs.dtype)
            if keyframe_attention_mask is not None:
                shared_kwargs["keyframe_attention_mask"] = keyframe_attention_mask

        return motion_inputs, shared_kwargs

    def _build_compressed_iframe_context(
        self,
        visual_features: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.LongTensor],
        *,
        diff_frames: Optional[torch.Tensor] = None,
        motion_vectors: Optional[torch.Tensor] = None,
        residuals: Optional[torch.Tensor] = None,
        num_diffs_per_keyframe: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if not getattr(self.config, "compressed_use_iframe_context", False):
            return None
        if visual_features is None or video_grid_thw is None:
            return None

        source_tensor = motion_vectors if motion_vectors is not None else residuals
        if source_tensor is None:
            source_tensor = diff_frames
        if source_tensor is None or source_tensor.dim() < 2:
            return None

        batch_size = int(source_tensor.shape[0])
        motion_len = int(source_tensor.shape[1])
        if batch_size == 0 or motion_len == 0:
            return None

        if num_diffs_per_keyframe is None:
            default_repeat = getattr(self, "num_motion_tokens_per_keyframe", None)
            num_diffs_per_keyframe = default_repeat if isinstance(default_repeat, int) else 4
        repeat_unit = max(int(num_diffs_per_keyframe), 1)

        per_sample_features = self._split_video_features_per_sample(
            visual_features.detach(),
            video_grid_thw,
        )
        if not per_sample_features:
            return None

        hidden_size = int(visual_features.shape[-1])
        context_dim = hidden_size * 4
        iframe_context = visual_features.new_zeros((batch_size, motion_len, context_dim))

        for sample_idx in range(batch_size):
            if sample_idx >= len(per_sample_features):
                continue

            frame_feats = per_sample_features[sample_idx]
            if frame_feats.shape[0] == 0:
                continue
            if frame_feats.shape[0] == 1:
                start_feat = frame_feats[0]
                end_feat = frame_feats[0]
                delta_feat = torch.zeros_like(start_feat)
                interaction = start_feat * end_feat
                context_token = torch.cat([start_feat, end_feat, delta_feat, interaction], dim=-1)
                iframe_context[sample_idx] = context_token.unsqueeze(0).repeat(motion_len, 1)
                continue

            n_visual_intervals = frame_feats.shape[0] - 1
            n_motion_intervals = max(1, (motion_len + repeat_unit - 1) // repeat_unit)
            produced = 0
            last_context = None
            sample_tokens = []

            for motion_idx in range(n_motion_intervals):
                visual_idx = min(
                    int(motion_idx * n_visual_intervals / n_motion_intervals),
                    n_visual_intervals - 1,
                )
                start_feat = frame_feats[visual_idx]
                end_feat = frame_feats[visual_idx + 1]
                delta_feat = end_feat - start_feat
                interaction = start_feat * end_feat
                context_token = torch.cat([start_feat, end_feat, delta_feat, interaction], dim=-1)
                last_context = context_token

                repeat_count = min(repeat_unit, motion_len - produced)
                if repeat_count <= 0:
                    break
                sample_tokens.append(context_token.unsqueeze(0).repeat(repeat_count, 1))
                produced += repeat_count

            if produced < motion_len:
                if last_context is None:
                    last_context = visual_features.new_zeros((context_dim,))
                sample_tokens.append(last_context.unsqueeze(0).repeat(motion_len - produced, 1))

            if sample_tokens:
                iframe_context[sample_idx] = torch.cat(sample_tokens, dim=0)

        return iframe_context

    def _get_embed_tokens(self):
        """
        获取 text embedding 层, 兼容不同版本 transformers 的嵌套结构。
        可能的路径: self.model.embed_tokens, self.model.language_model.embed_tokens, etc.
        """
        inner = self.model
        # 路径 1: self.model.embed_tokens (旧版)
        if hasattr(inner, "embed_tokens"):
            return inner.embed_tokens
        # 路径 2: self.model.language_model.embed_tokens (新版 composite)
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "embed_tokens"):
            return inner.language_model.embed_tokens
        # 路径 3: self.model.model.embed_tokens
        if hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
            return inner.model.embed_tokens
        raise AttributeError(
            f"无法找到 embed_tokens。self.model type={type(inner).__name__}, "
            f"children={[n for n, _ in inner.named_children()]}"
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # ---- 新增参数: 运动输入 ----
        diff_frames: Optional[torch.Tensor] = None,
        diff_pixel_values_videos: Optional[torch.Tensor] = None,
        diff_video_grid_thw: Optional[torch.LongTensor] = None,
        diff_pixel_values: Optional[torch.Tensor] = None,
        diff_image_grid_thw: Optional[torch.LongTensor] = None,
        diff_image_counts: Optional[torch.LongTensor] = None,
        motion_vectors: Optional[torch.Tensor] = None,
        residuals: Optional[torch.Tensor] = None,
        iframe_frames: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple[object, ...], CausalLMOutputWithPast]:
        """
        扩展的 forward, 增加了 diff_frames 参数。

        策略 (兼容 gradient checkpointing):
        1. 手动调用 ViT 编码视觉 tokens
        2. 手动调用 embed_tokens 得到 text embedding
        3. 将视觉 tokens 替换到对应位置 (模仿父类逻辑)
        4. 将 motion tokens 替换到 motion_pad 位置 (我们的逻辑)
        5. 传 inputs_embeds (而非 input_ids + pixel_values) 给父类
           这样 motion tokens 在计算图的显式路径上, 不受 checkpointing 影响
        """
        motion_encoder_type = getattr(self.motion_module, "motion_encoder_type", "resnet")

        # Step 1: 先取 keyframe 的视觉特征；shared_vit 还会复用它作为 keyframe context
        visual_features = self._get_visual_features(
            pixel_values, pixel_values_videos,
            image_grid_thw, video_grid_thw,
        )

        # Step 2: 编码运动输入 -> Motion Tokens
        motion_tokens = None
        has_motion_inputs = (
            diff_frames is not None
            or diff_pixel_values_videos is not None
            or diff_pixel_values is not None
            or motion_vectors is not None
            or residuals is not None
            or iframe_frames is not None
        )
        if has_motion_inputs:
            if motion_encoder_type == "shared_vit":
                shared_motion_inputs, shared_kwargs = self._prepare_shared_vit_motion_inputs(
                    diff_pixel_values_videos=diff_pixel_values_videos,
                    diff_video_grid_thw=diff_video_grid_thw,
                    diff_pixel_values=diff_pixel_values,
                    diff_image_grid_thw=diff_image_grid_thw,
                    diff_image_counts=diff_image_counts,
                    keyframe_visual_features=visual_features,
                    keyframe_video_grid_thw=video_grid_thw,
                )
                if shared_motion_inputs is not None:
                    motion_tokens = self.encode_motion(
                        diff_frames=shared_motion_inputs,
                        **shared_kwargs,
                    )
            else:
                iframe_context = self._build_compressed_iframe_context(
                    visual_features,
                    video_grid_thw,
                    diff_frames=diff_frames,
                    motion_vectors=motion_vectors,
                    residuals=residuals,
                    num_diffs_per_keyframe=self.num_motion_tokens_per_keyframe,
                ) if getattr(self.config, "motion_input_mode", "rgb_diff") == "compressed" else None
                motion_tokens = self.encode_motion(
                    diff_frames=diff_frames,
                    motion_vectors=motion_vectors,
                    residuals=residuals,
                    iframe_context=iframe_context,
                    iframe_frames=iframe_frames,
                )

        # 如果没有 motion tokens, 完全走父类原始逻辑
        if motion_tokens is None:
            return super().forward(
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

        # ---- 有 Motion Tokens: 手动做 embedding + 注入, 绕过 hook ----

        # Step 3: 获取 text embedding
        embed_fn = self._get_embed_tokens()
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = embed_fn(input_ids)

        # Step 4: 将视觉 tokens 替换到 inputs_embeds 中 (模仿父类行为)
        if visual_features is not None and input_ids is not None:
            if inputs_embeds is None:
                raise ValueError("存在 visual_features 但 inputs_embeds 为空，无法注入视觉 token")

            # 找到 image/video placeholder 位置并替换
            # Qwen2.5-VL 使用特定的 token range 来标记视觉区域
            # 参考父类: 通过 image_token_id / video_token_id 定位
            image_token_id = getattr(self.config, "image_token_id", None)
            video_token_id = getattr(self.config, "video_token_id", None)

            if video_token_id is not None and pixel_values_videos is not None:
                video_mask = input_ids == video_token_id
                inputs_embeds[video_mask] = visual_features.to(inputs_embeds.dtype)
            elif image_token_id is not None and pixel_values is not None:
                image_mask = input_ids == image_token_id
                inputs_embeds[image_mask] = visual_features.to(inputs_embeds.dtype)

        # Step 5: 注入 Motion Tokens (完全可微, 在计算图显式路径上)
        inputs_embeds = self._inject_motion_tokens(input_ids, inputs_embeds, motion_tokens)

        # Step 6: 调用父类 forward, 传 inputs_embeds 而非 input_ids
        # 注意: input_ids 设为 None, pixel_values 也设为 None (已手动处理)
        return super().forward(
            input_ids=None,  # 已手动 embed
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=None,  # 已手动处理
            pixel_values_videos=None,  # 已手动处理
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

    def forward_alignment(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        diff_frames: Optional[torch.Tensor] = None,
        diff_pixel_values_videos: Optional[torch.Tensor] = None,
        diff_video_grid_thw: Optional[torch.LongTensor] = None,
        diff_pixel_values: Optional[torch.Tensor] = None,
        diff_image_grid_thw: Optional[torch.LongTensor] = None,
        diff_image_counts: Optional[torch.LongTensor] = None,
        motion_vectors: Optional[torch.Tensor] = None,
        residuals: Optional[torch.Tensor] = None,
        iframe_frames: Optional[torch.Tensor] = None,
        target_pixel_values_videos: Optional[torch.Tensor] = None,
        target_video_grid_thw: Optional[torch.LongTensor] = None,
        target_interval_counts: Optional[torch.LongTensor] = None,
        num_diffs_per_keyframe: int = 4,
        **kwargs,
    ) -> dict[str, object]:
        """
        Stage 1 专用: InfoNCE 对比学习 Alignment Forward。

        不走 LLM 的 next-token prediction,
        而是用对比学习将 motion tokens 与 ViT visual tokens 对齐到同一表示空间。

        对齐策略 (改进版, 参考 FAME/MSCL/ViCLIP):
        1. ViT 编码关键帧 → visual_features [N_vis, D] (冻结, detach)
        2. ResNet 编码差分图 → motion_tokens [B, T_motion, D] (训练)
        3. 将 visual_features 按 keyframe interval 分组, 每组 mean pool → [N_kf, D]
        4. Target = ViT 帧间差: vf[i+1] - vf[i] (反映"变化"而非"内容")
        5. Motion interval i 的 tokens mean pool → query
        6. InfoNCE 对比学习: query_i 与 target_i 为正样本对, 其他为负样本

        Loss = 双向 InfoNCE + 轻量 cosine 辅助:
        $$\\mathcal{L} = \\frac{1}{2}(CE_{m→v} + CE_{v→m}) + 0.1 \\cdot (1 - cos(m, v))$$

        优势 (相比旧版 cosine + SmoothL1):
        - 帧差 target: 让 motion encoder 学习"什么变了", 而非"场景是什么"
        - InfoNCE 负样本: 防止 motion encoder 退化为 ViT 复读机
        - 可学习温度: 自动调节对比学习难度

        Returns:
            dict with keys: "loss", "contrastive_loss", "cosine_loss",
                           "avg_cosine_sim", "logit_scale", "num_pairs"
        """
        device = next(self.motion_module.parameters()).device

        motion_encoder_type = getattr(self.motion_module, "motion_encoder_type", "resnet")
        if (
            diff_frames is None
            and diff_pixel_values_videos is None
            and diff_pixel_values is None
            and motion_vectors is None
            and residuals is None
            and iframe_frames is None
        ):
            logger.warning("[Alignment] no motion inputs provided")
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        # Step 1: ViT 编码 keyframe visual features (冻结, detach)
        with torch.no_grad():
            visual_features = self._get_visual_features(
                pixel_values, pixel_values_videos,
                image_grid_thw, video_grid_thw,
            )

        if visual_features is None:
            logger.warning("[Alignment] visual_features is None (pixel_values=%s, pixel_values_videos=%s)",
                          pixel_values is not None, pixel_values_videos is not None)
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        # visual_features: [N_vis_total, D] (所有帧的 patch tokens 拼在一起)
        visual_features = visual_features.detach()  # [N_vis_total, D]

        # Step 2: 编码 motion tokens (Stage 1 alignment 用 align_proj 对齐到 ViT 空间)
        motion_stream_tokens = None
        if motion_encoder_type == "shared_vit":
            shared_motion_inputs, shared_kwargs = self._prepare_shared_vit_motion_inputs(
                diff_pixel_values_videos=diff_pixel_values_videos,
                diff_video_grid_thw=diff_video_grid_thw,
                diff_pixel_values=diff_pixel_values,
                diff_image_grid_thw=diff_image_grid_thw,
                diff_image_counts=diff_image_counts,
                keyframe_visual_features=visual_features,
                keyframe_video_grid_thw=video_grid_thw,
            )
            if shared_motion_inputs is None:
                logger.warning("[Alignment] shared_vit 的 diff visual features 为空")
                return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}
            motion_tokens = self.motion_module.encode_motion_for_alignment(
                diff_frames=shared_motion_inputs,
                **shared_kwargs,
            )
        else:
            iframe_context = self._build_compressed_iframe_context(
                visual_features,
                video_grid_thw,
                diff_frames=diff_frames,
                motion_vectors=motion_vectors,
                residuals=residuals,
                num_diffs_per_keyframe=num_diffs_per_keyframe,
            ) if getattr(self.config, "motion_input_mode", "rgb_diff") == "compressed" else None
            if getattr(self.config, "motion_input_mode", "rgb_diff") == "compressed" and hasattr(self.motion_module, "encode_motion_for_alignment_components"):
                motion_stream_tokens = self.motion_module.encode_motion_for_alignment_components(
                    diff_frames=diff_frames,
                    motion_vectors=motion_vectors,
                    residuals=residuals,
                    iframe_context=iframe_context,
                    iframe_frames=iframe_frames,
                )
                motion_tokens = motion_stream_tokens["fused"]
            else:
                motion_tokens = self.motion_module.encode_motion_for_alignment(
                    diff_frames=diff_frames,
                    motion_vectors=motion_vectors,
                    residuals=residuals,
                    iframe_context=iframe_context,
                    iframe_frames=iframe_frames,
                )  # [B, T_motion, vit_D]
        if motion_tokens is None:
            logger.warning("[Alignment] motion_tokens is None after motion encoding")
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}
        B, T_motion = motion_tokens.shape[:2]

        logger.info(
            "[Alignment] visual_features=%s, motion_tokens=%s, video_grid_thw=%s, image_grid_thw=%s, diff_frames=%s, motion_vectors=%s, residuals=%s",
            visual_features.shape,
            motion_tokens.shape,
            video_grid_thw if video_grid_thw is not None else "None",
            image_grid_thw if image_grid_thw is not None else "None",
            diff_frames.shape if diff_frames is not None else "None",
            motion_vectors.shape if motion_vectors is not None else "None",
            residuals.shape if residuals is not None else "None",
        )

        # Step 3: 将 visual tokens 按帧分组并 mean pool
        spatial_merge_size = getattr(self.config, "vision_config", None)
        if spatial_merge_size is not None:
            spatial_merge_size = getattr(spatial_merge_size, "spatial_merge_size", 2)
        else:
            spatial_merge_size = 2

        if video_grid_thw is not None:
            thw = video_grid_thw  # [num_videos, 3]
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
                        frame_feat = visual_features[start:end].mean(dim=0)  # [D]
                        per_frame_targets.append(frame_feat)
                offset += num_temporal * tokens_per_frame

            if not per_frame_targets:
                logger.warning("[Alignment] per_frame_targets is empty after ViT split "
                              "(spatial_merge_size=%d, thw=%s, visual_features=%s)",
                              spatial_merge_size, thw, visual_features.shape)
                return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

            per_frame_targets = torch.stack(per_frame_targets)  # [N_frames, D]
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
            per_frame_targets = torch.stack(per_frame_targets)  # [N_frames, D]
        else:
            logger.warning("[Alignment] both video_grid_thw and image_grid_thw are None")
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        n_visual_frames = per_frame_targets.shape[0]
        logger.info("[Alignment] n_visual_frames=%d, B=%d, T_motion=%d, num_diffs_per_keyframe=%d",
                    n_visual_frames, B, T_motion, num_diffs_per_keyframe)

        decoded_interval_targets: List[torch.Tensor] = []
        if target_pixel_values_videos is not None and target_video_grid_thw is not None:
            with torch.no_grad():
                decoded_visual_features = self._get_visual_features(
                    pixel_values=None,
                    pixel_values_videos=target_pixel_values_videos,
                    image_grid_thw=None,
                    video_grid_thw=target_video_grid_thw,
                )
            if decoded_visual_features is None:
                logger.warning("[Alignment] decoded target visual features is None")
            else:
                decoded_interval_targets = self._split_video_features_per_sample(
                    decoded_visual_features.detach(),
                    target_video_grid_thw,
                )

        # Step 4: 构造两路正样本
        # 1) interval-diff: pooled motion token -> ViT 相邻关键帧特征差
        # 2) decoded-frame distill: (start I-frame + k motion tokens) -> 对应 k+1 张解码帧特征
        alignment_pool = getattr(self.config, "alignment_motion_pool", "mean")
        interval_diff_weight = float(getattr(self.config, "alignment_interval_diff_weight", 1.0))
        decoded_frame_weight = float(getattr(self.config, "alignment_decoded_frame_weight", 1.0))
        stream_token_map = motion_stream_tokens or {"fused": motion_tokens}
        stream_token_map = {k: v for k, v in stream_token_map.items() if v is not None}

        interval_query_buckets = {k: [] for k in stream_token_map}
        interval_visual_targets = []
        interval_target_group_ids = []
        interval_sample_ids = []
        interval_ids = []

        decoded_query_buckets = {k: [] for k in stream_token_map}
        decoded_visual_target_list = []
        decoded_target_group_ids = []
        decoded_sample_ids = []
        decoded_interval_ids = []

        target_interval_counts_list = target_interval_counts.tolist() if target_interval_counts is not None else None
        decoded_cursor = 0
        group_offset = 0

        def _build_temporal_negative_mask(
            group_ids_list: List[int],
            sample_ids_list: List[int],
            interval_ids_list: List[int],
            device: torch.device,
        ) -> Optional[torch.Tensor]:
            if not group_ids_list:
                return None
            group_ids_tensor = torch.tensor(group_ids_list, device=device, dtype=torch.long)
            sample_ids_tensor = torch.tensor(sample_ids_list, device=device, dtype=torch.long)
            interval_ids_tensor = torch.tensor(interval_ids_list, device=device, dtype=torch.long)
            positive_mask = group_ids_tensor[:, None].eq(group_ids_tensor[None, :])
            same_sample = sample_ids_tensor[:, None].eq(sample_ids_tensor[None, :])
            far_temporal = (interval_ids_tensor[:, None] - interval_ids_tensor[None, :]).abs() > 1
            negative_mask = (~same_sample) | (same_sample & far_temporal)
            negative_mask = negative_mask & (~positive_mask)
            return negative_mask

        def _metric_tensor(result: dict[str, object], key: str = "loss") -> torch.Tensor:
            value = result.get(key)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"alignment result[{key!r}] 不是 Tensor: {type(value)!r}")
            return value

        def _run_alignment_route(
            route_name: str,
            query_buckets: dict[str, list[torch.Tensor]],
            visual_target_list: list[torch.Tensor],
            group_ids_list: list[int],
            sample_ids_list: list[int],
            interval_ids_list: list[int],
        ) -> Optional[dict[str, object]]:
            fused_queries = query_buckets.get("fused")
            if not fused_queries or not visual_target_list:
                return None

            motion_queries = torch.stack(fused_queries)
            motion_query_map = {
                name: torch.stack(queries)
                for name, queries in query_buckets.items()
                if queries and len(queries) == len(fused_queries)
            }
            visual_targets = torch.stack(visual_target_list).detach()
            target_group_ids_tensor = torch.tensor(group_ids_list, device=motion_queries.device, dtype=torch.long)
            negative_mask = _build_temporal_negative_mask(
                group_ids_list,
                sample_ids_list,
                interval_ids_list,
                motion_queries.device,
            )

            compute_dtype = motion_queries.dtype
            visual_targets = visual_targets.to(dtype=compute_dtype)
            motion_query_map = {
                name: queries.to(dtype=compute_dtype)
                for name, queries in motion_query_map.items()
            }
            if negative_mask is not None:
                negative_mask = negative_mask.to(device=motion_queries.device, dtype=torch.bool)
            num_unique_targets = int(target_group_ids_tensor.unique().numel())

            logger.info(
                "[Alignment][%s] num_pairs=%d, unique_targets=%d, streams=%s",
                route_name,
                motion_queries.shape[0],
                num_unique_targets,
                sorted(motion_query_map.keys()),
            )

            if getattr(self.config, "motion_input_mode", "rgb_diff") == "compressed" and hasattr(self.motion_module, "compute_multi_stream_alignment_loss"):
                route_result = self.motion_module.compute_multi_stream_alignment_loss(
                    motion_query_map,
                    visual_targets,
                    target_group_ids=target_group_ids_tensor,
                    negative_mask=negative_mask,
                )
            else:
                route_result = self.motion_module.compute_alignment_loss(
                    motion_queries,
                    visual_targets,
                    target_group_ids=target_group_ids_tensor,
                    negative_mask=negative_mask,
                )

            route_result["num_pairs"] = int(motion_queries.shape[0])
            route_result["num_unique_targets"] = num_unique_targets
            route_result["negative_mode"] = "local_temporal_batch"
            return route_result

        for b in range(B):
            sample_streams = {name: tokens[b] for name, tokens in stream_token_map.items()}
            mt = sample_streams["fused"]

            n_vf = n_visual_frames // max(B, 1)
            start_vf = b * n_vf
            vf_targets = per_frame_targets[start_vf:start_vf + n_vf]
            sample_group_count = max(n_vf - 1, 1)

            sample_target_count = 0
            sample_decoded_targets: List[torch.Tensor] = []
            if target_interval_counts_list is not None and b < len(target_interval_counts_list):
                sample_target_count = int(target_interval_counts_list[b])
                sample_decoded_targets = decoded_interval_targets[decoded_cursor:decoded_cursor + sample_target_count]
                decoded_cursor += sample_target_count

            if n_vf < 2 or T_motion == 0:
                if n_vf >= 1 and T_motion > 0:
                    for i in range(T_motion):
                        for stream_name, stream_tokens in sample_streams.items():
                            if i < stream_tokens.shape[0]:
                                interval_query_buckets[stream_name].append(stream_tokens[i])
                        interval_visual_targets.append(vf_targets[0])
                        interval_target_group_ids.append(group_offset)
                        interval_sample_ids.append(b)
                        interval_ids.append(i)
                group_offset += sample_group_count
                continue

            n_motion_intervals = T_motion // max(num_diffs_per_keyframe, 1)
            if n_motion_intervals == 0:
                n_motion_intervals = 1
            n_visual_intervals = n_vf - 1

            if sample_target_count not in {0, n_motion_intervals}:
                logger.warning(
                    "[Alignment] sample %d decoded target interval 数不匹配: decoded=%d, motion=%d",
                    b,
                    sample_target_count,
                    n_motion_intervals,
                )

            for mi in range(n_motion_intervals):
                mt_start = mi * num_diffs_per_keyframe
                mt_end = min(mt_start + num_diffs_per_keyframe, T_motion)
                if mt_start >= T_motion:
                    break

                vi = min(int(mi * n_visual_intervals / n_motion_intervals), n_visual_intervals - 1)
                target_group_id = group_offset + vi
                interval_target = vf_targets[vi + 1] - vf_targets[vi]

                for stream_name, stream_tokens in sample_streams.items():
                    stream_end = min(mt_end, stream_tokens.shape[0])
                    interval_tokens = stream_tokens[mt_start:stream_end]
                    if interval_tokens.shape[0] == 0:
                        continue
                    interval_query_buckets[stream_name].append(
                        pool_motion_tokens(interval_tokens, mode=alignment_pool)
                    )

                    if decoded_frame_weight > 0 and mi < len(sample_decoded_targets):
                        anchor_tokens = torch.cat(
                            [vf_targets[vi].to(dtype=interval_tokens.dtype).unsqueeze(0), interval_tokens],
                            dim=0,
                        )
                        decoded_query_buckets[stream_name].append(
                            pool_motion_tokens(anchor_tokens, mode=alignment_pool)
                        )

                interval_visual_targets.append(interval_target)
                interval_target_group_ids.append(target_group_id)
                interval_sample_ids.append(b)
                interval_ids.append(vi)

                if decoded_frame_weight > 0 and mi < len(sample_decoded_targets):
                    decoded_target_frames = sample_decoded_targets[mi]
                    if decoded_target_frames.shape[0] > 0:
                        decoded_visual_target_list.append(decoded_target_frames.mean(dim=0))
                        decoded_target_group_ids.append(target_group_id)
                        decoded_sample_ids.append(b)
                        decoded_interval_ids.append(vi)

            group_offset += sample_group_count

        if decoded_cursor != len(decoded_interval_targets):
            logger.warning(
                "[Alignment] decoded target interval 消耗数不一致: consumed=%d, total=%d",
                decoded_cursor,
                len(decoded_interval_targets),
            )

        interval_result = _run_alignment_route(
            "interval_diff",
            interval_query_buckets,
            interval_visual_targets,
            interval_target_group_ids,
            interval_sample_ids,
            interval_ids,
        ) if interval_diff_weight > 0 else None
        decoded_result = _run_alignment_route(
            "decoded_frame",
            decoded_query_buckets,
            decoded_visual_target_list,
            decoded_target_group_ids,
            decoded_sample_ids,
            decoded_interval_ids,
        ) if decoded_frame_weight > 0 else None

        if interval_result is None and decoded_result is None:
            logger.warning("[Alignment] no alignment pairs constructed for both routes")
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        loss_terms: list[torch.Tensor] = []
        if interval_result is not None and interval_diff_weight > 0:
            loss_terms.append(_metric_tensor(interval_result).float() * interval_diff_weight)
        if decoded_result is not None and decoded_frame_weight > 0:
            loss_terms.append(_metric_tensor(decoded_result).float() * decoded_frame_weight)

        if not loss_terms:
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        base_result = interval_result if interval_result is not None else decoded_result
        if base_result is None:
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        align_result = dict(base_result)
        base_loss = _metric_tensor(base_result)
        align_result["loss"] = torch.stack(loss_terms).sum().to(base_loss.dtype)
        align_result["alignment_interval_diff_weight"] = interval_diff_weight
        align_result["alignment_decoded_frame_weight"] = decoded_frame_weight

        if interval_result is not None:
            align_result["interval_diff_alignment_loss"] = _metric_tensor(interval_result).to(base_loss.dtype)
            align_result["interval_diff_avg_cosine_sim"] = interval_result.get("avg_cosine_sim")
            align_result["interval_num_pairs"] = interval_result.get("num_pairs", 0)
            align_result["interval_num_unique_targets"] = interval_result.get("num_unique_targets", 0)
        if decoded_result is not None:
            align_result["decoded_frame_alignment_loss"] = _metric_tensor(decoded_result).to(base_loss.dtype)
            align_result["decoded_frame_avg_cosine_sim"] = decoded_result.get("avg_cosine_sim")
            align_result["decoded_frame_num_pairs"] = decoded_result.get("num_pairs", 0)
            align_result["decoded_frame_num_unique_targets"] = decoded_result.get("num_unique_targets", 0)

        if interval_result is not None:
            align_result["num_pairs"] = interval_result.get("num_pairs", 0)
            align_result["num_unique_targets"] = interval_result.get("num_unique_targets", 0)
        elif decoded_result is not None:
            align_result["num_pairs"] = decoded_result.get("num_pairs", 0)
            align_result["num_unique_targets"] = decoded_result.get("num_unique_targets", 0)

        return align_result

    def configure_training(self, stage: int = 1, sft_mode: str = "lora", freeze_spatial_encoder: bool = False):
        """
        配置训练阶段的参数冻结策略。

        Stage 1 (Alignment, ~3h):
            - 冻结: LLM 全部, Qwen-ViT 全部
            - 训练: MotionEncoder + MotionProjector
            - 目的: 把 ResNet 特征对齐到 LLM 语义空间

        Stage 2 (SFT):
            sft_mode="lora" (~15h):
                - 冻结: Qwen-ViT
                - 训练: MotionEncoder + MotionProjector + LLM (LoRA)
            sft_mode="full":
                - 冻结: Qwen-ViT
                - 训练: MotionEncoder + MotionProjector + LLM (全量)
        """
        # 先全部冻结
        freeze_module(self)

        if stage == 1:
            self.motion_module.configure_training_state(
                train_alignment_head=True,
                train_projector=False,
                train_alignment_projector=True,
                freeze_spatial_encoder=freeze_spatial_encoder,
            )
            logger.info("=== Stage 1: Alignment ===")
            logger.info("训练: MotionSSM + Alignment MLP + AlignmentHead")
            logger.info("冻结: LLM + Qwen-ViT")
            if freeze_spatial_encoder:
                logger.info("Stage 1 warmup: 先冻结 MotionSSM spatial_encoder，仅训练 temporal + alignment 分支")

        elif stage == 2:
            self.motion_module.configure_training_state(
                train_alignment_head=False,
                train_projector=True,
                train_alignment_projector=False,
            )
            if sft_mode == "full":
                # 全量微调: 解冻 LLM 全部参数 (ViT 保持冻结)
                unfreeze_module(self.model)  # Qwen2.5-VL 的 LLM 部分
                visual_encoder = self._get_visual_encoder()
                if visual_encoder is not None:
                    freeze_module(visual_encoder)  # 重新冻结 ViT
                logger.info("=== Stage 2: Full SFT ===")
                logger.info("训练: MotionEncoder + MotionProjector + LLM (全量)")
                logger.info("冻结: Qwen-ViT")
            else:
                logger.info("=== Stage 2: SFT with LoRA ===")
                logger.info("训练: MotionEncoder + MotionProjector + LLM (LoRA)")
                logger.info("冻结: Qwen-ViT (原始参数)")

        else:
            raise ValueError(f"未知的训练阶段: {stage}")

        info = get_trainable_params_info(self)
        logger.info("参数统计: %s", info)

        return self

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        diff_frames: Optional[torch.Tensor] = None,
        diff_pixel_values_videos: Optional[torch.Tensor] = None,
        diff_video_grid_thw: Optional[torch.LongTensor] = None,
        diff_pixel_values: Optional[torch.Tensor] = None,
        diff_image_grid_thw: Optional[torch.LongTensor] = None,
        diff_image_counts: Optional[torch.LongTensor] = None,
        motion_vectors: Optional[torch.Tensor] = None,
        residuals: Optional[torch.Tensor] = None,
        iframe_frames: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ):
        """
        Generate with motion tokens injected into the initial inputs_embeds.

        对于有 diff_frames 的情况, 先手动构建 inputs_embeds (含 motion tokens),
        再调用父类 generate (传 inputs_embeds 而非 input_ids)。
        """
        motion_encoder_type = getattr(self.motion_module, "motion_encoder_type", "resnet")
        if (
            diff_frames is None
            and diff_pixel_values_videos is None
            and diff_pixel_values is None
            and motion_vectors is None
            and residuals is None
            and iframe_frames is None
        ):
            return super().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                **generate_kwargs,
            )

        # 编码 motion tokens
        visual_features = self._get_visual_features(
            pixel_values, pixel_values_videos,
            image_grid_thw, video_grid_thw,
        )
        if motion_encoder_type == "shared_vit":
            shared_motion_inputs, shared_kwargs = self._prepare_shared_vit_motion_inputs(
                diff_pixel_values_videos=diff_pixel_values_videos,
                diff_video_grid_thw=diff_video_grid_thw,
                diff_pixel_values=diff_pixel_values,
                diff_image_grid_thw=diff_image_grid_thw,
                diff_image_counts=diff_image_counts,
                keyframe_visual_features=visual_features,
                keyframe_video_grid_thw=video_grid_thw,
            )
            if shared_motion_inputs is None:
                raise ValueError("shared_vit 无法从 diff 输入构造 motion tokens")
            motion_tokens = self.encode_motion(
                diff_frames=shared_motion_inputs,
                **shared_kwargs,
            )
        else:
            iframe_context = self._build_compressed_iframe_context(
                visual_features,
                video_grid_thw,
                diff_frames=diff_frames,
                motion_vectors=motion_vectors,
                residuals=residuals,
                num_diffs_per_keyframe=self.num_motion_tokens_per_keyframe,
            ) if getattr(self.config, "motion_input_mode", "rgb_diff") == "compressed" else None
            motion_tokens = self.encode_motion(
                diff_frames=diff_frames,
                motion_vectors=motion_vectors,
                residuals=residuals,
                iframe_context=iframe_context,
                iframe_frames=iframe_frames,
            )

        # 手动构建 inputs_embeds

        embed_fn = self._get_embed_tokens()
        inputs_embeds = embed_fn(input_ids)

        # 替换视觉 tokens
        if visual_features is not None and input_ids is not None:
            image_token_id = getattr(self.config, "image_token_id", None)
            video_token_id = getattr(self.config, "video_token_id", None)
            if video_token_id is not None and pixel_values_videos is not None:
                video_mask = input_ids == video_token_id
                inputs_embeds[video_mask] = visual_features.to(inputs_embeds.dtype)
            elif image_token_id is not None and pixel_values is not None:
                image_mask = input_ids == image_token_id
                inputs_embeds[image_mask] = visual_features.to(inputs_embeds.dtype)

        # 注入 motion tokens
        inputs_embeds = self._inject_motion_tokens(input_ids, inputs_embeds, motion_tokens)

        return super().generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **generate_kwargs,
        )
