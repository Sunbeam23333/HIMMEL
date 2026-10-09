#!/usr/bin/env python3
"""
Himmel-VLM: LLaVA-OneVision 双流魔改
=======================================
核心改动:
1. 继承 LlavaOnevisionForConditionalGeneration (transformers >= 4.45)
2. 新增 MotionEncoder + Projector
3. 在 prepare_inputs_for_generation / forward 阶段交错拼接
   [Keyframe_1] [Motion_1_1] [Motion_1_2] ... [Keyframe_2] [Motion_2_1] ...

LLaVA-OneVision 与 Qwen2.5-VL 的关键差异:
- LLaVA 使用 SigLIP 作为视觉编码器, 输出是 flat 的 image tokens
- LLaVA 使用标准 RoPE (1D), 不需要处理 3D 位置编码
- LLaVA 的 multi_modal_projector 将 SigLIP 特征映射到 LLM 空间
- 视频处理: LLaVA-OneVision 把多帧拼成一个序列送入 SigLIP
"""
from __future__ import annotations

import logging
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionConfig
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

# 特殊 Token
MOTION_START_TOKEN = "<|motion_start|>"
MOTION_END_TOKEN = "<|motion_end|>"
MOTION_PAD_TOKEN = "<|motion_pad|>"


class LlavaDeltaForConditionalGeneration(LlavaOnevisionForConditionalGeneration):
    """
    LLaVA-OneVision + 双流运动编码。

    相比 Qwen 版本更简单:
    - 不需要处理 M-RoPE, 标准 1D RoPE 自动递增
    - 在 inputs_embeds 层面直接替换 motion_pad 占位符即可

    架构:
    ┌─────────────┐    ┌──────────────┐
    │ Keyframes   │    │  Diff-Maps   │
    │ (稀疏, 1fps)│    │ (密集, 4x)   │
    └──────┬──────┘    └──────┬───────┘
           │                  │
      SigLIP (冻结)       ResNet-18 (训练)
           │                  │
      Image Tokens       Motion Feat
           │                  │
      MM Projector       MLP Projector (训练)
      (冻结/微调)             │
           │            Motion Tokens
           └────────┬─────────┘
                    │
              交错拼接 (1D RoPE 自动处理)
                    │
              Qwen2-7B LLM (LoRA)
                    │
                  Output
    """

    def __init__(self, config: LlavaOnevisionConfig):
        super().__init__(config)

        # 新增: 运动编码模块
        # LLaVA-OneVision-7B 使用 Qwen2-7B 作为 LLM, hidden_size=3584
        # 但不同版本可能不同, 从 config 获取
        llm_hidden_size = config.text_config.hidden_size

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
            raise NotImplementedError("This archived LLaVA wrapper does not forward appearance-branch inputs; use the Qwen2.5 tri-stream implementation.")
        # LLaVA-OneVision 的 SigLIP ViT hidden_size: 1152 (用于 shared_vit)
        vit_hidden_size = getattr(config, "motion_vit_hidden_size", 1152)
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

        # 特殊 token IDs
        self.motion_start_token_id: Optional[int] = None
        self.motion_end_token_id: Optional[int] = None
        self.motion_pad_token_id: Optional[int] = None

    def setup_motion_token_ids(self, tokenizer):
        """注册特殊 motion tokens。"""
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
            logger.info("添加 %d 个特殊 token, 词表: %d", num_added, len(tokenizer))

        self.motion_start_token_id = tokenizer.convert_tokens_to_ids(MOTION_START_TOKEN)
        self.motion_end_token_id = tokenizer.convert_tokens_to_ids(MOTION_END_TOKEN)
        self.motion_pad_token_id = tokenizer.convert_tokens_to_ids(MOTION_PAD_TOKEN)

    def encode_motion(
        self,
        diff_frames: Optional[torch.Tensor] = None,
        motion_vectors: Optional[torch.Tensor] = None,
        residuals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """编码 RGB diff 或真实压缩域双流为 Motion Tokens。"""
        return self.motion_module(
            diff_frames=diff_frames,
            motion_vectors=motion_vectors,
            residuals=residuals,
        )

    def _replace_motion_placeholders(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        motion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 motion_pad 占位符替换为真正的 Motion Tokens。

        LLaVA 的流程比 Qwen 简单:
        - 不需要处理 3D position_ids
        - 1D RoPE 会自动按序列位置递增
        - 只需要在 embedding 层面替换即可

        注意: 使用 scatter 而非 in-place 索引赋值以保持梯度流。
        Stage 1 冻结 LLM 时, inputs_embeds 无 requires_grad,
        in-place 赋值不会为 motion_tokens 建立梯度图。
        """
        B, S, D = inputs_embeds.shape
        device = inputs_embeds.device
        new_embeds_list = []

        for b in range(B):
            motion_mask = (input_ids[b] == self.motion_pad_token_id)
            motion_indices = motion_mask.nonzero(as_tuple=True)[0]

            if len(motion_indices) == 0:
                new_embeds_list.append(inputs_embeds[b])
                continue

            mt = motion_tokens[b] if motion_tokens.dim() == 3 else motion_tokens
            num_motion = min(len(motion_indices), mt.shape[0])

            embeds = inputs_embeds[b].detach()
            padded_mt = mt[:num_motion].to(dtype=embeds.dtype)  # [num_motion, D]
            one_hot = torch.zeros(S, num_motion, device=device, dtype=embeds.dtype)
            one_hot[motion_indices[:num_motion], torch.arange(num_motion, device=device)] = 1.0
            motion_contrib = one_hot @ padded_mt  # [S, D]
            mask_f = motion_mask.unsqueeze(-1).to(embeds.dtype)
            embeds = embeds * (1.0 - mask_f) + motion_contrib
            new_embeds_list.append(embeds)

        return torch.stack(new_embeds_list, dim=0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # ---- 新增参数 ----
        diff_frames: Optional[torch.Tensor] = None,
        motion_vectors: Optional[torch.Tensor] = None,
        residuals: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        扩展 forward, 增加双流运动编码。

        处理流程:
        1. 编码差分图 -> Motion Tokens
        2. 父类处理视觉输入 (SigLIP + MM Projector)
        3. 在 inputs_embeds 中替换 motion 占位符
        4. 送入 LLM
        """
        # Step 1: 编码差分图
        motion_tokens = None
        if diff_frames is not None:
            motion_tokens = self.encode_motion(diff_frames)

        # 无 motion tokens 时, 直接走原始逻辑
        if motion_tokens is None:
            return super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # ---- 有 Motion Tokens: 手动控制流程 ----

        # Step 2: 生成 inputs_embeds (文本 + 视觉)
        # LLaVA-OneVision 的逻辑:
        # a) SigLIP 编码 pixel_values -> image_features
        # b) MM Projector 映射到 LLM 空间
        # c) 替换 input_ids 中的 image token 占位符
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                # 用 vision_tower 编码图像
                image_features = self._get_vision_features(
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )

                # 替换图像占位符
                special_image_mask = (
                    (input_ids == self.config.image_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_features = image_features.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_image_mask, image_features
                )

        # Step 3: 替换 motion 占位符
        inputs_embeds = self._replace_motion_placeholders(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            motion_tokens=motion_tokens,
        )

        # Step 4: 送入 LLM
        return super().forward(
            input_ids=None,  # 用 inputs_embeds 代替
            pixel_values=None,  # 已处理
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def _get_vision_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: Optional[torch.Tensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ) -> torch.Tensor:
        """
        提取视觉特征并通过 MM Projector 映射。

        LLaVA-OneVision 的 pixel_values 可能是:
          - 4D: [B, C, H, W]  (单 patch)
          - 5D: [num_images, num_patches, C, H, W]  (anyres, 每帧多个 patch)
        SigLIP 只接受 4D, 因此 5D 时需要先 flatten 再 reshape 回来。
        """
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        # 处理 5D pixel_values: [N, P, C, H, W] -> [N*P, C, H, W]
        needs_reshape = pixel_values.dim() == 5
        if needs_reshape:
            N, P, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(N * P, C, H, W)

        # 兼容不同版本 transformers: vision_tower 可能在 self 或 self.model 下
        vision_tower = getattr(self, "vision_tower", None)
        if vision_tower is None:
            vision_tower = getattr(getattr(self, "model", None), "vision_tower", None)
        if vision_tower is None:
            raise AttributeError("无法找到 vision_tower (self.vision_tower / self.model.vision_tower)")

        image_outputs = vision_tower(
            pixel_values,
            output_hidden_states=True,
        )
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature

        # 兼容不同版本 transformers: multi_modal_projector 可能在 self 或 self.model 下
        projector = getattr(self, "multi_modal_projector", None)
        if projector is None:
            projector = getattr(getattr(self, "model", None), "multi_modal_projector", None)
        if projector is None:
            raise AttributeError("无法找到 multi_modal_projector")

        image_features = projector(selected_image_feature)

        # 5D 时 reshape 回: [N*P, tokens, D] -> [N, P*tokens, D]
        if needs_reshape:
            NP, T, D = image_features.shape
            image_features = image_features.view(N, P * T, D)

        return image_features

    def forward_alignment(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        diff_frames: Optional[torch.Tensor] = None,
        num_diffs_per_keyframe: int = 4,
        **kwargs,
    ) -> dict:
        """
        Stage 1 专用: Contrastive/Regression Alignment Forward。

        LLaVA-OneVision 使用 SigLIP 编码器:
        - 无 temporal merge (每帧独立编码)
        - 无 spatial merge (SigLIP 直接输出 patch tokens)
        - 每帧 ~729 tokens (27x27, 去掉 CLS, 或根据 select_strategy)
        - 通过 MM Projector 映射到 LLM 空间后做 alignment

        Returns:
            dict with keys: "loss", "avg_cosine_sim", "num_pairs"
        """
        device = next(self.motion_module.parameters()).device

        if diff_frames is None and motion_vectors is None and residuals is None:
            logger.warning("[Alignment] no motion inputs provided")
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        # Step 1: SigLIP 编码 visual features (冻结, detach)
        # LLaVA 把视频帧当多张图片处理, pixel_values=[N_frames, 3, H, W]
        if pixel_values is None:
            logger.warning("[Alignment] pixel_values is None")
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        with torch.no_grad():
            # image_features: [N_frames, tokens_per_frame, D]
            image_features = self._get_vision_features(pixel_values, image_sizes)

        image_features = image_features.detach()

        # 诊断: 检查 SigLIP 输出是否有 NaN/Inf
        if torch.isnan(image_features).any() or torch.isinf(image_features).any():
            nan_cnt = torch.isnan(image_features).sum().item()
            inf_cnt = torch.isinf(image_features).sum().item()
            logger.warning("[Alignment] image_features 含 NaN=%d, Inf=%d (shape=%s, dtype=%s)",
                          nan_cnt, inf_cnt, image_features.shape, image_features.dtype)
            # 用 float32 重算以避免 bf16 精度问题
            logger.info("[Alignment] 尝试用 float32 重新编码 visual features")
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                pv_fp32 = pixel_values.float()
                image_features = self._get_vision_features(pv_fp32, image_sizes)
            image_features = image_features.detach()
            if torch.isnan(image_features).any():
                logger.warning("[Alignment] float32 重算后仍有 NaN, 跳过此 batch")
                return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        # per-frame mean pool: [N_frames, D]
        per_frame_targets = image_features.mean(dim=1)

        D = per_frame_targets.shape[-1]
        n_visual_frames = per_frame_targets.shape[0]

        # Step 2: 编码 motion tokens (训练)
        motion_tokens = self.encode_motion(
            diff_frames=diff_frames,
            motion_vectors=motion_vectors,
            residuals=residuals,
        )
        B, T_motion = motion_tokens.shape[:2]

        # 诊断: 检查 motion_tokens 是否有 NaN
        if torch.isnan(motion_tokens).any():
            logger.warning("[Alignment] motion_tokens 含 NaN (shape=%s)", motion_tokens.shape)
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        logger.info("[Alignment] per_frame_targets=%s, motion_tokens=%s, n_visual_frames=%d",
                    per_frame_targets.shape, motion_tokens.shape, n_visual_frames)
        logger.info("[Alignment] per_frame_targets range=[%.4f, %.4f], motion_tokens range=[%.4f, %.4f]",
                    per_frame_targets.min().item(), per_frame_targets.max().item(),
                    motion_tokens.min().item(), motion_tokens.max().item())

        # Step 3: 构造对齐 pairs (帧差 target, InfoNCE 对比学习)
        # target = vf[i+1] - vf[i] (ViT 特征空间的"语义级变化")
        # LLaVA 无 temporal merge, 所以 n_visual_frames = 实际关键帧数
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

        # Step 4: 计算 InfoNCE 对比学习 alignment loss
        align_result = self.motion_module.compute_alignment_loss(motion_queries, visual_targets)
        align_result["num_pairs"] = len(all_motion_queries)
        return align_result

    def configure_training(self, stage: int = 1, sft_mode: str = "lora", freeze_spatial_encoder: bool = False):
        """
        配置训练阶段。

        Stage 1 (Alignment):
            训练: MotionEncoder + MotionProjector
            冻结: SigLIP + MM Projector + LLM

        Stage 2 (SFT):
            sft_mode="lora": 训练 MotionEncoder + MotionProjector + LLM LoRA
            sft_mode="full": 训练 MotionEncoder + MotionProjector + LLM (全量)
            冻结: SigLIP + MM Projector (原始参数)
        """
        freeze_module(self)

        if stage == 1:
            self.motion_module.configure_training_state(
                train_alignment_head=True,
                freeze_spatial_encoder=freeze_spatial_encoder,
            )
            logger.info("=== Stage 1: Alignment ===")
            logger.info("训练: MotionEncoder + MotionProjector")
            logger.info("冻结: SigLIP + MM Projector + LLM")

        elif stage == 2:
            self.motion_module.configure_training_state(train_alignment_head=False)
            if sft_mode == "full":
                unfreeze_module(self.language_model)
                logger.info("=== Stage 2: Full SFT ===")
                logger.info("训练: MotionEncoder + MotionProjector + LLM (全量)")
                logger.info("冻结: SigLIP + MM Projector (原始参数)")
            else:
                logger.info("=== Stage 2: SFT with LoRA ===")
                logger.info("训练: MotionEncoder + MotionProjector + LLM LoRA")
                logger.info("冻结: SigLIP + MM Projector (原始参数)")

        else:
            raise ValueError(f"未知训练阶段: {stage}")

        info = get_trainable_params_info(self)
        logger.info("参数统计: %s", info)
        return self

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        diff_frames: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ):
        """
        Generate with motion tokens injected into inputs_embeds.
        """
        if diff_frames is None and motion_vectors is None and residuals is None:
            return super().generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        motion_tokens = self.encode_motion(
            diff_frames=diff_frames,
            motion_vectors=motion_vectors,
            residuals=residuals,
        )

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self._get_vision_features(pixel_values, image_sizes)
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        inputs_embeds = self._replace_motion_placeholders(input_ids, inputs_embeds, motion_tokens)

        return super().generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            pixel_values=None,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
