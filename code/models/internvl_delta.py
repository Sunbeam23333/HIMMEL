#!/usr/bin/env python3
"""
Himmel-VLM: InternVL3 双流魔改
===================================
基于 InternVLChatModel (trust_remote_code) 的双流运动编码改造。

InternVL3 架构要点:
1. ViT-MLP-LLM 范式: InternViT-300M → MLP Projector → Qwen2.5-7B LLM
2. 自定义模型类: InternVLChatModel (非 transformers 标准类, 需 trust_remote_code)
3. 动态分辨率: 图像被切分为 448x448 网格 → Pixel Shuffle 降 4x → 视觉 tokens
4. V2PE (Variable Visual Position Encoding): 灵活位置增量, 类 1D RoPE
5. LLM 底座: Qwen2.5-7B (hidden_size=3584)

对 Himmel 双流改造的影响:
- InternVL3 本质上是 "图像 tokens → LLM" 的简单范式 (类似 LLaVA)
- Motion Token 在 embedding 层面直接替换 motion_pad 占位符即可
- 不需要处理复杂的 3D 位置编码 (V2PE 对新增 token 自动递增)
- forward 通过 InternVLChatModel.forward() → language_model.forward() 的链路

关键实现细节:
- InternVL3 的 forward 入口是 chat() 或直接 forward()
- 视觉编码在 extract_feature() 方法中完成
- 图像 token 通过 IMG_CONTEXT_TOKEN (占位符 ID) 标记, 后被替换
- 我们的 motion_pad 处理逻辑与此完全对齐
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

# 特殊 Token
MOTION_START_TOKEN = "<|motion_start|>"
MOTION_END_TOKEN = "<|motion_end|>"
MOTION_PAD_TOKEN = "<|motion_pad|>"

# InternVL3 的图像占位符 token (默认值, 可从 config 获取)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class InternVLDeltaChatModel(nn.Module):
    """
    InternVL3 + 双流运动编码。

    由于 InternVL3 使用自定义模型类 (InternVLChatModel, trust_remote_code=True),
    我们无法在 import 时静态继承。采用 **组合模式** 包装:

    策略:
    - 内部持有一个完整的 InternVLChatModel (self.base_model)
    - 新增 MotionEncoder + Projector
    - 重写 forward: 先让 base_model 处理视觉, 再替换 motion tokens

    架构:
    ┌─────────────┐    ┌──────────────┐
    │ Keyframes   │    │  Diff-Maps   │
    │ (稀疏, 1fps)│    │ (密集, 4x)   │
    └──────┬──────┘    └──────┬───────┘
           │                  │
    InternViT-300M (冻结)  ResNet-18 (训练)
     + Pixel Shuffle          │
           │            MLP Projector (训练)
      MLP Projector           │
      (冻结/微调)        Motion Tokens
           │                  │
      Image Tokens            │
           └────────┬─────────┘
                    │
              交错拼接 (V2PE 自动处理)
                    │
           Qwen2.5-7B LLM (LoRA)
                    │
                  Output
    """

    def __init__(
        self,
        base_model: nn.Module,
        motion_hidden_size: Optional[int] = None,
        motion_encoder_type: str = "resnet",
        siglip_model_path: str = "google/siglip-base-patch16-224",
        freeze_motion_encoder: bool = False,
        motion_vit_hidden_size: int = 3584,
        motion_use_cross_attention: bool = False,
    ):
        """
        Args:
            base_model: 已加载的 InternVLChatModel 实例
            motion_hidden_size: Motion Projector 输出维度,
                                默认从 base_model.config 获取
            motion_encoder_type: 运动编码器类型 ("resnet"/"siglip"/"shared_vit")
            siglip_model_path: SigLIP 模型路径 (仅 siglip 模式)
            freeze_motion_encoder: 是否冻结运动编码器 (仅 siglip 模式)
            motion_vit_hidden_size: ViT hidden_size (仅 shared_vit 模式)
            motion_use_cross_attention: 是否使用 cross-attention (仅 shared_vit 模式)
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        # 获取 LLM hidden_size
        # InternVL3-8B: language_model 是 Qwen2.5-7B, hidden_size=3584
        if motion_hidden_size is None:
            if hasattr(self.config, "llm_config"):
                motion_hidden_size = self.config.llm_config.hidden_size
            elif hasattr(base_model, "language_model"):
                motion_hidden_size = base_model.language_model.config.hidden_size
            else:
                motion_hidden_size = 3584  # Qwen2.5-7B 默认值
                logger.warning("无法自动获取 hidden_size, 使用默认值 %d", motion_hidden_size)

        logger.info("InternVL3 LLM hidden_size: %d", motion_hidden_size)

        # 新增: 运动编码模块 (可插拔 / 可切换到真实压缩域双流)
        motion_input_mode = getattr(self.config, "motion_input_mode", "rgb_diff")
        use_motion_vectors = getattr(self.config, "use_motion_vectors", True)
        use_residuals = getattr(self.config, "use_residuals", True)
        compressed_fusion_mode = getattr(self.config, "compressed_fusion_mode", "gated")
        compressed_use_iframe_context = getattr(self.config, "compressed_use_iframe_context", False)
        if compressed_use_iframe_context or getattr(self.config, "compressed_use_iframe_encoder", False):
            raise NotImplementedError("This archived InternVL wrapper does not forward appearance-branch inputs; use the Qwen2.5 tri-stream implementation.")
        if motion_input_mode == "compressed":
            self.motion_module = CompressedMotionAdapter(
                hidden_size=motion_hidden_size,
                motion_encoder_type=motion_encoder_type,
                siglip_model_path=siglip_model_path,
                freeze_motion_encoder=freeze_motion_encoder,
                vit_hidden_size=motion_vit_hidden_size,
                use_cross_attention=motion_use_cross_attention,
                use_motion_vectors=use_motion_vectors,
                use_residuals=use_residuals,
                use_iframe_context=compressed_use_iframe_context,
                fusion_mode=compressed_fusion_mode,
            )
        else:
            self.motion_module = MotionEncoderWithProjector(
                hidden_size=motion_hidden_size,
                motion_encoder_type=motion_encoder_type,
                siglip_model_path=siglip_model_path,
                freeze_motion_encoder=freeze_motion_encoder,
                vit_hidden_size=motion_vit_hidden_size,
                use_cross_attention=motion_use_cross_attention,
            )

        self.num_motion_tokens_per_keyframe = 4

        # 特殊 token IDs
        self.motion_start_token_id: Optional[int] = None
        self.motion_end_token_id: Optional[int] = None
        self.motion_pad_token_id: Optional[int] = None

        # InternVL3 的图像占位符 token ID
        self.img_context_token_id: Optional[int] = None

    def setup_motion_token_ids(self, tokenizer):
        """注册特殊 motion tokens。"""
        special_tokens = {
            "additional_special_tokens": [
                MOTION_START_TOKEN, MOTION_END_TOKEN, MOTION_PAD_TOKEN,
            ]
        }
        num_added = tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            # 扩展 base_model 的 embedding 层
            if hasattr(self.base_model, "language_model"):
                self.base_model.language_model.resize_token_embeddings(len(tokenizer))
            logger.info("添加 %d 个特殊 token, 词表: %d", num_added, len(tokenizer))

        self.motion_start_token_id = tokenizer.convert_tokens_to_ids(MOTION_START_TOKEN)
        self.motion_end_token_id = tokenizer.convert_tokens_to_ids(MOTION_END_TOKEN)
        self.motion_pad_token_id = tokenizer.convert_tokens_to_ids(MOTION_PAD_TOKEN)

        # 获取 InternVL3 的图像占位符 ID
        img_ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        if img_ctx_id != tokenizer.unk_token_id:
            self.img_context_token_id = img_ctx_id

        logger.info(
            "Motion token IDs - start: %d, end: %d, pad: %d",
            self.motion_start_token_id, self.motion_end_token_id, self.motion_pad_token_id,
        )

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
        将 motion_pad 占位符替换为 Motion Tokens。

        InternVL3 的视觉 token 处理流程:
        1. Tokenizer 在图像位置插入 <IMG_CONTEXT> 占位符
        2. forward 中用 InternViT 的输出替换这些位置
        3. 我们的 motion_pad 替换逻辑与此完全并行

        V2PE 位置编码会自动为序列中的所有 token 分配递增位置,
        不需要我们手动干预 (与 LLaVA 的 1D RoPE 类似)。

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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_flags: Optional[torch.LongTensor] = None,
        # ---- Himmel 新增 ----
        diff_frames: Optional[torch.Tensor] = None,
    ):
        """
        扩展的 forward, 新增 diff_frames 参数。

        InternVL3 的 forward 核心流程:
        1. pixel_values → InternViT → extract_feature() → vit_embeds
        2. vit_embeds → MLP Projector → image_tokens (与 LLM hidden 同维)
        3. input_ids → embedding layer → inputs_embeds
        4. 用 image_tokens 替换 inputs_embeds 中 img_context_token 的位置
        5. 送入 language_model (Qwen2.5-7B) 做 causal LM

        我们在 Step 4 之后插入 Step 4.5: 替换 motion tokens
        """
        # Step 1: 编码差分图
        motion_tokens = None
        if diff_frames is not None:
            motion_tokens = self.encode_motion(diff_frames)

        # 无 motion tokens -> 完全走原始逻辑
        if motion_tokens is None:
            return self.base_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                image_flags=image_flags,
            )

        # ---- 有 Motion Tokens: 手动控制 InternVL3 的编码流程 ----

        # Step 2: 视觉编码 (InternViT + Pixel Shuffle + MLP Projector)
        if inputs_embeds is None and input_ids is not None:
            # 先获取文本 embedding
            language_model = self.base_model.language_model
            inputs_embeds = language_model.get_input_embeddings()(input_ids)

            # 编码视觉
            if pixel_values is not None:
                # InternViT 权重为 bf16, 需要将 pixel_values 转为相同 dtype
                vit_dtype = next(self.base_model.vision_model.parameters()).dtype
                if pixel_values.dtype != vit_dtype:
                    pixel_values = pixel_values.to(dtype=vit_dtype)
                # InternVL3 的视觉编码管线
                vit_embeds = self.base_model.extract_feature(pixel_values)
                # vit_embeds: [num_patches, num_visual_tokens, hidden_size]

                # Pixel Shuffle 后通过 MLP Projector
                # (extract_feature 内部已完成)

                # 处理 image_flags (区分真实图像和 padding 图像)
                if image_flags is not None:
                    # image_flags: [num_images], 1=真实图像, 0=padding
                    vit_embeds = vit_embeds * image_flags[:, None, None].to(
                        vit_embeds.dtype
                    )

                # 替换 IMG_CONTEXT 占位符
                B, S, D = inputs_embeds.shape
                input_ids_flat = input_ids.reshape(-1)

                if self.img_context_token_id is not None:
                    selected = (input_ids_flat == self.img_context_token_id)
                else:
                    # 降级: 从 base_model config 获取
                    img_token_id = getattr(self.base_model, "img_context_token_id", None)
                    if img_token_id is not None:
                        selected = (input_ids_flat == img_token_id)
                    else:
                        selected = torch.zeros_like(input_ids_flat, dtype=torch.bool)

                if selected.sum() > 0:
                    vit_flat = vit_embeds.reshape(-1, vit_embeds.shape[-1])
                    n_select = min(selected.sum().item(), vit_flat.shape[0])
                    inputs_embeds_flat = inputs_embeds.reshape(-1, D)
                    inputs_embeds_flat[
                        selected.nonzero(as_tuple=True)[0][:n_select]
                    ] = vit_flat[:n_select].to(inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds_flat.reshape(B, S, D)

        # Step 4.5: 替换 Motion Tokens
        inputs_embeds = self._replace_motion_placeholders(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            motion_tokens=motion_tokens,
        )

        # Step 5: 送入 LLM
        return self.base_model.language_model(
            input_ids=None,  # 已有 inputs_embeds
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def forward_alignment(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        diff_frames: Optional[torch.Tensor] = None,
        num_diffs_per_keyframe: int = 4,
        **kwargs,
    ) -> dict:
        """
        Stage 1 专用: Contrastive/Regression Alignment Forward。

        InternVL3 架构特点:
        - InternViT-300M + Pixel Shuffle 4x → 每张 448x448 tile 输出 256 tokens
        - extract_feature() 返回 [num_images, 256, D] (D = llm_hidden_size)
        - 视频帧作为多张独立图片输入, 无 temporal merge
        - 每帧 mean pool → [N_frames, D] 作为 visual targets

        Returns:
            dict with keys: "loss", "avg_cosine_sim", "num_pairs"
        """
        device = next(self.motion_module.parameters()).device

        if diff_frames is None and motion_vectors is None and residuals is None:
            logger.warning("[Alignment] no motion inputs provided")
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        # Step 1: InternViT 编码 visual features (冻结, detach)
        if pixel_values is None:
            logger.warning("[Alignment] pixel_values is None")
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}

        with torch.no_grad():
            # extract_feature 内部完成 InternViT + Pixel Shuffle + MLP Projector
            # 返回 [num_images, 256, D]
            # InternViT 权重为 bf16, 需要将 pixel_values 转为相同 dtype
            vit_dtype = next(self.base_model.vision_model.parameters()).dtype
            if pixel_values.dtype != vit_dtype:
                pixel_values = pixel_values.to(dtype=vit_dtype)
            vit_embeds = self.base_model.extract_feature(pixel_values)

            # 处理 image_flags (区分真实图像和 padding 图像)
            if image_flags is not None:
                vit_embeds = vit_embeds * image_flags[:, None, None].to(vit_embeds.dtype)

        vit_embeds = vit_embeds.detach()

        # per-frame mean pool: [num_images, 256, D] → [num_images, D]
        per_frame_targets = vit_embeds.mean(dim=1)
        n_visual_frames = per_frame_targets.shape[0]
        D = per_frame_targets.shape[-1]

        # Step 2: 编码 motion tokens (训练)
        motion_tokens = self.encode_motion(diff_frames)
        B, T_motion = motion_tokens.shape[:2]

        logger.info("[Alignment] per_frame_targets=%s, motion_tokens=%s, n_visual_frames=%d",
                    per_frame_targets.shape, motion_tokens.shape, n_visual_frames)

        # Step 3: 构造对齐 pairs (帧差 target, InfoNCE 对比学习)
        # target = vf[i+1] - vf[i] (ViT 特征空间的"语义级变化")
        # InternVL3 无 temporal merge, n_visual_frames = 实际关键帧数
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

        Stage 1 (Alignment, ~3h):
            训练: MotionEncoder + MotionProjector
            冻结: InternViT + MLP Projector + LLM

        Stage 2 (SFT):
            sft_mode="lora": 训练 MotionEncoder + MotionProjector + LLM LoRA
            sft_mode="full": 训练 MotionEncoder + MotionProjector + LLM (全量)
            冻结: InternViT + MLP Projector (原始参数)
        """
        freeze_module(self)

        if stage == 1:
            self.motion_module.configure_training_state(
                train_alignment_head=True,
                freeze_spatial_encoder=freeze_spatial_encoder,
            )
            logger.info("=== InternVL3 Stage 1: Alignment ===")
            logger.info("训练: MotionEncoder + MotionProjector")
            logger.info("冻结: InternViT + MLP Projector + LLM")

        elif stage == 2:
            self.motion_module.configure_training_state(train_alignment_head=False)
            if sft_mode == "full":
                unfreeze_module(self.base_model.language_model)
                logger.info("=== InternVL3 Stage 2: Full SFT ===")
                logger.info("训练: MotionEncoder + MotionProjector + LLM (全量)")
                logger.info("冻结: InternViT + MLP Projector (原始参数)")
            else:
                logger.info("=== InternVL3 Stage 2: SFT with LoRA ===")
                logger.info("训练: MotionEncoder + MotionProjector + LLM LoRA")
                logger.info("冻结: InternViT + MLP Projector (原始参数)")

        else:
            raise ValueError(f"未知训练阶段: {stage}")

        info = get_trainable_params_info(self)
        logger.info("参数统计: %s", info)
        return self

    # ========================================================================
    # 委托方法: 让外部代码能像使用标准 transformers 模型一样使用
    # ========================================================================

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    def get_input_embeddings(self):
        return self.base_model.language_model.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens):
        return self.base_model.language_model.resize_token_embeddings(new_num_tokens)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        diff_frames: Optional[torch.Tensor] = None,
        motion_vectors: Optional[torch.Tensor] = None,
        residuals: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ):
        """
        Generate with motion tokens injected into inputs_embeds.
        """
        if diff_frames is None and motion_vectors is None and residuals is None:
            return self.base_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_flags=image_flags,
                **generate_kwargs,
            )

        motion_tokens = self.encode_motion(
            diff_frames=diff_frames,
            motion_vectors=motion_vectors,
            residuals=residuals,
        )

        language_model = self.base_model.language_model
        inputs_embeds = language_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            vit_dtype = next(self.base_model.vision_model.parameters()).dtype
            if pixel_values.dtype != vit_dtype:
                pixel_values = pixel_values.to(dtype=vit_dtype)
            vit_embeds = self.base_model.extract_feature(pixel_values)
            if image_flags is not None:
                vit_embeds = vit_embeds * image_flags[:, None, None].to(vit_embeds.dtype)

            B, S, D = inputs_embeds.shape
            input_ids_flat = input_ids.reshape(-1)
            if self.img_context_token_id is not None:
                selected = (input_ids_flat == self.img_context_token_id)
            else:
                img_token_id = getattr(self.base_model, "img_context_token_id", None)
                selected = (input_ids_flat == img_token_id) if img_token_id else torch.zeros_like(input_ids_flat, dtype=torch.bool)

            if selected.sum() > 0:
                vit_flat = vit_embeds.reshape(-1, vit_embeds.shape[-1])
                n_select = min(selected.sum().item(), vit_flat.shape[0])
                inputs_embeds_flat = inputs_embeds.reshape(-1, D)
                inputs_embeds_flat[selected.nonzero(as_tuple=True)[0][:n_select]] = vit_flat[:n_select].to(inputs_embeds.dtype)
                inputs_embeds = inputs_embeds_flat.reshape(B, S, D)

        inputs_embeds = self._replace_motion_placeholders(input_ids, inputs_embeds, motion_tokens)

        return language_model.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )


# ============================================================================
# 工厂函数
# ============================================================================

def load_internvl3_delta(
    model_path: str,
    dtype=torch.bfloat16,
    motion_encoder_type: str = "resnet",
    siglip_model_path: str = "google/siglip-base-patch16-224",
    freeze_motion_encoder: bool = False,
    motion_vit_hidden_size: int = 3584,
    motion_use_cross_attention: bool = False,
    **kwargs,
):
    """
    便捷加载函数。

    用法:
        model = load_internvl3_delta("/path/to/InternVL3-8B")
        model = load_internvl3_delta("/path/to/InternVL3-8B", motion_encoder_type="siglip")

    注意: InternVL3 必须设置 trust_remote_code=True
    """
    from transformers import AutoModel

    delta_runtime_config = {
        "motion_input_mode": kwargs.pop("motion_input_mode", "rgb_diff"),
        "use_motion_vectors": kwargs.pop("use_motion_vectors", True),
        "use_residuals": kwargs.pop("use_residuals", True),
        "compressed_fusion_mode": kwargs.pop("compressed_fusion_mode", "gated"),
        "compressed_use_iframe_context": kwargs.pop("compressed_use_iframe_context", False),
        "compressed_use_iframe_encoder": kwargs.pop("compressed_use_iframe_encoder", False),
        "alignment_motion_pool": kwargs.pop("alignment_motion_pool", "mean"),
        "diff_mode": kwargs.pop("diff_mode", "abs"),
    }

    logger.info("加载 InternVL3 base model: %s", model_path)
    # InternVL3 的 trust_remote_code 模型在 meta device 上初始化会崩溃
    # (InternVisionEncoder.__init__ 中 torch.linspace(...).item() 不兼容 meta tensor)
    # 必须显式关闭 low_cpu_mem_usage
    base_model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=False,
        **kwargs,
    )
    for key, value in delta_runtime_config.items():
        setattr(base_model.config, key, value)

    model = InternVLDeltaChatModel(
        base_model,
        motion_encoder_type=motion_encoder_type,
        siglip_model_path=siglip_model_path,
        freeze_motion_encoder=freeze_motion_encoder,
        motion_vit_hidden_size=motion_vit_hidden_size,
        motion_use_cross_attention=motion_use_cross_attention,
    )
    logger.info("InternVL3 Delta 模型创建完成 (motion_encoder_type=%s)", motion_encoder_type)
    return model
