#!/usr/bin/env python3
"""
Himmel-VLM: 真实压缩域运动适配器
=================================

职责:
1. 分别编码 I-frame context / motion vector / residual 三路信息；
2. 先融合 MV + residual，再与 I-frame context 做二级 gated fusion；
3. 维持与现有 MotionEncoderWithProjector 相同的接口，便于无缝替换。
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from .delta_components import (
    AlignmentHead,
    MotionEncoderWithProjector,
    freeze_module,
    unfreeze_module,
)

logger = logging.getLogger(__name__)


class CompressedStreamFusion(nn.Module):
    """通用双流融合模块，可用于 MV/residual 或 motion/I-frame 的二级融合。"""

    def __init__(self, hidden_size: int, mode: str = "gated"):
        super().__init__()
        self.mode = mode
        fusion_in = hidden_size * 4
        self.pre_norm = nn.LayerNorm(hidden_size)
        self.primary_norm = nn.LayerNorm(hidden_size)
        self.secondary_norm = nn.LayerNorm(hidden_size)

        if mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(fusion_in, hidden_size),
                nn.Sigmoid(),
            )
            self.delta = nn.Sequential(
                nn.Linear(fusion_in, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        elif mode == "concat_mlp":
            self.proj = nn.Sequential(
                nn.Linear(fusion_in, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        elif mode == "sum":
            self.scale = nn.Parameter(torch.tensor(0.5))
        else:
            raise ValueError(f"不支持的 compressed_fusion_mode: {mode}")

    def forward(
        self,
        primary_tokens: torch.Tensor | None,
        secondary_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if primary_tokens is None and secondary_tokens is None:
            raise ValueError("CompressedStreamFusion 至少需要一个输入流")
        if primary_tokens is None:
            assert secondary_tokens is not None
            return self.pre_norm(secondary_tokens)
        if secondary_tokens is None:
            return self.pre_norm(primary_tokens)

        assert primary_tokens is not None
        assert secondary_tokens is not None
        primary = primary_tokens
        secondary = secondary_tokens
        if primary.shape[:2] != secondary.shape[:2]:
            time_dim = min(primary.shape[1], secondary.shape[1])
            primary = primary[:, :time_dim]
            secondary = secondary[:, :time_dim]

        # Retain equally sized residual branches after temporal alignment.
        primary_tokens = primary
        secondary_tokens = secondary

        primary = self.primary_norm(primary)
        secondary = self.secondary_norm(secondary)
        primary_delta = primary - secondary
        primary_interaction = primary * secondary
        fusion_input = torch.cat(
            [primary, secondary, primary_delta, primary_interaction],
            dim=-1,
        )

        if self.mode == "gated":
            gate = self.gate(fusion_input)
            fused = (
                gate * primary_tokens
                + (1.0 - gate) * secondary_tokens
                + self.delta(fusion_input)
            )
        elif self.mode == "concat_mlp":
            fused = self.proj(fusion_input)
        else:  # sum
            alpha = torch.clamp(self.scale, 0.0, 1.0)
            fused = alpha * primary_tokens + (1.0 - alpha) * secondary_tokens

        return self.pre_norm(fused)


class IframeContextProjector(nn.Module):
    """把 I-frame interval context 投影到 motion hidden / alignment 空间。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        in_dim = hidden_size * 4
        mid_dim = hidden_size * 2
        self.input_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, iframe_context: torch.Tensor) -> torch.Tensor:
        hidden = self.input_norm(iframe_context)
        hidden = self.proj(hidden)
        return self.output_norm(hidden)


def _metric_tensor(result: dict[str, object], key: str) -> torch.Tensor:
    value = result.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"alignment result[{key!r}] 不是 Tensor: {type(value).__name__}")
    return value


class CompressedMotionAdapter(nn.Module):
    """显式 I-frame/MV/residual 三流压缩域适配器。

    支持两种 I-frame 融入方式（互斥）:
    - ``use_iframe_context=True`` (旧): 从 ViT 视觉特征派生高层 context，通过 MLP
      投影后与 MV+Residual 做二级 gated fusion。I-frame 信息来自宿主 ViT。
    - ``use_iframe_encoder=True`` (新, 对齐 CoViAR): I-frame 拥有独立的像素级编码器
      分支 (``MotionEncoderWithProjector``)，和 MV/Residual 地位对等，编码压缩域
      原始 I-frame 图像后参与三路融合。这是论文中标准的三流架构。
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        motion_encoder_type: str = "resnet",
        siglip_model_path: str = "google/siglip-base-patch16-224",
        freeze_motion_encoder: bool = False,
        vit_hidden_size: int = 1152,
        use_cross_attention: bool = False,
        use_motion_vectors: bool = True,
        use_residuals: bool = True,
        use_iframe_context: bool = False,
        use_iframe_encoder: bool = False,
        fusion_mode: str = "gated",
        siglip_unfreeze_last_n_layers: int = 0,
        fused_alignment_weight: float = 1.0,
        mv_alignment_weight: float = 0.25,
        residual_alignment_weight: float = 0.25,
        iframe_alignment_weight: float = 0.25,
    ):
        super().__init__()
        if not use_motion_vectors and not use_residuals:
            raise ValueError("CompressedMotionAdapter 至少需要一个启用的压缩域流")
        if use_iframe_context and use_iframe_encoder:
            raise ValueError(
                "use_iframe_context 和 use_iframe_encoder 互斥，不能同时启用。"
                "use_iframe_context 使用 ViT 派生 context；"
                "use_iframe_encoder 使用独立像素编码器（CoViAR 标准三流）。"
            )

        self.hidden_size = hidden_size
        self.motion_encoder_type = motion_encoder_type
        self.use_motion_vectors = use_motion_vectors
        self.use_residuals = use_residuals
        self.use_iframe_context = use_iframe_context
        self.use_iframe_encoder = use_iframe_encoder
        self.freeze_motion_encoder = freeze_motion_encoder
        self.fusion_mode = fusion_mode
        self.fused_alignment_weight = float(fused_alignment_weight)
        self.mv_alignment_weight = float(mv_alignment_weight)
        self.residual_alignment_weight = float(residual_alignment_weight)
        self.iframe_alignment_weight = float(iframe_alignment_weight)

        self.mv_branch = MotionEncoderWithProjector(
            hidden_size=hidden_size,
            motion_encoder_type=motion_encoder_type,
            siglip_model_path=siglip_model_path,
            freeze_motion_encoder=freeze_motion_encoder,
            vit_hidden_size=vit_hidden_size,
            use_cross_attention=use_cross_attention,
            siglip_unfreeze_last_n_layers=siglip_unfreeze_last_n_layers,
        ) if use_motion_vectors else None
        self.residual_branch = MotionEncoderWithProjector(
            hidden_size=hidden_size,
            motion_encoder_type=motion_encoder_type,
            siglip_model_path=siglip_model_path,
            freeze_motion_encoder=freeze_motion_encoder,
            vit_hidden_size=vit_hidden_size,
            use_cross_attention=use_cross_attention,
            siglip_unfreeze_last_n_layers=siglip_unfreeze_last_n_layers,
        ) if use_residuals else None

        # ---- CoViAR 真三流: 独立 I-frame 编码器分支 ----
        self.iframe_branch = MotionEncoderWithProjector(
            hidden_size=hidden_size,
            motion_encoder_type=motion_encoder_type,
            siglip_model_path=siglip_model_path,
            freeze_motion_encoder=freeze_motion_encoder,
            vit_hidden_size=vit_hidden_size,
            use_cross_attention=use_cross_attention,
            siglip_unfreeze_last_n_layers=siglip_unfreeze_last_n_layers,
        ) if use_iframe_encoder else None

        self.hidden_fusion = CompressedStreamFusion(hidden_size=hidden_size, mode=fusion_mode)
        self.align_fusion = CompressedStreamFusion(hidden_size=hidden_size, mode=fusion_mode)

        # ---- 旧方案: ViT 派生 I-frame context ----
        self.iframe_hidden_projector = IframeContextProjector(hidden_size) if use_iframe_context else None
        self.iframe_align_projector = IframeContextProjector(hidden_size) if use_iframe_context else None

        # 三路融合 (旧 context 方案或新 encoder 方案共用)
        _need_tri_fusion = use_iframe_context or use_iframe_encoder
        self.hidden_tri_fusion = CompressedStreamFusion(hidden_size=hidden_size, mode=fusion_mode) if _need_tri_fusion else None
        self.align_tri_fusion = CompressedStreamFusion(hidden_size=hidden_size, mode=fusion_mode) if _need_tri_fusion else None
        self.alignment_head = AlignmentHead(hidden_size=hidden_size)

        logger.info(
            "CompressedMotionAdapter: iframe_context=%s iframe_encoder=%s mv=%s residual=%s "
            "fusion=%s hidden=%d align_weights=(fused=%.2f,iframe=%.2f,mv=%.2f,res=%.2f)",
            use_iframe_context,
            use_iframe_encoder,
            use_motion_vectors,
            use_residuals,
            fusion_mode,
            hidden_size,
            self.fused_alignment_weight,
            self.iframe_alignment_weight,
            self.mv_alignment_weight,
            self.residual_alignment_weight,
        )

    def _select_legacy_branch(self) -> MotionEncoderWithProjector:
        if self.residual_branch is not None:
            return self.residual_branch
        assert self.mv_branch is not None
        return self.mv_branch

    def _encode_iframe_context(
        self,
        iframe_context: torch.Tensor | None,
        *,
        alignment: bool,
    ) -> torch.Tensor | None:
        """旧方案: ViT 派生 context → MLP 投影."""
        if iframe_context is None or not self.use_iframe_context:
            return None
        projector = self.iframe_align_projector if alignment else self.iframe_hidden_projector
        if projector is None:
            return None
        return projector(iframe_context)

    def _encode_iframe_pixels(
        self,
        iframe_frames: torch.Tensor | None,
        *,
        alignment: bool,
        **kwargs,
    ) -> torch.Tensor | None:
        """新方案 (CoViAR 真三流): 独立编码器编码 I-frame 像素."""
        if iframe_frames is None or self.iframe_branch is None:
            return None
        if alignment:
            return self.iframe_branch.encode_motion_for_alignment(iframe_frames, **kwargs)
        return self.iframe_branch(iframe_frames, **kwargs)

    def _get_iframe_tokens(
        self,
        *,
        iframe_context: torch.Tensor | None = None,
        iframe_frames: torch.Tensor | None = None,
        alignment: bool = False,
        **kwargs,
    ) -> torch.Tensor | None:
        """统一获取 iframe tokens（自动选择 context 或 encoder 方案）."""
        if self.use_iframe_encoder:
            return self._encode_iframe_pixels(iframe_frames, alignment=alignment, **kwargs)
        if self.use_iframe_context:
            return self._encode_iframe_context(iframe_context, alignment=alignment)
        return None

    def _encode_stream(
        self,
        *,
        diff_frames: torch.Tensor | None = None,
        motion_vectors: torch.Tensor | None = None,
        residuals: torch.Tensor | None = None,
        iframe_context: torch.Tensor | None = None,
        iframe_frames: torch.Tensor | None = None,
        alignment: bool = False,
        return_components: bool = False,
        **kwargs,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        iframe_tokens = self._get_iframe_tokens(
            iframe_context=iframe_context,
            iframe_frames=iframe_frames,
            alignment=alignment,
            **kwargs,
        )

        if motion_vectors is None and residuals is None:
            if diff_frames is None:
                raise ValueError("未提供任何 motion 输入")
            branch = self._select_legacy_branch()
            base_tokens = (
                branch.encode_motion_for_alignment(diff_frames, **kwargs)
                if alignment else
                branch(diff_frames, **kwargs)
            )
            if iframe_tokens is not None:
                tri_fusion = self.align_tri_fusion if alignment else self.hidden_tri_fusion
                if tri_fusion is None:
                    raise ValueError("iframe tri-fusion 尚未初始化")
                fused_tokens = tri_fusion(base_tokens, iframe_tokens)
            else:
                fused_tokens = base_tokens
            if return_components:
                return {
                    "fused": fused_tokens,
                    "iframe": iframe_tokens,
                    "motion_vectors": None,
                    "residuals": None,
                }
            return fused_tokens

        mv_tokens = None
        residual_tokens = None
        if motion_vectors is not None and self.mv_branch is not None:
            mv_tokens = (
                self.mv_branch.encode_motion_for_alignment(motion_vectors, **kwargs)
                if alignment else
                self.mv_branch(motion_vectors, **kwargs)
            )
        if residuals is not None and self.residual_branch is not None:
            residual_tokens = (
                self.residual_branch.encode_motion_for_alignment(residuals, **kwargs)
                if alignment else
                self.residual_branch(residuals, **kwargs)
            )

        base_fusion = self.align_fusion if alignment else self.hidden_fusion
        fused_tokens = base_fusion(mv_tokens, residual_tokens)
        if iframe_tokens is not None:
            tri_fusion = self.align_tri_fusion if alignment else self.hidden_tri_fusion
            if tri_fusion is None:
                raise ValueError("iframe tri-fusion 尚未初始化")
            fused_tokens = tri_fusion(fused_tokens, iframe_tokens)

        if return_components:
            return {
                "fused": fused_tokens,
                "iframe": iframe_tokens,
                "motion_vectors": mv_tokens,
                "residuals": residual_tokens,
            }
        return fused_tokens

    def reload_pretrained_weights(self) -> bool:
        reloaded = False
        for name, branch in (("mv", self.mv_branch), ("residual", self.residual_branch), ("iframe", self.iframe_branch)):
            if branch is None:
                continue
            branch_reloaded = branch.reload_pretrained_weights()
            logger.info("CompressedMotionAdapter.%s_branch reload=%s", name, branch_reloaded)
            reloaded = branch_reloaded or reloaded

        init_modules: list[nn.Module] = [self.hidden_fusion, self.align_fusion, self.alignment_head]
        for module in (
            self.iframe_hidden_projector,
            self.iframe_align_projector,
            self.hidden_tri_fusion,
            self.align_tri_fusion,
        ):
            if module is not None:
                init_modules.append(module)

        for module in init_modules:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

        with torch.no_grad():
            self.alignment_head.logit_scale.fill_(math.log(1.0 / 0.07))
        return reloaded

    def configure_training_state(
        self,
        train_alignment_head: bool = True,
        *,
        train_projector: bool = True,
        train_alignment_projector: bool = True,
        freeze_spatial_encoder: bool = False,
    ) -> None:
        freeze_module(self)

        for branch in (self.mv_branch, self.residual_branch, self.iframe_branch):
            if branch is None:
                continue
            if not branch.freeze_motion_encoder:
                if freeze_spatial_encoder and branch.motion_encoder_type == "motionssm":
                    branch.encoder.set_trainable(spatial=False, temporal=True)
                else:
                    unfreeze_module(branch.encoder)
            if train_projector:
                unfreeze_module(branch.projector)
            if train_alignment_projector:
                unfreeze_module(branch.align_proj)
            freeze_module(branch.alignment_head)  # adapter 自己维护唯一的 alignment head

        unfreeze_module(self.hidden_fusion)
        unfreeze_module(self.align_fusion)
        if self.use_iframe_context:
            for module in (
                self.iframe_hidden_projector,
                self.iframe_align_projector,
                self.hidden_tri_fusion,
                self.align_tri_fusion,
            ):
                if module is not None:
                    unfreeze_module(module)
        if self.use_iframe_encoder:
            for module in (self.hidden_tri_fusion, self.align_tri_fusion):
                if module is not None:
                    unfreeze_module(module)
        if train_alignment_head:
            unfreeze_module(self.alignment_head)

        logger.info(
            "CompressedMotionAdapter 训练策略: encoder_trainable=%s, train_projector=%s, "
            "train_alignment_projector=%s, freeze_spatial_encoder=%s, iframe_context=%s, "
            "iframe_encoder=%s, alignment_head_trainable=%s",
            not self.freeze_motion_encoder,
            train_projector,
            train_alignment_projector,
            freeze_spatial_encoder,
            self.use_iframe_context,
            self.use_iframe_encoder,
            train_alignment_head,
        )

    def set_alignment_options(
        self,
        *,
        gather_global_negatives: bool | None = None,
        fused_weight: float | None = None,
        mv_weight: float | None = None,
        residual_weight: float | None = None,
        iframe_weight: float | None = None,
    ) -> None:
        if gather_global_negatives is not None:
            self.alignment_head.set_gather_global_negatives(gather_global_negatives)
        if fused_weight is not None:
            self.fused_alignment_weight = float(fused_weight)
        if mv_weight is not None:
            self.mv_alignment_weight = float(mv_weight)
        if residual_weight is not None:
            self.residual_alignment_weight = float(residual_weight)
        if iframe_weight is not None:
            self.iframe_alignment_weight = float(iframe_weight)

    def encode_motion(
        self,
        diff_frames: torch.Tensor | None = None,
        *,
        motion_vectors: torch.Tensor | None = None,
        residuals: torch.Tensor | None = None,
        iframe_context: torch.Tensor | None = None,
        iframe_frames: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._encode_stream(
            diff_frames=diff_frames,
            motion_vectors=motion_vectors,
            residuals=residuals,
            iframe_context=iframe_context,
            iframe_frames=iframe_frames,
            alignment=False,
            **kwargs,
        )

    def encode_motion_for_alignment(
        self,
        diff_frames: torch.Tensor | None = None,
        *,
        motion_vectors: torch.Tensor | None = None,
        residuals: torch.Tensor | None = None,
        iframe_context: torch.Tensor | None = None,
        iframe_frames: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._encode_stream(
            diff_frames=diff_frames,
            motion_vectors=motion_vectors,
            residuals=residuals,
            iframe_context=iframe_context,
            iframe_frames=iframe_frames,
            alignment=True,
            **kwargs,
        )

    def encode_motion_for_alignment_components(
        self,
        diff_frames: torch.Tensor | None = None,
        *,
        motion_vectors: torch.Tensor | None = None,
        residuals: torch.Tensor | None = None,
        iframe_context: torch.Tensor | None = None,
        iframe_frames: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        encoded = self._encode_stream(
            diff_frames=diff_frames,
            motion_vectors=motion_vectors,
            residuals=residuals,
            iframe_context=iframe_context,
            iframe_frames=iframe_frames,
            alignment=True,
            return_components=True,
            **kwargs,
        )
        assert isinstance(encoded, dict)
        return encoded

    def compute_alignment_loss(
        self,
        motion_tokens: torch.Tensor,
        visual_targets: torch.Tensor,
        target_group_ids: torch.Tensor | None = None,
        negative_mask: torch.Tensor | None = None,
    ) -> dict[str, object]:
        return self.alignment_head(
            motion_tokens,
            visual_targets,
            target_group_ids=target_group_ids,
            negative_mask=negative_mask,
        )

    def compute_multi_stream_alignment_loss(
        self,
        stream_queries: dict[str, torch.Tensor],
        visual_targets: torch.Tensor,
        target_group_ids: torch.Tensor | None = None,
        negative_mask: torch.Tensor | None = None,
    ) -> dict[str, object]:
        stream_specs = [
            ("fused", self.fused_alignment_weight),
            ("iframe", self.iframe_alignment_weight),
            ("motion_vectors", self.mv_alignment_weight),
            ("residuals", self.residual_alignment_weight),
        ]

        per_stream_results: dict[str, dict[str, object]] = {}
        weighted_losses: list[torch.Tensor] = []
        aux_streams: list[str] = []

        for stream_name, weight in stream_specs:
            stream_tokens = stream_queries.get(stream_name)
            if stream_tokens is None or weight <= 0:
                continue

            result = self.alignment_head(
                stream_tokens,
                visual_targets,
                target_group_ids=target_group_ids,
                negative_mask=negative_mask,
            )
            per_stream_results[stream_name] = result
            weighted_losses.append(_metric_tensor(result, "loss").float() * float(weight))
            if stream_name != "fused":
                aux_streams.append(stream_name)

        if not weighted_losses:
            raise ValueError("compute_multi_stream_alignment_loss 未收到可用的压缩域流")

        if "fused" in per_stream_results:
            base_metrics = dict(per_stream_results["fused"])
            fused_loss = _metric_tensor(per_stream_results["fused"], "loss")
            aux_loss_value = fused_loss.float().new_zeros(())
            for name, weight in stream_specs:
                if name == "fused" or name not in per_stream_results or weight <= 0:
                    continue
                aux_loss_value = aux_loss_value + _metric_tensor(per_stream_results[name], "loss").float() * float(weight)
        else:
            first_stream_name = next(iter(per_stream_results))
            base_metrics = dict(per_stream_results[first_stream_name])
            aux_loss_value = weighted_losses[0].new_zeros(())
            for loss_value in weighted_losses:
                aux_loss_value = aux_loss_value + loss_value

        total_loss = torch.stack(weighted_losses).sum()
        base_loss = _metric_tensor(base_metrics, "loss")
        base_metrics["loss"] = total_loss.to(base_loss.dtype)
        base_metrics["compressed_aux_streams"] = aux_streams
        base_metrics["compressed_aux_loss"] = aux_loss_value.to(base_loss.dtype)

        for stream_name in aux_streams:
            stream_result = per_stream_results[stream_name]
            if stream_name == "motion_vectors":
                prefix = "mv"
            elif stream_name == "residuals":
                prefix = "residual"
            else:
                prefix = stream_name
            base_metrics[f"{prefix}_alignment_loss"] = _metric_tensor(stream_result, "loss").to(base_loss.dtype)
            base_metrics[f"{prefix}_avg_cosine_sim"] = stream_result["avg_cosine_sim"]

        return base_metrics

    def forward(
        self,
        diff_frames: torch.Tensor | None = None,
        *,
        motion_vectors: torch.Tensor | None = None,
        residuals: torch.Tensor | None = None,
        iframe_context: torch.Tensor | None = None,
        iframe_frames: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.encode_motion(
            diff_frames=diff_frames,
            motion_vectors=motion_vectors,
            residuals=residuals,
            iframe_context=iframe_context,
            iframe_frames=iframe_frames,
            **kwargs,
        )
