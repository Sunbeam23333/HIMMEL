#!/usr/bin/env python3
"""
Himmel-VLM: 共享运动编码组件
=================================
支持多种可插拔的运动编码器:
  - "resnet": ResNet-18 (默认, ImageNet 预训练, 11M 参数)
  - "motionssm": ResNet 空间 stem + Mamba/MotionSSM 风格时序状态混合
  - "siglip": SigLIP ViT (预训练视觉编码器, embedding 空间天然接近语言模型)
  - "shared_vit": 复用宿主 VLM 的 ViT 编码差分图 (零额外参数, 天然同空间)

MotionProjector: 将编码器特征映射到 LLM 隐层空间
  - 两层 MLP (带 GELU) 比单层 Linear 对齐效果更好 (参考 LLaVA-1.5)

训练策略:
- Stage-1: 只训练 MotionEncoder + MotionProjector (对齐到 LLM 语义空间)
- Stage-2: 继续训练 MotionEncoder + MotionProjector + LLM LoRA

通过配置文件中的 "motion_encoder_type" 字段选择编码器类型。
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision.models import resnet18

# ResNet-18 预训练权重的默认本地路径 (由 download_himmel.py 下载)
# __file__ = .../code/models/delta_components.py
# dirname x2 = .../code/ , dirname x3 = .../Video_Horizon/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_RESNET18_CKPT = os.environ.get(
    "HIMMEL_RESNET18_CHECKPOINT",
    os.path.join(_PROJECT_ROOT, "models", "resnet18-imagenet1k-v1.pth"),
)

# 支持的 Motion Encoder 类型
MOTION_ENCODER_TYPES = ["resnet", "motionssm", "siglip", "shared_vit"]


def _set_parameter_by_name(module: nn.Module, name: str, param: nn.Parameter) -> None:
    """按层级路径回写参数, 用于修复 meta device 初始化后的空参数。"""
    parent = module
    parts = name.split(".")
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], param)


def _reinitialize_module_parameters(module: nn.Module, module_name: str, special_init=None) -> bool:
    """若子模块处于 meta/全零状态, 则做一次通用重初始化。"""
    params = list(module.named_parameters())
    if not params:
        return False

    _, first_param = params[0]
    needs_reinit = first_param.is_meta or (
        first_param.device.type != "meta" and first_param.detach().abs().sum().item() == 0
    )
    if not needs_reinit:
        return False

    logger = logging.getLogger(__name__)
    logger.warning("检测到 %s 权重无效 (meta/全零), 重新初始化...", module_name)
    device = first_param.device if not first_param.is_meta else torch.device("cpu")
    named_params = dict(module.named_parameters())

    for name, param in params:
        if param.is_meta:
            new_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
        else:
            new_tensor = param.data

        handled = False
        if special_init is not None:
            handled = bool(special_init(name, new_tensor, named_params))

        if not handled:
            if "bias" in name:
                nn.init.zeros_(new_tensor)
            elif new_tensor.dim() >= 2:
                nn.init.kaiming_uniform_(new_tensor, a=5 ** 0.5)
            elif "weight" in name:
                nn.init.ones_(new_tensor)
            else:
                nn.init.zeros_(new_tensor)

        if param.is_meta:
            _set_parameter_by_name(module, name, nn.Parameter(new_tensor, requires_grad=param.requires_grad))

    logger.info("%s 权重重新初始化完成", module_name)
    return True


class MotionEncoder(nn.Module):
    """
    基于 ResNet-18 的轻量运动编码器。

    输入: 差分图 [B, T_diff, 3, 224, 224]
    输出: 运动特征 [B, T_diff, 512]

    为什么选 ResNet-18:
    1. 差分图是低级视觉信号 (边缘/运动), 不需要深层语义理解
    2. ResNet-18 只有 11M 参数, 全参训练也很快
    3. ImageNet 预训练权重提供了良好的低级特征初始化

    注意: 权重从本地文件加载, 不联网。由 download_himmel.py 预先下载。

    重要: 当宿主模型通过 from_pretrained(low_cpu_mem_usage=True) 加载时,
    __init__ 中的 load_state_dict 在 meta device 上是 no-op。
    必须在 from_pretrained 完成后调用 reload_pretrained_weights() 重新加载。
    """

    def __init__(self, resnet_ckpt: str | None = _DEFAULT_RESNET18_CKPT):
        super().__init__()
        self._resnet_ckpt = resnet_ckpt  # 记住路径, 以便后续重新加载

        # 先创建空的 ResNet-18 (不联网)
        backbone = resnet18(weights=None)

        # 从本地文件加载 ImageNet 预训练权重
        self._try_load_resnet_weights(backbone, resnet_ckpt)

        # 去掉最后的 FC 层和 avgpool, 只保留特征提取部分
        # ResNet-18 最终输出: [B, 512, 7, 7] (输入 224x224 时)
        self.features = nn.Sequential(
            backbone.conv1,     # -> [B, 64, 112, 112]
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,   # -> [B, 64, 56, 56]
            backbone.layer1,    # -> [B, 64, 56, 56]
            backbone.layer2,    # -> [B, 128, 28, 28]
            backbone.layer3,    # -> [B, 256, 14, 14]
            backbone.layer4,    # -> [B, 512, 7, 7]
        )
        # 全局平均池化 -> [B, 512]
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.out_dim = 512  # ResNet-18 最后一层通道数

    @staticmethod
    def _try_load_resnet_weights(backbone, resnet_ckpt: str | None):
        """尝试加载 ResNet-18 预训练权重。"""
        if resnet_ckpt is not None and os.path.isfile(resnet_ckpt):
            state_dict = torch.load(resnet_ckpt, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state_dict)
            logging.getLogger(__name__).info(
                "MotionEncoder: 从本地加载 ResNet-18 权重: %s", resnet_ckpt
            )
        elif resnet_ckpt is not None:
            logging.getLogger(__name__).warning(
                "MotionEncoder: ResNet-18 权重文件不存在: %s, 使用随机初始化", resnet_ckpt
            )

    def reload_pretrained_weights(self):
        """
        无条件重新加载 ResNet-18 ImageNet 预训练权重 (包括 BN running stats)。

        为什么无条件重新加载:
        不同 VLM 的 from_pretrained 对新增子模块的处理方式不同:
        - 有些会把参数放在 meta device (全零检测可以捕获)
        - 有些会用 _init_weights 做默认初始化 (参数非零但不是 ImageNet 权重)
        - 但无论哪种, BN 的 running_mean/running_var (buffers) 都会丢失

        丢失 BN running stats 会导致 BN 输出值域异常 → 后续层 NaN。
        因此这里直接无条件重新加载, 确保 parameters + buffers 都正确。
        """
        _logger = logging.getLogger(__name__)

        if self._resnet_ckpt is not None and os.path.isfile(self._resnet_ckpt):
            _logger.info(
                "MotionEncoder: 加载 ResNet-18 ImageNet 权重 (含 BN running stats): %s",
                self._resnet_ckpt,
            )

            full_state_dict = torch.load(self._resnet_ckpt, map_location="cpu", weights_only=True)

            # 确定当前模型的目标 dtype (通常是 bfloat16, 因为宿主模型用 torch_dtype=bfloat16 加载)
            # ImageNet 权重是 float32, 需要转为目标 dtype 后再 assign, 否则 features 的参数会变回 float32
            target_dtype = None
            for p in self.features.parameters():
                if not p.is_meta:
                    target_dtype = p.dtype
                    break

            # 构建 self.features 对应的 key 映射
            # nn.Sequential 索引: 0=conv1, 1=bn1, 2=relu, 3=maxpool, 4=layer1, ...
            prefix_map = {
                "conv1": "0", "bn1": "1",
                "layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7",
            }

            features_state = {}
            for key, value in full_state_dict.items():
                prefix = key.split(".")[0]
                if prefix in prefix_map:
                    new_key = key.replace(prefix, prefix_map[prefix], 1)
                    # 将 float32 权重转为目标 dtype, 避免 assign=True 后 features 变回 float32
                    if target_dtype is not None and value.dtype != target_dtype:
                        value = value.to(dtype=target_dtype)
                    features_state[new_key] = value

            # assign=True: 直接替换参数张量 (避免 meta tensor 的 copy_ no-op)
            # strict=False: BN 的 num_batches_tracked 等可选 key 不报错
            self.features.load_state_dict(features_state, strict=False, assign=True)

            _logger.info(
                "MotionEncoder: 成功加载 %d 个 parameters + buffers (含 BN running_mean/var)",
                len(features_state),
            )
            return True
        else:
            # 权重文件不存在, 用 kaiming 初始化修复
            _logger.warning(
                "reload_pretrained_weights: 权重文件不可用: %s, "
                "用 kaiming 初始化 (非 ImageNet 预训练, 可能影响收敛)",
                self._resnet_ckpt,
            )
            first_param = next(self.features.parameters())
            device = first_param.device if not first_param.is_meta else torch.device("cpu")
            for name, param in self.features.named_parameters():
                new_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
                if "weight" in name and new_tensor.dim() >= 2:
                    nn.init.kaiming_normal_(new_tensor, mode="fan_out", nonlinearity="relu")
                elif "bias" in name:
                    nn.init.zeros_(new_tensor)
                elif "weight" in name:  # BN weight
                    nn.init.ones_(new_tensor)
                else:
                    nn.init.zeros_(new_tensor)
                parts = name.split(".")
                parent = self.features
                for part in parts[:-1]:
                    if part.isdigit():
                        parent = parent[int(part)]
                    else:
                        parent = getattr(parent, part)
                setattr(parent, parts[-1], nn.Parameter(new_tensor, requires_grad=param.requires_grad))

            # 也重置 BN running stats buffers
            for module in self.features.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean = torch.zeros(module.num_features, device=device)
                    module.running_var = torch.ones(module.num_features, device=device)
                    module.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=device)

            _logger.info("MotionEncoder: 使用 kaiming 初始化修复完成 (含 BN reset)")
            return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 3, 224, 224] 差分图序列
               或 [B*T, 3, 224, 224] 已展平的输入
        Returns:
            [B, T, 512] 或 [B*T, 512] 运动特征
        """
        has_time_dim = x.dim() == 5
        batch_size = num_frames = 0
        if has_time_dim:
            batch_size, num_frames = x.shape[:2]
            x = x.reshape(batch_size * num_frames, *x.shape[2:])  # -> [B*T, 3, 224, 224]

        feat = self.features(x)      # -> [B*T, 512, 7, 7]
        feat = self.pool(feat)        # -> [B*T, 512, 1, 1]
        feat = feat.flatten(1)        # -> [B*T, 512]

        if has_time_dim:
            feat = feat.reshape(batch_size, num_frames, -1)  # -> [B, T, 512]

        return feat


# ============================================================================
# MotionSSM Encoder (Mamba/MotionSSM 风格时序建模)
# ============================================================================

class MotionSSMBlock(nn.Module):
    """轻量 Mamba 风格时序混合块: depthwise temporal mixing + gated residual。"""

    def __init__(self, dim: int, expand: int = 2, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * expand
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, inner_dim * 2)
        self.forward_conv = nn.Conv1d(inner_dim, inner_dim, kernel_size, padding=kernel_size - 1, groups=inner_dim)
        self.backward_conv = nn.Conv1d(inner_dim, inner_dim, kernel_size, padding=kernel_size - 1, groups=inner_dim)
        self.mix_gate = nn.Linear(dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        u, gate = self.in_proj(h).chunk(2, dim=-1)
        u_t = u.transpose(1, 2)
        seq_len = u_t.shape[-1]

        fw = self.forward_conv(u_t)[..., :seq_len]
        bw = torch.flip(self.backward_conv(torch.flip(u_t, dims=[-1]))[..., :seq_len], dims=[-1])
        selective_gate = torch.sigmoid(self.mix_gate(h)).transpose(1, 2)
        mixed = fw + selective_gate * bw
        mixed = mixed.transpose(1, 2)
        mixed = mixed * torch.sigmoid(gate)
        mixed = self.out_proj(mixed)
        mixed = self.dropout(mixed)
        return residual + mixed


class MotionSSMEncoder(nn.Module):
    """
    MotionSSM: 先用 ResNet-18 提取每帧差分图空间特征, 再用 Mamba 风格时序块建模跨帧运动状态。

    设计动机:
    1. 复用 ResNet 的稳定低级运动边缘特征, 避免从零学空间 stem
    2. 用时序状态混合块替代纯 CNN, 补上 ResNet-18 缺失的 temporal memory
    3. 不依赖额外 CUDA 扩展, 便于直接接入现有训练链路
    """

    def __init__(
        self,
        resnet_ckpt: str | None = _DEFAULT_RESNET18_CKPT,
        depth: int = 4,
        kernel_size: int = 5,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spatial_encoder = MotionEncoder(resnet_ckpt=resnet_ckpt)
        self.temporal_proj = nn.Linear(self.spatial_encoder.out_dim, self.spatial_encoder.out_dim)
        self.temporal_blocks = nn.ModuleList([
            MotionSSMBlock(
                dim=self.spatial_encoder.out_dim,
                expand=expand,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.temporal_norm = nn.LayerNorm(self.spatial_encoder.out_dim)
        self.out_dim = self.spatial_encoder.out_dim

    def reload_pretrained_weights(self):
        spatial_reloaded = self.spatial_encoder.reload_pretrained_weights()
        proj_reinit = _reinitialize_module_parameters(self.temporal_proj, "MotionSSM temporal projection")
        block_reinit = _reinitialize_module_parameters(self.temporal_blocks, "MotionSSM temporal blocks")
        norm_reinit = _reinitialize_module_parameters(self.temporal_norm, "MotionSSM temporal norm")
        return spatial_reloaded or proj_reinit or block_reinit or norm_reinit

    def set_trainable(self, *, spatial: bool = True, temporal: bool = True) -> None:
        """分别控制 spatial / temporal 子模块的可训练状态。"""
        if spatial:
            unfreeze_module(self.spatial_encoder)
        else:
            freeze_module(self.spatial_encoder)

        for module in (self.temporal_proj, self.temporal_blocks, self.temporal_norm):
            if temporal:
                unfreeze_module(module)
            else:
                freeze_module(module)

    def iter_spatial_parameters(self):
        return self.spatial_encoder.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.spatial_encoder(x)
        feat = self.temporal_proj(feat)
        for block in self.temporal_blocks:
            feat = block(feat)
        feat = self.temporal_norm(feat)
        return feat


class MotionProjector(nn.Module):
    """
    将编码器的特征映射到 LLM 的隐层维度。

    结构: Linear(motion_dim -> hidden) -> GELU -> Linear(hidden -> hidden)

    参考 LLaVA-1.5: 两层 MLP + GELU 的 projector 比单层 Linear
    在 alignment 阶段收敛更快, 最终效果也更好。
    """

    def __init__(self, motion_dim: int = 512, hidden_size: int = 3584):
        """
        Args:
            motion_dim: 运动编码器输出维度 (ResNet-18=512, SigLIP-B=768, etc.)
            hidden_size: LLM 隐层维度
                - Qwen2.5-VL-7B: 3584
                - LLaVA-OneVision-7B: 4096
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(motion_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, motion_dim] 运动特征
        Returns:
            [B, T, hidden_size] 映射后的 Motion Tokens
        """
        return self.proj(x)


# ============================================================================
# SigLIP Motion Encoder (方案 B1)
# ============================================================================

class SigLIPMotionEncoder(nn.Module):
    """
    基于 SigLIP 预训练视觉编码器的运动编码器。

    优势:
    1. SigLIP 在大量图文对上预训练, embedding 空间天然接近语言模型
    2. 对语义信息的编码远强于 ImageNet ResNet-18
    3. alignment 更容易收敛

    输入: 差分图 [B, T_diff, 3, 224, 224]
    输出: 运动特征 [B, T_diff, 768] (SigLIP-B/16 的输出维度)

    使用 google/siglip-base-patch16-224 作为默认模型。
    Stage 1 可选冻结/训练 SigLIP。
    """

    def __init__(
        self,
        model_name_or_path: str = "google/siglip-base-patch16-224",
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 0,
    ):
        super().__init__()
        self._model_name_or_path = model_name_or_path
        self._freeze_encoder = freeze_encoder
        self._unfreeze_last_n_layers = max(0, int(unfreeze_last_n_layers))
        self._loaded = False  # 延迟加载标记

        # SigLIP 输出维度 (base=768, large=1024)
        self.out_dim = 768  # 默认 base

        # 延迟初始化: 实际模型在 reload_pretrained_weights 中加载
        # 这样在 from_pretrained 阶段不会触发下载
        self.vision_model = None
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _apply_trainable_policy(self) -> None:
        _logger = logging.getLogger(__name__)
        if self.vision_model is None or not hasattr(self.vision_model, "parameters"):
            return

        for p in self.vision_model.parameters():
            p.requires_grad = not self._freeze_encoder

        if not self._freeze_encoder:
            if self._unfreeze_last_n_layers > 0:
                _logger.info(
                    "SigLIPMotionEncoder: 全量可训练，忽略 unfreeze_last_n_layers=%d",
                    self._unfreeze_last_n_layers,
                )
            return

        if self._unfreeze_last_n_layers <= 0:
            return

        vision_tower = getattr(self.vision_model, "vision_model", None)
        encoder = getattr(vision_tower, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None:
            _logger.warning(
                "SigLIPMotionEncoder: 无法定位 encoder.layers，跳过最后 %d 层解冻",
                self._unfreeze_last_n_layers,
            )
            return

        num_layers = len(layers)
        if num_layers == 0:
            return

        n = min(self._unfreeze_last_n_layers, num_layers)
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

        post_layernorm = getattr(vision_tower, "post_layernorm", None)
        if post_layernorm is not None:
            for p in post_layernorm.parameters():
                p.requires_grad = True

        _logger.info(
            "SigLIPMotionEncoder: 冻结 SigLIP，仅解冻最后 %d/%d 个 vision layers + post_layernorm",
            n,
            num_layers,
        )

    def reload_pretrained_weights(self):
        """加载 SigLIP 预训练权重。"""
        _logger = logging.getLogger(__name__)

        if self._loaded and self.vision_model is not None:
            _logger.info("SigLIPMotionEncoder: 已加载, 跳过重新加载")
            return True

        try:
            from transformers import SiglipVisionModel

            _logger.info("SigLIPMotionEncoder: 加载 SigLIP 模型: %s", self._model_name_or_path)
            self.vision_model = SiglipVisionModel.from_pretrained(
                self._model_name_or_path,
                torch_dtype=torch.bfloat16,
            )
            # 获取实际输出维度
            self.out_dim = self.vision_model.config.hidden_size
            self._apply_trainable_policy()
            if self._freeze_encoder:
                if self._unfreeze_last_n_layers > 0:
                    _logger.info(
                        "SigLIPMotionEncoder: 冻结主体，仅解冻最后 %d 层 (out_dim=%d)",
                        self._unfreeze_last_n_layers,
                        self.out_dim,
                    )
                else:
                    _logger.info("SigLIPMotionEncoder: 冻结 SigLIP (out_dim=%d)", self.out_dim)
            else:
                _logger.info("SigLIPMotionEncoder: SigLIP 可训练 (out_dim=%d)", self.out_dim)

            self._loaded = True
            return True
        except Exception as e:
            raise RuntimeError(
                "Unable to load the requested SigLIP encoder; this release will not substitute a random CNN."
            ) from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 3, 224, 224] 差分图序列
        Returns:
            [B, T, out_dim] 运动特征
        """
        if self.vision_model is None:
            _ = self.reload_pretrained_weights()
        assert self.vision_model is not None

        has_time_dim = x.dim() == 5
        batch_size = num_frames = 0
        if has_time_dim:
            batch_size, num_frames = x.shape[:2]
            x = x.reshape(batch_size * num_frames, *x.shape[2:])

        # SigLIP 需要特定的归一化 (已在 DIFF_TRANSFORM 中用 ImageNet 归一化)
        # SigLIP 接受 [N, 3, 224, 224] 输入
        weight_dtype = next(self.vision_model.parameters()).dtype
        if x.dtype != weight_dtype:
            x = x.to(dtype=weight_dtype)

        if hasattr(self.vision_model, 'config'):
            # 真正的 SigLIP model
            outputs = self.vision_model(pixel_values=x)
            # 使用 pooler_output 或 last_hidden_state 的 mean pool
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                feat = outputs.pooler_output  # [N, hidden_size]
            else:
                feat = outputs.last_hidden_state.mean(dim=1)  # [N, hidden_size]
        else:
            # 回退的简单网络
            feat = self.vision_model(x)  # [N, out_dim]

        if has_time_dim:
            feat = feat.reshape(batch_size, num_frames, -1)

        return feat


# ============================================================================
# Shared ViT Motion Encoder (方案 B3)
# ============================================================================

class SharedViTMotionEncoder(nn.Module):
    """
    复用宿主 VLM 的 ViT 编码差分图 (最简方案)。

    优势:
    1. 零额外视觉编码器参数 — motion embedding 天然在同一空间
    2. 只需要一个 Cross-Attention 层来捕获差分图与关键帧的关系
    3. 如果不加 cross-attention, 最简形式只需要 MLP projector

    策略:
    - 将差分图送入同一个 ViT (冻结), 得到 per-frame 特征
    - 通过 mean pool 得到每帧一个 token
    - 可选: 加一层 cross-attention 与关键帧特征交互
    - 通过 MLP Projector 映射到 motion token 空间

    注意: 这个类不持有 ViT 实例, 而是在 forward 时接收 ViT 的输出。
    具体的 ViT 调用在各 delta model (qwen_delta, llava_delta, internvl_delta) 中完成。
    """

    def __init__(self, vit_hidden_size: int = 1152, use_cross_attention: bool = False):
        """
        Args:
            vit_hidden_size: ViT 输出的 hidden_size (经 PatchMerger 后的维度)
                - Qwen2.5-VL: 3584 (原始 ViT 1280 → PatchMerger → 3584)
                - Qwen3-VL: 4096 (原始 ViT hidden_size → PatchMerger → 4096)
                - SigLIP (LLaVA): 1152 (不经过 PatchMerger)
                - InternViT-300M: 3584 (经 MLP projector 后与 LLM 对齐)
            use_cross_attention: 是否加 cross-attention 层
        """
        super().__init__()
        self.out_dim = vit_hidden_size
        self.use_cross_attention = use_cross_attention

        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=vit_hidden_size,
                num_heads=8,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(vit_hidden_size)

        # Shared ViT 不需要自己做 reload
        self._vit_ref = None  # 由外部设置

    def reload_pretrained_weights(self):
        """SharedViT 不需要加载权重 (复用宿主 ViT)。"""
        _logger = logging.getLogger(__name__)
        _logger.info("SharedViTMotionEncoder: 复用宿主 VLM 的 ViT, 无需单独加载权重")

        # 初始化 cross_attention (如果有) 的权重
        if self.use_cross_attention:
            # xavier 初始化
            nn.init.xavier_uniform_(self.cross_attn.in_proj_weight)
            nn.init.xavier_uniform_(self.cross_attn.out_proj.weight)
        return True

    def set_vit_reference(self, vit_module: nn.Module):
        """设置对宿主 ViT 的引用 (在模型初始化后调用)。"""
        self._vit_ref = vit_module

    def forward(self, x: torch.Tensor, keyframe_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对于 SharedViT, 输入 x 已经是 ViT 编码后的特征。

        在 qwen_delta/llava_delta/internvl_delta 中, 会:
        1. 用冻结的 ViT 编码差分图 → 得到 per-patch features
        2. Mean pool → per-frame features [B, T, vit_hidden_size]
        3. 传入这里做可选的 cross-attention 处理

        Args:
            x: [B, T, vit_hidden_size] 差分图经 ViT 编码后的 per-frame 特征
            keyframe_features: [B, N_kf, vit_hidden_size] 可选的关键帧特征 (用于 cross-attention)
        Returns:
            [B, T, vit_hidden_size]
        """
        if self.use_cross_attention and keyframe_features is not None:
            # Cross-attention: motion tokens attend to keyframe features
            residual = x
            attn_out, _ = self.cross_attn(
                query=x,
                key=keyframe_features,
                value=keyframe_features,
            )
            x = self.cross_norm(residual + attn_out)

        return x


def pool_motion_tokens(tokens: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """将一个 motion interval 的 tokens 聚合成单个 query。

    H.264 accumulated residual 启发:
    - mean:   旧策略, 所有中间 diff 一视同仁
    - last:   直接保留区间末端 token, 更贴近 end-start residual
    - linear: 对后段 token 赋更大权重, 在保留累计趋势的同时比 last 更稳
    """
    if tokens.ndim != 2:
        raise ValueError(f"pool_motion_tokens 期望 2D tensor, 实际是 {tokens.shape}")
    if tokens.shape[0] == 0:
        raise ValueError("pool_motion_tokens 收到空 interval")

    if tokens.shape[0] == 1:
        return tokens[0]

    if mode == "mean":
        return tokens.mean(dim=0)
    if mode == "last":
        return tokens[-1]
    if mode == "linear":
        weights = torch.arange(1, tokens.shape[0] + 1, device=tokens.device, dtype=tokens.dtype)
        weights = weights / weights.sum()
        return (tokens * weights[:, None]).sum(dim=0)

    raise ValueError(f"不支持的 alignment_motion_pool: {mode}")


class AlignmentHead(nn.Module):
    """
    InfoNCE 对比学习 Alignment Loss (Stage 1 专用)。

    核心思想 (参考 FAME, MSCL, ViCLIP 等):
    - Motion interval i 的 token 应该和对应 visual interval 的特征相似 (正样本对)
    - 当多个 motion interval 映射到同一个 visual interval 时, 它们共享同一组正样本
    - 只有来自不同 visual interval 的 pair 才视为负样本
    - 这样 motion tokens 学到的是"区分不同时间段运动"的能力, 而不是被假负样本压制

    Loss = 双向 multi-positive InfoNCE + 轻量 cosine 辅助:

    $$\\mathcal{L} = \\frac{1}{2}(\\text{CE}_{m \\to v}^{multi+} + \\text{CE}_{v \\to m}^{multi+}) + \\alpha \\cdot (1 - \\cos(m, v))$$

    相比旧版 cosine + SmoothL1 回归:
    1. 引入负样本, 同时避免一对多对齐时的假负样本
    2. 可学习温度 logit_scale → 自动调节对比难度
    3. 去掉 SmoothL1 → 不强制幅值相等, 只要求语义空间一致
    """

    _DIST_GROUP_ID_SHIFT = 1_000_000_000

    def __init__(
        self,
        hidden_size: int = 3584,
        cosine_weight: float = 0.1,
        gather_global_negatives: bool = True,
    ):
        super().__init__()
        # 可学习温度: exp(logit_scale) ≈ 1/0.07 ≈ 14.3
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cosine_weight = cosine_weight
        self.gather_global_negatives = gather_global_negatives

    def set_gather_global_negatives(self, enabled: bool) -> None:
        self.gather_global_negatives = bool(enabled)

    @staticmethod
    def _dist_ready() -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    @staticmethod
    def _build_positive_mask(
        num_pairs: int,
        device: torch.device,
        target_group_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target_group_ids is None:
            return torch.eye(num_pairs, device=device, dtype=torch.bool)

        assert target_group_ids is not None
        group_ids = target_group_ids.reshape(-1).to(device=device)
        if group_ids.shape[0] != num_pairs:
            raise ValueError(
                f"target_group_ids 长度与 pair 数不一致: {group_ids.shape[0]} vs {num_pairs}"
            )
        return group_ids[:, None].eq(group_ids[None, :])

    @classmethod
    def _all_gather_lengths(cls, local_length: int, device: torch.device) -> list[int]:
        world_size = dist.get_world_size()
        local_tensor = torch.tensor([local_length], device=device, dtype=torch.long)
        gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor)
        return [int(x.item()) for x in gathered]

    @classmethod
    def _gather_tensor_with_padding(
        cls,
        local_tensor: torch.Tensor,
        *,
        keep_local_grad: bool,
    ) -> tuple[torch.Tensor, list[int], int]:
        if local_tensor.ndim == 0:
            local_tensor = local_tensor.unsqueeze(0)

        device = local_tensor.device
        sizes = cls._all_gather_lengths(local_tensor.shape[0], device)
        max_size = max(sizes, default=local_tensor.shape[0])
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if local_tensor.shape[0] < max_size:
            pad_shape = (max_size - local_tensor.shape[0],) + tuple(local_tensor.shape[1:])
            pad = local_tensor.new_zeros(pad_shape)
            padded_local = torch.cat([local_tensor.detach(), pad], dim=0)
        else:
            padded_local = local_tensor.detach()

        gathered = [torch.zeros_like(padded_local) for _ in range(world_size)]
        dist.all_gather(gathered, padded_local)
        if keep_local_grad:
            gathered[rank][:local_tensor.shape[0]] = local_tensor

        chunks = [tensor[:sizes[idx]] for idx, tensor in enumerate(gathered) if sizes[idx] > 0]
        if chunks:
            global_tensor = torch.cat(chunks, dim=0)
        else:
            global_tensor = local_tensor.new_zeros((0,) + tuple(local_tensor.shape[1:]))
        return global_tensor, sizes, rank

    @classmethod
    def _gather_group_ids(
        cls,
        target_group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
        group_ids = target_group_ids.reshape(-1).to(dtype=torch.long)
        rank = dist.get_rank()
        local_group_ids = group_ids + rank * cls._DIST_GROUP_ID_SHIFT
        global_group_ids, sizes, _ = cls._gather_tensor_with_padding(
            local_group_ids,
            keep_local_grad=False,
        )
        return local_group_ids, global_group_ids.to(dtype=torch.long), sizes, rank

    @staticmethod
    def _multi_positive_info_nce(
        logits: torch.Tensor,
        positive_mask: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if candidate_mask is None:
            candidate_mask = torch.ones_like(positive_mask, dtype=torch.bool)
        candidate_mask = candidate_mask | positive_mask
        valid_rows = positive_mask.any(dim=-1) & candidate_mask.any(dim=-1)
        if not valid_rows.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        masked_pos_logits = logits.masked_fill(~positive_mask, torch.finfo(logits.dtype).min)
        masked_all_logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
        pos_logsumexp = torch.logsumexp(masked_pos_logits, dim=-1)
        all_logsumexp = torch.logsumexp(masked_all_logits, dim=-1)
        loss = -(pos_logsumexp - all_logsumexp)
        return loss[valid_rows].mean()

    def forward(
        self,
        motion_tokens: torch.Tensor,
        visual_targets: torch.Tensor,
        target_group_ids: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, object]:
        """
        计算 InfoNCE 对比学习 alignment loss。

        Args:
            motion_tokens: [N, D] 每个 interval 的 pooled motion token
            visual_targets: [N, D] 每个 interval 对应的 visual target (帧差特征)
            target_group_ids: [N], 同一 visual interval 的 pair 共享同一个 group id
            negative_mask: [N, N], True 表示该 target 可作为当前 query 的负例
        Returns:
            dict: {"loss": scalar, "contrastive_loss": scalar, "cosine_loss": scalar,
                   "avg_cosine_sim": float, "logit_scale": float}
        """
        N = motion_tokens.shape[0]

        # 保存原始 dtype, 在 float32 下计算避免 bf16 精度问题
        orig_dtype = motion_tokens.dtype
        motion_tokens = motion_tokens.float()
        visual_targets = visual_targets.float()

        # L2 归一化
        motion_norm = nn.functional.normalize(motion_tokens, dim=-1)  # [N, D]
        target_norm = nn.functional.normalize(visual_targets, dim=-1)  # [N, D]

        # 可学习温度, clamp 防止梯度爆炸
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        global_num_pairs = N
        avg_candidates = 0.0
        use_global_negatives = (
            negative_mask is None
            and self.gather_global_negatives
            and N > 0
            and self._dist_ready()
        )

        if use_global_negatives:
            global_target_norm, sizes, rank = self._gather_tensor_with_padding(
                target_norm,
                keep_local_grad=False,
            )
            global_motion_norm, _, _ = self._gather_tensor_with_padding(
                motion_norm,
                keep_local_grad=True,
            )
            global_num_pairs = int(sum(sizes))

            if target_group_ids is None:
                col_offset = sum(sizes[:rank])
                positive_mask = torch.zeros(
                    (N, global_num_pairs),
                    device=motion_tokens.device,
                    dtype=torch.bool,
                )
                positive_mask[:, col_offset:col_offset + N] = torch.eye(
                    N,
                    device=motion_tokens.device,
                    dtype=torch.bool,
                )
            else:
                local_group_ids, global_group_ids, _, _ = self._gather_group_ids(target_group_ids)
                positive_mask = local_group_ids[:, None].eq(global_group_ids[None, :])

            candidate_mask = None
            if global_num_pairs >= 2:
                logits_m2v = logit_scale * motion_norm @ global_target_norm.T
                logits_v2m = logit_scale * target_norm @ global_motion_norm.T
                loss_m2v = self._multi_positive_info_nce(logits_m2v, positive_mask)
                loss_v2m = self._multi_positive_info_nce(logits_v2m, positive_mask)
                contrastive_loss = (loss_m2v + loss_v2m) / 2.0
            else:
                contrastive_loss = torch.tensor(0.0, device=motion_tokens.device, dtype=torch.float32)
        else:
            positive_mask = self._build_positive_mask(N, motion_tokens.device, target_group_ids)
            candidate_mask = None
            if negative_mask is not None:
                candidate_mask = negative_mask.to(device=motion_tokens.device, dtype=torch.bool)
                if candidate_mask.shape != positive_mask.shape:
                    raise ValueError(
                        f"negative_mask 形状与 positive_mask 不一致: {candidate_mask.shape} vs {positive_mask.shape}"
                    )
                candidate_mask = candidate_mask & (~positive_mask)
            if N >= 2:
                # ---- multi-positive InfoNCE 对比学习 ----
                logits = logit_scale * motion_norm @ target_norm.T  # [N, N]
                loss_m2v = self._multi_positive_info_nce(logits, positive_mask, candidate_mask=candidate_mask)
                loss_v2m = self._multi_positive_info_nce(logits.T, positive_mask.T, candidate_mask=None if candidate_mask is None else candidate_mask.T)
                contrastive_loss = (loss_m2v + loss_v2m) / 2.0
            else:
                # N < 2: 没有负样本, 退化为 cosine loss (fallback)
                contrastive_loss = torch.tensor(0.0, device=motion_tokens.device, dtype=torch.float32)

        effective_candidate_mask = positive_mask if candidate_mask is None else (candidate_mask | positive_mask)

        # ---- 轻量 cosine 辅助 loss ----
        # 逐对 cosine similarity (不依赖负样本, 即使 N=1 也有效)
        cosine_sim = (motion_norm * target_norm).sum(dim=-1)  # [N]
        cosine_loss = (1.0 - cosine_sim).mean()

        # 总 loss
        loss = contrastive_loss + self.cosine_weight * cosine_loss

        # 统计信息
        with torch.no_grad():
            avg_cosine = cosine_sim.mean().item()
            scale_val = logit_scale.item()
            avg_positive = positive_mask.float().sum(dim=-1).mean().item()
            avg_candidates = effective_candidate_mask.float().sum(dim=-1).mean().item()

        return {
            "loss": loss.to(orig_dtype),
            "contrastive_loss": contrastive_loss.to(orig_dtype),
            "cosine_loss": cosine_loss.to(orig_dtype),
            "avg_cosine_sim": avg_cosine,
            "logit_scale": scale_val,
            "avg_positive_targets": avg_positive,
            "avg_candidate_targets": avg_candidates,
            "global_num_pairs": global_num_pairs,
        }


class MotionEncoderWithProjector(nn.Module):
    """
    打包的运动编码 pipeline: Encoder + MLP Projector。

    支持多种编码器类型:
    - "resnet": ResNet-18 (默认, ImageNet 预训练, 11M 参数)
    - "motionssm": ResNet 空间 stem + MotionSSM 时序状态混合
    - "siglip": SigLIP ViT (预训练视觉编码器, embedding 天然接近语言模型)
    - "shared_vit": 复用宿主 VLM 的 ViT (零额外参数)

    通过构造参数 motion_encoder_type 选择编码器类型。
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        resnet_ckpt: str | None = _DEFAULT_RESNET18_CKPT,
        motion_encoder_type: str = "resnet",
        siglip_model_path: str = "google/siglip-base-patch16-224",
        freeze_motion_encoder: bool = False,
        vit_hidden_size: int = 1152,
        use_cross_attention: bool = False,
        siglip_unfreeze_last_n_layers: int = 0,
    ):
        super().__init__()
        self.motion_encoder_type = motion_encoder_type
        self._hidden_size = hidden_size
        self.freeze_motion_encoder = freeze_motion_encoder

        _logger = logging.getLogger(__name__)
        _logger.info(
            "MotionEncoderWithProjector: type=%s, hidden_size=%d",
            motion_encoder_type, hidden_size,
        )

        # 根据类型创建编码器
        if motion_encoder_type == "resnet":
            self.encoder = MotionEncoder(resnet_ckpt=resnet_ckpt)
        elif motion_encoder_type == "motionssm":
            self.encoder = MotionSSMEncoder(resnet_ckpt=resnet_ckpt)
        elif motion_encoder_type == "siglip":
            self.encoder = SigLIPMotionEncoder(
                model_name_or_path=siglip_model_path,
                freeze_encoder=freeze_motion_encoder,
                unfreeze_last_n_layers=siglip_unfreeze_last_n_layers,
            )
        elif motion_encoder_type == "shared_vit":
            self.encoder = SharedViTMotionEncoder(
                vit_hidden_size=vit_hidden_size,
                use_cross_attention=use_cross_attention,
            )
        else:
            raise ValueError(
                f"不支持的 motion_encoder_type: {motion_encoder_type}. "
                f"支持: {MOTION_ENCODER_TYPES}"
            )

        # Projector: 编码器输出维度 → LLM 隐层维度
        self.projector = MotionProjector(
            motion_dim=self.encoder.out_dim,
            hidden_size=hidden_size,
        )
        # Stage 1 alignment projector:
        # 对齐目标是 ViT 经 PatchMerger 后的 token 空间 (hidden_size)
        # 使用与 projector 同结构的两层 MLP，避免单层 Linear 扛过重的语义映射。
        self._align_out_dim = hidden_size
        self.align_proj = MotionProjector(
            motion_dim=self.encoder.out_dim,
            hidden_size=hidden_size,
        )
        # Stage 1 alignment head
        self.alignment_head = AlignmentHead(hidden_size=hidden_size)

    def reload_pretrained_weights(self):
        """
        重新加载编码器的预训练权重, 并重新初始化 Projector 和 AlignmentHead。

        必须在宿主模型 from_pretrained 完成后调用,
        以修复 meta device 导致的权重丢失问题。

        Returns:
            bool: 是否执行了操作
        """
        encoder_reloaded = self.encoder.reload_pretrained_weights()
        proj_reinitialized = _reinitialize_module_parameters(self.projector, "Projector")
        align_proj_reinitialized = _reinitialize_module_parameters(self.align_proj, "AlignProj")
        ah_reinitialized = _reinitialize_module_parameters(
            self.alignment_head,
            "AlignmentHead",
            special_init=lambda name, tensor, _: ((tensor.fill_(np.log(1 / 0.07))), True)[1] if "logit_scale" in name else False,
        )
        # 无条件强制重置 logit_scale，彻底绕开 needs_reinit 误判
        # DeepSpeed ZeRO 在 materialize 时可能把标量参数置零，导致 exp(0)=1.0
        correct_val = float(np.log(1.0 / 0.07))  # ≈ 2.659
        with torch.no_grad():
            self.alignment_head.logit_scale.fill_(correct_val)
        logging.getLogger(__name__).info(
            "AlignmentHead.logit_scale 强制重置为 %.4f (exp=%.2f ≈ 1/0.07)",
            correct_val, np.exp(correct_val),
        )
        return encoder_reloaded or proj_reinitialized or align_proj_reinitialized or ah_reinitialized

    def configure_training_state(
        self,
        train_alignment_head: bool = True,
        *,
        train_projector: bool = True,
        train_alignment_projector: bool = True,
        freeze_spatial_encoder: bool = False,
    ) -> None:
        """按训练阶段恢复 motion module 的可训练状态。"""
        freeze_module(self)
        encoder_trainable = not self.freeze_motion_encoder
        spatial_frozen = False

        if encoder_trainable:
            if freeze_spatial_encoder and self.motion_encoder_type == "motionssm":
                self.encoder.set_trainable(spatial=False, temporal=True)
                spatial_frozen = True
            else:
                unfreeze_module(self.encoder)

        if train_projector:
            unfreeze_module(self.projector)
        if train_alignment_projector:
            unfreeze_module(self.align_proj)
        if train_alignment_head:
            unfreeze_module(self.alignment_head)

        logging.getLogger(__name__).info(
            "MotionEncoderWithProjector 训练策略: encoder_trainable=%s, spatial_frozen=%s, projector_trainable=%s, align_projector_trainable=%s, alignment_head_trainable=%s",
            encoder_trainable,
            spatial_frozen,
            train_projector,
            train_alignment_projector,
            train_alignment_head,
        )

    def set_spatial_encoder_trainable(self, trainable: bool) -> bool:
        if self.motion_encoder_type != "motionssm":
            return False
        self.encoder.set_trainable(spatial=trainable, temporal=True)
        logging.getLogger(__name__).info(
            "MotionEncoderWithProjector: MotionSSM spatial_encoder trainable=%s",
            trainable,
        )
        return True

    def bootstrap_projector_from_alignment(self) -> bool:
        if type(self.projector) is not type(self.align_proj):
            logging.getLogger(__name__).warning(
                "projector 与 align_proj 类型不一致，跳过权重复制"
            )
            return False
        self.projector.load_state_dict(self.align_proj.state_dict())
        logging.getLogger(__name__).info(
            "已将 Stage1 alignment projector 权重复制到 Stage2 projector"
        )
        return True

    def encode_motion(self, diff_frames: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        编码差分图, 返回 LLM 空间的 motion tokens (Stage 2 推理/SFT 用)。

        Args:
            diff_frames: [B, T_diff, 3, 224, 224]
        Returns:
            [B, T_diff, hidden_size]
        """
        return self.forward(diff_frames, **kwargs)

    def encode_motion_for_alignment(self, diff_frames: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        编码差分图, 返回 ViT 空间的 motion features (Stage 1 alignment 用)。

        同样走 encoder, 但经过 align_proj 而非 projector, 保持与 ViT 特征同维度。

        Args:
            diff_frames: [B, T_diff, 3, 224, 224]
        Returns:
            [B, T_diff, vit_hidden_size]
        """
        encoder_params = list(self.encoder.parameters())
        if encoder_params:
            weight_dtype = encoder_params[0].dtype
            if diff_frames.dtype != weight_dtype:
                diff_frames = diff_frames.to(dtype=weight_dtype)

        if self.motion_encoder_type == "shared_vit":
            # SharedViTMotionEncoder.forward() 只接受 keyframe_features 参数
            shared_kwargs = {
                k: v for k, v in kwargs.items()
                if k in {"keyframe_features"}
            }
            motion_feat = self.encoder(diff_frames, **shared_kwargs)
        else:
            motion_feat = self.encoder(diff_frames)  # [B, T, encoder_out_dim]

        return self.align_proj(motion_feat)  # [B, T, vit_hidden_size]

    def forward(self, diff_frames: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        差分图 -> Motion Tokens (与 LLM 同维度)。

        Args:
            diff_frames: [B, T_diff, 3, 224, 224]
            **kwargs: 额外参数 (如 shared_vit 的 keyframe_features)
        Returns:
            [B, T_diff, hidden_size]
        """
        # 将输入 cast 到与 encoder 权重相同的 dtype
        encoder_params = list(self.encoder.parameters())
        if encoder_params:
            weight_dtype = encoder_params[0].dtype
            if diff_frames.dtype != weight_dtype:
                diff_frames = diff_frames.to(dtype=weight_dtype)

        if self.motion_encoder_type == "shared_vit":
            # SharedViT: diff_frames 已经是 ViT 编码后的 per-frame 特征
            shared_kwargs = {
                k: v for k, v in kwargs.items()
                if k in {"keyframe_features", "keyframe_attention_mask", "motion_attention_mask"}
            }
            motion_feat = self.encoder(diff_frames, **shared_kwargs)
        else:
            motion_feat = self.encoder(diff_frames)  # [B, T, encoder_out_dim]

        # 诊断: 检查 encoder 输出
        if torch.isnan(motion_feat).any():
            logging.getLogger(__name__).warning(
                "[MotionModule] encoder (%s) 输出含 NaN (shape=%s)",
                self.motion_encoder_type, motion_feat.shape,
            )

        motion_tokens = self.projector(motion_feat)  # [B, T, hidden_size]

        # 诊断: 检查 projector 输出
        if torch.isnan(motion_tokens).any() and not torch.isnan(motion_feat).any():
            logging.getLogger(__name__).warning(
                "[MotionModule] projector 输出含 NaN (输入正常), projector 权重可能无效"
            )
        return motion_tokens

    def set_alignment_gather_global_negatives(self, enabled: bool) -> None:
        self.alignment_head.set_gather_global_negatives(enabled)

    def compute_alignment_loss(
        self,
        motion_tokens: torch.Tensor,
        visual_targets: torch.Tensor,
        target_group_ids: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, object]:
        """
        计算 alignment loss (Stage 1 专用)。

        Args:
            motion_tokens: [N, D] 展平的 motion tokens
            visual_targets: [N, D] 对应的 visual targets
            target_group_ids: [N], 同一 visual interval 的 pair 共享同一个 group id
            negative_mask: [N, N], True 表示该 target 可作为当前 query 的负例
        Returns:
            dict: {"loss": scalar, "contrastive_loss": scalar, "cosine_loss": scalar,
                   "avg_cosine_sim": float, "logit_scale": float}
        """
        return self.alignment_head(
            motion_tokens,
            visual_targets,
            target_group_ids=target_group_ids,
            negative_mask=negative_mask,
        )


# ============================================================================
# 工具函数
# ============================================================================

def get_trainable_params_info(model: nn.Module) -> dict[str, object]:
    """统计模型可训练参数量, 方便日志打印。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio": f"{trainable / total * 100:.2f}%" if total > 0 else "0%",
        "total_mb": f"{total * 4 / 1024 / 1024:.1f} MB (fp32)",
        "trainable_mb": f"{trainable * 4 / 1024 / 1024:.1f} MB (fp32)",
    }


def freeze_module(module: nn.Module) -> None:
    """冻结模块的所有参数。"""
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """解冻模块的所有参数。"""
    for p in module.parameters():
        p.requires_grad = True
