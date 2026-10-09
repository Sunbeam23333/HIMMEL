# Himmel-VLM: Efficient Long Video Understanding via Compressed Domain Adaptation
# 模型组件包
#
# 支持的模型:
#   - QwenDeltaForConditionalGeneration     (Qwen2.5-VL + 双流)
#   - Qwen3DeltaForConditionalGeneration    (Qwen3-VL + 双流)
#   - LlavaDeltaForConditionalGeneration    (LLaVA-OneVision + 双流)
#   - InternVLDeltaChatModel                (InternVL3 + 双流, 组合模式)
#
# 共享组件:
#   - MotionEncoder (ResNet-18)
#   - MotionProjector (2-layer MLP)
#   - MotionEncoderWithProjector
