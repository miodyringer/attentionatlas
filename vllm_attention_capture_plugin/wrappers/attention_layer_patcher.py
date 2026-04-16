"""
Patch Attention layer to capture attention weights.

This module patches vLLM's Attention layer (not the projection layers) to capture
attention weights. At this level, Q/K/V have all model-specific transformations
applied (RoPE, QK normalization, etc), so we capture the ACTUAL attention values.
"""
import logging
from typing import Any
import numpy as np

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context

from vllm_attention_capture_plugin.hooks.attention_hook import AttentionCaptureHook
from vllm_attention_capture_plugin.wrappers import compute_attention_with_capture

logger = logging.getLogger(__name__)


def patch_attention_layer(
    attention_layer: Any,
    layer_idx: int,
    capture_hook: AttentionCaptureHook,
) -> None:
    """Patch an Attention layer to capture attention weights.

    This hooks at the Attention.forward() level, where Q/K/V have all
    model-specific transformations applied (RoPE, QK norm, etc).

    Args:
        attention_layer: The vLLM Attention layer instance (attn_module.attn)
        layer_idx: Layer index (0-indexed)
        capture_hook: The capture hook to use
    """
    # Store original forward method
    original_forward = attention_layer.forward

    # Get attention parameters
    num_heads = getattr(attention_layer, "num_heads", None)
    num_kv_heads = getattr(attention_layer, "num_kv_heads", None)
    head_size = getattr(attention_layer, "head_size", None)
    scale = getattr(attention_layer, "scale", None)

    if num_heads is None or head_size is None:
        logger.warning(
            "Cannot patch layer %d: missing num_heads or head_size", layer_idx
        )
        return

    if num_kv_heads is None:
        num_kv_heads = num_heads

    if scale is None:
        scale = head_size**-0.5

    logger.info(
        "Patching Attention layer %d: num_heads=%d, num_kv_heads=%d, head_size=%d, scale=%.4f",
        layer_idx,
        num_heads,
        num_kv_heads,
        head_size,
        scale,
    )

    def forward_with_capture(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """Wrapped forward method that captures attention weights.

        Signature matches vLLM's actual Attention.forward:
          forward(query, key, value, output_shape=None)

        Args:
            query: [num_tokens, num_heads * head_size] - Q after RoPE
            key: [num_tokens, num_kv_heads * head_size] - K after RoPE
            value: [num_tokens, num_kv_heads * head_size] - V
            output_shape: Optional output shape specification

        Note: kv_cache and attn_metadata are NOT parameters - they're managed
        internally by the Attention class. We can access them via:
        - attention_layer.kv_cache (instance attribute)
        - get_attention_context() (context variable)

        Returns:
            Attention output tensor
        """
        num_tokens = query.shape[0]
        is_decode = (num_tokens == 1)

        # Check if we should capture this layer
        if not capture_hook.should_capture(layer_idx):
            return original_forward(query, key, value, output_shape)

        # IMPORTANT: Store raw K/V BEFORE vLLM reshapes them into cache
        # We'll use these for attention calculation instead of extracting from cache
        if key is not None and value is not None:
            # Initialize accumulator if needed
            if not hasattr(attention_layer, '_raw_kv_accumulator'):
                attention_layer._raw_kv_accumulator = {
                    'keys': [],
                    'values': []
                }

            # Reshape: [num_tokens, num_kv_heads * head_size] -> [num_tokens, num_kv_heads, head_size]
            k_reshaped = key.view(num_tokens, num_kv_heads, head_size)
            v_reshaped = value.view(num_tokens, num_kv_heads, head_size)

            if is_decode:
                # Decode: append new token to accumulator
                attention_layer._raw_kv_accumulator['keys'].append(k_reshaped)
                attention_layer._raw_kv_accumulator['values'].append(v_reshaped)
            else:
                # Prefill: reset and store all tokens
                attention_layer._raw_kv_accumulator = {
                    'keys': [k_reshaped],
                    'values': [v_reshaped]
                }

        # Call vLLM's original forward first (don't modify behavior)
        result = original_forward(query, key, value, output_shape)

        # === ATTENTION COMPUTATION FOR CAPTURE ===
        try:
            if not is_decode:
                # ============================================================
                # PREFILL PHASE: All tokens in prompt
                # ============================================================
                # Reshape Q, K, V: [num_tokens, num_heads * head_size] -> [num_tokens, num_heads, head_size]
                q = query.view(num_tokens, num_heads, head_size)
                k = key.view(num_tokens, num_kv_heads, head_size)
                v = value.view(num_tokens, num_kv_heads, head_size)

                # Handle GQA: repeat KV heads to match Q heads
                if num_kv_heads < num_heads:
                    num_queries_per_kv = num_heads // num_kv_heads
                    k = k.repeat_interleave(num_queries_per_kv, dim=1)
                    v = v.repeat_interleave(num_queries_per_kv, dim=1)

                # Transpose for attention computation
                k_t = k.transpose(0, 1).transpose(1, 2)  # [num_heads, head_size, num_tokens]
                q_transposed = q.transpose(0, 1)  # [num_heads, num_tokens, head_size]

                # Compute attention scores: Q @ K^T * scale
                scores = torch.bmm(q_transposed, k_t) * scale

                # Apply causal mask
                causal_mask = torch.triu(torch.ones(num_tokens, num_tokens, device=scores.device), diagonal=1).bool()
                scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

                # Apply softmax to get attention weights
                attn_weights = torch.softmax(scores, dim=-1)

            else:
                # ============================================================
                # DECODE PHASE: One new token, attend to all previous + self
                # ============================================================
                # Use accumulated raw K/V instead of extracting from cache
                if not hasattr(attention_layer, '_raw_kv_accumulator'):
                    logger.warning(f"Layer {layer_idx}: No raw K/V accumulator, skipping decode capture")
                    return result

                # Concatenate all accumulated keys and values
                all_keys = torch.cat(attention_layer._raw_kv_accumulator['keys'], dim=0)
                all_values = torch.cat(attention_layer._raw_kv_accumulator['values'], dim=0)
                context_len = all_keys.shape[0]

                # Reshape query: [1, num_heads * head_size] -> [1, num_heads, head_size]
                q = query.view(1, num_heads, head_size)

                # Handle GQA: expand KV heads to match Q heads
                if num_heads != num_kv_heads:
                    num_queries_per_kv = num_heads // num_kv_heads
                    # Expand: [context_len, num_kv_heads, head_size] -> [context_len, num_heads, head_size]
                    all_keys = all_keys.unsqueeze(2).expand(
                        context_len, num_kv_heads, num_queries_per_kv, head_size
                    ).reshape(context_len, num_heads, head_size)
                    all_values = all_values.unsqueeze(2).expand(
                        context_len, num_kv_heads, num_queries_per_kv, head_size
                    ).reshape(context_len, num_heads, head_size)

                # Transpose for attention computation
                # Q: [1, num_heads, head_size] -> [num_heads, 1, head_size]
                # K: [context_len, num_heads, head_size] -> [num_heads, context_len, head_size]
                q_t = q.transpose(0, 1)  # [num_heads, 1, head_size]
                k_t = all_keys.transpose(0, 1)  # [num_heads, context_len, head_size]

                # Compute attention scores: Q @ K^T
                # [num_heads, 1, head_size] @ [num_heads, head_size, context_len]
                # -> [num_heads, 1, context_len]
                attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1))

                # Scale
                attn_scores = attn_scores * scale

                # Softmax to get attention weights
                attn_weights = torch.softmax(attn_scores, dim=-1)

            # Store attention weights via capture hook
            request_id = "default_request"
            capture_hook.capture_attention_weights(
                layer_idx=layer_idx,
                attn_weights=attn_weights,
                request_id=request_id,
            )

        except Exception as e:
            import traceback
            logger.error(
                f"Layer {layer_idx}: Failed to compute attention weights: {e}\n{traceback.format_exc()}"
            )

        return result

    # Replace forward method
    attention_layer.forward = forward_with_capture

    # Store reference for potential restoration
    attention_layer._original_forward = original_forward
    attention_layer._capture_hook = capture_hook
    attention_layer._layer_idx = layer_idx

    logger.info("✓ Successfully patched Attention layer %d", layer_idx)


def patch_model_for_attention_capture(
    model: Any,
    capture_hook: AttentionCaptureHook,
) -> None:
    """Patch a model's Attention layers to enable capture.

    Args:
        model: The vLLM model instance
        capture_hook: The capture hook to use
    """
    logger.info("Patching model for attention capture...")

    # Find all layers to patch
    # Different architectures use different attribute names:
    # - Qwen, Llama, Mistral: model.layers
    # - GPT-2: transformer.h
    # - GPT-J: transformer.h
    # - BLOOM: transformer.h
    layers = None

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Qwen/Llama/Mistral architecture
        layers = model.model.layers
        logger.info("Found Qwen/Llama architecture: model.layers")
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2/GPT-J/BLOOM architecture
        layers = model.transformer.h
        logger.info("Found GPT-2/GPT-J architecture: transformer.h")
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        # GPT-NeoX architecture
        layers = model.gpt_neox.layers
        logger.info("Found GPT-NeoX architecture: gpt_neox.layers")
    else:
        logger.error("Model structure not recognized, cannot patch")
        logger.info("Model type: %s, attributes: %s", type(model), dir(model))
        return

    logger.info("Found %d layers in model", len(layers))

    # Patch each layer that should be captured
    num_patched = 0
    for layer_idx in capture_hook.capture_layers:
        if layer_idx >= len(layers):
            logger.warning("Layer %d out of range, skipping", layer_idx)
            continue

        layer = layers[layer_idx]

        # Find the attention module
        # Different model architectures use different names:
        # - Qwen, Llama: self_attn
        # - GPT-2: attn
        attn_module = None
        if hasattr(layer, "self_attn"):
            attn_module = layer.self_attn
        elif hasattr(layer, "attn"):
            attn_module = layer.attn
        else:
            logger.warning("Layer %d has no attention module, skipping", layer_idx)
            continue

        # The actual Attention layer is usually inside the attention module
        # For Qwen: attn_module.attn
        # For GPT-2: attn_module.attn
        if hasattr(attn_module, "attn"):
            attention_layer = attn_module.attn
            logger.info(
                "Found Attention layer at layers[%d].%s.attn",
                layer_idx,
                "self_attn" if hasattr(layer, "self_attn") else "attn",
            )
        else:
            logger.warning(
                "Attention module at layer %d has no .attn attribute, skipping",
                layer_idx,
            )
            continue

        # Patch the Attention layer
        try:
            patch_attention_layer(attention_layer, layer_idx, capture_hook)
            num_patched += 1
        except Exception as e:
            logger.error("Failed to patch layer %d: %s", layer_idx, e)
            import traceback

            logger.debug("Traceback: %s", traceback.format_exc())

    logger.info(
        "✓ Successfully patched %d/%d layers",
        num_patched,
        len(capture_hook.capture_layers),
    )


__all__ = ["patch_attention_layer", "patch_model_for_attention_capture"]
