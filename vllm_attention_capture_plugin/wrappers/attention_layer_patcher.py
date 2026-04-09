"""
Patch Attention layer to capture attention weights.

This module patches vLLM's Attention layer (not the projection layers) to capture
attention weights. At this level, Q/K/V have all model-specific transformations
applied (RoPE, QK normalization, etc), so we capture the ACTUAL attention values.
"""
import logging
from typing import Any

import torch

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
        # Log EVERY forward call to layer (even if not capturing)
        num_tokens = query.shape[0]
        is_decode = (num_tokens == 1)

        # === PARAMETER LOGGING FOR DEBUGGING ===
        logger.info("=" * 80)
        logger.info("🔍 HOOK CALLED - Layer %d forward", layer_idx)
        logger.info("=" * 80)
        logger.info("Phase: %s", "DECODE" if is_decode else "PREFILL")
        logger.info("Query shape: %s, dtype: %s", query.shape, query.dtype)
        logger.info("Key shape: %s, dtype: %s", key.shape, key.dtype)
        logger.info("Value shape: %s, dtype: %s", value.shape, value.dtype)
        logger.info("output_shape: %s", output_shape)

        # Try to access kv_cache and attn_metadata from the Attention instance
        logger.info("\n🔎 Attempting to access kv_cache/attn_metadata from instance:")

        kv_cache = getattr(attention_layer, "kv_cache", None)
        logger.info("  attention_layer.kv_cache: %s", type(kv_cache))
        if isinstance(kv_cache, torch.Tensor):
            logger.info("    shape: %s", kv_cache.shape)
        elif isinstance(kv_cache, list) and len(kv_cache) > 0:
            logger.info("    list with %d elements", len(kv_cache))
            if isinstance(kv_cache[0], torch.Tensor):
                logger.info("    first element shape: %s", kv_cache[0].shape)

        # Try to get from context
        try:
            from vllm.model_executor.layers.attention.attention import (
                get_attention_context,
            )
            layer_name = getattr(attention_layer, "layer_name", None)
            logger.info("  attention_layer.layer_name: %s", layer_name)

            if layer_name:
                context_result = get_attention_context(layer_name)
                logger.info("  get_attention_context returned: %s", type(context_result))
                if context_result:
                    logger.info("    context tuple length: %d", len(context_result) if isinstance(context_result, tuple) else 0)
        except Exception as e:
            logger.info("  Failed to get attention context: %s", e)

        logger.info("=" * 80)

        # Check if we should capture this layer
        if not capture_hook.should_capture(layer_idx):
            # Use vLLM's optimized backend
            logger.info("❌ Layer %d: NOT in capture_layers, skipping", layer_idx)
            return original_forward(query, key, value, output_shape)

        logger.info("✅ Layer %d: IN capture_layers - hook is active!", layer_idx)

        # Call vLLM's original forward first (don't modify behavior)
        result = original_forward(query, key, value, output_shape)
        logger.info("✅ Original forward completed, result shape: %s", result.shape)

        # === ATTENTION COMPUTATION FOR CAPTURE ===
        try:
            # Use attention parameters from closure (extracted during patching)
            # num_heads, num_kv_heads, head_size, scale are already in scope

            logger.info("Computing attention weights: heads=%d, kv_heads=%d, head_size=%d",
                       num_heads, num_kv_heads, head_size)

            if not is_decode:
                # ============================================================
                # PREFILL PHASE: All tokens in prompt
                # ============================================================
                logger.info("🧮 PREFILL: Computing attention for %d tokens", num_tokens)

                # Reshape Q, K, V: [num_tokens, num_heads * head_size] -> [num_tokens, num_heads, head_size]
                q = query.view(num_tokens, num_heads, head_size)
                k = key.view(num_tokens, num_kv_heads, head_size)
                v = value.view(num_tokens, num_kv_heads, head_size)

                logger.info("  Q: %s, K: %s, V: %s", q.shape, k.shape, v.shape)

                # Handle GQA: repeat KV heads to match Q heads
                if num_kv_heads < num_heads:
                    num_queries_per_kv = num_heads // num_kv_heads
                    k = k.repeat_interleave(num_queries_per_kv, dim=1)
                    v = v.repeat_interleave(num_queries_per_kv, dim=1)
                    logger.info("  GQA: Repeated K/V to %s", k.shape)

                # Transpose for attention computation
                # Q: [num_tokens, num_heads, head_size]
                # K: [num_tokens, num_heads, head_size] -> [num_heads, head_size, num_tokens]
                k_t = k.transpose(0, 1).transpose(1, 2)  # [num_heads, head_size, num_tokens]
                q_transposed = q.transpose(0, 1)  # [num_heads, num_tokens, head_size]

                # Compute attention scores: Q @ K^T * scale
                # [num_heads, num_tokens, head_size] @ [num_heads, head_size, num_tokens]
                # = [num_heads, num_tokens, num_tokens]
                scores = torch.bmm(q_transposed, k_t) * scale
                logger.info("  Scores: %s", scores.shape)

                # Apply causal mask: token i can only attend to tokens [0, i]
                # Create mask where position (i, j) is True if j > i (future positions)
                causal_mask = torch.triu(torch.ones(num_tokens, num_tokens, device=scores.device), diagonal=1).bool()
                # Set future positions to -inf so softmax gives them 0 probability
                scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
                logger.info("  Applied causal mask")

                # Apply softmax to get attention weights
                attn_weights = torch.softmax(scores, dim=-1)
                logger.info("  Attention weights: %s, range=[%.4f, %.4f]",
                           attn_weights.shape, attn_weights.min().item(), attn_weights.max().item())

            else:
                # ============================================================
                # DECODE PHASE: One new token, attend to all previous + self
                # ============================================================
                logger.info("🧮 DECODE: Computing attention for new token")

                # Get context to access KV cache and metadata
                from vllm.model_executor.layers.attention.attention import (
                    get_attention_context,
                )
                layer_name = getattr(attention_layer, "layer_name", None)
                if not layer_name:
                    logger.warning("No layer_name, cannot get context for DECODE")
                    logger.info("=" * 80)
                    return result

                attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
                if attn_metadata is None:
                    logger.warning("No attn_metadata available")
                    logger.info("=" * 80)
                    return result

                # Get sequence info
                seq_len = attn_metadata.seq_lens[0].item()
                block_table = attn_metadata.block_table[0]
                block_size = kv_cache.shape[3]  # [2, num_blocks, num_heads, block_size, head_dim]

                logger.info("  seq_len=%d, block_size=%d, block_table=%s",
                           seq_len, block_size, block_table.tolist())

                # Reshape new token's Q, K, V
                new_q = query.view(1, num_heads, head_size)
                new_k = key.view(1, num_kv_heads, head_size)
                new_v = value.view(1, num_kv_heads, head_size)

                # Extract past keys and values from KV cache
                num_past_tokens = seq_len - 1

                if num_past_tokens == 0:
                    # First token - no past context, just use new K
                    logger.info("  First token, no past keys to extract")
                    # Handle GQA
                    if num_kv_heads < num_heads:
                        num_queries_per_kv = num_heads // num_kv_heads
                        new_k_repeated = new_k.repeat_interleave(num_queries_per_kv, dim=1)
                    else:
                        new_k_repeated = new_k
                    all_keys = new_k_repeated.squeeze(0).unsqueeze(1)  # [num_heads, 1, head_size]
                else:
                    num_blocks_needed = (num_past_tokens + block_size - 1) // block_size

                    logger.info("  Extracting %d past tokens from %d blocks",
                               num_past_tokens, num_blocks_needed)

                    # Extract keys from cache
                    past_keys_list = []
                    tokens_remaining = num_past_tokens
                    for i in range(num_blocks_needed):
                        block_idx = block_table[i].item()
                        tokens_in_this_block = min(tokens_remaining, block_size)

                        # kv_cache[0] = keys, shape: [num_blocks, num_kv_heads, block_size, head_size]
                        block_keys = kv_cache[0, block_idx, :, :tokens_in_this_block, :]
                        past_keys_list.append(block_keys)
                        tokens_remaining -= tokens_in_this_block

                    # Concatenate along token dimension: [num_kv_heads, num_past_tokens, head_size]
                    past_keys = torch.cat(past_keys_list, dim=1)
                    logger.info("  Extracted past keys: %s", past_keys.shape)

                    # Handle GQA: repeat KV heads to match Q heads
                    if num_kv_heads < num_heads:
                        num_queries_per_kv = num_heads // num_kv_heads
                        past_keys = past_keys.repeat_interleave(num_queries_per_kv, dim=0)
                        new_k_repeated = new_k.repeat_interleave(num_queries_per_kv, dim=1)
                        logger.info("  GQA: Repeated to %s", past_keys.shape)
                    else:
                        new_k_repeated = new_k

                    # Concatenate past + new keys: [num_heads, seq_len, head_size]
                    # past_keys: [num_heads, num_past_tokens, head_size]
                    # new_k_repeated: [1, num_heads, head_size] -> squeeze(0).unsqueeze(1) -> [num_heads, 1, head_size]
                    new_k_reshaped = new_k_repeated.squeeze(0).unsqueeze(1)  # [num_heads, 1, head_size]
                    all_keys = torch.cat([past_keys, new_k_reshaped], dim=1)
                    logger.info("  All keys (past + new): %s", all_keys.shape)

                # Compute attention: new query @ all keys
                # new_q: [1, num_heads, head_size]
                # all_keys: [num_heads, seq_len, head_size] -> transpose -> [num_heads, head_size, seq_len]
                all_keys_t = all_keys.transpose(1, 2)  # [num_heads, head_size, seq_len]
                new_q_squeezed = new_q.squeeze(0)  # [num_heads, head_size]

                # [num_heads, 1, head_size] @ [num_heads, head_size, seq_len] = [num_heads, 1, seq_len]
                scores = torch.bmm(new_q_squeezed.unsqueeze(1), all_keys_t).squeeze(1) * scale
                logger.info("  Scores: %s", scores.shape)

                # Apply softmax: [num_heads, seq_len]
                attn_weights = torch.softmax(scores, dim=-1)
                logger.info("  Attention weights: %s, range=[%.4f, %.4f]",
                           attn_weights.shape, attn_weights.min().item(), attn_weights.max().item())

                # Reshape to [num_heads, 1, seq_len] for consistency with PREFILL format
                # PREFILL: [num_heads, num_tokens, seq_len]
                # DECODE: [num_heads, 1, seq_len] where num_tokens=1
                attn_weights = attn_weights.unsqueeze(1)

            # Store attention weights via capture hook
            request_id = "default_request"  # TODO: Extract from attn_metadata
            capture_hook.capture_attention_weights(
                layer_idx=layer_idx,
                attn_weights=attn_weights,
                request_id=request_id,
            )
            logger.info("✅ Stored attention weights for layer %d", layer_idx)

        except Exception as e:
            logger.error("❌ Failed to compute attention weights: %s", e)
            import traceback
            traceback_str = traceback.format_exc()
            logger.debug("Traceback: %s", traceback_str)
            # Log first 500 chars of traceback to INFO for debugging
            logger.info("Error details: %s", traceback_str[:500])

        logger.info("=" * 80)
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
