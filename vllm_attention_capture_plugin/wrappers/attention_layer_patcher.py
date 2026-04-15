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
        num_tokens = query.shape[0]
        is_decode = (num_tokens == 1)

        # Only log for layer 0
        if layer_idx == 0:
            phase = "DECODE" if is_decode else "PREFILL"
            logger.info(f"\n[Layer {layer_idx}] {phase} phase: {num_tokens} token(s)")
            logger.info(f"  Input Q: {query.shape}, K: {key.shape}, V: {value.shape}")

        # Check if we should capture this layer
        if not capture_hook.should_capture(layer_idx):
            return original_forward(query, key, value, output_shape)

        # Call vLLM's original forward first (don't modify behavior)
        result = original_forward(query, key, value, output_shape)

        # === ATTENTION COMPUTATION FOR CAPTURE ===
        try:
            if not is_decode:
                # ============================================================
                # PREFILL PHASE: All tokens in prompt
                # ============================================================
                # Save RAW K before any processing
                raw_k_input = key.clone()

                # Reshape Q, K, V: [num_tokens, num_heads * head_size] -> [num_tokens, num_heads, head_size]
                q = query.view(num_tokens, num_heads, head_size)
                k = key.view(num_tokens, num_kv_heads, head_size)
                v = value.view(num_tokens, num_kv_heads, head_size)

                if layer_idx == 0:
                    logger.info(f"  Reshaped Q: {q.shape}, K: {k.shape}")
                    logger.info(f"  PREFILL Q[0,0,:5] = {q[0, 0, :5].float().cpu().numpy()}")
                    logger.info(f"  PREFILL K[0,0,:5] (raw from input) = {k[0, 0, :5].float().cpu().numpy()}")
                    logger.info(f"  PREFILL K[-1,0,:5] (raw from input) = {k[-1, 0, :5].float().cpu().numpy()}")

                    # Save raw K for comparison with cache later
                    try:
                        import pickle
                        with open('/tmp/vllm_prefill_k_raw.pkl', 'wb') as f:
                            pickle.dump(k.float().detach().cpu(), f)
                    except: pass

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

                if layer_idx == 0:
                    logger.info(f"  Scores: {scores.shape}, range=[{scores[0].min().item():.3f}, {scores[0].max().item():.3f}]")
                    logger.info(f"  Attention: {attn_weights.shape}, sum={attn_weights[0, 0].sum().item():.3f}")

                    # IMPORTANT: Check what's in cache AFTER original_forward() in prefill
                    # This will show us the transformation vLLM applies
                    try:
                        from vllm.model_executor.layers.attention.attention import get_attention_context
                        layer_name = getattr(attention_layer, "layer_name", None)
                        if layer_name:
                            attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
                            if kv_cache is not None and attn_metadata is not None:
                                seq_len = attn_metadata.seq_lens[0].item()
                                block_table = attn_metadata.block_table[0]
                                block_size = kv_cache.shape[3]

                                # Extract first token from cache to compare with raw input
                                block_idx = block_table[0].item()
                                first_token_k_cached = kv_cache[0, block_idx, :, 0, :]  # [num_kv_heads, head_size]

                                logger.info(f"\n  PREFILL CACHE CHECK:")
                                logger.info(f"  Cache shape: {kv_cache.shape}")
                                logger.info(f"  Extracting: kv_cache[0, block_idx={block_idx}, head=0, token=0, :]")

                                # Extract first token from cache
                                block_idx = block_table[0].item()
                                first_token_k_cached = kv_cache[0, block_idx, 0, 0, :]  # [head_size]

                                # Also check second token to see pattern
                                second_token_k_cached = kv_cache[0, block_idx, 0, 1, :]  # [head_size]

                                logger.info(f"  Token 0 K from cache (full): {first_token_k_cached.float().cpu().numpy()}")
                                logger.info(f"  Token 0 K from raw (full):   {k[0, 0, :].float().cpu().numpy()}")
                                logger.info(f"  Token 1 K from cache (full): {second_token_k_cached.float().cpu().numpy()}")
                                logger.info(f"  Token 1 K from raw (full):   {k[1, 0, :].float().cpu().numpy()}")

                                # Check if the pattern is consistent
                                diff0 = (first_token_k_cached - k[0, 0, :]).float().cpu().numpy()
                                diff1 = (second_token_k_cached - k[1, 0, :]).float().cpu().numpy()

                                cache0_np = first_token_k_cached.float().cpu().numpy()
                                cache1_np = second_token_k_cached.float().cpu().numpy()

                                logger.info(f"  Token 0: zeros in cache: {(abs(cache0_np) < 1e-6).sum()}/{len(cache0_np)}, max diff: {abs(diff0).max():.6f}")
                                logger.info(f"  Token 1: zeros in cache: {(abs(cache1_np) < 1e-6).sum()}/{len(cache1_np)}, max diff: {abs(diff1).max():.6f}")

                                # Check which dimensions are zero
                                zero_dims_0 = np.where(abs(cache0_np) < 1e-6)[0]
                                zero_dims_1 = np.where(abs(cache1_np) < 1e-6)[0]
                                logger.info(f"  Token 0 zero dimensions: {zero_dims_0}")
                                logger.info(f"  Token 1 zero dimensions: {zero_dims_1}")
                                logger.info(f"  Zero dims match: {np.array_equal(zero_dims_0, zero_dims_1)}")

                                # Save both for detailed analysis
                                import pickle
                                with open('/tmp/vllm_prefill_k_cached.pkl', 'wb') as f:
                                    pickle.dump(first_token_k_cached.float().detach().cpu(), f)
                                with open('/tmp/vllm_prefill_k_raw_comparison.pkl', 'wb') as f:
                                    pickle.dump({'cached': first_token_k_cached.float().detach().cpu(),
                                               'raw': k[0, 0, :].float().detach().cpu(),
                                               'diff': diff0}, f)
                    except Exception as e:
                        logger.info(f"  Failed to check cache: {e}")

            else:
                # ============================================================
                # DECODE PHASE: One new token, attend to all previous + self
                # ============================================================
                # Reshape new token's Q from input parameter (Q is not cached)
                new_q = query.view(1, num_heads, head_size)

                # Get context to access KV cache
                from vllm.model_executor.layers.attention.attention import (
                    get_attention_context,
                )
                layer_name = getattr(attention_layer, "layer_name", None)

                if not layer_name or get_attention_context(layer_name)[0] is None:
                    return result

                attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)

                # Get sequence info
                seq_len = attn_metadata.seq_lens[0].item()
                block_table = attn_metadata.block_table[0]
                block_size = kv_cache.shape[3]

                if layer_idx == 0:
                    logger.info(f"  seq_len={seq_len} (total tokens including new one)")
                    logger.info(f"  New Q: {new_q.shape}")
                    logger.info(f"  DECODE Q[0,0,:5] = {new_q[0, 0, :5].float().cpu().numpy()}")

                # === SAVE Q FOR COMPARISON ===
                try:
                    import pickle
                    q_to_save = new_q.float() if new_q.dtype == torch.bfloat16 else new_q
                    with open('/tmp/vllm_debug_q.pkl', 'wb') as f:
                        pickle.dump(q_to_save.detach().cpu(), f)
                except: pass

                # Extract ALL keys from KV cache (including the transformed new key)
                # The cache has been updated by original_forward(), so it contains all seq_len tokens
                num_blocks_needed = (seq_len + block_size - 1) // block_size
                all_keys_list = []
                tokens_remaining = seq_len

                for i in range(num_blocks_needed):
                    block_idx = block_table[i].item()
                    tokens_in_this_block = min(tokens_remaining, block_size)
                    block_keys = kv_cache[0, block_idx, :, :tokens_in_this_block, :]
                    all_keys_list.append(block_keys)
                    tokens_remaining -= tokens_in_this_block

                # Concatenate: [num_kv_heads, seq_len, head_size]
                all_keys = torch.cat(all_keys_list, dim=1)

                # Handle GQA
                if num_kv_heads < num_heads:
                    num_queries_per_kv = num_heads // num_kv_heads
                    all_keys = all_keys.repeat_interleave(num_queries_per_kv, dim=0)

                # All keys are now consistently transformed (by vLLM)

                # === SAVE K FOR COMPARISON ===
                try:
                    import pickle
                    k_to_save = all_keys.float() if all_keys.dtype == torch.bfloat16 else all_keys
                    with open('/tmp/vllm_debug_k.pkl', 'wb') as f:
                        pickle.dump(k_to_save.detach().cpu(), f)
                except: pass

                if layer_idx == 0:
                    logger.info(f"  All K: {all_keys.shape}")
                    logger.info(f"  DECODE K[0,0,:5] (1st token) = {all_keys[0, 0, :5].float().cpu().numpy()}")
                    logger.info(f"  DECODE K[0,-1,:5] (new token from input) = {all_keys[0, -1, :5].float().cpu().numpy()}")

                    # Check for zero dimensions
                    first_token_k = all_keys[0, 0, :].float().cpu().numpy()
                    zero_dims = np.where(np.abs(first_token_k) < 1e-6)[0]
                    logger.info(f"  First token K zero dimensions: {len(zero_dims)}/{len(first_token_k)} (should be few/none)")
                    if len(zero_dims) > 0:
                        logger.info(f"    Zero dim indices: {zero_dims}")

                # Compute attention: new query @ all keys
                all_keys_t = all_keys.transpose(1, 2)  # [num_heads, head_size, seq_len]
                new_q_squeezed = new_q.squeeze(0)  # [num_heads, head_size]

                scores = torch.bmm(new_q_squeezed.unsqueeze(1), all_keys_t).squeeze(1) * scale

                # Apply softmax
                attn_weights = torch.softmax(scores, dim=-1)

                if layer_idx == 0:
                    logger.info(f"  Scores: {scores.shape}, range=[{scores[0].min().item():.3f}, {scores[0].max().item():.3f}]")
                    logger.info(f"  Attention: {attn_weights.shape}, sum={attn_weights[0].sum().item():.3f}")

                # === SAVE SCORES AND ATTENTION FOR COMPARISON ===
                try:
                    import pickle
                    scores_to_save = scores.float() if scores.dtype == torch.bfloat16 else scores
                    attn_to_save = attn_weights.float() if attn_weights.dtype == torch.bfloat16 else attn_weights
                    with open('/tmp/vllm_debug_scores.pkl', 'wb') as f:
                        pickle.dump(scores_to_save.detach().cpu(), f)
                    with open('/tmp/vllm_debug_attn.pkl', 'wb') as f:
                        pickle.dump(attn_to_save.detach().cpu(), f)
                except: pass

                # Reshape to [num_heads, 1, seq_len] for consistency with PREFILL format
                attn_weights = attn_weights.unsqueeze(1)

            # Store attention weights via capture hook
            request_id = "default_request"
            capture_hook.capture_attention_weights(
                layer_idx=layer_idx,
                attn_weights=attn_weights,
                request_id=request_id,
            )

        except Exception as e:
            if layer_idx == 0:
                logger.error(f"  ❌ Failed to compute attention: {e}")

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
