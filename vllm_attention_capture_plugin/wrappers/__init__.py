"""
Attention patching utilities for capturing attention scores.

This module provides functionality to patch attention layers to capture
attention weights using SDPA fallback or manual attention computation.
"""

from typing import Any

import torch
import torch.nn.functional as F

from vllm_attention_capture_plugin.hooks.attention_hook import AttentionCaptureHook


def compute_attention_with_capture(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention manually to capture attention weights.

    This is a fallback implementation used only for layers where we want to
    capture attention scores. It matches vLLM's SDPA implementation as closely
    as possible to avoid numerical differences.

    Args:
        query: [num_tokens, num_heads, head_dim]
        key: [num_tokens, num_kv_heads, head_dim]
        value: [num_tokens, num_kv_heads, head_dim]
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        attn_mask: Optional attention mask
        is_causal: Whether to apply causal masking

    Returns:
        output: Attention output [num_tokens, num_heads, head_dim]
        attn_weights: Attention weights [num_heads, num_tokens, num_tokens]
    """
    num_tokens, num_heads, head_dim = query.shape
    _, num_kv_heads, _ = key.shape

    # Use PyTorch SDPA for prefill to match vLLM's implementation exactly
    # For GQA, SDPA handles KV head repetition internally with enable_gqa=True
    if num_tokens > 1:  # Prefill
        # Reshape to match SDPA expected format: [batch, heads, seq, dim]
        q_sdpa = query.unsqueeze(0).transpose(1, 2)  # [1, num_heads, num_tokens, head_dim]
        k_sdpa = key.unsqueeze(0).transpose(1, 2)    # [1, num_kv_heads, num_tokens, head_dim]
        v_sdpa = value.unsqueeze(0).transpose(1, 2)  # [1, num_kv_heads, num_tokens, head_dim]

        # Use SDPA with enable_gqa (matches vLLM's CPU backend)
        output_sdpa = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=(num_heads > num_kv_heads),
        )

        # Reshape back: [1, num_heads, num_tokens, head_dim] -> [num_tokens, num_heads, head_dim]
        output = output_sdpa.squeeze(0).transpose(0, 1)

        # Compute attention weights manually for capture
        # Use einsum for exact numerical match with vLLM reference
        # (SDPA doesn't return weights)
        # Handle GQA: repeat KV heads if needed
        if num_kv_heads < num_heads:
            assert num_heads % num_kv_heads == 0
            num_repeats = num_heads // num_kv_heads
            key = key.repeat_interleave(num_repeats, dim=1)
            value = value.repeat_interleave(num_repeats, dim=1)

        # Compute scores using einsum (matches vLLM reference exactly)
        # einsum("qhd,khd->hqk") computes: result[h,q,k] = sum_d(query[q,h,d] * key[k,h,d])
        attn_scores = (scale * torch.einsum("qhd,khd->hqk", query, key)).float()

        # Apply causal mask (already in float32)
        if is_causal:
            causal_mask = torch.triu(
                torch.full((num_tokens, num_tokens), float('-inf'),
                          device=query.device, dtype=torch.float32),
                diagonal=1
            )
            attn_scores = attn_scores + causal_mask

        # Softmax in float32, keep in float32 for accurate capture
        # DO NOT convert back to lower precision (bfloat16/float16) as it causes:
        # 1. Row sums != 1.0 (precision loss in normalization)
        # 2. Inaccurate attention values for analysis
        attn_weights = F.softmax(attn_scores, dim=-1)

        return output, attn_weights

    else:  # Decode (single token)
        # For decode, manually compute (SDPA doesn't help much for single token)
        # Handle GQA: repeat KV heads if needed
        if num_kv_heads < num_heads:
            assert num_heads % num_kv_heads == 0
            num_repeats = num_heads // num_kv_heads
            key = key.repeat_interleave(num_repeats, dim=1)
            value = value.repeat_interleave(num_repeats, dim=1)

        # Compute scores using einsum for exact match with vLLM reference
        attn_scores = (scale * torch.einsum("qhd,khd->hqk", query, key)).float()

        # Apply mask if provided (convert to float32 if needed)
        if attn_mask is not None:
            if attn_mask.dtype != torch.float32:
                attn_mask = attn_mask.float()
            attn_scores = attn_scores + attn_mask

        # Softmax in float32, keep in float32 for accurate capture
        # DO NOT convert back to lower precision (bfloat16/float16) as it causes:
        # 1. Row sums != 1.0 (precision loss in normalization)
        # 2. Inaccurate attention values for analysis
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values using einsum
        output = torch.einsum("hqk,khd->qhd", attn_weights, value)

        return output, attn_weights


class AttentionLayerWrapper:
    """Wrapper for attention layers to enable capture.

    This wrapper intercepts the forward pass of attention layers and:
    1. For capture layers: Uses manual attention computation to extract weights
    2. For non-capture layers: Uses standard optimized backend

    The wrapper is designed to be minimal and non-invasive.
    """

    def __init__(
        self,
        layer_idx: int,
        original_forward: Any,
        capture_hook: AttentionCaptureHook,
        num_heads: int,
        head_dim: int,
        scale: float,
    ):
        self.layer_idx = layer_idx
        self.original_forward = original_forward
        self.capture_hook = capture_hook
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Wrapped forward method."""

        # Check if we should capture this layer
        if self.capture_hook.should_capture(self.layer_idx):
            # Use manual attention computation to capture weights
            # Note: This is a simplified version for PoC
            # In production, we need to handle KV cache, various masks, etc.

            # For now, create a dummy request_id (in production, get from ForwardContext)
            request_id = "dummy_request"

            # Compute attention with capture
            output, attn_weights = compute_attention_with_capture(
                query, key, value, self.scale, is_causal=True
            )

            # Capture the attention weights
            self.capture_hook.capture_attention_weights(
                self.layer_idx,
                attn_weights,
                request_id
            )

            return output
        else:
            # Use optimized backend (original forward)
            return self.original_forward(query, key, value, **kwargs)


def patch_attention_layer(
    attention_layer: Any,
    layer_idx: int,
    capture_hook: AttentionCaptureHook,
    num_heads: int,
    head_dim: int,
    scale: float,
) -> None:
    """Patch an attention layer to enable capture.

    Args:
        attention_layer: The attention layer instance to patch
        layer_idx: Layer index (0-indexed)
        capture_hook: The capture hook to use
        num_heads: Number of attention heads
        head_dim: Head dimension
        scale: Attention scale factor
    """
    # Store original forward method
    original_forward = attention_layer.forward

    # Create wrapper
    wrapper = AttentionLayerWrapper(
        layer_idx=layer_idx,
        original_forward=original_forward,
        capture_hook=capture_hook,
        num_heads=num_heads,
        head_dim=head_dim,
        scale=scale,
    )

    # Replace forward method with wrapper
    attention_layer.forward = wrapper
