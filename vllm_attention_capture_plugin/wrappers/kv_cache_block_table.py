"""
KV Cache extraction using block_table (the correct approach for decode phase).

During decode phase, we need to extract the full cached sequence from vLLM's
paged KV cache. The block_table tells us which blocks contain the cached tokens.

Key insight: slot_mapping is for WRITING new tokens, block_table is for READING
cached tokens.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def extract_kv_from_cache_using_block_table(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    request_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract cached K and V tensors using block_table.

    This is the correct way to extract cached KV during decode phase.
    The block_table maps sequence positions to cache blocks.

    Args:
        kv_cache: [2, num_blocks, num_kv_heads, block_size, head_dim]
                  Paged KV cache where dim 0 = [keys, values]
        block_table: [num_requests, max_num_blocks_per_seq]
                     Block indices for each request's sequence
        seq_len: Total sequence length including current decode token
        num_kv_heads: Number of KV heads
        head_dim: Head dimension
        request_idx: Which request in the batch (0 for single request)

    Returns:
        keys: [cache_len, num_kv_heads, head_dim] - cached keys (excluding current token)
        values: [cache_len, num_kv_heads, head_dim] - cached values (excluding current token)

    Notes:
        - cache_len = seq_len - 1 (we extract history, not the new token)
        - The new token's K/V is provided separately and will be concatenated by caller
    """
    # Extract dimensions
    block_size = kv_cache.shape[3]
    cache_len = seq_len - 1  # Number of cached tokens (excluding current decode token)

    if cache_len <= 0:
        # No cached tokens (first token in sequence)
        return (
            torch.empty((0, num_kv_heads, head_dim), dtype=kv_cache.dtype, device=kv_cache.device),
            torch.empty((0, num_kv_heads, head_dim), dtype=kv_cache.dtype, device=kv_cache.device),
        )

    # Get key and value caches
    key_cache = kv_cache[0]  # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache = kv_cache[1]

    # Get block table for this request
    if block_table.ndim == 2:
        # Multi-request batch: [num_requests, max_blocks]
        request_blocks = block_table[request_idx]  # [max_blocks]
    else:
        # Single request: [max_blocks]
        request_blocks = block_table

    # Calculate how many blocks we need for cache_len tokens
    num_blocks_needed = (cache_len + block_size - 1) // block_size

    # Get the actual block indices we need
    block_indices = request_blocks[:num_blocks_needed]

    # Validate block indices
    max_block_idx = block_indices.max().item() if len(block_indices) > 0 else -1
    if max_block_idx >= key_cache.shape[0]:
        logger.error(
            f"Block index {max_block_idx} out of bounds (cache has {key_cache.shape[0]} blocks)"
        )
        # Return empty tensors rather than crashing
        return (
            torch.empty((0, num_kv_heads, head_dim), dtype=kv_cache.dtype, device=kv_cache.device),
            torch.empty((0, num_kv_heads, head_dim), dtype=kv_cache.dtype, device=kv_cache.device),
        )

    # Gather blocks from cache
    # key_cache[block_indices] -> [num_blocks_needed, num_kv_heads, block_size, head_dim]
    selected_key_blocks = key_cache[block_indices]
    selected_value_blocks = value_cache[block_indices]

    # Reshape to sequence format:
    # [num_blocks, num_kv_heads, block_size, head_dim]
    # -> [num_blocks, block_size, num_kv_heads, head_dim] (transpose)
    # -> [num_blocks * block_size, num_kv_heads, head_dim] (reshape)
    keys = selected_key_blocks.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
    values = selected_value_blocks.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)

    # Trim to exact cache length (last block might be partially filled)
    keys = keys[:cache_len]
    values = values[:cache_len]

    return keys, values


def extract_kv_for_decode_batch(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract cached KV for a batch of decode requests.

    Args:
        kv_cache: [2, num_blocks, num_kv_heads, block_size, head_dim]
        block_table: [num_requests, max_blocks]
        seq_lens: [num_requests] - sequence length for each request
        num_kv_heads: Number of KV heads
        head_dim: Head dimension

    Returns:
        List of (keys, values) tuples, one per request
    """
    num_requests = seq_lens.shape[0]
    results = []

    for request_idx in range(num_requests):
        seq_len = seq_lens[request_idx].item()

        keys, values = extract_kv_from_cache_using_block_table(
            kv_cache=kv_cache,
            block_table=block_table,
            seq_len=seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            request_idx=request_idx,
        )

        results.append((keys, values))

    return results


__all__ = [
    "extract_kv_from_cache_using_block_table",
    "extract_kv_for_decode_batch",
]
