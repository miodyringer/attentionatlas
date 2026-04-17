---
name: vLLM Attention Capture Implementation
description: Complete guide for implementing attention weight capture in vLLM, including CPU backend cache layout issues and solutions
type: reference
originSessionId: a3561707-6896-4660-a3bb-404ccc66cefe
---
# vLLM Attention Capture Implementation Guide

## Problem: CPU Backend KV Cache Layout

### Root Cause
vLLM's CPU backend uses **hardware-optimized KV cache layouts** that cannot be easily reversed:

```python
# vLLM stores keys/values using a custom reshape operation
ops.cpu_attn_reshape_and_cache(
    key, value, key_cache, value_cache,
    slot_mapping, isa  # ISA determines layout: AMX, AVX, NEON, VXE, etc.
)
```

**Cache Layout Characteristics:**
- Original shape: `[2, num_blocks, num_kv_heads, block_size, head_size]`
- After unbind: `[num_blocks, num_kv_heads, block_size, head_size]`
- **BUT**: The actual data layout is ISA-specific (not standard PyTorch layout)
- Example: For head_size=64, dimensions [16-31] and [48-63] may contain zeros (blocked/interleaved layout)
- Pattern varies by CPU architecture (AMX uses 16-element blocks, etc.)

### Why Direct Cache Extraction Fails

```python
# This DOES NOT WORK on CPU backend:
key_cache, value_cache = kv_cache.unbind(0)
block_id = block_table[req_idx, block_idx].item()
key_block = key_cache[block_id].permute(1, 0, 2)  # ❌ Wrong layout!
```

**Symptoms:**
- Extracted keys don't match input keys
- Half the dimensions are zero
- Attention weights differ significantly from ground truth (mean diff > 0.04)
- Last token in decode phase has largest error

## Solution: Bypass Cache Extraction

### Strategy
**Capture raw K/V BEFORE vLLM reshapes them**, then use accumulated values for attention computation.

### Implementation

```python
def forward_with_capture(query, key, value, output_shape=None):
    # Step 1: Initialize accumulator (do this once per layer)
    if not hasattr(attention_layer, '_raw_kv_accumulator'):
        attention_layer._raw_kv_accumulator = {
            'keys': [],
            'values': []
        }
    
    # Step 2: Capture raw K/V BEFORE vLLM processes them
    if key is not None and value is not None:
        k_reshaped = key.view(num_tokens, num_kv_heads, head_size)
        v_reshaped = value.view(num_tokens, num_kv_heads, head_size)
        
        if is_decode:
            # Decode: append new token
            attention_layer._raw_kv_accumulator['keys'].append(k_reshaped)
            attention_layer._raw_kv_accumulator['values'].append(v_reshaped)
        else:
            # Prefill: reset and store all tokens
            attention_layer._raw_kv_accumulator = {
                'keys': [k_reshaped],
                'values': [v_reshaped]
            }
    
    # Step 3: Call original forward (vLLM processes with its cache)
    result = original_forward(query, key, value, output_shape)
    
    # Step 4: Use accumulated K/V for attention calculation
    try:
        if is_decode:
            # Concatenate all accumulated K/V
            all_keys = torch.cat(attention_layer._raw_kv_accumulator['keys'], dim=0)
            all_values = torch.cat(attention_layer._raw_kv_accumulator['values'], dim=0)
            
            # Handle GQA if needed
            if num_heads != num_kv_heads:
                num_queries_per_kv = num_heads // num_kv_heads
                all_keys = all_keys.unsqueeze(2).expand(
                    context_len, num_kv_heads, num_queries_per_kv, head_size
                ).reshape(context_len, num_heads, head_size)
                all_values = all_values.unsqueeze(2).expand(
                    context_len, num_kv_heads, num_queries_per_kv, head_size
                ).reshape(context_len, num_heads, head_size)
            
            # Compute attention
            q = query.view(1, num_heads, head_size)
            q_t = q.transpose(0, 1)  # [num_heads, 1, head_size]
            k_t = all_keys.transpose(0, 1)  # [num_heads, context_len, head_size]
            
            attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Store weights
            capture_hook.capture_attention_weights(layer_idx, attn_weights, request_id)
    
    return result
```

## Results

**Before fix (extracting from CPU cache):**
- Mean absolute difference: 0.041349
- Max absolute difference: 0.325586
- 42% of values with diff > 0.01

**After fix (using raw K/V accumulator):**
- Mean absolute difference: 0.000246 ✅ (167x improvement)
- Max absolute difference: 0.003413 ✅ (95x improvement)
- 0% of values with diff > 0.01 ✅

## Critical Learnings

### 1. Backend-Specific Cache Layouts
- **CPU backend**: Uses `cpu_attn_reshape_and_cache()` with ISA-specific layouts
- **GPU backends**: May have different layouts (e.g., FlashAttention, PagedAttention)
- **Never assume** you can directly extract from KV cache - always verify format

### 2. Where to Capture
- ❌ **DON'T** extract from KV cache after `original_forward()`
- ✅ **DO** capture raw tensors BEFORE they enter vLLM's processing pipeline
- ✅ Capture at `Attention.forward()` entry point where K/V are still in standard PyTorch format

### 3. Accumulator Pattern
- **Prefill phase**: Reset accumulator with all prompt tokens
- **Decode phase**: Append each new token
- **Per-layer**: Each attention layer needs its own accumulator
- **Cleanup**: Clear accumulator when starting new sequence

### 4. Testing Strategy
```python
# Always compare against HuggingFace Transformers
# Check both prefill AND decode phases
# Look for these specific patterns:

# Prefill tokens (0-3): Usually work fine
# Decode tokens (4+): This is where cache extraction fails

# Good: mean diff < 0.001
# Acceptable: mean diff < 0.01
# Bad: mean diff > 0.01 (indicates wrong cache extraction)
```

### 5. Configuration Discovery
Check logs for backend type:
```
WARNING [cpu.py:136] VLLM_CPU_KVCACHE_SPACE not set.  # CPU backend
INFO [attention_selector.py:...] Using FlashAttention   # GPU backend
```

### 6. Block Size Retrieval
**Wrong approach:**
```python
forward_context = get_forward_context()
block_size = forward_context.vllm_context.cache_config.block_size  # ❌ No vllm_context field
```

**Correct approaches:**
```python
# Option 1: From cache shape (if you must extract from cache)
block_size = kv_cache.shape[3]

# Option 2: From vllm_config (during initialization)
from vllm.config import get_current_vllm_config
vllm_config = get_current_vllm_config()
block_size = vllm_config.cache_config.block_size

# Option 3: From get_kv_cache_spec method
def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
    block_size = vllm_config.cache_config.block_size
```

## Future Optimizations

### For GPU Backends
- Test if GPU backends (FlashAttention, PagedAttention) have similar cache layout issues
- May need backend-specific extraction logic
- Consider checking `attention_layer.attn_backend.get_name()` and branching logic

### Memory Optimization
Current approach stores all K/V in memory:
```python
# Memory usage: O(sequence_length * num_kv_heads * head_size * num_layers)
# For 100 tokens, 32 heads, 128 head_size, 32 layers: ~12 MB per sequence
```

**Potential optimizations:**
1. **Selective storage**: Only store K/V for layers being captured
2. **Windowed storage**: Only keep last N tokens for decode
3. **On-demand computation**: Compute attention incrementally without storing all K/V

### Multi-Request Support
Current implementation assumes single request (simple accumulator):
```python
# TODO: Support multiple concurrent requests
# Need to track per-request accumulators
attention_layer._raw_kv_accumulator = {
    'request_123': {'keys': [...], 'values': [...]},
    'request_456': {'keys': [...], 'values': [...]},
}
```

## References

**vLLM Source Files:**
- `/vllm/v1/attention/backends/cpu_attn.py` - CPU backend implementation
- `/csrc/cpu/cpu_attn.cpp` - C++ implementation of reshape_and_cache
- `/vllm/model_executor/layers/attention/attention.py` - Attention layer interface

**Key Functions:**
- `ops.cpu_attn_reshape_and_cache()` - Reshapes K/V into cache (ISA-specific)
- `ops.cpu_attention_with_kv_cache()` - Computes attention using reshaped cache
- `get_attention_context()` - Helper to get attention metadata and cache

**Testing:**
- Compare against HuggingFace Transformers with `output_attentions=True`
- Test both prefill (multiple tokens) and decode (single token) phases
- Verify row sums equal 1.0 and match reference implementation within 0.001
