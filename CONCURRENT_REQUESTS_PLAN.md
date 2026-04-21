# Implementation Plan: Concurrent Request Support

## Overview

To support concurrent requests with separate attention tracking, we need to:
1. Access batch boundary information from vLLM's attention metadata
2. Split concatenated attention tensors by request boundaries
3. Store attention separately for each request in the batch

## Required Changes

### 1. Access Attention Metadata in Forward Hook

**File**: `vllm_attention_capture_plugin/wrappers/attention_layer_patcher.py`

The attention forward hook needs to access `attn_metadata` which contains batch boundary information.

**Current signature**:
```python
def forward_with_capture(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
) -> torch.Tensor:
```

**Problem**: `attn_metadata` is NOT a parameter. It's accessed internally by the Attention class via:
- `get_forward_context()` - Returns ForwardContext
- `attention_layer.attn_metadata` - Instance attribute set by model runner

**Solution**: Access via the Attention instance:
```python
# Inside forward_with_capture:
attn_metadata = getattr(attention_layer, 'attn_metadata', None)
if attn_metadata is None:
    # Fallback to current behavior (single request)
    pass
```

### 2. Extract Batch Boundary Information

**Key fields in `AttentionMetadata`**:

```python
class FlashAttentionMetadata:
    num_actual_tokens: int          # Total tokens in batch
    query_start_loc: torch.Tensor   # [num_reqs+1] cumulative start positions
    seq_lens: torch.Tensor          # [num_reqs] sequence length per request
```

**Example for 2 concurrent requests**:
```python
# Request 0: 6 tokens (2 prompt + 4 generated)
# Request 1: 3 tokens (2 prompt + 1 generated)

query_start_loc = [0, 6, 9]  # Start positions: req0 at 0, req1 at 6, end at 9
seq_lens = [6, 3]             # Lengths: req0 has 6 tokens, req1 has 3
num_actual_tokens = 9         # Total tokens in batch
```

**Extract per-request ranges**:
```python
if attn_metadata and len(batch_mapping) > 1:
    # Multi-request batch
    query_start_loc = attn_metadata.query_start_loc.cpu().numpy()
    num_reqs = len(query_start_loc) - 1
    
    request_ranges = []
    for i in range(num_reqs):
        start = int(query_start_loc[i])
        end = int(query_start_loc[i + 1])
        request_ranges.append((start, end))
    # request_ranges = [(0, 6), (6, 9)]
```

### 3. Split Attention Computation by Request

**Current approach** (single request):
```python
# Compute attention for entire batch
attn_weights = torch.softmax(scores, dim=-1)  # [num_heads, total_tokens, seq_len]

# Store under single request ID
capture_hook.capture_attention_weights(layer_idx, attn_weights, request_id)
```

**New approach** (multi-request):
```python
# Compute attention for entire batch (same as before)
attn_weights = torch.softmax(scores, dim=-1)  # [num_heads, 9, 9]

# Split by request boundaries
for req_idx, (start, end) in enumerate(request_ranges):
    req_id = batch_mapping[req_idx]
    
    # Extract attention for this request's tokens
    # Shape: [num_heads, num_tokens_this_req, full_seq_len]
    req_attn = attn_weights[:, start:end, :]
    
    # Also need to extract only the relevant context for this request
    # This is trickier - need to know which context positions belong to this request
    
    capture_hook.capture_attention_weights(layer_idx, req_attn, req_id)
```

### 4. Handle Context Length per Request

**Challenge**: Each request may have a different context length, but the attention tensor has a fixed `seq_len` dimension.

**During prefill** (easier):
- Each request attends only to its own tokens
- Request 0: attends to positions [0:6]
- Request 1: attends to positions [6:9]

```python
# Prefill: Extract per-request attention
for req_idx, (start, end) in enumerate(request_ranges):
    req_id = batch_mapping[req_idx]
    num_tokens = end - start
    
    # Extract this request's attention to its own context
    req_attn = attn_weights[:, start:end, start:end]  # [num_heads, num_tokens, num_tokens]
    
    capture_hook.capture_attention_weights(layer_idx, req_attn, req_id)
```

**During decode** (more complex):
- Each request generates 1 token
- Attends to its accumulated K/V (different lengths per request)
- Need to track accumulated context per request separately

```python
# Decode: Need separate K/V accumulators per request
for req_idx, (start, end) in enumerate(request_ranges):
    req_id = batch_mapping[req_idx]
    
    # Get this request's accumulated K/V
    accumulator = attention_layer._raw_kv_accumulators[req_id]
    context_len = len(accumulator['keys'])  # This request's context length
    
    # Extract this request's query and context attention
    q_req = query[start:end]  # Should be 1 token during decode
    
    # Compute attention for this specific request
    # ... (similar to current decode logic but per-request)
```

### 5. Update K/V Accumulator Logic

**Current**: All K/V stored under one request ID  
**Needed**: Split K/V by request boundaries before storing

```python
# During prefill with multiple requests
for req_idx, (start, end) in enumerate(request_ranges):
    req_id = batch_mapping[req_idx]
    
    # Extract K/V for this request
    k_req = key[start:end]  # [num_tokens_req, num_kv_heads * head_size]
    v_req = value[start:end]
    
    # Store in this request's accumulator
    if req_id not in attention_layer._raw_kv_accumulators:
        attention_layer._raw_kv_accumulators[req_id] = {
            'keys': [k_req.view(end-start, num_kv_heads, head_size)],
            'values': [v_req.view(end-start, num_kv_heads, head_size)]
        }
```

## Implementation Steps

1. **Add metadata access** (~10 lines)
   - Access `attn_metadata` from attention layer
   - Extract `query_start_loc` and `seq_lens`
   - Compute request ranges

2. **Split prefill attention** (~30 lines)
   - Loop over request ranges
   - Extract per-request attention slices
   - Store under correct request ID

3. **Split decode attention** (~50 lines)
   - Loop over request ranges (should be 1 token per request)
   - Use per-request K/V accumulators
   - Compute attention separately for each request
   - Store under correct request ID

4. **Update K/V accumulator logic** (~20 lines)
   - Split K/V tensors by request boundaries during prefill
   - Store in separate accumulators per request ID

5. **Add fallback logic** (~10 lines)
   - If `attn_metadata` unavailable or `query_start_loc` missing
   - Fall back to current single-request behavior

## Complexity Estimate

- **Code changes**: ~120 lines of modifications
- **Testing**: Need to verify:
  - Single request still works (regression test)
  - Concurrent requests get separate attention data
  - Request IDs are correctly mapped
  - Attention shapes are correct per request
  
## Alternative: Simpler Approach

If full separation is too complex, a **simpler option** is:

**Store under all request IDs with metadata**:
```python
# Store the concatenated attention under all request IDs
# Include metadata about which tokens belong to which request
for req_idx, req_id in enumerate(batch_mapping.values()):
    capture_hook.capture_attention_weights(
        layer_idx, 
        attn_weights,  # Full batched attention
        req_id,
        metadata={'request_ranges': request_ranges, 'my_index': req_idx}
    )
```

Then during retrieval, users can:
```python
scores = get_attention_scores(request_id)
# scores contains metadata about how to slice for this specific request
my_idx = scores['metadata']['my_index']
start, end = scores['metadata']['request_ranges'][my_idx]
my_attention = scores[layer][..., start:end, start:end]
```

This is much simpler (~30 lines) but pushes the splitting logic to the user.

## Recommendation

Start with the **simpler approach** (store with metadata) because:
1. Much less code to maintain
2. Easier to test and debug
3. User has full control over how to interpret batched data
4. Can upgrade to full separation later if needed

The full separation approach should be implemented only if there's a strong use case for automatic per-request splitting.
