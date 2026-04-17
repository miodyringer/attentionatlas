# Multi-Request Support - Quick Start Guide

## Problem Statement

The current vLLM Attention Capture Plugin is limited to single-request scenarios because it uses a hardcoded request ID:

```python
request_id = "default_request"  # ❌ All requests use the same ID
```

This means:
- ❌ Sequential requests overwrite each other's data
- ❌ Batched inference (multiple prompts) gets mixed together  
- ❌ Concurrent requests corrupt each other's captures
- ❌ No way to isolate attention patterns per request

## Solution Overview

To support multiple requests, we need to:

1. **Extract real request IDs** from vLLM's runtime context
2. **Store K/V accumulators per-request** (not shared across all requests)
3. **Map tokens to requests** in batched inference
4. **Clean up** per-request state when generation completes

## Critical Code Changes

### Change 1: Extract Request ID (attention_layer_patcher.py)

**Current:**
```python
# Line ~204
request_id = "default_request"  # ❌ HARDCODED
capture_hook.capture_attention_weights(
    layer_idx=layer_idx,
    attn_weights=attn_weights,
    request_id=request_id,
)
```

**Fixed:**
```python
# Extract from vLLM context
request_id = extract_request_id_from_vllm(attention_layer, attn_metadata)
capture_hook.capture_attention_weights(
    layer_idx=layer_idx,
    attn_weights=attn_weights,
    request_id=request_id,  # ✅ Real request ID
)
```

### Change 2: Per-Request K/V Accumulator (attention_layer_patcher.py)

**Current (BROKEN):**
```python
# Lines ~109-128 - ONE accumulator shared by ALL requests ❌
if not hasattr(attention_layer, '_raw_kv_accumulator'):
    attention_layer._raw_kv_accumulator = {
        'keys': [],
        'values': []
    }

# All requests write to the same accumulator
accumulator = attention_layer._raw_kv_accumulator
```

**Fixed:**
```python
# Dictionary of accumulators, keyed by request_id ✅
if not hasattr(attention_layer, '_raw_kv_accumulators'):
    attention_layer._raw_kv_accumulators = {}

# Get or create accumulator for THIS request
if request_id not in attention_layer._raw_kv_accumulators:
    attention_layer._raw_kv_accumulators[request_id] = {
        'keys': [],
        'values': []
    }

accumulator = attention_layer._raw_kv_accumulators[request_id]
```

### Change 3: Helper Function to Extract Request ID

Add this new function to `attention_layer_patcher.py`:

```python
def extract_request_id_from_vllm(
    attention_layer: Any,
    attn_metadata: Any = None,
) -> str:
    """Extract request ID from vLLM context.
    
    Args:
        attention_layer: The attention layer (may have layer_name)
        attn_metadata: Optional pre-fetched attention metadata
    
    Returns:
        Request ID string
    """
    # Try to get attention metadata if not provided
    if attn_metadata is None:
        try:
            layer_name = getattr(attention_layer, "layer_name", None)
            if layer_name:
                from vllm.model_executor.layers.attention.attention import get_attention_context
                attn_metadata, _, _, _ = get_attention_context(layer_name)
        except Exception:
            pass
    
    # Try to extract request ID from metadata
    if attn_metadata is not None:
        # Check for request_ids attribute (list of IDs for batch)
        if hasattr(attn_metadata, 'request_ids') and attn_metadata.request_ids:
            # For now, take first request in batch
            # TODO: Handle batched inference properly
            request_ids = attn_metadata.request_ids
            if isinstance(request_ids, (list, tuple)) and len(request_ids) > 0:
                return str(request_ids[0])
            else:
                return str(request_ids)
        
        # Check for seq_ids as fallback
        if hasattr(attn_metadata, 'seq_ids') and attn_metadata.seq_ids:
            seq_ids = attn_metadata.seq_ids
            if isinstance(seq_ids, (list, tuple)) and len(seq_ids) > 0:
                return f"seq_{seq_ids[0]}"
            else:
                return f"seq_{seq_ids}"
    
    # Try forward context
    try:
        from vllm.forward_context import get_forward_context
        forward_ctx = get_forward_context()
        if hasattr(forward_ctx, 'request_id'):
            return str(forward_ctx.request_id)
        if hasattr(forward_ctx, 'request_ids') and forward_ctx.request_ids:
            return str(forward_ctx.request_ids[0])
    except Exception:
        pass
    
    # Fallback: generate unique ID based on timestamp
    # This ensures different requests don't collide even if we can't
    # extract the real ID from vLLM
    import time
    fallback_id = f"unknown_{int(time.time() * 1000000)}"
    
    logger.warning(
        "Could not extract request ID from vLLM context. "
        f"Using fallback: {fallback_id}"
    )
    
    return fallback_id
```

## Implementation Steps

### Step 1: Update forward_with_capture()

In `attention_layer_patcher.py`, modify the `forward_with_capture()` function:

```python
def forward_with_capture(...):
    num_tokens = query.shape[0]
    is_decode = (num_tokens == 1)
    
    # Check if we should capture this layer
    if not capture_hook.should_capture(layer_idx):
        return original_forward(query, key, value, output_shape)
    
    # ✅ EXTRACT REQUEST ID
    request_id = extract_request_id_from_vllm(attention_layer)
    
    # ✅ PER-REQUEST K/V ACCUMULATOR
    if key is not None and value is not None:
        # Initialize per-request accumulator storage
        if not hasattr(attention_layer, '_raw_kv_accumulators'):
            attention_layer._raw_kv_accumulators = {}
        
        # Get or create accumulator for this request
        if request_id not in attention_layer._raw_kv_accumulators:
            attention_layer._raw_kv_accumulators[request_id] = {
                'keys': [],
                'values': []
            }
        
        accumulator = attention_layer._raw_kv_accumulators[request_id]
        
        # Reshape K/V
        k_reshaped = key.view(num_tokens, num_kv_heads, head_size)
        v_reshaped = value.view(num_tokens, num_kv_heads, head_size)
        
        if is_decode:
            # Decode: append to this request's accumulator
            accumulator['keys'].append(k_reshaped)
            accumulator['values'].append(v_reshaped)
        else:
            # Prefill: reset this request's accumulator
            attention_layer._raw_kv_accumulators[request_id] = {
                'keys': [k_reshaped],
                'values': [v_reshaped]
            }
    
    # Call vLLM's original forward
    result = original_forward(query, key, value, output_shape)
    
    # Compute attention...
    try:
        if not is_decode:
            # Prefill logic...
            pass
        else:
            # ✅ USE THIS REQUEST'S ACCUMULATOR
            if not hasattr(attention_layer, '_raw_kv_accumulators'):
                logger.warning(f"Layer {layer_idx}: No K/V accumulators")
                return result
            
            if request_id not in attention_layer._raw_kv_accumulators:
                logger.warning(
                    f"Layer {layer_idx}: No accumulator for request {request_id}"
                )
                return result
            
            accumulator = attention_layer._raw_kv_accumulators[request_id]
            all_keys = torch.cat(accumulator['keys'], dim=0)
            all_values = torch.cat(accumulator['values'], dim=0)
            
            # Compute attention with this request's K/V...
        
        # ✅ STORE WITH REAL REQUEST ID
        capture_hook.capture_attention_weights(
            layer_idx=layer_idx,
            attn_weights=attn_weights,
            request_id=request_id,  # Real ID, not "default_request"
        )
    
    except Exception as e:
        logger.error(f"Layer {layer_idx}: Failed to compute attention: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return result
```

### Step 2: Add Cleanup Logic

Add cleanup to `api.py` to clear stale accumulators:

```python
def clear_request_state(llm: Any, request_id: str) -> None:
    """Clear per-request state from all patched layers.
    
    Call this after generation completes to free memory.
    
    Args:
        llm: The vLLM LLM instance
        request_id: The request ID to clean up
    """
    llm_id = id(llm)
    hook = _CAPTURE_HOOKS.get(llm_id)
    
    if hook:
        # Clear captured attention scores
        hook.clear_request(request_id)
    
    # TODO: Clear K/V accumulators from patched layers
    # This requires traversing the model, which is complex in v1
    # For now, rely on Python GC to clean up old accumulators
```

### Step 3: Update API Usage

The API already works! Just need users to use the real request ID:

```python
from vllm import LLM, SamplingParams
from vllm_attention_capture_plugin import enable_attention_capture, get_attention_scores

llm = LLM(model="gpt2")
enable_attention_capture(llm, capture_layers=[0, 1, 2])

# Single request
outputs = llm.generate("Hello", SamplingParams(max_tokens=10))
scores = get_attention_scores(outputs[0].request_id)  # ✅ Works!

# Batched requests
outputs = llm.generate(
    ["Hello", "Goodbye"],
    SamplingParams(max_tokens=10)
)
for output in outputs:
    scores = get_attention_scores(output.request_id)  # ✅ Each request isolated
    print(f"Request {output.request_id}: {scores[0].shape}")
```

## Testing Checklist

- [ ] Single request works (backward compatibility)
- [ ] Sequential requests don't interfere (data isolation)
- [ ] Batched requests are isolated (separate captures)
- [ ] Request IDs are correctly extracted from vLLM
- [ ] Fallback ID generation works when vLLM context unavailable
- [ ] Memory cleanup prevents leaks

## Next Steps

1. **Run investigation script** to confirm what vLLM provides
2. **Implement the helper function** (`extract_request_id_from_vllm`)
3. **Update forward_with_capture()** with per-request accumulator
4. **Test thoroughly** with single, sequential, and batched requests
5. **Add cleanup logic** to prevent memory leaks

## Estimated Effort

- **Investigation**: 1 hour (running script + analyzing output)
- **Core implementation**: 3-4 hours (helper function + accumulator fix)
- **Testing**: 2-3 hours (single, sequential, batch tests)
- **Documentation**: 1 hour (update docs)

**Total: 7-9 hours**

---

## Fallback Strategy

If vLLM doesn't expose request IDs in the context, we can use a **timestamp-based unique ID**:

```python
import time

def generate_unique_request_id() -> str:
    """Generate a unique request ID based on timestamp."""
    return f"req_{int(time.time() * 1000000)}"
```

This ensures different requests don't collide, even if we can't map back to vLLM's request ID. Users would need to track the mapping themselves.

---

**End of Quick Start Guide**
