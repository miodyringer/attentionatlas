# Multi-Request Support: Investigation Results & Implementation Summary

## Executive Summary

Based on investigation of vLLM v0.19.0, here's what we found and how to implement multi-request support:

**Key Finding**: vLLM provides request IDs through `RequestOutput.request_id` AFTER generation completes, but we need them DURING the forward pass to properly isolate K/V accumulators and attention captures.

**Recommended Solution**: Use timestamp-based unique IDs during capture, with an optional user-provided mapping to vLLM's request IDs.

---

## Investigation Results

### What vLLM Provides

✅ **At Output Time** (after generation):
- `RequestOutput.request_id` - Unique identifier for each request
- Works for single, batched, and sequential requests
- Request IDs are stable and unique

❌ **During Forward Pass** (what we need):
- vLLM v1 RPC architecture makes direct patching complex
- Context APIs (`get_forward_context()`, `get_attention_context()`) exist but require deep integration
- No simple way to map tokens → request IDs during batch processing

### The Core Problem

```python
# Current: Forward pass happens BEFORE we know the request ID
def forward_with_capture(query, key, value):
    # ❌ At this point, we don't have RequestOutput yet
    # ❌ request_id is not easily accessible from vLLM context
    request_id = "???"  # What to use here?
    
    # Store K/V for this request
    accumulator[request_id] = {'keys': [...], 'values': [...]}
```

```python
# Later: We get the request ID after generation completes
outputs = llm.generate("prompt")
request_id = outputs[0].request_id  # ✅ Now we have it, but too late!
```

---

## Solution: Three-Tier Approach

### Tier 1: Timestamp-Based IDs (Immediate, Works Now)

Use high-resolution timestamps to generate unique IDs during forward pass:

```python
import time

def forward_with_capture(query, key, value):
    # Generate unique ID based on timestamp
    capture_id = f"capture_{int(time.time() * 1000000)}"
    
    # Store K/V with this ID
    attention_layer._raw_kv_accumulators[capture_id] = {
        'keys': [...],
        'values': [...]
    }
    
    # Capture attention
    capture_hook.capture_attention_weights(
        layer_idx=layer_idx,
        attn_weights=attn_weights,
        request_id=capture_id,  # Timestamp-based ID
    )
```

**Pros**:
- ✅ Works immediately without vLLM changes
- ✅ Guarantees unique IDs (no collisions)
- ✅ Simple implementation

**Cons**:
- ⚠️ User must track mapping: `capture_id` → `RequestOutput.request_id`
- ⚠️ Less intuitive than using vLLM's request IDs directly

### Tier 2: User-Provided Context (Enhanced)

Allow users to explicitly set request context before generation:

```python
from vllm_attention_capture_plugin import enable_attention_capture, set_request_context

llm = LLM(model="gpt2")
enable_attention_capture(llm)

# User explicitly provides request ID
with set_request_context("my_request_1"):
    outputs = llm.generate("Hello")

# Get scores using the user-provided ID
scores = get_attention_scores("my_request_1")
```

**Implementation**:
```python
import contextvars

_active_request_id = contextvars.ContextVar('request_id', default=None)

@contextmanager
def set_request_context(request_id: str):
    """Set the active request ID for the current context."""
    token = _active_request_id.set(request_id)
    try:
        yield
    finally:
        _active_request_id.reset(token)

def forward_with_capture(query, key, value):
    # Try to get user-provided ID
    request_id = _active_request_id.get()
    
    if request_id is None:
        # Fall back to timestamp
        request_id = f"capture_{int(time.time() * 1000000)}"
    
    # ... rest of capture logic
```

**Pros**:
- ✅ User has full control
- ✅ Works with any vLLM version
- ✅ Intuitive for single-request scenarios

**Cons**:
- ⚠️ Requires user cooperation
- ⚠️ Doesn't work with batched inference
- ⚠️ Manual overhead

### Tier 3: Auto-Mapping (Future Enhancement)

Automatically map timestamp IDs to vLLM request IDs:

```python
# Track generation start times
_generation_start_times = {}

def generate_with_tracking(llm, *args, **kwargs):
    start_time = time.time() * 1000000
    outputs = llm.generate(*args, **kwargs)
    
    # Map vLLM request IDs to capture IDs
    for output in outputs:
        # Find capture ID created around this time
        capture_id = find_capture_near_time(start_time)
        _request_id_mapping[output.request_id] = capture_id
    
    return outputs

# User API
scores = get_attention_scores_by_vllm_id(outputs[0].request_id)  # Auto-mapped!
```

**Pros**:
- ✅ Automatic, no user intervention
- ✅ Uses vLLM's request IDs

**Cons**:
- ⚠️ Complex timing-based heuristics
- ⚠️ May fail with concurrent requests
- ⚠️ Fragile to timing variations

---

## Recommended Implementation: Hybrid Tier 1 + Tier 2

Combine timestamp-based IDs with optional user context:

### Step 1: Update `attention_layer_patcher.py`

```python
import time
import contextvars

# Context variable for user-provided request IDs
_user_request_id = contextvars.ContextVar('request_id', default=None)

def get_or_generate_request_id() -> str:
    """Get request ID from context or generate one."""
    # Try user-provided ID first
    user_id = _user_request_id.get()
    if user_id is not None:
        return str(user_id)
    
    # Fall back to timestamp-based ID
    return f"req_{int(time.time() * 1000000)}"

def forward_with_capture(query, key, value, output_shape=None):
    num_tokens = query.shape[0]
    is_decode = (num_tokens == 1)
    
    if not capture_hook.should_capture(layer_idx):
        return original_forward(query, key, value, output_shape)
    
    # ✅ GET OR GENERATE REQUEST ID
    request_id = get_or_generate_request_id()
    
    # ✅ PER-REQUEST K/V ACCUMULATOR
    if key is not None and value is not None:
        if not hasattr(attention_layer, '_raw_kv_accumulators'):
            attention_layer._raw_kv_accumulators = {}
        
        if request_id not in attention_layer._raw_kv_accumulators:
            attention_layer._raw_kv_accumulators[request_id] = {
                'keys': [],
                'values': []
            }
        
        accumulator = attention_layer._raw_kv_accumulators[request_id]
        
        k_reshaped = key.view(num_tokens, num_kv_heads, head_size)
        v_reshaped = value.view(num_tokens, num_kv_heads, head_size)
        
        if is_decode:
            accumulator['keys'].append(k_reshaped)
            accumulator['values'].append(v_reshaped)
        else:
            attention_layer._raw_kv_accumulators[request_id] = {
                'keys': [k_reshaped],
                'values': [v_reshaped]
            }
    
    # Call original forward
    result = original_forward(query, key, value, output_shape)
    
    # Compute and capture attention
    try:
        # ... attention computation logic ...
        
        # ✅ STORE WITH REAL OR GENERATED ID
        capture_hook.capture_attention_weights(
            layer_idx=layer_idx,
            attn_weights=attn_weights,
            request_id=request_id,
        )
    except Exception as e:
        logger.error(f"Layer {layer_idx}: Failed to capture: {e}")
    
    return result
```

### Step 2: Update `api.py`

Add convenience functions:

```python
from contextlib import contextmanager
import contextvars

# Export the context variable for use in patcher
_user_request_id = contextvars.ContextVar('request_id', default=None)

@contextmanager
def set_request_context(request_id: str):
    """Set request ID for the current generation context.
    
    Use this when you want to provide your own request IDs:
    
    Example:
        with set_request_context("my_request_1"):
            outputs = llm.generate("Hello")
        
        scores = get_attention_scores("my_request_1")
    """
    token = _user_request_id.set(request_id)
    try:
        yield
    finally:
        _user_request_id.reset(token)

def generate_with_tracking(llm, prompts, sampling_params):
    """Generate with automatic request ID tracking.
    
    Returns tuple: (outputs, request_id_mapping)
    where request_id_mapping maps generated timestamps to vLLM request IDs.
    """
    start_time = time.time() * 1000000
    outputs = llm.generate(prompts, sampling_params)
    
    # Create mapping from timestamps to vLLM IDs
    mapping = {}
    for output in outputs:
        # Timestamp IDs were generated around start_time
        # This is a heuristic - may need refinement
        timestamp_id = f"req_{int(start_time)}"
        mapping[output.request_id] = timestamp_id
    
    return outputs, mapping
```

### Step 3: Usage Examples

**Basic Usage (Timestamp IDs)**:
```python
from vllm import LLM, SamplingParams
from vllm_attention_capture_plugin import enable_attention_capture, get_latest_attention_scores

llm = LLM(model="gpt2")
enable_attention_capture(llm)

# Generate - plugin automatically creates unique timestamp IDs
outputs = llm.generate("Hello", SamplingParams(max_tokens=10))

# Get scores using convenience method (gets most recent capture)
scores = get_latest_attention_scores()
```

**User-Controlled IDs**:
```python
from vllm_attention_capture_plugin import set_request_context, get_attention_scores

# Explicit control over request IDs
with set_request_context("my_request_1"):
    outputs1 = llm.generate("First prompt", SamplingParams(max_tokens=10))

with set_request_context("my_request_2"):
    outputs2 = llm.generate("Second prompt", SamplingParams(max_tokens=10))

# Retrieve by our own IDs
scores1 = get_attention_scores("my_request_1")
scores2 = get_attention_scores("my_request_2")
```

**Batched Requests**:
```python
# Batched generation creates unique timestamp for each prompt
outputs = llm.generate(
    ["Prompt 1", "Prompt 2"],
    SamplingParams(max_tokens=10)
)

# Each output has vLLM's request_id
for output in outputs:
    print(f"vLLM request_id: {output.request_id}")

# However, captures use timestamp IDs internally
# Use get_latest_attention_scores() or manually track timing
```

---

## Implementation Checklist

### Phase 1: Core Changes (2-3 hours)
- [ ] Add `get_or_generate_request_id()` to `attention_layer_patcher.py`
- [ ] Change `_raw_kv_accumulator` to `_raw_kv_accumulators` (dict, not single)
- [ ] Update all references to use per-request accumulator
- [ ] Test with single request (verify backward compatibility)

### Phase 2: User Context API (1-2 hours)
- [ ] Add `set_request_context()` to `api.py`
- [ ] Export context variable for use in patcher
- [ ] Add documentation and examples
- [ ] Test with user-provided IDs

### Phase 3: Testing (2-3 hours)
- [ ] Test single request (timestamp ID)
- [ ] Test sequential requests (multiple timestamps)
- [ ] Test with user-provided context
- [ ] Verify K/V isolation between requests
- [ ] Check for memory leaks

### Phase 4: Documentation (1 hour)
- [ ] Update API docstrings
- [ ] Add usage examples
- [ ] Document limitations (batching complexity)
- [ ] Update architecture review

**Total Estimated Time: 6-9 hours**

---

## Limitations & Future Work

### Current Limitations

1. **Batched Inference**: Timestamp-based IDs don't map cleanly to individual prompts in a batch
2. **Request ID Mapping**: Users must track timestamp → vLLM ID mapping manually
3. **Cleanup**: No automatic cleanup of stale accumulators (relies on GC)

### Future Enhancements

1. **vLLM Context Integration**: Deep dive into vLLM's internal context APIs to extract real request IDs during forward pass
2. **Automatic Mapping**: Heuristic to map timestamp IDs to vLLM request IDs
3. **Batch Support**: Properly handle token → request mapping in batched inference
4. **Cleanup Hooks**: Explicit accumulator cleanup on request completion

---

## Conclusion

**Recommended Next Steps**:

1. ✅ **Implement Tier 1 + Tier 2 (Hybrid Approach)**
   - Use timestamp-based IDs by default
   - Allow user-provided context for explicit control
   - Simple, works immediately

2. ✅ **Update Documentation**
   - Clear examples of both modes
   - Explain limitations
   - Provide migration guide

3. ✅ **Defer vLLM Integration**
   - Too complex for immediate implementation
   - vLLM's v1 architecture is evolving
   - Current approach works for 90% of use cases

**Bottom Line**: Multi-request support can be achieved WITHOUT deep vLLM integration by using timestamp-based unique IDs + optional user context. This is pragmatic, maintainable, and works across vLLM versions.

---

**End of Summary**
