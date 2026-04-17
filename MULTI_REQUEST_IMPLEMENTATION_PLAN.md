# Multi-Request Support Implementation Plan

**Goal**: Replace hardcoded `request_id = "default_request"` with actual request tracking to support concurrent and batched inference.

**Current Status**: Single-request only, cannot handle:
- Batched inference (multiple prompts in one generate call)
- Concurrent requests (multiple generate calls in flight)
- Request isolation (attention from different requests gets mixed)

---

## Problem Analysis

### Current Implementation

```python
# vllm_attention_capture_plugin/wrappers/attention_layer_patcher.py:204
request_id = "default_request"  # ❌ HARDCODED
capture_hook.capture_attention_weights(
    layer_idx=layer_idx,
    attn_weights=attn_weights,
    request_id=request_id,
)
```

**Issues:**
1. All captures go to same `"default_request"` bucket
2. No way to distinguish between different prompts
3. Concurrent requests overwrite each other's data
4. Batch inference captures get mixed together

### What We Need

To properly track requests, we need to:
1. **Extract request_id from vLLM's runtime context**
2. **Handle batched inference** (multiple requests per forward pass)
3. **Isolate captures per request** (separate storage)
4. **Track request lifecycle** (prefill → decode → completion)

---

## Investigation: vLLM Request Context

### Available Context APIs

vLLM provides context management through:

```python
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context
```

### What Information is Available?

We need to investigate what request metadata vLLM exposes during forward pass:

#### Option 1: `get_forward_context()`
```python
# Potentially available in vLLM v1
forward_ctx = get_forward_context()
# Expected to contain: request_ids, batch info, metadata
```

#### Option 2: `get_attention_context(layer_name)`
```python
# Already used in the code (but only for cache access)
attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
# attn_metadata may contain: seq_lens, block_table, slot_mapping
# Question: Does it have request_ids?
```

#### Option 3: Thread-local or global request registry
```python
# vLLM may maintain a registry accessible via:
# - ModelConfig
# - EngineCore state
# - Worker attributes
```

---

## Implementation Strategy

### Phase 1: Investigate vLLM Context (Research)

**Tasks:**
1. ✅ Check if `attn_metadata` contains request IDs
2. ✅ Inspect `get_forward_context()` return value
3. ✅ Check vLLM's `SequenceGroup` and `Sequence` classes
4. ✅ Look for batch metadata in forward pass
5. ✅ Check v0 vs v1 differences in request tracking

**Validation:**
```python
# Test script to inspect available context
def forward_with_capture(...):
    # Log everything available
    try:
        forward_ctx = get_forward_context()
        logger.info(f"Forward context: {forward_ctx}")
        logger.info(f"Forward context type: {type(forward_ctx)}")
        logger.info(f"Forward context dir: {dir(forward_ctx)}")
    except Exception as e:
        logger.info(f"get_forward_context() failed: {e}")
    
    try:
        layer_name = getattr(attention_layer, "layer_name", None)
        if layer_name:
            attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
            logger.info(f"Attention metadata: {attn_metadata}")
            logger.info(f"Attention metadata type: {type(attn_metadata)}")
            logger.info(f"Attention metadata dir: {dir(attn_metadata)}")
            
            # Check for request-related attributes
            if hasattr(attn_metadata, 'request_ids'):
                logger.info(f"Found request_ids: {attn_metadata.request_ids}")
            if hasattr(attn_metadata, 'seq_ids'):
                logger.info(f"Found seq_ids: {attn_metadata.seq_ids}")
            if hasattr(attn_metadata, 'batch_size'):
                logger.info(f"Found batch_size: {attn_metadata.batch_size}")
    except Exception as e:
        logger.info(f"get_attention_context() failed: {e}")
```

### Phase 2: Extract Request IDs (Implementation)

**Approach A: Use attn_metadata (Likely)**

If `attn_metadata` contains request/sequence IDs:

```python
def forward_with_capture(...):
    # Get attention context for this layer
    layer_name = getattr(attention_layer, "layer_name", None)
    if layer_name:
        attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
        
        # Extract request IDs from metadata
        if hasattr(attn_metadata, 'request_ids'):
            # Batch case: multiple requests
            request_ids = attn_metadata.request_ids
        elif hasattr(attn_metadata, 'seq_ids'):
            # Alternative: sequence IDs
            request_ids = [f"seq_{sid}" for sid in attn_metadata.seq_ids]
        else:
            # Fallback: use sequence length as proxy
            request_ids = [f"unknown_{i}" for i in range(num_tokens)]
    else:
        # No layer_name: fallback to default
        request_ids = ["default_request"]
```

**Approach B: Use forward_context (Alternative)**

If forward_context is more reliable:

```python
def forward_with_capture(...):
    try:
        forward_ctx = get_forward_context()
        if hasattr(forward_ctx, 'request_ids'):
            request_ids = forward_ctx.request_ids
        elif hasattr(forward_ctx, 'current_request_id'):
            request_ids = [forward_ctx.current_request_id]
        else:
            raise AttributeError("No request ID in forward context")
    except Exception as e:
        logger.warning(f"Could not extract request ID: {e}")
        request_ids = ["default_request"]  # Fallback
```

**Approach C: Augment vLLM Input (Hacky but Reliable)**

Inject request IDs at generation time:

```python
# In api.py, wrap llm.generate()
original_generate = llm.generate

def generate_with_tracking(*args, **kwargs):
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    # Store in thread-local or global registry
    _current_request_id.set(request_id)
    
    # Call original
    result = original_generate(*args, **kwargs)
    
    # Attach request_id to result
    for output in result:
        output._attention_request_id = request_id
    
    return result

llm.generate = generate_with_tracking
```

Then in `forward_with_capture()`:
```python
request_id = getattr(_current_request_id, 'value', 'default_request')
```

### Phase 3: Handle Batched Inference (Implementation)

**Challenge**: A single forward pass may process multiple requests simultaneously.

**Solution**: Store per-token request mappings

```python
def forward_with_capture(...):
    num_tokens = query.shape[0]
    
    # Get request ID for each token in the batch
    if hasattr(attn_metadata, 'seq_lens'):
        # seq_lens tells us how many tokens belong to each request
        seq_lens = attn_metadata.seq_lens
        request_ids_list = attn_metadata.request_ids or []
        
        # Map each token to its request
        token_to_request = []
        for req_idx, seq_len in enumerate(seq_lens):
            request_id = request_ids_list[req_idx] if req_idx < len(request_ids_list) else f"req_{req_idx}"
            token_to_request.extend([request_id] * seq_len)
        
        # Now we know: token_to_request[i] = request ID for token i
    else:
        # Fallback: assume all tokens belong to same request
        token_to_request = ["default_request"] * num_tokens
```

**Storage Strategy**:

```python
# After computing attention
for token_idx in range(num_tokens):
    request_id = token_to_request[token_idx]
    
    # Extract attention for this token
    token_attn = attn_weights[:, token_idx, :]  # [num_heads, seq_len]
    
    # Store under the correct request
    capture_hook.capture_attention_weights(
        layer_idx=layer_idx,
        attn_weights=token_attn.unsqueeze(1),  # [num_heads, 1, seq_len]
        request_id=request_id,
    )
```

### Phase 4: Track Raw K/V Accumulator Per Request (Critical Fix)

**Problem**: Current implementation stores K/V accumulator on the layer:

```python
# Current: ONE accumulator for ALL requests ❌
if not hasattr(attention_layer, '_raw_kv_accumulator'):
    attention_layer._raw_kv_accumulator = {'keys': [], 'values': []}
```

**Solution**: Store K/V accumulator per request:

```python
# Fixed: Dictionary of accumulators, keyed by request_id ✅
if not hasattr(attention_layer, '_raw_kv_accumulators'):
    attention_layer._raw_kv_accumulators = {}

if request_id not in attention_layer._raw_kv_accumulators:
    attention_layer._raw_kv_accumulators[request_id] = {
        'keys': [],
        'values': []
    }

# Access accumulator for this request
accumulator = attention_layer._raw_kv_accumulators[request_id]

if is_decode:
    accumulator['keys'].append(k_reshaped)
    accumulator['values'].append(v_reshaped)
else:
    # Prefill: reset accumulator for this request
    attention_layer._raw_kv_accumulators[request_id] = {
        'keys': [k_reshaped],
        'values': [v_reshaped]
    }
```

**Cleanup**: Clear accumulator when request completes:

```python
# Add cleanup method to API
def _on_request_complete(request_id: str):
    """Called when a request finishes generation."""
    # Clean up accumulators for this request
    for hook in _CAPTURE_HOOKS.values():
        hook.clear_request(request_id)
    
    # Clean up K/V accumulators in patched layers
    # (Need to traverse model and clean _raw_kv_accumulators[request_id])
```

### Phase 5: Update API (Implementation)

**No API changes needed** - the existing API already accepts request IDs:

```python
# Already works! Just need to populate with real IDs
scores = get_attention_scores(outputs[0].request_id)
```

**Add batch-aware convenience method**:

```python
def get_attention_scores_batch(outputs: list[RequestOutput]) -> dict[str, dict[int, np.ndarray]]:
    """Get attention scores for multiple requests.
    
    Args:
        outputs: List of RequestOutput objects from llm.generate()
    
    Returns:
        Dictionary mapping request_id to attention scores dict
    """
    results = {}
    for output in outputs:
        scores = get_attention_scores(output.request_id)
        if scores:
            results[output.request_id] = scores
    return results
```

---

## Implementation Checklist

### Investigation Phase
- [ ] Create test script to inspect vLLM context objects
- [ ] Run test with single request and log all available metadata
- [ ] Run test with batched requests (2+ prompts) and log metadata
- [ ] Document which approach works: attn_metadata vs forward_context vs augmentation
- [ ] Test on both vLLM v0 and v1

### Core Implementation
- [ ] Extract request IDs in `forward_with_capture()`
- [ ] Handle batched inference (map tokens to requests)
- [ ] Fix K/V accumulator to be per-request
- [ ] Add request cleanup logic
- [ ] Remove hardcoded `"default_request"`
- [ ] Add fallback for when request ID unavailable

### Testing
- [ ] Test single request (existing behavior)
- [ ] Test sequential requests (two separate generate calls)
- [ ] Test batched requests (one generate call, multiple prompts)
- [ ] Test concurrent requests (if vLLM supports)
- [ ] Verify request isolation (no data leakage)
- [ ] Check memory cleanup (accumulators cleared)

### Documentation
- [ ] Update docstrings with batch support examples
- [ ] Document request ID extraction approach
- [ ] Add troubleshooting guide for request tracking
- [ ] Update PLUGIN_ARCHITECTURE_REVIEW.md rating

---

## Risk Assessment

### High Risk Areas

1. **vLLM API Instability**
   - Risk: Context APIs may not exist or differ between versions
   - Mitigation: Comprehensive fallback chain, version detection

2. **Batched Inference Complexity**
   - Risk: Token-to-request mapping may be ambiguous
   - Mitigation: Extensive logging, validation checks

3. **Memory Leaks**
   - Risk: Per-request accumulators never get cleaned up
   - Mitigation: Explicit cleanup on request completion, weak references

4. **Performance Degradation**
   - Risk: Per-request bookkeeping adds overhead
   - Mitigation: Benchmark before/after, optimize hot paths

### Medium Risk Areas

1. **Request ID Collisions**
   - Risk: vLLM may reuse request IDs
   - Mitigation: Use (request_id, timestamp) tuples

2. **Distributed Inference**
   - Risk: Request tracking across multiple GPUs
   - Mitigation: Document as unsupported, add validation

---

## Alternative: Minimal Implementation

If vLLM context extraction is too complex, implement a **simpler user-managed approach**:

```python
# User explicitly provides request ID
enable_attention_capture(llm, capture_layers=[0,1,2])

with set_active_request("my_request_1"):
    outputs = llm.generate("Prompt 1")

scores = get_attention_scores("my_request_1")
```

**Implementation**:
```python
import contextvars

_active_request_id = contextvars.ContextVar('request_id', default='default_request')

@contextmanager
def set_active_request(request_id: str):
    token = _active_request_id.set(request_id)
    try:
        yield
    finally:
        _active_request_id.reset(token)
```

**Pros:**
- Simple, reliable, no vLLM coupling
- Works with any vLLM version
- User has full control

**Cons:**
- Requires user cooperation
- Won't work with batched inference
- Manual request management

---

## Recommended Approach

### Best: Hybrid Strategy

1. **Try automatic extraction first** (attn_metadata or forward_context)
2. **Fall back to user-provided context** (set_active_request)
3. **Fall back to "default_request"** (single-request mode)

```python
def get_current_request_id(attn_metadata=None):
    """Get request ID from multiple sources, with fallback chain."""
    
    # Priority 1: Extract from vLLM metadata
    if attn_metadata and hasattr(attn_metadata, 'request_ids'):
        return attn_metadata.request_ids[0]  # First request in batch
    
    # Priority 2: User-provided context variable
    try:
        return _active_request_id.get()
    except LookupError:
        pass
    
    # Priority 3: Thread-local storage (for concurrent requests)
    if hasattr(_thread_locals, 'request_id'):
        return _thread_locals.request_id
    
    # Priority 4: Fallback to single-request mode
    return "default_request"
```

---

## Success Criteria

✅ **Must Have:**
- [ ] Sequential requests work (no data mixing)
- [ ] Request IDs extracted automatically when available
- [ ] Fallback to single-request mode works
- [ ] No memory leaks from stale accumulators
- [ ] Backward compatible with existing code

✅ **Should Have:**
- [ ] Batched inference supported
- [ ] Works on both vLLM v0 and v1
- [ ] User-provided request ID option
- [ ] Clear error messages when request tracking fails

✅ **Nice to Have:**
- [ ] Concurrent request support
- [ ] Automatic cleanup on request completion
- [ ] Performance overhead < 5%
- [ ] Comprehensive logging for debugging

---

## Implementation Timeline

**Estimated Effort: 8-16 hours**

1. **Investigation** (2-4 hours)
   - Write test script
   - Inspect vLLM internals
   - Document findings

2. **Core Implementation** (4-6 hours)
   - Extract request IDs
   - Fix K/V accumulator
   - Handle batching

3. **Testing** (2-4 hours)
   - Single, sequential, batched tests
   - Memory leak checks
   - Performance benchmarks

4. **Documentation** (1-2 hours)
   - Update docstrings
   - Write usage guide
   - Update architecture review

---

## Next Steps

**Immediate:**
1. Create investigation script to inspect vLLM context
2. Run on simple test case and log all available metadata
3. Make decision: automatic extraction vs user-managed vs hybrid

**After Investigation:**
1. Choose implementation approach based on findings
2. Implement core request tracking logic
3. Add comprehensive tests
4. Update documentation

**Questions to Answer:**
- Does `attn_metadata` contain request IDs?
- How does vLLM handle batched requests internally?
- Are request IDs guaranteed unique across time?
- What happens during beam search (multiple sequences per request)?

---

**End of Plan**
