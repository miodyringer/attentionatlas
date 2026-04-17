# How to Test Multi-Request Support

## Quick Test

Run the simple test script:

```bash
python test_multi_request_simple.py
```

This will:
1. ✅ Test sequential requests with auto-generated IDs
2. ✅ Test user-provided custom IDs
3. ✅ Verify no data mixing between requests

**Expected output**:
```
================================================================================
MULTI-REQUEST SUPPORT TEST
================================================================================

1. Creating LLM...
   ✓ LLM created

2. Enabling attention capture...
   ✓ Capture enabled for layer 0

================================================================================
TEST 1: Sequential Requests (Auto-Generated IDs)
================================================================================

Generating request 1...
   Generated: '...'
   ✓ Captured attention, shape: (12, 7, 5)

Generating request 2...
   Generated: '...'
   ✓ Captured attention, shape: (12, 7, 5)

   ✓ Scores are different (requests properly isolated)

✅ TEST 1 PASSED: Sequential requests work!

================================================================================
TEST 2: User-Provided Request IDs
================================================================================

Generating with custom ID 'my_request_A'...
   Generated: '...'

Generating with custom ID 'my_request_B'...
   Generated: '...'

   ✓ Retrieved 'my_request_A', shape: (12, 5, 5)
   ✓ Retrieved 'my_request_B', shape: (12, 5, 5)
   ✓ Scores are different (requests properly isolated)

✅ TEST 2 PASSED: User-provided IDs work!

================================================================================
ALL TESTS PASSED! 🎉
================================================================================

Multi-request support is working correctly:
  ✓ Sequential requests are isolated
  ✓ Auto-generated IDs work
  ✓ User-provided IDs work
  ✓ No data mixing between requests

Your plugin is ready to use!
```

---

## Manual Testing

### Test 1: Basic Usage (Auto-Generated IDs)

```python
import os
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

from vllm import LLM, SamplingParams
from vllm_attention_capture_plugin import (
    enable_attention_capture,
    get_latest_attention_scores
)

# Setup
llm = LLM(model="gpt2", max_model_len=512, gpu_memory_utilization=0.3)
enable_attention_capture(llm, capture_layers=[0], attention_window=5)

# Generate multiple requests
for i in range(3):
    outputs = llm.generate(f"Request {i}", SamplingParams(max_tokens=5))
    scores = get_latest_attention_scores()
    print(f"Request {i}: scores shape = {scores[0].shape}")
```

**What to check**:
- ✅ Each request should return attention scores
- ✅ Scores should have shape `(num_heads, num_tokens, window_size)`
- ✅ Each request should work independently

---

### Test 2: User-Provided IDs

```python
from vllm_attention_capture_plugin import set_request_context, get_attention_scores

# Use custom IDs
with set_request_context("my_request_1"):
    outputs1 = llm.generate("First", SamplingParams(max_tokens=5))

with set_request_context("my_request_2"):
    outputs2 = llm.generate("Second", SamplingParams(max_tokens=5))

# Retrieve by custom IDs
scores1 = get_attention_scores("my_request_1")
scores2 = get_attention_scores("my_request_2")

print(f"Request 1 shape: {scores1[0].shape}")
print(f"Request 2 shape: {scores2[0].shape}")
```

**What to check**:
- ✅ Both requests should return scores
- ✅ Should be able to retrieve by custom ID
- ✅ IDs should be isolated from each other

---

### Test 3: Verify No Data Mixing

```python
import numpy as np

# Generate two different requests
with set_request_context("request_A"):
    outputs_a = llm.generate("Short", SamplingParams(max_tokens=2))

with set_request_context("request_B"):
    outputs_b = llm.generate("This is longer", SamplingParams(max_tokens=3))

# Get scores
scores_a = get_attention_scores("request_A")
scores_b = get_attention_scores("request_B")

# Check they're different
are_different = not np.array_equal(scores_a[0], scores_b[0])
print(f"Scores are different: {are_different}")  # Should be True

# Check different lengths (if prompts are different lengths)
print(f"Request A tokens: {scores_a[0].shape[1]}")
print(f"Request B tokens: {scores_b[0].shape[1]}")
```

**What to check**:
- ✅ Scores should be different (not identical)
- ✅ Different sequence lengths should be captured separately
- ✅ No data from one request should appear in another

---

## Troubleshooting

### Issue: "No scores captured"

**Cause**: vLLM v1 RPC serialization issue

**Fix**: Set environment variable before importing vLLM:
```python
import os
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

# Then import vLLM
from vllm import LLM
```

Or run with environment variable:
```bash
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python your_script.py
```

---

### Issue: "Cannot retrieve request by ID"

**Cause**: Request ID doesn't match

**Fix**: Either:
1. Use `get_latest_attention_scores()` to get most recent
2. Verify you're using the correct ID with `set_request_context()`
3. Check captured IDs: `hook.captured_scores.keys()`

---

### Issue: Test hangs on "Creating LLM"

**Cause**: Multiprocessing needs main guard

**Fix**: Wrap your code in `if __name__ == "__main__":`
```python
if __name__ == "__main__":
    # Your test code here
    llm = LLM(...)
```

---

## What Gets Tested

The test verifies:

1. **Request Isolation**: Each request's K/V accumulator is separate
2. **Auto-Generated IDs**: Timestamp-based IDs work
3. **User-Provided IDs**: Custom IDs via `set_request_context()` work
4. **No Data Mixing**: Scores from different requests are different
5. **Backward Compatibility**: Existing usage patterns still work

---

## Expected Behavior

### Before (Broken):
```python
# Request 1
outputs1 = llm.generate("Hello", SamplingParams(max_tokens=5))
scores1 = get_latest_attention_scores()

# Request 2 - OVERWRITES request 1's data ❌
outputs2 = llm.generate("World", SamplingParams(max_tokens=5))
scores2 = get_latest_attention_scores()

# scores1 and scores2 might be identical (BUG!)
```

### After (Fixed):
```python
# Request 1
outputs1 = llm.generate("Hello", SamplingParams(max_tokens=5))
scores1 = get_latest_attention_scores()

# Request 2 - properly isolated ✅
outputs2 = llm.generate("World", SamplingParams(max_tokens=5))
scores2 = get_latest_attention_scores()

# scores1 and scores2 are different (CORRECT!)
```

---

## Quick Commands

```bash
# Run simple test
python test_multi_request_simple.py

# Run with environment variable
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python test_multi_request_simple.py

# Run full test suite (more comprehensive)
python test_multi_request_support.py
```

---

## Success Criteria

✅ Test passes if:
- Both test sections pass (TEST 1 and TEST 2)
- No "FAILED" messages appear
- Final message shows "ALL TESTS PASSED! 🎉"
- Scores are different between requests (no data mixing)

❌ Test fails if:
- "No scores captured" message appears
- Cannot retrieve by custom ID
- Scores are identical between different requests
- Any assertion error or exception occurs

---

**Ready to test!** Just run: `python test_multi_request_simple.py`
