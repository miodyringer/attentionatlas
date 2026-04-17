#!/usr/bin/env python3
"""
Test script for multi-request support in vLLM Attention Capture Plugin.

Tests:
1. Single request (backward compatibility)
2. Sequential requests with auto-generated IDs
3. User-provided request IDs with set_request_context
4. Request isolation (verify no data mixing)
"""

import numpy as np
from vllm import LLM, SamplingParams

from vllm_attention_capture_plugin import (
    enable_attention_capture,
    get_attention_scores,
    get_latest_attention_scores,
    set_request_context,
    clear_all_captures,
)


def test_single_request():
    """Test single request (backward compatibility)."""
    print("\n" + "="*80)
    print("TEST 1: Single Request (Backward Compatibility)")
    print("="*80 + "\n")

    llm = LLM(model="gpt2", max_model_len=512, gpu_memory_utilization=0.3)
    enable_attention_capture(llm, capture_layers=[0], attention_window=5)

    outputs = llm.generate("Hello world", SamplingParams(max_tokens=5))

    # Should work with get_latest_attention_scores()
    scores = get_latest_attention_scores()

    if scores is None:
        print("❌ FAILED: No scores captured")
        return False

    print(f"✓ Captured attention for layer 0")
    print(f"  Shape: {scores[0].shape}")
    print(f"  Expected: (num_heads, num_tokens, window=5)")

    # Verify shape
    num_heads, num_tokens, window = scores[0].shape
    if window != 5:
        print(f"❌ FAILED: Expected window=5, got {window}")
        return False

    print("✅ PASSED: Single request works\n")
    return True


def test_sequential_requests():
    """Test sequential requests with auto-generated IDs."""
    print("\n" + "="*80)
    print("TEST 2: Sequential Requests (Auto-generated IDs)")
    print("="*80 + "\n")

    llm = LLM(model="gpt2", max_model_len=512, gpu_memory_utilization=0.3)
    enable_attention_capture(llm, capture_layers=[0], attention_window=5)

    # Generate first request
    print("Request 1: Generating...")
    outputs1 = llm.generate("First request", SamplingParams(max_tokens=3))
    scores1 = get_latest_attention_scores()

    if scores1 is None:
        print("❌ FAILED: No scores for request 1")
        return False

    print(f"✓ Request 1 captured, shape: {scores1[0].shape}")

    # Generate second request
    print("\nRequest 2: Generating...")
    outputs2 = llm.generate("Second request", SamplingParams(max_tokens=3))
    scores2 = get_latest_attention_scores()

    if scores2 is None:
        print("❌ FAILED: No scores for request 2")
        return False

    print(f"✓ Request 2 captured, shape: {scores2[0].shape}")

    # Verify they're different (different number of tokens)
    if scores1[0].shape == scores2[0].shape:
        print("⚠️ WARNING: Shapes are identical (may be expected if prompts same length)")

    # Verify data is different
    if np.array_equal(scores1[0], scores2[0]):
        print("❌ FAILED: Scores are identical (data mixing detected)")
        return False

    print("\n✓ Scores are different (requests properly isolated)")
    print("✅ PASSED: Sequential requests work\n")
    return True


def test_user_provided_ids():
    """Test user-provided request IDs."""
    print("\n" + "="*80)
    print("TEST 3: User-Provided Request IDs")
    print("="*80 + "\n")

    llm = LLM(model="gpt2", max_model_len=512, gpu_memory_utilization=0.3)
    enable_attention_capture(llm, capture_layers=[0], attention_window=5)

    # Generate with custom IDs
    print("Request 'my_request_1': Generating...")
    with set_request_context("my_request_1"):
        outputs1 = llm.generate("First prompt", SamplingParams(max_tokens=3))

    print("Request 'my_request_2': Generating...")
    with set_request_context("my_request_2"):
        outputs2 = llm.generate("Second prompt", SamplingParams(max_tokens=3))

    # Retrieve by custom IDs
    scores1 = get_attention_scores("my_request_1")
    scores2 = get_attention_scores("my_request_2")

    if scores1 is None:
        print("❌ FAILED: Cannot retrieve 'my_request_1'")
        return False

    if scores2 is None:
        print("❌ FAILED: Cannot retrieve 'my_request_2'")
        return False

    print(f"✓ Retrieved 'my_request_1', shape: {scores1[0].shape}")
    print(f"✓ Retrieved 'my_request_2', shape: {scores2[0].shape}")

    # Verify data is different
    if np.array_equal(scores1[0], scores2[0]):
        print("❌ FAILED: Scores are identical (data mixing detected)")
        return False

    print("\n✓ Scores are different (requests properly isolated)")
    print("✅ PASSED: User-provided IDs work\n")
    return True


def test_request_isolation():
    """Test that requests don't interfere with each other."""
    print("\n" + "="*80)
    print("TEST 4: Request Isolation (Verify No Data Mixing)")
    print("="*80 + "\n")

    llm = LLM(model="gpt2", max_model_len=512, gpu_memory_utilization=0.3)
    enable_attention_capture(llm, capture_layers=[0], attention_window=5)

    # Generate with different prompt lengths to ensure different K/V sizes
    print("Request A (short): Generating...")
    with set_request_context("request_a"):
        outputs_a = llm.generate("Hi", SamplingParams(max_tokens=2))

    print("Request B (long): Generating...")
    with set_request_context("request_b"):
        outputs_b = llm.generate("This is a longer prompt", SamplingParams(max_tokens=3))

    scores_a = get_attention_scores("request_a")
    scores_b = get_attention_scores("request_b")

    if scores_a is None or scores_b is None:
        print("❌ FAILED: Missing scores")
        return False

    print(f"✓ Request A shape: {scores_a[0].shape}")
    print(f"✓ Request B shape: {scores_b[0].shape}")

    # Verify different shapes (different sequence lengths)
    if scores_a[0].shape[1] == scores_b[0].shape[1]:
        print("⚠️ WARNING: Same number of tokens (may be expected)")
    else:
        print(f"✓ Different sequence lengths: {scores_a[0].shape[1]} vs {scores_b[0].shape[1]}")

    # Verify data is different
    min_tokens = min(scores_a[0].shape[1], scores_b[0].shape[1])
    if np.array_equal(scores_a[0][:, :min_tokens, :], scores_b[0][:, :min_tokens, :]):
        print("❌ FAILED: Overlapping data is identical (K/V mixing detected)")
        return False

    print("\n✓ Data is properly isolated")
    print("✅ PASSED: Request isolation works\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MULTI-REQUEST SUPPORT TEST SUITE")
    print("="*80)

    results = []

    try:
        results.append(("Single request", test_single_request()))
    except Exception as e:
        print(f"❌ TEST 1 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Single request", False))

    try:
        results.append(("Sequential requests", test_sequential_requests()))
    except Exception as e:
        print(f"❌ TEST 2 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Sequential requests", False))

    try:
        results.append(("User-provided IDs", test_user_provided_ids()))
    except Exception as e:
        print(f"❌ TEST 3 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("User-provided IDs", False))

    try:
        results.append(("Request isolation", test_request_isolation()))
    except Exception as e:
        print(f"❌ TEST 4 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Request isolation", False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
