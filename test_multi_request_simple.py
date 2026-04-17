#!/usr/bin/env python3
"""
Simple test to verify multi-request support is working.

This script demonstrates:
1. Sequential requests with auto-generated IDs
2. User-provided request IDs
3. Request isolation (no data mixing)
"""

import os
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

from vllm import LLM, SamplingParams
from vllm_attention_capture_plugin import (
    enable_attention_capture,
    get_latest_attention_scores,
    get_attention_scores,
    set_request_context,
)


def main():
    print("=" * 80)
    print("MULTI-REQUEST SUPPORT TEST")
    print("=" * 80)

    # Create LLM once
    print("\n1. Creating LLM...")
    llm = LLM(model="gpt2", max_model_len=512, gpu_memory_utilization=0.3)
    print("   ✓ LLM created")

    # Enable capture
    print("\n2. Enabling attention capture...")
    enable_attention_capture(llm, capture_layers=[0], attention_window=5)
    print("   ✓ Capture enabled for layer 0")

    # Test 1: Sequential requests with auto-generated IDs
    print("\n" + "=" * 80)
    print("TEST 1: Sequential Requests (Auto-Generated IDs)")
    print("=" * 80)

    print("\nGenerating request 1...")
    outputs1 = llm.generate("Hello world", SamplingParams(max_tokens=5))
    print(f"   Generated: '{outputs1[0].outputs[0].text}'")

    scores1 = get_latest_attention_scores()
    if scores1 is None:
        print("   ❌ FAILED: No scores captured for request 1")
        return False
    print(f"   ✓ Captured attention, shape: {scores1[0].shape}")

    print("\nGenerating request 2...")
    outputs2 = llm.generate("Goodbye world", SamplingParams(max_tokens=5))
    print(f"   Generated: '{outputs2[0].outputs[0].text}'")

    scores2 = get_latest_attention_scores()
    if scores2 is None:
        print("   ❌ FAILED: No scores captured for request 2")
        return False
    print(f"   ✓ Captured attention, shape: {scores2[0].shape}")

    # Verify they're different
    import numpy as np
    if np.array_equal(scores1[0], scores2[0]):
        print("   ❌ FAILED: Scores are identical (data mixing detected!)")
        return False
    print("   ✓ Scores are different (requests properly isolated)")

    print("\n✅ TEST 1 PASSED: Sequential requests work!")

    # Test 2: User-provided IDs
    print("\n" + "=" * 80)
    print("TEST 2: User-Provided Request IDs")
    print("=" * 80)

    print("\nGenerating with custom ID 'my_request_A'...")
    with set_request_context("my_request_A"):
        outputs_a = llm.generate("First request", SamplingParams(max_tokens=3))
    print(f"   Generated: '{outputs_a[0].outputs[0].text}'")

    print("\nGenerating with custom ID 'my_request_B'...")
    with set_request_context("my_request_B"):
        outputs_b = llm.generate("Second request", SamplingParams(max_tokens=3))
    print(f"   Generated: '{outputs_b[0].outputs[0].text}'")

    # Retrieve by custom IDs
    scores_a = get_attention_scores("my_request_A")
    scores_b = get_attention_scores("my_request_B")

    if scores_a is None:
        print("   ❌ FAILED: Cannot retrieve 'my_request_A'")
        return False
    if scores_b is None:
        print("   ❌ FAILED: Cannot retrieve 'my_request_B'")
        return False

    print(f"\n   ✓ Retrieved 'my_request_A', shape: {scores_a[0].shape}")
    print(f"   ✓ Retrieved 'my_request_B', shape: {scores_b[0].shape}")

    # Verify they're different
    if np.array_equal(scores_a[0], scores_b[0]):
        print("   ❌ FAILED: Scores are identical (data mixing detected!)")
        return False
    print("   ✓ Scores are different (requests properly isolated)")

    print("\n✅ TEST 2 PASSED: User-provided IDs work!")

    # Summary
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! 🎉")
    print("=" * 80)
    print("\nMulti-request support is working correctly:")
    print("  ✓ Sequential requests are isolated")
    print("  ✓ Auto-generated IDs work")
    print("  ✓ User-provided IDs work")
    print("  ✓ No data mixing between requests")
    print("\nYour plugin is ready to use!")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
