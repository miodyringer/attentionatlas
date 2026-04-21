"""
Simple test to verify request ID tracking works correctly.
Tests that a single generate() call produces only 1 request ID.
"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

try:
    from vllm import LLM, SamplingParams
    from vllm_attention_capture_plugin import (
        enable_attention_capture,
        get_capture_hook,
    )
except ImportError as e:
    print(f"ERROR: Missing dependencies: {e}")
    print("\nThis test requires vllm to be installed.")
    print("Please install with: pip install vllm")
    sys.exit(1)

def test_single_request():
    """Test that a single generate() call uses only 1 request ID"""
    print("\n" + "=" * 80)
    print("TEST: Single request ID for single generate() call")
    print("=" * 80)

    # Initialize model
    llm = LLM(model="gpt2", enforce_eager=True)

    # Enable attention capture
    enable_attention_capture(
        llm,
        capture_layers=[0],  # Just first layer for speed
        attention_window=None,
        auto_clear=False
    )

    # Get the hook to inspect captured data
    hook = get_capture_hook(llm)

    # Generate text
    prompt = "The quick brown fox"
    print(f"\nPrompt: {prompt}")
    print("Generating with max_tokens=50...")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
    )

    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text

    print(f"Generated: {generated_text[:50]}...")

    # Check request IDs in hook
    print(f"\n🔍 Request IDs in hook: {list(hook.captured_scores.keys())}")

    num_request_ids = len(hook.captured_scores)

    if num_request_ids == 1:
        print(f"✅ PASS: Only 1 request ID (expected)")

        # Check chunk counts
        for req_id, layers in hook.captured_scores.items():
            print(f"\nRequest ID: {req_id}")
            for layer_id, chunks in layers.items():
                print(f"  Layer {layer_id}: {len(chunks)} chunks captured")

        return True
    else:
        print(f"❌ FAIL: Found {num_request_ids} request IDs (expected 1)")
        print("This indicates the request ID tracking is not working correctly.")

        # Show details
        for req_id, layers in hook.captured_scores.items():
            print(f"\n  Request ID: {req_id}")
            for layer_id, chunks in layers.items():
                print(f"    Layer {layer_id}: {len(chunks)} chunks")

        return False


def test_concurrent_requests():
    """Test that concurrent requests get different IDs"""
    print("\n" + "=" * 80)
    print("TEST: Different request IDs for concurrent requests")
    print("=" * 80)

    # Initialize model
    llm = LLM(model="gpt2", enforce_eager=True)

    # Enable attention capture
    enable_attention_capture(
        llm,
        capture_layers=[0],
        attention_window=None,
        auto_clear=False
    )

    hook = get_capture_hook(llm)

    # Generate with multiple prompts
    prompts = ["Hello world", "The quick brown fox"]
    print(f"\nPrompts: {prompts}")
    print("Generating concurrently with max_tokens=10...")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
    )

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"Output {i}: {output.outputs[0].text[:30]}...")

    # Check request IDs
    print(f"\n🔍 Request IDs in hook: {list(hook.captured_scores.keys())}")

    num_request_ids = len(hook.captured_scores)

    if num_request_ids == len(prompts):
        print(f"✅ PASS: {num_request_ids} request IDs for {len(prompts)} prompts (expected)")
        return True
    else:
        print(f"❌ FAIL: Found {num_request_ids} request IDs (expected {len(prompts)})")
        return False


if __name__ == "__main__":
    print("Testing vLLM Attention Capture Plugin Request ID Tracking")
    print("=" * 80)

    test1_passed = test_single_request()
    print("\n")
    test2_passed = test_concurrent_requests()

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Single request test: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Concurrent requests test: {'✅ PASS' if test2_passed else '❌ FAIL'}")

    if test1_passed and test2_passed:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
