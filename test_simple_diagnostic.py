#!/usr/bin/env python3
"""Simple diagnostic test to check if attention capture is working at all."""

import os
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

from vllm import LLM, SamplingParams
from vllm_attention_capture_plugin import (
    enable_attention_capture,
    get_latest_attention_scores,
    get_capture_hook,
)

print("Creating LLM...")
llm = LLM(model="gpt2", max_model_len=512, gpu_memory_utilization=0.3)

print("\nEnabling attention capture...")
try:
    enable_attention_capture(llm, capture_layers=[0], attention_window=5)
    print("✓ Patching succeeded")
except Exception as e:
    print(f"❌ Patching failed: {e}")
    exit(1)

print("\nChecking capture hook...")
hook = get_capture_hook(llm)
if hook:
    print(f"✓ Capture hook exists")
    print(f"  Capture layers: {hook.capture_layers}")
    print(f"  Attention window: {hook.attention_window}")
    print(f"  Enabled: {hook.enabled}")
else:
    print("❌ No capture hook found")
    exit(1)

print("\nGenerating text...")
outputs = llm.generate("Hello world", SamplingParams(max_tokens=5))
print(f"✓ Generated: {outputs[0].outputs[0].text}")

print("\nChecking captured scores...")
scores = get_latest_attention_scores()

if scores is None:
    print("❌ No scores captured")
    print(f"\nHook's captured_scores dict: {hook.captured_scores}")
    print(f"Keys in dict: {list(hook.captured_scores.keys())}")
    exit(1)

print(f"✓ Scores captured!")
print(f"  Layer 0 shape: {scores[0].shape}")
print(f"  Expected: (num_heads, num_tokens, window=5)")

print("\n🎉 SUCCESS: Attention capture is working!")
