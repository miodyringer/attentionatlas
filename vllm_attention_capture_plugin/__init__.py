"""
vLLM Attention Capture Plugin

A plugin-based implementation for capturing attention scores in vLLM
without modifying core vLLM code.

This plugin:
1. Hooks into attention computation to extract windowed attention scores
2. Stores scores in-memory during generation
3. Returns scores in RequestOutput for analysis

Usage:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        # Plugin will be auto-loaded via entry points
    )

    # Enable capture via SamplingParams (future)
    outputs = llm.generate(
        "Hello world",
        sampling_params=SamplingParams(
            capture_attention=True,
            capture_layers=[0, 1, 2],
            attention_window=5
        )
    )

    # Access captured scores
    scores = outputs[0].outputs[0].attention_scores
"""

__version__ = "0.1.0"

from vllm_attention_capture_plugin.api import (
    clear_all_captures,
    disable_attention_capture,
    enable_attention_capture,
    get_attention_scores,
    get_capture_config,
    get_capture_hook,
    get_latest_attention_scores,
    set_request_context,  # New API for multi-request support
)
from vllm_attention_capture_plugin.hooks.attention_hook import AttentionCaptureHook

__all__ = [
    "AttentionCaptureHook",
    "enable_attention_capture",
    "disable_attention_capture",
    "get_attention_scores",
    "get_latest_attention_scores",
    "get_capture_hook",
    "get_capture_config",
    "clear_all_captures",
    "set_request_context",  # New API for multi-request support
]
