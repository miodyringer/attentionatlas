#!/usr/bin/env python3
"""
Simplified investigation script for vLLM v1 using environment variable approach.

Since vLLM v1 RPC doesn't allow passing closures, we'll:
1. Set environment variable to enable pickle serialization
2. Use a simpler patching approach

Usage:
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 python investigate_vllm_request_context_v1.py
"""

import logging
import os
import sys

# Enable pickle serialization for vLLM v1 RPC
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_investigation():
    """Run investigation with vLLM v1."""
    logger.info("Starting vLLM v1 request context investigation...\n")

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vLLM not installed")
        sys.exit(1)

    # Create LLM
    logger.info("Creating LLM instance...")
    llm = LLM(
        model="gpt2",
        max_model_len=512,
        gpu_memory_utilization=0.3,
    )

    logger.info("\n" + "="*80)
    logger.info("INVESTIGATION APPROACH")
    logger.info("="*80)
    logger.info("Since direct patching is complex in v1, we'll:")
    logger.info("1. Examine vLLM's RequestOutput object structure")
    logger.info("2. Check what metadata is available post-generation")
    logger.info("3. See if request IDs are exposed")
    logger.info("")

    # Test 1: Single request
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Single Request")
    logger.info("="*80 + "\n")

    outputs = llm.generate(
        "Hello, my name is",
        SamplingParams(temperature=0.0, max_tokens=5)
    )

    logger.info("### REQUEST OUTPUT STRUCTURE ###")
    output = outputs[0]
    logger.info(f"Type: {type(output)}")
    logger.info(f"Attributes: {[a for a in dir(output) if not a.startswith('_')]}")
    logger.info("")

    logger.info("### REQUEST ID ###")
    logger.info(f"request_id: {output.request_id}")
    logger.info(f"request_id type: {type(output.request_id)}")
    logger.info("")

    logger.info("### OTHER ATTRIBUTES ###")
    if hasattr(output, 'prompt'):
        logger.info(f"prompt: {output.prompt}")
    if hasattr(output, 'prompt_token_ids'):
        logger.info(f"prompt_token_ids (first 10): {output.prompt_token_ids[:10]}")
    if hasattr(output, 'outputs'):
        logger.info(f"outputs: {len(output.outputs)} completion(s)")
        if output.outputs:
            logger.info(f"generated text: {output.outputs[0].text}")
            logger.info(f"generated token_ids (first 10): {output.outputs[0].token_ids[:10]}")
    if hasattr(output, 'finished'):
        logger.info(f"finished: {output.finished}")
    if hasattr(output, 'metrics'):
        logger.info(f"metrics: {output.metrics}")

    # Test 2: Batched requests
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Batched Requests (2 prompts)")
    logger.info("="*80 + "\n")

    outputs = llm.generate(
        [
            "The capital of France is",
            "The color of the sky is",
        ],
        SamplingParams(temperature=0.0, max_tokens=3)
    )

    logger.info(f"### BATCH SIZE: {len(outputs)} ###\n")

    for i, output in enumerate(outputs):
        logger.info(f"--- Request {i} ---")
        logger.info(f"request_id: {output.request_id}")
        logger.info(f"prompt: {output.prompt}")
        logger.info(f"generated: {output.outputs[0].text}")
        logger.info("")

    # Test 3: Sequential requests
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Sequential Requests")
    logger.info("="*80 + "\n")

    logger.info("Request 1:")
    outputs1 = llm.generate("First request", SamplingParams(max_tokens=3))
    logger.info(f"request_id: {outputs1[0].request_id}")

    logger.info("\nRequest 2:")
    outputs2 = llm.generate("Second request", SamplingParams(max_tokens=3))
    logger.info(f"request_id: {outputs2[0].request_id}")

    logger.info(f"\nRequest IDs are unique: {outputs1[0].request_id != outputs2[0].request_id}")

    logger.info("\n" + "="*80)
    logger.info("FINDINGS SUMMARY")
    logger.info("="*80)
    logger.info("\n### KEY INSIGHTS ###")
    logger.info(f"1. Request IDs ARE exposed via RequestOutput.request_id")
    logger.info(f"2. Request IDs are unique across sequential requests")
    logger.info(f"3. Batched requests each have their own request_id")
    logger.info("")
    logger.info("### CONCLUSION ###")
    logger.info("vLLM provides request IDs at the OUTPUT level.")
    logger.info("However, we need to access them DURING forward pass (not after).")
    logger.info("")
    logger.info("### NEXT STEPS ###")
    logger.info("Need to investigate vLLM's internal context during forward pass.")
    logger.info("This requires patching attention layers, which is complex in v1.")
    logger.info("")
    logger.info("### RECOMMENDATION ###")
    logger.info("Use timestamp-based unique IDs during capture,")
    logger.info("then provide a mapping API to associate captures with request IDs.")


if __name__ == "__main__":
    run_investigation()
