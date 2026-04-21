"""
Patch Attention layer to capture attention weights.

This module patches vLLM's Attention layer (not the projection layers) to capture
attention weights. At this level, Q/K/V have all model-specific transformations
applied (RoPE, QK normalization, etc), so we capture the ACTUAL attention values.
"""
import contextvars
import logging
import time
from typing import Any
import numpy as np

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context

from vllm_attention_capture_plugin.hooks.attention_hook import AttentionCaptureHook
from vllm_attention_capture_plugin.wrappers import compute_attention_with_capture

logger = logging.getLogger(__name__)

# Context variable for user-provided request IDs
_user_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    'request_id', default=None
)

# Context variable for batch_idx -> vLLM request_id mapping
# This is populated by patch_model_runner() before each forward pass
_batch_req_id_mapping: contextvars.ContextVar[dict[int, str] | None] = contextvars.ContextVar(
    'batch_req_id_mapping', default=None
)

# Session-level fallback request ID (one per generate() call)
# Used when we can't get vLLM's native request ID
_session_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    'session_request_id', default=None
)


def get_or_generate_request_id(batch_idx: int = 0) -> str:
    """Get request ID using vLLM's native request tracking or fallbacks.

    Priority order:
    1. User-provided ID from context variable (set_request_context)
    2. vLLM's native request_id from batch mapping (populated by model runner patch)
    3. Session-level fallback ID (one per generate() call)
    4. Generate new timestamp-based ID as last resort

    Args:
        batch_idx: Batch index for multi-request batches (default: 0 for single request)

    Returns:
        Request ID string that will be consistent across all layers and decode steps
    """
    # Priority 1: User-provided ID (explicit override)
    user_id = _user_request_id.get()
    if user_id is not None:
        return str(user_id)

    # Priority 2: vLLM's native request ID from batch mapping
    batch_mapping = _batch_req_id_mapping.get()
    if batch_mapping is not None and batch_idx in batch_mapping:
        return batch_mapping[batch_idx]

    # Priority 3: Session-level fallback (one per generate() call)
    session_id = _session_request_id.get()
    if session_id is not None:
        return session_id

    # Priority 4: Generate new timestamp-based ID and store as session ID
    # This ensures all forward passes in this session use the same ID
    new_session_id = f"req_{int(time.time() * 1000000)}"
    _session_request_id.set(new_session_id)

    logger.warning(
        f"Could not get vLLM native request ID (batch_idx={batch_idx}), "
        f"using session fallback: {new_session_id}"
    )

    return new_session_id


def extract_request_ranges(
    attention_layer: Any,
    batch_mapping: dict[int, str] | None
) -> list[tuple[int, int, str]] | None:
    """Extract per-request token ranges from attention metadata.

    This function accesses vLLM's AttentionMetadata to determine where each
    request's tokens are located in the concatenated batch tensor.

    Args:
        attention_layer: The vLLM Attention layer instance with attn_metadata attribute
        batch_mapping: Dict mapping batch_idx → request_id (from _batch_req_id_mapping)

    Returns:
        List of (start_pos, end_pos, request_id) tuples for each request in batch,
        or None if batch splitting is not possible/needed.

        Example: [(0, 6, 'req-123'), (6, 9, 'req-456')]
        Means: Request 'req-123' uses tokens [0:6], Request 'req-456' uses tokens [6:9]

    Returns None when:
        - Single request batch (len(batch_mapping) <= 1)
        - AttentionMetadata unavailable
        - query_start_loc attribute missing
        - query_start_loc is None or too short
    """
    # Skip if single request or no batch mapping
    if not batch_mapping or len(batch_mapping) <= 1:
        return None

    # Try to access attention metadata
    attn_metadata = getattr(attention_layer, 'attn_metadata', None)
    if attn_metadata is None:
        return None

    # Check for query_start_loc attribute (contains cumulative token positions)
    if not hasattr(attn_metadata, 'query_start_loc'):
        return None

    query_start_loc = attn_metadata.query_start_loc
    if query_start_loc is None or len(query_start_loc) < 2:
        # Need at least 2 elements to define a range
        return None

    # Convert to CPU numpy for indexing
    query_start_loc = query_start_loc.cpu().numpy()
    num_reqs = len(query_start_loc) - 1

    # Extract ranges for each request
    request_ranges = []
    for req_idx in range(num_reqs):
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        req_id = batch_mapping.get(req_idx)

        if req_id:
            request_ranges.append((start, end, req_id))

    return request_ranges if request_ranges else None


def patch_attention_layer(
    attention_layer: Any,
    layer_idx: int,
    capture_hook: AttentionCaptureHook,
) -> None:
    """Patch an Attention layer to capture attention weights.

    This hooks at the Attention.forward() level, where Q/K/V have all
    model-specific transformations applied (RoPE, QK norm, etc).

    Args:
        attention_layer: The vLLM Attention layer instance (attn_module.attn)
        layer_idx: Layer index (0-indexed)
        capture_hook: The capture hook to use
    """
    # Store original forward method
    original_forward = attention_layer.forward

    # Get attention parameters
    num_heads = getattr(attention_layer, "num_heads", None)
    num_kv_heads = getattr(attention_layer, "num_kv_heads", None)
    head_size = getattr(attention_layer, "head_size", None)
    scale = getattr(attention_layer, "scale", None)

    if num_heads is None or head_size is None:
        logger.warning(
            "Cannot patch layer %d: missing num_heads or head_size", layer_idx
        )
        return

    if num_kv_heads is None:
        num_kv_heads = num_heads

    if scale is None:
        scale = head_size**-0.5

    logger.info(
        "Patching Attention layer %d: num_heads=%d, num_kv_heads=%d, head_size=%d, scale=%.4f",
        layer_idx,
        num_heads,
        num_kv_heads,
        head_size,
        scale,
    )

    def forward_with_capture(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """Wrapped forward method that captures attention weights.

        Signature matches vLLM's actual Attention.forward:
          forward(query, key, value, output_shape=None)

        Args:
            query: [num_tokens, num_heads * head_size] - Q after RoPE
            key: [num_tokens, num_kv_heads * head_size] - K after RoPE
            value: [num_tokens, num_kv_heads * head_size] - V
            output_shape: Optional output shape specification

        Note: kv_cache and attn_metadata are NOT parameters - they're managed
        internally by the Attention class. We can access them via:
        - attention_layer.kv_cache (instance attribute)
        - get_attention_context() (context variable)

        Returns:
            Attention output tensor
        """
        num_tokens = query.shape[0]
        is_decode = (num_tokens == 1)

        # Check if we should capture this layer
        if not capture_hook.should_capture(layer_idx):
            return original_forward(query, key, value, output_shape)

        # Get fallback request ID for single-request scenarios
        # For multi-request batches, IDs are determined per-request in split logic
        request_id = get_or_generate_request_id()

        # IMPORTANT: Store raw K/V BEFORE vLLM reshapes them into cache
        # We use per-request accumulators to isolate different requests
        if key is not None and value is not None:
            # Initialize per-request accumulator storage
            if not hasattr(attention_layer, '_raw_kv_accumulators'):
                attention_layer._raw_kv_accumulators = {}

            batch_mapping = _batch_req_id_mapping.get()
            request_ranges = extract_request_ranges(attention_layer, batch_mapping)

            if request_ranges and not is_decode:
                # Multi-request prefill: split K/V by request boundaries
                print(f"🔹 Layer {layer_idx}: Prefill with {len(request_ranges)} requests")
                for start, end, req_id in request_ranges:
                    num_tokens_req = end - start
                    k_req = key[start:end].view(num_tokens_req, num_kv_heads, head_size)
                    v_req = value[start:end].view(num_tokens_req, num_kv_heads, head_size)

                    # Reset this request's accumulator (prefill)
                    attention_layer._raw_kv_accumulators[req_id] = {
                        'keys': [k_req],
                        'values': [v_req]
                    }
                    print(f"  Request {req_id}: stored {num_tokens_req} tokens in accumulator")
            elif request_ranges and is_decode:
                # Multi-request decode: split K/V by request boundaries
                print(f"🔹 Layer {layer_idx}: Decode with {len(request_ranges)} requests")
                for start, end, req_id in request_ranges:
                    num_tokens_req = end - start
                    k_req = key[start:end].view(num_tokens_req, num_kv_heads, head_size)
                    v_req = value[start:end].view(num_tokens_req, num_kv_heads, head_size)

                    # Append to this request's accumulator
                    if req_id not in attention_layer._raw_kv_accumulators:
                        print(f"  ⚠️  Layer {layer_idx}: No accumulator for request {req_id} during decode K/V append")
                        attention_layer._raw_kv_accumulators[req_id] = {
                            'keys': [],
                            'values': []
                        }

                    attention_layer._raw_kv_accumulators[req_id]['keys'].append(k_req)
                    attention_layer._raw_kv_accumulators[req_id]['values'].append(v_req)
                    print(f"  Request {req_id}: appended {num_tokens_req} tokens to accumulator")
            else:
                # Single request or no metadata: use original logic
                request_id = get_or_generate_request_id()
                print(f"🔹 Layer {layer_idx}: Single request mode, request_id={request_id}, is_decode={is_decode}")

                # Get or create accumulator for this specific request
                if request_id not in attention_layer._raw_kv_accumulators:
                    attention_layer._raw_kv_accumulators[request_id] = {
                        'keys': [],
                        'values': []
                    }

                accumulator = attention_layer._raw_kv_accumulators[request_id]

                # Reshape: [num_tokens, num_kv_heads * head_size] -> [num_tokens, num_kv_heads, head_size]
                k_reshaped = key.view(num_tokens, num_kv_heads, head_size)
                v_reshaped = value.view(num_tokens, num_kv_heads, head_size)

                if is_decode:
                    # Decode: append new token to this request's accumulator
                    accumulator['keys'].append(k_reshaped)
                    accumulator['values'].append(v_reshaped)
                else:
                    # Single-request prefill: reset accumulator
                    attention_layer._raw_kv_accumulators[request_id] = {
                        'keys': [k_reshaped],
                        'values': [v_reshaped]
                    }

        # Call vLLM's original forward first (don't modify behavior)
        result = original_forward(query, key, value, output_shape)

        # === ATTENTION COMPUTATION FOR CAPTURE ===
        try:
            if not is_decode:
                # ============================================================
                # PREFILL PHASE: All tokens in prompt
                # ============================================================
                # Reshape Q, K, V: [num_tokens, num_heads * head_size] -> [num_tokens, num_heads, head_size]
                q = query.view(num_tokens, num_heads, head_size)
                k = key.view(num_tokens, num_kv_heads, head_size)
                v = value.view(num_tokens, num_kv_heads, head_size)

                # Handle GQA: repeat KV heads to match Q heads
                if num_kv_heads < num_heads:
                    num_queries_per_kv = num_heads // num_kv_heads
                    k = k.repeat_interleave(num_queries_per_kv, dim=1)
                    v = v.repeat_interleave(num_queries_per_kv, dim=1)

                # Transpose for attention computation
                k_t = k.transpose(0, 1).transpose(1, 2)  # [num_heads, head_size, num_tokens]
                q_transposed = q.transpose(0, 1)  # [num_heads, num_tokens, head_size]

                # Compute attention scores: Q @ K^T * scale
                scores = torch.bmm(q_transposed, k_t) * scale

                # Apply causal mask
                causal_mask = torch.triu(torch.ones(num_tokens, num_tokens, device=scores.device), diagonal=1).bool()
                scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

                # Apply softmax to get attention weights
                attn_weights = torch.softmax(scores, dim=-1)

            else:
                # ============================================================
                # DECODE PHASE: One new token, attend to all previous + self
                # ============================================================
                # Use accumulated raw K/V for this specific request
                if not hasattr(attention_layer, '_raw_kv_accumulators'):
                    logger.warning(f"Layer {layer_idx}: No K/V accumulators, skipping decode capture")
                    return result

                if request_id not in attention_layer._raw_kv_accumulators:
                    logger.warning(
                        f"Layer {layer_idx}: No accumulator for request {request_id}, "
                        "skipping decode capture"
                    )
                    return result

                # Get this request's accumulator
                accumulator = attention_layer._raw_kv_accumulators[request_id]

                # Concatenate all accumulated keys and values for this request
                # This includes all prefill tokens + all previously generated decode tokens
                all_keys = torch.cat(accumulator['keys'], dim=0)
                all_values = torch.cat(accumulator['values'], dim=0)
                context_len = all_keys.shape[0]

                # Reshape query: [1, num_heads * head_size] -> [1, num_heads, head_size]
                q = query.view(1, num_heads, head_size)

                # Handle GQA: expand KV heads to match Q heads
                if num_heads != num_kv_heads:
                    num_queries_per_kv = num_heads // num_kv_heads
                    # Expand: [context_len, num_kv_heads, head_size] -> [context_len, num_heads, head_size]
                    all_keys = all_keys.unsqueeze(2).expand(
                        context_len, num_kv_heads, num_queries_per_kv, head_size
                    ).reshape(context_len, num_heads, head_size)
                    all_values = all_values.unsqueeze(2).expand(
                        context_len, num_kv_heads, num_queries_per_kv, head_size
                    ).reshape(context_len, num_heads, head_size)

                # Transpose for attention computation
                # Q: [1, num_heads, head_size] -> [num_heads, 1, head_size]
                # K: [context_len, num_heads, head_size] -> [num_heads, context_len, head_size]
                q_t = q.transpose(0, 1)  # [num_heads, 1, head_size]
                k_t = all_keys.transpose(0, 1)  # [num_heads, context_len, head_size]

                # Compute attention scores: Q @ K^T
                # [num_heads, 1, head_size] @ [num_heads, head_size, context_len]
                # -> [num_heads, 1, context_len]
                attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1))

                # Scale
                attn_scores = attn_scores * scale

                # Softmax to get attention weights
                # Shape: [num_heads, 1, context_len] where context_len includes ALL tokens (prefill + previous decodes + current)
                attn_weights = torch.softmax(attn_scores, dim=-1)

            # Store attention weights - split by request if batch has multiple requests
            batch_mapping = _batch_req_id_mapping.get()
            request_ranges = extract_request_ranges(attention_layer, batch_mapping)

            print(f"🔸 Layer {layer_idx}: is_decode={is_decode}, batch_mapping={batch_mapping}, request_ranges={request_ranges}")

            if request_ranges:
                if not is_decode:
                    # Multi-request prefill: split attention by request boundaries
                    # Each request attends only to its own tokens (diagonal blocks)
                    print(f"🔸 Layer {layer_idx}: Multi-request prefill with {len(request_ranges)} requests")
                    for start, end, req_id in request_ranges:
                        # Extract this request's attention to its own context
                        # Shape: [num_heads, num_tokens_this_req, num_tokens_this_req]
                        req_attn = attn_weights[:, start:end, start:end]

                        print(f"  Storing prefill attention for request {req_id}: shape {req_attn.shape}")
                        capture_hook.capture_attention_weights(
                            layer_idx=layer_idx,
                            attn_weights=req_attn,
                            request_id=req_id,
                        )
                else:
                    # Multi-request decode: process each request's token separately
                    print(f"🔸 Layer {layer_idx}: Multi-request decode with {len(request_ranges)} requests")
                    # Check for accumulators
                    if not hasattr(attention_layer, '_raw_kv_accumulators'):
                        print(f"  ⚠️  Layer {layer_idx}: No K/V accumulators, skipping multi-request decode capture")
                        return result

                    print(f"  Accumulators present: {list(attention_layer._raw_kv_accumulators.keys())}")

                    for start, end, req_id in request_ranges:
                        if req_id not in attention_layer._raw_kv_accumulators:
                            print(f"  ⚠️  Layer {layer_idx}: No accumulator for request {req_id}, skipping decode capture for this request")
                            continue

                        print(f"  Processing decode for request {req_id}, tokens [{start}:{end}]")

                        # This request's query (should be 1 token during decode)
                        q_req = query[start:end]  # [1, num_heads * head_size]
                        num_tokens_req = end - start

                        # Validate: should be 1 token per request during decode
                        if num_tokens_req != 1:
                            print(f"  ⚠️  Layer {layer_idx}: Decode phase but request {req_id} has {num_tokens_req} tokens (expected 1)")

                        # Get this request's accumulated K/V
                        accumulator = attention_layer._raw_kv_accumulators[req_id]
                        all_keys = torch.cat(accumulator['keys'], dim=0)  # [context_len, num_kv_heads, head_size]
                        all_values = torch.cat(accumulator['values'], dim=0)
                        context_len = all_keys.shape[0]

                        print(f"    Accumulator has {len(accumulator['keys'])} K/V chunks, context_len={context_len}")

                        # Reshape query
                        q = q_req.view(num_tokens_req, num_heads, head_size)  # [1, num_heads, head_size]

                        # Handle GQA: expand KV heads
                        if num_heads != num_kv_heads:
                            num_queries_per_kv = num_heads // num_kv_heads
                            all_keys = all_keys.unsqueeze(2).expand(
                                context_len, num_kv_heads, num_queries_per_kv, head_size
                            ).reshape(context_len, num_heads, head_size)
                            all_values = all_values.unsqueeze(2).expand(
                                context_len, num_kv_heads, num_queries_per_kv, head_size
                            ).reshape(context_len, num_heads, head_size)

                        # Transpose for attention
                        q_t = q.transpose(0, 1)  # [num_heads, 1, head_size]
                        k_t = all_keys.transpose(0, 1)  # [num_heads, context_len, head_size]

                        # Compute attention scores
                        attn_scores_req = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # [num_heads, 1, context_len]
                        attn_weights_req = torch.softmax(attn_scores_req, dim=-1)

                        print(f"    Storing decode attention for request {req_id}: shape {attn_weights_req.shape}")
                        # Store attention for this specific request
                        capture_hook.capture_attention_weights(
                            layer_idx=layer_idx,
                            attn_weights=attn_weights_req,
                            request_id=req_id,
                        )
            else:
                # Single request, no metadata, or decode phase without splitting
                # Use the already-computed attn_weights
                capture_hook.capture_attention_weights(
                    layer_idx=layer_idx,
                    attn_weights=attn_weights,
                    request_id=request_id,
                )

        except Exception as e:
            import traceback
            logger.error(
                f"Layer {layer_idx}: Failed to compute attention weights: {e}\n{traceback.format_exc()}"
            )

        return result

    # Replace forward method
    attention_layer.forward = forward_with_capture

    # Store reference for potential restoration
    attention_layer._original_forward = original_forward
    attention_layer._capture_hook = capture_hook
    attention_layer._layer_idx = layer_idx

    logger.info("✓ Successfully patched Attention layer %d", layer_idx)


def patch_model_for_attention_capture(
    model: Any,
    capture_hook: AttentionCaptureHook,
) -> None:
    """Patch a model's Attention layers to enable capture.

    Args:
        model: The vLLM model instance
        capture_hook: The capture hook to use
    """
    logger.info("Patching model for attention capture...")

    # Find all layers to patch
    # Different architectures use different attribute names:
    # - Qwen, Llama, Mistral: model.layers
    # - GPT-2: transformer.h
    # - GPT-J: transformer.h
    # - BLOOM: transformer.h
    layers = None

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Qwen/Llama/Mistral architecture
        layers = model.model.layers
        logger.info("Found Qwen/Llama architecture: model.layers")
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2/GPT-J/BLOOM architecture
        layers = model.transformer.h
        logger.info("Found GPT-2/GPT-J architecture: transformer.h")
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        # GPT-NeoX architecture
        layers = model.gpt_neox.layers
        logger.info("Found GPT-NeoX architecture: gpt_neox.layers")
    else:
        logger.error("Model structure not recognized, cannot patch")
        logger.info("Model type: %s, attributes: %s", type(model), dir(model))
        return

    logger.info("Found %d layers in model", len(layers))

    # Patch each layer that should be captured
    num_patched = 0
    for layer_idx in capture_hook.capture_layers:
        if layer_idx >= len(layers):
            logger.warning("Layer %d out of range, skipping", layer_idx)
            continue

        layer = layers[layer_idx]

        # Find the attention module
        # Different model architectures use different names:
        # - Qwen, Llama: self_attn
        # - GPT-2: attn
        attn_module = None
        if hasattr(layer, "self_attn"):
            attn_module = layer.self_attn
        elif hasattr(layer, "attn"):
            attn_module = layer.attn
        else:
            logger.warning("Layer %d has no attention module, skipping", layer_idx)
            continue

        # The actual Attention layer is usually inside the attention module
        # For Qwen: attn_module.attn
        # For GPT-2: attn_module.attn
        if hasattr(attn_module, "attn"):
            attention_layer = attn_module.attn
            logger.info(
                "Found Attention layer at layers[%d].%s.attn",
                layer_idx,
                "self_attn" if hasattr(layer, "self_attn") else "attn",
            )
        else:
            logger.warning(
                "Attention module at layer %d has no .attn attribute, skipping",
                layer_idx,
            )
            continue

        # Patch the Attention layer
        try:
            patch_attention_layer(attention_layer, layer_idx, capture_hook)
            num_patched += 1
        except Exception as e:
            logger.error("Failed to patch layer %d: %s", layer_idx, e)
            import traceback

            logger.debug("Traceback: %s", traceback.format_exc())

    logger.info(
        "✓ Successfully patched %d/%d layers",
        num_patched,
        len(capture_hook.capture_layers),
    )


def patch_model_runner(llm: Any) -> None:
    """Patch the model runner to populate request ID mapping before forward passes.

    This extracts vLLM's native request IDs from InputBatch and stores them in a
    context variable so attention hooks can access them.

    Args:
        llm: The vLLM LLM instance
    """
    logger.info("Patching model runner for request ID tracking...")

    try:
        # Try v0 path first (direct access to model_executor)
        if hasattr(llm.llm_engine, "model_executor"):
            model_executor = llm.llm_engine.model_executor
            if hasattr(model_executor, "driver_worker"):
                model_runner = model_executor.driver_worker.model_runner
                _patch_model_runner_execute(model_runner)
                logger.info("✓ Successfully patched v0 model runner")
                return

        # v1 architecture - use RPC
        if hasattr(llm.llm_engine, "engine_core"):
            logger.info("Detected vLLM v1 - attempting RPC-based model runner patching")

            def _patch_model_runner_on_engine(worker=None):
                """Execute in EngineCore process to patch model runner"""
                try:
                    import logging
                    logger = logging.getLogger(__name__)

                    # Find model runner in engine process
                    model_runner = None

                    # Try to get from worker
                    if worker is not None and hasattr(worker, 'model_runner'):
                        model_runner = worker.model_runner
                    else:
                        # Fallback: search via gc
                        try:
                            from vllm.v1.worker.gpu_model_runner import GPUModelRunner
                            import gc

                            for obj in gc.get_objects():
                                if isinstance(obj, GPUModelRunner):
                                    model_runner = obj
                                    break
                        except Exception:
                            pass

                    if model_runner is None:
                        return {"success": False, "error": "Could not find model runner"}

                    # Import and patch
                    from vllm_attention_capture_plugin.wrappers.attention_layer_patcher import _patch_model_runner_execute
                    _patch_model_runner_execute(model_runner)

                    return {"success": True}

                except Exception as e:
                    import traceback
                    return {
                        "success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }

            result = llm.llm_engine.collective_rpc(_patch_model_runner_on_engine)

            if isinstance(result, list):
                result = result[0]

            if not result.get("success"):
                error = result.get("error", "Unknown error")
                tb = result.get("traceback", "")
                raise RuntimeError(f"{error}\n{tb}")

            logger.info("✓ Successfully patched v1 model runner via RPC")
            return

        logger.warning("Could not patch model runner - unknown vLLM architecture")

    except Exception as e:
        logger.error(f"Failed to patch model runner: {e}")
        logger.warning("Request ID tracking will fall back to session-level IDs")


def _patch_model_runner_execute(model_runner: Any) -> None:
    """Patch a model runner's execute_model to populate request ID mapping.

    Args:
        model_runner: The vLLM model runner instance (GPU or CPU)
    """
    if not hasattr(model_runner, 'execute_model'):
        logger.warning("Model runner has no execute_model method, skipping patch")
        return

    original_execute = model_runner.execute_model

    def execute_model_with_req_id_mapping(*args, **kwargs):
        """Wrapped execute_model that extracts and stores request ID mapping"""

        # Extract request IDs from SchedulerOutput
        mapping = None
        debug_phase = None

        if args:
            arg0 = args[0]

            # Try 1: Direct req_ids attribute (v1 InputBatch - old API)
            if hasattr(arg0, 'req_ids'):
                mapping = {idx: req_id for idx, req_id in enumerate(arg0.req_ids)}
                debug_phase = "InputBatch"

            # Try 2: SchedulerOutput (v1 new API)
            elif hasattr(arg0, 'scheduled_new_reqs') or hasattr(arg0, 'scheduled_cached_reqs'):
                # Collect all request IDs from scheduled requests
                req_ids = []

                # New requests (list) - these appear during prefill
                if hasattr(arg0, 'scheduled_new_reqs') and arg0.scheduled_new_reqs:
                    debug_phase = "prefill"
                    if isinstance(arg0.scheduled_new_reqs, list):
                        for req in arg0.scheduled_new_reqs:
                            if hasattr(req, 'req_id'):
                                req_ids.append(req.req_id)
                    elif hasattr(arg0.scheduled_new_reqs, 'req_id'):
                        req_ids.append(arg0.scheduled_new_reqs, 'req_id')

                # Cached requests (dict/list/object) - these appear during decode
                if hasattr(arg0, 'scheduled_cached_reqs') and arg0.scheduled_cached_reqs is not None:
                    if not debug_phase:
                        debug_phase = "decode"

                    # Check if it's a dict (req_id -> data mapping)
                    if isinstance(arg0.scheduled_cached_reqs, dict):
                        req_ids.extend(list(arg0.scheduled_cached_reqs.keys()))
                    # Check if it's a list
                    elif isinstance(arg0.scheduled_cached_reqs, list):
                        for req in arg0.scheduled_cached_reqs:
                            if hasattr(req, 'req_id'):
                                req_ids.append(req.req_id)
                            elif hasattr(req, 'req_ids'):  # CachedRequestData
                                req_ids.extend(req.req_ids)
                    # Single object
                    elif hasattr(arg0.scheduled_cached_reqs, 'req_id'):
                        req_ids.append(arg0.scheduled_cached_reqs.req_id)
                    elif hasattr(arg0.scheduled_cached_reqs, 'req_ids'):  # CachedRequestData
                        req_ids.extend(arg0.scheduled_cached_reqs.req_ids)

                if req_ids:
                    mapping = {idx: req_id for idx, req_id in enumerate(req_ids)}

        if mapping and debug_phase == "prefill":
            # Only log once per request during prefill
            logger.info(f"✓ Mapped request IDs ({debug_phase}): {list(mapping.values())}")

        if mapping:
            _batch_req_id_mapping.set(mapping)
        else:
            # Clear mapping if we can't extract it
            _batch_req_id_mapping.set(None)

        # Call original execute_model
        return original_execute(*args, **kwargs)

    # Replace the method
    model_runner.execute_model = execute_model_with_req_id_mapping
    logger.info("✓ Patched model_runner.execute_model for request ID extraction")


__all__ = [
    "patch_attention_layer",
    "patch_model_for_attention_capture",
    "patch_model_runner",
    "_user_request_id",
    "_batch_req_id_mapping",
    "_session_request_id",
]
