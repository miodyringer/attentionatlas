"""
Attention capture API for end users.

This module provides the main API for enabling attention capture in vLLM.
"""

import logging
from typing import Any

from vllm_attention_capture_plugin.hooks.attention_hook import AttentionCaptureHook
from vllm_attention_capture_plugin.wrappers.attention_layer_patcher import (
    patch_model_for_attention_capture,
)

logger = logging.getLogger(__name__)


# Global registry for capture hooks (one per LLM instance)
_CAPTURE_HOOKS: dict[int, AttentionCaptureHook] = {}


def enable_attention_capture(
    llm: Any,  # vllm.LLM
    capture_layers: list[int] | None = None,
    attention_window: int | None = None,
    auto_clear: bool = True,
) -> None:
    """Enable attention capture for a vLLM LLM instance.

    This function patches the model's attention layers to capture attention weights
    during generation. Captured attention scores will be available in RequestOutput.

    Args:
        llm: A vLLM LLM instance
        capture_layers: List of layer indices to capture (e.g., [0, 1, 2]).
                       If None, captures first 3 layers.
        attention_window: Number of recent tokens to capture attention for.
                         If None, captures full attention matrix (higher memory).
                         If set (e.g., 5), only captures last N positions (memory efficient).
        auto_clear: If True, automatically clear captured scores after retrieval.
                   If False, scores remain in memory until explicitly cleared.

    Example:
        ```python
        from vllm import LLM, SamplingParams
        from vllm_attention_capture_plugin import enable_attention_capture

        # Create LLM
        llm = LLM(model="gpt2")

        # Enable capture with full attention (no windowing)
        enable_attention_capture(
            llm,
            capture_layers=[0, 1, 2],
            attention_window=None  # Capture full attention
        )

        # Or with windowing for memory efficiency
        enable_attention_capture(
            llm,
            capture_layers=[0, 1, 2],
            attention_window=5  # Only last 5 tokens
        )

        # Generate text
        outputs = llm.generate("Hello world", SamplingParams(max_tokens=10))

        # Access attention scores
        attention = get_attention_scores(outputs[0].request_id)
        ```

    Memory usage (for 1000 tokens, 3 layers, 32 heads):
        - window=None (full): ~384 MB per sequence
        - window=10: ~4 MB per sequence
        - window=5: ~2 MB per sequence

    Performance impact:
        - Capture layers: +20-30% slower per layer
        - Non-capture layers: No overhead
        - Overall: <5-10% slowdown for typical config
    """
    if capture_layers is None:
        capture_layers = [0, 1, 2]  # Default: first 3 layers

    logger.info(
        "Enabling attention capture: layers=%s, window=%s, auto_clear=%s",
        capture_layers,
        attention_window if attention_window is not None else "full",
        auto_clear,
    )

    # Create capture hook
    hook = AttentionCaptureHook(attention_window, capture_layers)
    hook.auto_clear = auto_clear  # Store config in hook

    # Store in global registry (keyed by LLM instance id)
    llm_id = id(llm)
    _CAPTURE_HOOKS[llm_id] = hook

    # Patch the model's attention layers
    try:
        # Access model - handle both v0 and v1 engine structures
        model = None

        # Try v0 path first (direct access)
        try:
            if hasattr(llm.llm_engine, "model_executor"):
                model = llm.llm_engine.model_executor.driver_worker.model_runner.model
                logger.info("Using v0 engine model path - direct access")
                patch_model_for_attention_capture(model, hook)
                logger.info("✅ Attention capture enabled successfully (v0)")
                return  # Success via v0
        except AttributeError:
            pass

        # v1 architecture - use collective_rpc
        if model is None and hasattr(llm.llm_engine, "engine_core"):
            logger.info("Detected vLLM v1 - attempting RPC-based patching")

            try:
                # Capture variables in closure for serialization
                capture_layers = hook.capture_layers
                attention_window = hook.attention_window
                auto_clear = hook.auto_clear
                hook_llm_id = llm_id

                # Define function to execute on engine process
                def _patch_model_on_engine(worker=None):
                    """Execute in EngineCore process to patch the model

                    Args:
                        worker: Worker instance passed by collective_rpc (may be None)
                    """
                    try:
                        # Import in engine process
                        from vllm_attention_capture_plugin.hooks.attention_hook import AttentionCaptureHook
                        from vllm_attention_capture_plugin.wrappers.attention_layer_patcher import patch_model_for_attention_capture
                        import vllm_attention_capture_plugin.api as plugin_api

                        # Recreate hook in engine process using closure variables
                        engine_hook = AttentionCaptureHook(attention_window, capture_layers)
                        engine_hook.auto_clear = auto_clear

                        # Find model in engine process
                        model = None

                        # If worker provided, try to get model from it
                        if worker is not None:
                            if hasattr(worker, 'model_runner'):
                                if hasattr(worker.model_runner, 'model'):
                                    model = worker.model_runner.model

                        # Fallback: search for model via gc
                        if model is None:
                            try:
                                from vllm.v1.worker.cpu_model_runner import CPUModelRunner
                                import gc

                                for obj in gc.get_objects():
                                    if isinstance(obj, CPUModelRunner):
                                        if hasattr(obj, 'model'):
                                            model = obj.model
                                            break
                            except Exception:
                                pass

                        # Try GPU model runner too
                        if model is None:
                            try:
                                from vllm.v1.worker.gpu_model_runner import GPUModelRunner
                                import gc

                                for obj in gc.get_objects():
                                    if isinstance(obj, GPUModelRunner):
                                        if hasattr(obj, 'model'):
                                            model = obj.model
                                            break
                            except Exception:
                                pass

                        if model is None:
                            return {"success": False, "error": "Could not find model in engine process"}

                        # Store hook in engine process registry
                        plugin_api._CAPTURE_HOOKS[hook_llm_id] = engine_hook

                        # Patch the model
                        patch_model_for_attention_capture(model, engine_hook)

                        return {"success": True, "model_type": type(model).__name__}

                    except Exception as e:
                        import traceback
                        return {
                            "success": False,
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        }

                # Execute via RPC
                result = llm.llm_engine.collective_rpc(_patch_model_on_engine)

                if isinstance(result, list):
                    result = result[0]  # Get first result from DP ranks

                if not result.get("success"):
                    error = result.get("error", "Unknown error")
                    tb = result.get("traceback", "")
                    raise RuntimeError(f"{error}\n{tb}")

                logger.info(f"✅ Attention capture enabled successfully (v1 via RPC) - model: {result.get('model_type')}")
                return  # Success via v1 RPC

            except Exception as e:
                logger.error(f"v1 RPC patching failed: {e}")
                raise RuntimeError(
                    f"vLLM v1 RPC-based patching failed: {e}\n\n"
                    "The plugin attempted to patch the model via RPC but encountered an error. "
                    "This may indicate that your vLLM version's RPC API is incompatible."
                ) from e

        # Neither v0 nor v1 worked
        if model is None:
            raise RuntimeError(
                "Could not access model from LLM instance. Unknown engine architecture."
            )

        logger.info("✅ Attention capture enabled successfully")
    except Exception as e:
        logger.error("Failed to enable attention capture: %s", e)
        del _CAPTURE_HOOKS[llm_id]
        raise RuntimeError(f"Failed to patch attention layers: {e}") from e


def disable_attention_capture(llm: Any) -> None:
    """Disable attention capture for a vLLM LLM instance.

    Args:
        llm: A vLLM LLM instance with capture previously enabled
    """
    llm_id = id(llm)
    if llm_id in _CAPTURE_HOOKS:
        _CAPTURE_HOOKS[llm_id].disable()
        del _CAPTURE_HOOKS[llm_id]
        logger.info("Attention capture disabled")
    else:
        logger.warning("Attention capture was not enabled for this LLM")


def get_attention_scores(request_id: str) -> dict[int, Any] | None:
    """Get captured attention scores for a completed request.

    Args:
        request_id: The request ID from RequestOutput

    Returns:
        Dictionary mapping layer_idx to attention scores (numpy array),
        or None if no scores were captured for this request.

        Shape of each array: [num_heads, num_tokens_generated, attention_window]

    Example:
        ```python
        outputs = llm.generate("Hello world")
        scores = get_attention_scores(outputs[0].request_id)

        if scores:
            layer_0_attention = scores[0]
            print(f"Shape: {layer_0_attention.shape}")  # (num_heads, tokens, window)

            # Analyze last token's attention
            last_token = layer_0_attention[:, -1, :]  # [num_heads, window]
            avg_attention = last_token.mean(axis=0)  # Average across heads
            print(f"Attention to last {len(avg_attention)} tokens: {avg_attention}")
        ```
    """
    # Search all hooks for this request_id
    for hook in _CAPTURE_HOOKS.values():
        if request_id in hook.captured_scores:
            return hook.get_captured_scores(request_id)

        # FALLBACK: For backward compatibility with hardcoded "default_request"
        # Try to find captures under "default_request" and return them for the requested ID
        if "default_request" in hook.captured_scores:
            logger.warning(
                "Found captures under 'default_request' instead of '%s'. "
                "This is a temporary workaround for Phase 2. "
                "Consider using get_latest_attention_scores() for single-request scenarios.",
                request_id
            )
            return hook.get_captured_scores("default_request")

    return None


def get_capture_config(llm: Any) -> dict[str, Any] | None:
    """Get the current capture configuration for a vLLM LLM instance.

    Args:
        llm: A vLLM LLM instance

    Returns:
        Dictionary with capture configuration, or None if capture not enabled.
        Keys: 'capture_layers', 'attention_window', 'enabled', 'auto_clear'

    Example:
        ```python
        config = get_capture_config(llm)
        if config:
            print(f"Capturing layers: {config['capture_layers']}")
            print(f"Window size: {config['attention_window']}")
        ```
    """
    llm_id = id(llm)
    hook = _CAPTURE_HOOKS.get(llm_id)

    if hook is None:
        return None

    return {
        "capture_layers": list(hook.capture_layers),
        "attention_window": hook.attention_window,
        "enabled": hook.enabled,
        "auto_clear": getattr(hook, "auto_clear", True),
    }


def clear_all_captures(llm: Any | None = None) -> None:
    """Clear all captured attention scores.

    Args:
        llm: If provided, clear captures only for this LLM instance.
             If None, clear captures for all LLM instances.

    Example:
        ```python
        # Clear captures for specific LLM
        clear_all_captures(llm)

        # Or clear all captures
        clear_all_captures()
        ```
    """
    if llm is not None:
        # Clear for specific LLM
        llm_id = id(llm)
        hook = _CAPTURE_HOOKS.get(llm_id)
        if hook:
            hook.captured_scores.clear()
            logger.info("Cleared captures for LLM instance %d", llm_id)
        else:
            logger.warning("No capture hook found for this LLM")
    else:
        # Clear for all LLMs
        for hook in _CAPTURE_HOOKS.values():
            hook.captured_scores.clear()
        logger.info("Cleared all captures")


def get_capture_hook(llm: Any) -> AttentionCaptureHook | None:
    """Get the capture hook for a vLLM LLM instance (for advanced usage).

    Args:
        llm: A vLLM LLM instance

    Returns:
        The AttentionCaptureHook instance, or None if capture not enabled
    """
    llm_id = id(llm)
    return _CAPTURE_HOOKS.get(llm_id)


def get_latest_attention_scores() -> dict[int, Any] | None:
    """Get captured attention scores from the most recent request.

    This is a convenience method for single-request scenarios where you don't
    want to track request IDs.

    Returns:
        Dictionary mapping layer_idx to attention scores, or None if nothing captured.

    Example:
        ```python
        # For single-request usage
        outputs = llm.generate("Hello world")
        scores = get_latest_attention_scores()  # No need for request ID
        ```
    """
    # Get the most recently captured request from all hooks
    for hook in _CAPTURE_HOOKS.values():
        if hook.captured_scores:
            # Return scores from the first available request
            # For single-request scenarios, there should only be one
            request_id = next(iter(hook.captured_scores))
            return hook.get_captured_scores(request_id)

    return None


__all__ = [
    "enable_attention_capture",
    "disable_attention_capture",
    "get_attention_scores",
    "get_latest_attention_scores",
    "get_capture_hook",
    "get_capture_config",
    "clear_all_captures",
]
