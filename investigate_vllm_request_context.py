#!/usr/bin/env python3
"""
Investigation script to determine how to extract request IDs from vLLM.

This script patches vLLM to log all available context information during
forward passes to help us understand how to implement multi-request support.

Usage:
    python investigate_vllm_request_context.py
"""

import logging
import sys
from typing import Any

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def inspect_object(obj: Any, name: str = "object", max_depth: int = 2) -> dict:
    """Recursively inspect an object and return its structure."""
    result = {
        "type": str(type(obj)),
        "str": str(obj)[:200],  # Limit length
        "attributes": {},
    }

    if max_depth <= 0:
        return result

    try:
        attrs = [a for a in dir(obj) if not a.startswith('_')]
        for attr in attrs[:20]:  # Limit to first 20 attributes
            try:
                val = getattr(obj, attr)
                # Don't recurse into methods/functions
                if callable(val):
                    result["attributes"][attr] = "<callable>"
                elif isinstance(val, (int, float, str, bool, type(None))):
                    result["attributes"][attr] = val
                elif isinstance(val, (list, tuple)) and len(val) < 10:
                    result["attributes"][attr] = str(val)[:200]
                else:
                    result["attributes"][attr] = str(type(val))
            except Exception as e:
                result["attributes"][attr] = f"<error: {e}>"
    except Exception as e:
        result["error"] = str(e)

    return result


def create_investigation_patch():
    """Create a patch that logs all available context during forward pass."""

    def investigation_forward_wrapper(original_forward):
        """Wrap forward method to log context information."""

        def forward_with_investigation(
            query,
            key,
            value,
            output_shape=None,
        ):
            """Log everything we can find about the current request context."""

            logger.info("\n" + "="*80)
            logger.info("INVESTIGATION: Forward pass called")
            logger.info("="*80)

            # 1. Inspect forward pass inputs
            logger.info("\n### INPUT TENSORS ###")
            logger.info(f"Query shape: {query.shape}, dtype: {query.dtype}, device: {query.device}")
            logger.info(f"Key shape: {key.shape}, dtype: {key.dtype}, device: {key.device}")
            logger.info(f"Value shape: {value.shape}, dtype: {value.dtype}, device: {value.device}")
            logger.info(f"Output shape: {output_shape}")

            num_tokens = query.shape[0]
            is_decode = (num_tokens == 1)
            logger.info(f"Phase: {'DECODE' if is_decode else 'PREFILL'}")
            logger.info(f"Num tokens: {num_tokens}")

            # 2. Try to get forward context
            logger.info("\n### FORWARD CONTEXT ###")
            try:
                from vllm.forward_context import get_forward_context
                forward_ctx = get_forward_context()
                logger.info(f"Forward context exists: YES")
                ctx_info = inspect_object(forward_ctx, "forward_context")
                logger.info(f"Forward context type: {ctx_info['type']}")
                logger.info(f"Forward context attributes: {ctx_info['attributes']}")
            except ImportError as e:
                logger.info(f"Cannot import get_forward_context: {e}")
            except Exception as e:
                logger.info(f"Error getting forward context: {e}")

            # 3. Try to get attention context
            logger.info("\n### ATTENTION CONTEXT ###")
            try:
                from vllm.model_executor.layers.attention.attention import get_attention_context

                # Need layer_name - try to get it from the attention layer
                layer_name = None
                if hasattr(forward_with_investigation, '__self__'):
                    attention_layer = forward_with_investigation.__self__
                    layer_name = getattr(attention_layer, "layer_name", None)

                if layer_name:
                    logger.info(f"Layer name: {layer_name}")
                    attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)

                    logger.info("Attention metadata exists: YES")
                    metadata_info = inspect_object(attn_metadata, "attn_metadata")
                    logger.info(f"Attention metadata type: {metadata_info['type']}")
                    logger.info(f"Attention metadata attributes: {metadata_info['attributes']}")

                    # Look for request-related attributes specifically
                    logger.info("\n### REQUEST-RELATED ATTRIBUTES ###")
                    for attr in ['request_ids', 'seq_ids', 'request_id', 'seq_id',
                                 'seq_lens', 'num_seqs', 'batch_size', 'num_queries',
                                 'block_table', 'slot_mapping', 'context_lens']:
                        if hasattr(attn_metadata, attr):
                            val = getattr(attn_metadata, attr)
                            logger.info(f"  {attr}: {val}")
                        else:
                            logger.info(f"  {attr}: NOT FOUND")
                else:
                    logger.info("Layer name not available, cannot get attention context")

            except ImportError as e:
                logger.info(f"Cannot import get_attention_context: {e}")
            except Exception as e:
                logger.info(f"Error getting attention context: {e}")
                import traceback
                logger.info(traceback.format_exc())

            # 4. Check vLLM config
            logger.info("\n### VLLM CONFIG ###")
            try:
                from vllm.config import get_current_vllm_config
                config = get_current_vllm_config()
                if config:
                    config_info = inspect_object(config, "vllm_config")
                    logger.info(f"Config type: {config_info['type']}")
                    logger.info(f"Config attributes: {config_info['attributes']}")
                else:
                    logger.info("Config is None")
            except ImportError:
                logger.info("Cannot import get_current_vllm_config")
            except Exception as e:
                logger.info(f"Error getting vLLM config: {e}")

            # 5. Check thread-local storage
            logger.info("\n### THREAD-LOCAL STORAGE ###")
            import threading
            current_thread = threading.current_thread()
            logger.info(f"Thread name: {current_thread.name}")
            logger.info(f"Thread ident: {current_thread.ident}")
            if hasattr(current_thread, '__dict__'):
                logger.info(f"Thread attributes: {list(current_thread.__dict__.keys())}")

            # 6. Check for global state
            logger.info("\n### GLOBAL STATE ###")
            try:
                import vllm
                vllm_module_attrs = [a for a in dir(vllm) if not a.startswith('_')]
                logger.info(f"vLLM module attributes (sample): {vllm_module_attrs[:10]}")
            except Exception as e:
                logger.info(f"Error inspecting vllm module: {e}")

            logger.info("\n" + "="*80 + "\n")

            # Call original forward
            return original_forward(query, key, value, output_shape)

        return forward_with_investigation

    return investigation_forward_wrapper


def patch_attention_for_investigation(attention_layer: Any) -> None:
    """Patch an attention layer with investigation wrapper."""
    original_forward = attention_layer.forward
    wrapper = create_investigation_patch()
    attention_layer.forward = wrapper(original_forward)
    logger.info(f"Patched attention layer for investigation")


def run_investigation():
    """Run the investigation with a simple vLLM example."""
    logger.info("Starting vLLM request context investigation...\n")

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vLLM not installed. Please install: pip install vllm")
        sys.exit(1)

    # Create LLM with small model
    logger.info("Creating LLM instance...")
    llm = LLM(
        model="gpt2",
        max_model_len=512,
        gpu_memory_utilization=0.3,
    )

    # Patch first attention layer
    logger.info("Patching attention layer for investigation...\n")
    patched_via_rpc = False  # Track whether we patched via RPC

    try:
        # Try to access the model
        if hasattr(llm.llm_engine, "model_executor"):
            # v0 path
            model = llm.llm_engine.model_executor.driver_worker.model_runner.model
            logger.info("Using vLLM v0 engine")
        elif hasattr(llm.llm_engine, "engine_core"):
            # v1 path - use RPC
            logger.info("Using vLLM v1 engine - patching via RPC")

            def _patch_model_on_engine(worker=None):
                """Execute in EngineCore process to patch the model."""
                try:
                    import logging
                    logger = logging.getLogger(__name__)

                    # Find model in engine process
                    model = None
                    if worker is not None and hasattr(worker, 'model_runner'):
                        if hasattr(worker.model_runner, 'model'):
                            model = worker.model_runner.model

                    if model is None:
                        # Fallback: search via gc
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

                    if model is None:
                        return {"success": False, "error": "Could not find model"}

                    # Find first attention layer
                    if hasattr(model, "model") and hasattr(model.model, "layers"):
                        layer = model.model.layers[0]
                    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                        layer = model.transformer.h[0]
                    else:
                        return {"success": False, "error": "Cannot find layers"}

                    # Get attention module
                    if hasattr(layer, "self_attn"):
                        attn_module = layer.self_attn
                    elif hasattr(layer, "attn"):
                        attn_module = layer.attn
                    else:
                        return {"success": False, "error": "Cannot find attention module"}

                    # Get attention layer
                    if hasattr(attn_module, "attn"):
                        attention_layer = attn_module.attn

                        # Create investigation wrapper in engine process
                        original_forward = attention_layer.forward

                        def forward_with_investigation(query, key, value, output_shape=None):
                            """Log everything we can find about the current request context."""

                            logger.info("\n" + "="*80)
                            logger.info("INVESTIGATION: Forward pass called")
                            logger.info("="*80)

                            # 1. Inspect forward pass inputs
                            logger.info("\n### INPUT TENSORS ###")
                            logger.info(f"Query shape: {query.shape}, dtype: {query.dtype}")
                            logger.info(f"Key shape: {key.shape}, dtype: {key.dtype}")
                            logger.info(f"Value shape: {value.shape}, dtype: {value.dtype}")

                            num_tokens = query.shape[0]
                            is_decode = (num_tokens == 1)
                            logger.info(f"Phase: {'DECODE' if is_decode else 'PREFILL'}")
                            logger.info(f"Num tokens: {num_tokens}")

                            # 2. Try to get forward context
                            logger.info("\n### FORWARD CONTEXT ###")
                            try:
                                from vllm.forward_context import get_forward_context
                                forward_ctx = get_forward_context()
                                logger.info(f"Forward context exists: YES")
                                logger.info(f"Forward context type: {type(forward_ctx)}")
                                logger.info(f"Forward context dir: {dir(forward_ctx)}")

                                # Check for request-related attributes
                                for attr in ['request_id', 'request_ids', 'seq_id', 'seq_ids']:
                                    if hasattr(forward_ctx, attr):
                                        logger.info(f"  {attr}: {getattr(forward_ctx, attr)}")
                            except Exception as e:
                                logger.info(f"Error getting forward context: {e}")

                            # 3. Try to get attention context
                            logger.info("\n### ATTENTION CONTEXT ###")
                            try:
                                from vllm.model_executor.layers.attention.attention import get_attention_context

                                layer_name = getattr(attention_layer, "layer_name", None)
                                if layer_name:
                                    logger.info(f"Layer name: {layer_name}")
                                    attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)

                                    logger.info("Attention metadata exists: YES")
                                    logger.info(f"Attention metadata type: {type(attn_metadata)}")

                                    # Look for request-related attributes
                                    logger.info("\n### REQUEST-RELATED ATTRIBUTES ###")
                                    for attr in ['request_ids', 'seq_ids', 'request_id', 'seq_id',
                                                'seq_lens', 'num_seqs', 'batch_size', 'num_queries',
                                                'num_prefill_tokens', 'num_decode_tokens']:
                                        if hasattr(attn_metadata, attr):
                                            val = getattr(attn_metadata, attr)
                                            logger.info(f"  {attr}: {val}")
                                else:
                                    logger.info("Layer name not available")
                            except Exception as e:
                                logger.info(f"Error getting attention context: {e}")
                                import traceback
                                logger.info(traceback.format_exc())

                            logger.info("\n" + "="*80 + "\n")

                            # Call original forward
                            return original_forward(query, key, value, output_shape)

                        attention_layer.forward = forward_with_investigation
                        return {"success": True, "message": "Patched successfully"}
                    else:
                        return {"success": False, "error": "Cannot find attention layer"}

                except Exception as e:
                    import traceback
                    return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

            # Execute via RPC
            result = llm.llm_engine.collective_rpc(_patch_model_on_engine)
            if isinstance(result, list):
                result = result[0]

            if not result.get("success"):
                logger.error(f"v1 RPC patching failed: {result.get('error')}")
                if 'traceback' in result:
                    logger.error(result['traceback'])
                return

            logger.info("Successfully patched via RPC")
            # Skip the manual model access below since we did it via RPC
            patched_via_rpc = True
        else:
            logger.error("Cannot access model - unknown engine structure")
            return

        # Only do manual patching if we didn't use RPC
        if not patched_via_rpc:
            # Find first attention layer
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layer = model.model.layers[0]
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                layer = model.transformer.h[0]
            else:
                logger.error("Cannot find attention layers")
                return

            # Get attention module
            if hasattr(layer, "self_attn"):
                attn_module = layer.self_attn
            elif hasattr(layer, "attn"):
                attn_module = layer.attn
            else:
                logger.error("Cannot find attention module")
                return

            # Get attention layer
            if hasattr(attn_module, "attn"):
                attention_layer = attn_module.attn
                patch_attention_for_investigation(attention_layer)
            else:
                logger.error("Cannot find attention layer")
                return

    except Exception as e:
        logger.error(f"Failed to patch attention: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Test 1: Single request
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Single Request")
    logger.info("="*80 + "\n")

    outputs = llm.generate(
        "Hello, my name is",
        SamplingParams(temperature=0.0, max_tokens=5)
    )

    logger.info("\n### OUTPUT ###")
    for output in outputs:
        logger.info(f"Request ID: {output.request_id}")
        logger.info(f"Prompt: {output.prompt}")
        logger.info(f"Generated: {output.outputs[0].text}")

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

    logger.info("\n### OUTPUTS ###")
    for output in outputs:
        logger.info(f"Request ID: {output.request_id}")
        logger.info(f"Prompt: {output.prompt}")
        logger.info(f"Generated: {output.outputs[0].text}")

    logger.info("\n" + "="*80)
    logger.info("Investigation complete!")
    logger.info("="*80)

    logger.info("\n### SUMMARY ###")
    logger.info("Review the logs above to identify:")
    logger.info("1. Does attn_metadata contain request_ids or seq_ids?")
    logger.info("2. How are batched requests represented?")
    logger.info("3. Is there a forward_context with request information?")
    logger.info("4. What other sources of request identification exist?")


if __name__ == "__main__":
    run_investigation()
