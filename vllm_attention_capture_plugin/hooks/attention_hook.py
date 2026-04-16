"""
Attention Capture Hook

Hooks into attention computation to extract and store windowed attention scores.
"""

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AttentionCaptureHook:
    """Hooks into attention computation to extract attention scores.

    This hook:
    1. Intercepts attention weights after softmax computation
    2. Optionally applies windowing to save memory (if attention_window is set)
    3. Accumulates scores per request across generation steps
    4. Returns accumulated scores on request completion

    Args:
        attention_window: Number of recent tokens to capture attention for.
                         If None, captures full attention matrix (more memory).
        capture_layers: List of layer indices to capture (e.g., [0, 1, 2])

    Memory usage example:
        - Full matrix (1000 tokens): 1000 × 32 heads × 1000 × 4 bytes = 128 MB
        - Windowed (1000 tokens, window=5): 1000 × 32 × 5 × 4 bytes = 640 KB (200× reduction)
    """

    def __init__(self, attention_window: int | None, capture_layers: list[int]):
        self.attention_window = attention_window
        self.capture_layers = set(capture_layers)
        # Store captured scores: {request_id: {layer_idx: [list of tensors]}}
        self.captured_scores: dict[str, dict[int, list[torch.Tensor]]] = {}
        self.enabled = True

    def should_capture(self, layer_idx: int) -> bool:
        """Check if we should capture this layer."""
        return self.enabled and layer_idx in self.capture_layers

    def capture_attention_weights(
        self,
        layer_idx: int,
        attn_weights: torch.Tensor,
        request_id: str,
    ) -> None:
        """Capture attention weights for a layer.

        Args:
            layer_idx: Layer index (0-indexed)
            attn_weights: Attention weights tensor [num_heads, num_tokens_batch, seq_len]
                         or [batch, num_heads, num_tokens, seq_len]
            request_id: Unique request identifier

        The function handles both batched and non-batched attention weights.
        If attention_window is set, only the last N positions are stored for memory efficiency.
        If attention_window is None, captures the full attention matrix.
        """
        if not self.should_capture(layer_idx):
            return

        # Handle different tensor shapes
        if attn_weights.ndim == 4:
            # [batch, num_heads, num_tokens, seq_len]
            # For now, assume batch=1 or take first item
            attn_weights = attn_weights[0]

        # Now shape is [num_heads, num_tokens_batch, seq_len]

        if self.attention_window is None:
            # Capture full attention matrix
            captured = attn_weights
        else:
            # Apply windowing for memory efficiency
            num_heads = attn_weights.shape[0]
            num_tokens = attn_weights.shape[1]
            seq_len = attn_weights.shape[2]

            # Apply windowing PER TOKEN to respect causal structure
            windowed_list = []
            for token_idx in range(num_tokens):
                # Determine the absolute position of this token in the sequence
                if num_tokens == 1:
                    # Decode phase: single token, its position is at the end
                    absolute_position = seq_len - 1
                else:
                    # Prefill phase: tokens are at positions [0, num_tokens-1]
                    absolute_position = token_idx

                # Valid range for this token: 0 to absolute_position (causal)
                valid_end = absolute_position + 1

                # Window: last N positions of valid range
                window_start = max(0, valid_end - self.attention_window)
                window_end = valid_end

                # Extract window: [num_heads, window_size]
                token_attn = attn_weights[:, token_idx, window_start:window_end]

                # Pad if window is smaller than attention_window (early tokens)
                actual_window_size = window_end - window_start
                if actual_window_size < self.attention_window:
                    pad_size = self.attention_window - actual_window_size
                    padding = torch.zeros(
                        (num_heads, pad_size),
                        dtype=token_attn.dtype,
                        device=token_attn.device
                    )
                    token_attn = torch.cat([padding, token_attn], dim=1)

                windowed_list.append(token_attn)

            # Stack into [num_heads, num_tokens, attention_window]
            captured = torch.stack(windowed_list, dim=1)

        # Initialize storage for this request if needed
        if request_id not in self.captured_scores:
            self.captured_scores[request_id] = {}
        if layer_idx not in self.captured_scores[request_id]:
            self.captured_scores[request_id][layer_idx] = []

        # Store on CPU to avoid GPU memory buildup
        self.captured_scores[request_id][layer_idx].append(
            captured.detach().cpu()
        )

    def get_captured_scores(self, request_id: str) -> dict[int, np.ndarray]:
        """Return accumulated scores for request, then clear storage.

        Args:
            request_id: Request identifier

        Returns:
            Dictionary mapping layer_idx to numpy array of shape:
            - If attention_window is None: [num_heads, total_tokens, max_seq_len]
            - If attention_window is set: [num_heads, total_tokens, attention_window]

            Each tensor represents the attention weights for all tokens (prefill + generated).
            For full attention, all tensors are padded to max_seq_len to enable concatenation.

        Example:
            If we have 7 prefill tokens + 100 generated tokens with 16 heads:
            - Full attention: {0: array of shape (16, 107, 107)}
            - Windowed (window=5): {0: array of shape (16, 107, 5)}

        Note:
            For full attention, tensors from prefill and decode phases have different seq_lens
            and must be padded to the maximum seq_len before concatenation.
        """
        if request_id not in self.captured_scores:
            return {}

        # For each layer, concatenate the captured tensors
        result = {}
        for layer_idx, score_list in self.captured_scores[request_id].items():
            try:
                # Convert bfloat16 tensors to float32
                converted_list = []
                for t in score_list:
                    if isinstance(t, torch.Tensor):
                        if t.dtype == torch.bfloat16:
                            converted_list.append(t.float())
                        else:
                            converted_list.append(t)
                    else:
                        converted_list.append(t)

                # Check if we need padding (full attention mode with varying seq_lens)
                shapes = [t.shape for t in converted_list]
                seq_lens = [s[2] for s in shapes]  # Third dimension is seq_len

                if len(set(seq_lens)) > 1:
                    # Need to pad to max seq_len
                    max_seq_len = max(seq_lens)
                    num_heads = shapes[0][0]

                    padded_list = []
                    for t in converted_list:
                        current_seq_len = t.shape[2]
                        if current_seq_len < max_seq_len:
                            # Pad along the seq_len dimension (dim=2)
                            # Shape: [num_heads, num_tokens, seq_len] -> [num_heads, num_tokens, max_seq_len]
                            pad_size = max_seq_len - current_seq_len
                            if isinstance(t, torch.Tensor):
                                padding = torch.zeros(
                                    (num_heads, t.shape[1], pad_size),
                                    dtype=t.dtype,
                                    device=t.device
                                )
                                t_padded = torch.cat([t, padding], dim=2)
                            else:
                                padding = np.zeros((num_heads, t.shape[1], pad_size), dtype=t.dtype)
                                t_padded = np.concatenate([t, padding], axis=2)
                            padded_list.append(t_padded)
                        else:
                            padded_list.append(t)

                    converted_list = padded_list

                # Concatenate along token dimension (axis=1)
                if isinstance(converted_list[0], torch.Tensor):
                    concatenated = torch.cat(converted_list, dim=1).numpy()
                else:
                    concatenated = np.concatenate(converted_list, axis=1)

                result[layer_idx] = concatenated
            except Exception as e:
                logger.error(
                    "Failed to concatenate scores for layer %d: %s. "
                    "Shapes: %s",
                    layer_idx, e, [arr.shape for arr in score_list]
                )
                # Fallback: convert individually and return as list
                numpy_list = []
                for t in score_list:
                    if isinstance(t, torch.Tensor):
                        numpy_list.append(t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy())
                    else:
                        numpy_list.append(np.array(t))
                result[layer_idx] = numpy_list

        # Clear captured scores to free memory
        del self.captured_scores[request_id]

        return result

    def clear_request(self, request_id: str) -> None:
        """Explicitly clear captured scores for a request."""
        if request_id in self.captured_scores:
            del self.captured_scores[request_id]

    def enable(self) -> None:
        """Enable attention capture."""
        self.enabled = True

    def disable(self) -> None:
        """Disable attention capture."""
        self.enabled = False

    def get_memory_usage(self) -> dict[str, Any]:
        """Get current memory usage statistics.

        Returns:
            Dictionary with memory stats per request.
        """
        stats = {}
        for request_id, layers in self.captured_scores.items():
            total_bytes = 0
            layer_stats = {}
            for layer_idx, score_list in layers.items():
                layer_bytes = sum(t.numel() * t.element_size() for t in score_list)
                total_bytes += layer_bytes
                layer_stats[layer_idx] = {
                    "num_chunks": len(score_list),
                    "bytes": layer_bytes,
                    "mb": layer_bytes / (1024 * 1024)
                }
            stats[request_id] = {
                "layers": layer_stats,
                "total_bytes": total_bytes,
                "total_mb": total_bytes / (1024 * 1024)
            }
        return stats
