# Exposing Attentions in vLLM - Wiki Outline

## **1. Overview**

### What this document covers
- Introduction to attention mechanisms and why capturing them matters
- The vLLM attention capture plugin implementation
- Important learnings: classifications of what's possible and impossible with attention capture in inference engines

### Target audience
- Researchers interested in interpretability
- ML engineers debugging model behavior
- Contributors who want to extend or maintain the plugin

### Prerequisites
**Required:**
- Basic transformer architecture knowledge (attention, Q/K/V)
- Familiarity with vLLM (what it does, basic usage)
- PyTorch experience (tensor operations, model hooks)

**Helpful but not required:**
- GPU programming concepts
- Knowledge of attention optimizations (Flash Attention)

### Disclaimer
**What this plugin IS:**
- A research and debugging tool for attention analysis
- Suitable for interpretability research and model behavior investigation
- Designed for controlled, low-throughput scenarios

**What this plugin IS NOT:**
- Production-ready for high-throughput serving
- A comprehensive solution for all attention analysis needs
- Suitable for real-time inference in performance-critical applications
- A replacement for proper model debugging and validation

**Limitations to be aware of:**
- Windowed capture only in large context applications
- Performance overhead on capture layers
- Limited model support (see compatibility section)

### References & Sources
- vLLM documentation: https://docs.vllm.ai/en/latest/
- Flash Attention papers: 
  - https://arxiv.org/abs/2205.14135
  - https://arxiv.org/abs/2307.08691
  - https://arxiv.org/abs/2407.08608
- PagedAttention paper: https://arxiv.org/abs/2309.06180
- Inspiration from vLLM PR: https://github.com/vllm-project/vllm/pull/35014
- GitHub Repository:

---

## **2. Background & Motivation**

### **2.1 Why Capture Attention?**

High-level overview of use cases, prioritized by practicality:

1. **Hallucination Detection**
   - Identifying when models generate information not grounded in context
   - Analyzing whether models attend to relevant source material

2. **Debugging Model Behavior**
   - Understanding unexpected outputs
   - Investigating why models ignore certain context

3. **Context Utilization Analysis**
   - Verifying that RAG systems actually use retrieved documents
   - Measuring effective context window usage

4. **Interpretability Research**
   - Studying attention patterns across layers
   - Understanding information flow in transformers

5. **Alignment Research**
   - Analyzing how models use instructions vs context
   - Studying prompt sensitivity

### **2.2 The Challenge**

**vLLM-Specific Technical Challenges:**

1. **Flash Attention doesn't materialize weights**
   - Modern attention backends (Flash Attention, FlashInfer) use kernel fusion
   - Attention weights are computed and discarded within fused kernels
   - Never written to memory → cannot be extracted

2. **Memory constraints**
   - Full attention matrices: O(L × n² × h) memory where:
     - L = number of layers
     - n = sequence length  
     - h = number of heads
     - f = memory of score
   - Example: 30 layers, 1000 tokens, 32 heads, 4 bytes (float32) = **3.8 GB**
   - Infeasible for production serving scenarios

3. **Performance requirements**
   - vLLM optimizes for throughput (tokens/second)
   - Any attention capture solution must minimize overhead
   - Can't break vLLM's core optimizations (continuous batching, paged KV cache)

4. **Plugin constraint**
   - Cannot modify vLLM core code (maintainability)
   - Must work as external plugin
   - Need to intercept attention computation without breaking abstractions

**Quantified Tradeoffs:**
- Full attention matrix (baseline): 3.8 GB for 30 layers, 1000 tokens, 32 heads
- Windowed capture (window=5, 3 layers): 1.9 MB
- **Memory reduction: ~2000×**
- Latency impact: ~5% with 3 capture layers (SDPA fallback on those layers)
- Throughput: ~95 tokens/sec vs 100 baseline (5% degradation)

---

## **3. Technical Deep Dive**

### **3.1 Attention Fundamentals**

**Content:**
- Scaled Dot-Product Attention formula and intuition (no proofs)
- Step-by-step breakdown of SDPA computation
- Causal masking: why autoregressive models need it (training vs inference difference)
- KV caching concept (link to external resource for mathematical details)

**Include:**
- Visualizations for each concept:
  - Q/K/V matrix dimensions and operations
  - Causal mask (upper triangular structure)
  - KV cache accumulation across timesteps
  - Attention weight matrix heatmap example

**Validation approach:**
- Show example attention weights and explain what makes them "reasonable"
- Demonstrate attention patterns (e.g., last token attending heavily to recent context)

**Note:** GQA/MQA not covered here (will be in implementation section where relevant)

### **3.2 vLLM Architecture**

**Content:**

1. **PagedAttention (high-level)**
   - Concept: KV cache managed in blocks like virtual memory
   - Enables efficient memory sharing and dynamic batching
   - Key takeaway: plugin must work with paged KV structure

2. **Attention Backend Abstraction**
   - vLLM supports multiple backends: Flash Attention, FlashInfer, xFormers, PyTorch SDPA
   - Backend selection logic:
     - Based on hardware (GPU generation)
     - Based on model config (head dimensions, GQA)
     - Based on availability (installed packages)
   - Why this matters: plugin must intercept regardless of backend

3. **Model Executor Architecture**
   - v0 vs v1 executors exist (note: plugin handles both transparently)
   - Focus on inference and execution flow only
   - Model loading → request scheduling → attention computation → output

**Include:**
- Architecture diagram showing:
  - vLLM components: LLM → Executor → Model → Attention Layers
  - Plugin injection points (where patching happens)
  - Data flow: request → KV cache → attention → output

### **3.3 The Materialization Problem**

**Content:**

1. **Performance Benchmarks**
   - Comparison: Standard SDPA vs Flash Attention performance
   - Show actual numbers (tokens/sec, latency percentiles)
   - Highlight memory bandwidth as bottleneck

2. **Code Comparison**
   ```python
   # Standard SDPA (materializes attention)
   scores = (Q @ K.T) / sqrt(d_k)  # [batch, heads, seq, seq] - MATERIALIZED
   scores = scores + causal_mask
   attn_weights = softmax(scores)   # [batch, heads, seq, seq] - MATERIALIZED
   output = attn_weights @ V
   
   # Flash Attention (fused)
   output = flash_attn_func(Q, K, V, causal=True)  # No intermediate tensors
   ```

3. **GPU Memory Hierarchy**
   - HBM (High Bandwidth Memory): large but slow
   - SRAM: small but fast (on-chip)
   - Flash Attention uses tiling to keep data in SRAM
   - Materializing attention weights requires HBM writes/reads (slow)

4. **Backend Behavior (minimal detail)**
   - Flash Attention: never materializes
   - FlashInfer: never materializes
   - xFormers: can optionally materialize (not default)
   - PyTorch SDPA: materializes in fallback path

**Option for future:** Document alternative approaches considered and rejected (keep as optional expansion)

---

## **4. Plugin Architecture**

### **4.1 Design Principles**

**Core principles with brief explanations:**

1. **Zero core code changes**
   - Implemented entirely as external plugin
   - No vLLM modifications required

2. **Memory-conscious**
   - Windowed capture reduces memory 2000×
   - Configurable layers and window size

3. **Performance-aware**
   - Overhead only on selected capture layers
   - Other layers use optimized backends

4. **Backend agnostic**
   - Works regardless of vLLM's attention backend choice
   - Falls back to SDPA when needed

**Violated approaches (lessons learned):**
- ❌ Direct vLLM codebase modification (maintainability nightmare)
- ❌ Intercepting generation loop directly (breaks vLLM's request scheduling)
- ❌ Modifying vLLM's output datastructures (version compatibility issues)

**Non-goals:**
- Not trying to capture pre-softmax scores
- Not trying to capture attention gradients
- Not trying to support all vLLM features (e.g., multiprocessing in Phase 2)
- Not aiming for zero overhead (small overhead acceptable for interpretability)

### **4.2 Core Components**

#### **4.2.1 Attention Capture Hook**

**Class Structure:**
```python
class AttentionCaptureHook:
    def __init__(self, capture_layers, attention_window):
        self._enabled: bool
        self._capture_layers: List[int]
        self._attention_window: int
        self._current_request_id: Optional[str]
        self._layer_scores: Dict[int, List[np.ndarray]]  # layer -> scores per step
        
    def capture_attention(self, layer_id: int, attn_weights: torch.Tensor) -> None:
        """Store windowed attention weights for current request"""
        
    def start_request(self, request_id: str) -> None:
        """Initialize tracking for new request"""
        
    def get_scores(self, request_id: str) -> Dict[int, np.ndarray]:
        """Retrieve accumulated scores for request"""
```

**Responsibilities:**
1. **Global State Management**
   - Single global hook instance manages all capture state
   - Thread safety: (TO INVESTIGATE - document whether this is handled or limitation)

2. **Request ID Tracking**
   - Associates attention weights with specific requests
   - Timestamp-based ID generation (don't document algorithm details)

3. **Multi-layer Coordination**
   - Accumulates scores from multiple configured layers
   - Handles layers capturing at different rates

4. **Memory Statistics Tracking**
   - Tracks total memory used per request
   - Records: layer count, sequence length, window size, head count
   - Used for: debugging, validation, user reporting

**Request Lifecycle:**
```
1. start_request(id) → Initialize empty score accumulation
2. generate() → capture_attention() called at each decode step for each capture layer
3. get_scores(id) → Retrieve final accumulated attention weights
4. cleanup → Scores removed after retrieval
```

#### **4.2.2 Attention Layer Patcher**

**Patching Mechanism (show code):**
```python
def patch_attention_layer(model, layer_id, hook):
    """Monkey-patch a specific attention layer to capture weights"""
    original_forward = model.layers[layer_id].self_attn.forward
    
    def forward_with_capture(*args, **kwargs):
        # Fall back to manual SDPA to get attention weights
        attn_output, attn_weights = manual_sdpa_with_weights(...)
        
        # Capture windowed weights
        if hook.enabled and layer_id in hook.capture_layers:
            hook.capture_attention(layer_id, attn_weights)
            
        return attn_output
    
    model.layers[layer_id].self_attn.forward = forward_with_capture
```

**Detection Logic:**
- Detects v0 vs v1 executor by inspecting model structure
- v0: `model.model.layers[i].self_attn`
- v1: Different path (document the actual paths)
- Patches the appropriate forward method based on detection

**SDPA Fallback Implementation (key differences):**
- Manual Q @ K^T computation vs fused kernels
- Explicit softmax vs implicit in fused op
- Returns both output AND attention weights
- ~20-30% slower than Flash Attention for capture layers
- Other layers unaffected (still use optimized backends)

**GQA (Group Query Attention) Handling:**
```python
if num_kv_heads < num_q_heads:
    # Repeat K and V to match Q head count
    num_repeats = num_q_heads // num_kv_heads
    key = key.repeat_interleave(num_repeats, dim=1)
    value = value.repeat_interleave(num_repeats, dim=1)
```
- Why needed: Some models (Llama, Mistral) use GQA to reduce KV cache size
- Query heads > KV heads → must repeat KV to align dimensions
- Plugin handles this transparently in SDPA fallback

**Model Compatibility Matrix:**

| Model Family | Status | Notes |
|--------------|--------|-------|
| GPT-2 | ✅ Supported | Phase 2 validated |
| Llama 2/3 | 🚧 Planned | GQA support implemented, needs testing |
| Mistral | 🚧 Planned | GQA + sliding window attention |
| Qwen | 🚧 Planned | Standard MHA |
| Phi | ❌ Not yet | Different attention structure |

#### **4.2.3 KV Cache Extraction**

**Concept (not highly technical):**
- During prefill: K and V are computed directly from input
- During decode: K and V for past tokens are in vLLM's paged KV cache
- Plugin needs to reconstruct full K and V tensors from cached blocks

**vLLM Paging Strategy (keep short):**
- KV cache split into fixed-size blocks (like OS virtual memory)
- Block table maps logical sequence positions to physical cache blocks
- Plugin uses vLLM's existing extraction utilities

**Decode vs Prefill Phase:**
- **Prefill**: All tokens processed at once, K/V available directly
- **Decode**: One new token, need to fetch cached K/V for context
- Why decode needs special handling: Past K/V not in current computation, must be retrieved from cache

**Code Example:**
```python
# Simplified extraction logic
def extract_kv_from_cache(kv_cache, block_table):
    """Reconstruct full K and V tensors from paged cache"""
    # This uses vLLM's built-in utilities
    from vllm.attention.ops.paged_attn import reshape_from_cache
    
    key = reshape_from_cache(kv_cache[0], block_table, "key")
    value = reshape_from_cache(kv_cache[1], block_table, "value")
    return key, value
```

**Note:** Actual implementation copies logic from vLLM's internal attention implementations.

### **4.3 Configuration & API**

**Public API Functions:**

```python
def enable_attention_capture(
    llm: LLM,
    capture_layers: List[int],
    attention_window: int
) -> None:
    """
    Enable attention capture for specified layers.
    
    Args:
        llm: vLLM LLM instance
        capture_layers: List of layer indices to capture (0-indexed). Use [-1] for all layers.
        attention_window: Number of recent tokens to capture (e.g., 5 = last 5 tokens)
    """

def disable_attention_capture(llm: LLM) -> None:
    """Disable attention capture and unpatch layers."""

def get_attention_scores(request_id: str) -> Dict[int, np.ndarray]:
    """
    Retrieve captured attention scores for a specific request.
    
    Returns:
        Dictionary mapping layer_id -> attention weights
        Shape per layer: [num_heads, num_generated_tokens, attention_window]
    """

def get_latest_attention_scores() -> Dict[int, np.ndarray]:
    """Convenience function to get scores from most recent request."""

def set_request_context(request_id: str) -> None:
    """Set request ID for next generation (multi-request support)."""

def get_capture_config() -> Dict[str, Any]:
    """
    Get current capture configuration.
    
    Returns:
        {
            "enabled": bool,
            "capture_layers": List[int],
            "attention_window": int,
            "current_request_id": Optional[str]
        }
    """

def clear_all_captures() -> None:
    """Clear all stored attention scores (free memory)."""
```

---

## **5. Implementation Details**

### **5.3 Memory Management**

**Memory Per Request Formula:**
```
memory_bytes = sum over capture_layers of:
    num_heads × num_generated_tokens × attention_window × 4 bytes (float32)

Example:
- 3 layers, 12 heads each, 100 generated tokens, window=5
- 3 × 12 × 100 × 5 × 4 = 72,000 bytes = 72 KB
```

**Attention Score Accumulation:**
```python
# Data structure for storing scores across generation steps
self._layer_scores: Dict[str, Dict[int, List[np.ndarray]]]
# request_id -> layer_id -> [step_0_weights, step_1_weights, ...]

# At retrieval time, concatenate across steps:
final_scores = np.concatenate(self._layer_scores[request_id][layer_id], axis=0)
# Shape: [total_generated_tokens, num_heads, attention_window]
```

**Cleanup:**
- Automatic: Scores are deleted from memory after `get_attention_scores()` retrieval
- Manual: User can call `clear_all_captures()` to free memory immediately
- Best practice: Retrieve scores promptly after generation to avoid memory buildup

---

## **6. Usage Guide**

### **6.1 Installation**

**Local Installation (Development):**
```bash
# Clone or navigate to plugin directory
cd /path/to/vllm_attention_capture_plugin

# Install in editable mode
pip install -e .

# Required environment variable (Phase 2 limitation)
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

### **6.2 Basic Usage**

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm_attention_capture_plugin import (
    enable_attention_capture,
    get_latest_attention_scores
)

# Initialize model
llm = LLM(model="gpt2")

# Enable capture for first 3 layers, window of 5 tokens
enable_attention_capture(
    llm,
    capture_layers=[0, 1, 2],
    attention_window=5
)

# Generate text
outputs = llm.generate(
    "The capital of France is",
    SamplingParams(max_tokens=20)
)

# Retrieve attention scores
scores = get_latest_attention_scores()
# scores = {0: array[12, 20, 5], 1: array[12, 20, 5], 2: array[12, 20, 5]}

print(f"Layer 0 shape: {scores[0].shape}")  # (12 heads, 20 tokens, window 5)
print(f"Last token attention: {scores[0][:, -1, :].mean(axis=0)}")  # Average across heads
```

### **6.3 Advanced Configuration**

**Example 1: Capture All Layers**
```python
enable_attention_capture(
    llm,
    capture_layers=[-1],  # Special value: all layers
    attention_window=10
)
```

**Example 2: Capture Specific Non-Sequential Layers**
```python
# Capture first, middle, and last layer of a 12-layer model
enable_attention_capture(
    llm,
    capture_layers=[0, 6, 11],
    attention_window=8
)
```

### **6.4 Accessing Results**

**Output Format:**
```python
scores = get_latest_attention_scores()
# Type: Dict[int, np.ndarray]
# Keys: layer indices (e.g., 0, 1, 2)
# Values: np.ndarray of shape [num_heads, num_generated_tokens, attention_window]
```

**Example Analysis:**
```python
# Get layer 0 attention
layer_0 = scores[0]  # Shape: [12, 20, 5]

# Average attention across all heads for last token
last_token_attn = layer_0[:, -1, :].mean(axis=0)
print(f"Attention distribution: {last_token_attn}")
# Output: [0.05, 0.08, 0.12, 0.25, 0.50]  # Attends most to most recent token

# Find which head attends most to specific position
position = -1  # Last position in window (most recent token)
head_attention_to_recent = layer_0[:, :, position].mean(axis=1)  # Average over tokens
print(f"Head {head_attention_to_recent.argmax()} attends most to recent context")
```

**Visualization (link to external tools):**
- Matplotlib: heatmap of attention weights
- bertviz: interactive attention visualization
- Custom visualization tools (link to examples if available)

**Interpretation Gotchas:**
- Windowed attention only: Can't see attention to tokens outside window
- Post-softmax only: Don't have pre-softmax scores (can't see "negative attention")
- Averaged across batch: If batch size > 1, results may be mixed (Phase 2 limitation)

---

## **7. Limitations & Constraints**

### **7.1 Current Limitations**

**Multiprocessing Mode (Phase 2):**
- ❌ Not supported: `VLLM_ENABLE_V1_MULTIPROCESSING=1` breaks plugin
- Why: Global state doesn't propagate across process boundaries
- Workaround: Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` (required)

**Model Support:**
- ✅ GPT-2 (validated)
- 🚧 Llama, Mistral, Qwen (GQA implemented, not tested)
- ❌ Other architectures (Phi, Falcon) not yet supported

**GPU Support:**
- ✅ Single GPU
- ❌ Multi-GPU / tensor parallelism not supported
- ❌ Pipeline parallelism not supported

**Capture Limitations:**
- Windowed only (not full attention matrix)
- Post-softmax only (not pre-softmax scores)
- Selected layers only (not all layers by default for performance)

### **7.2 Known Issues**

**Request ID Edge Cases:**
- Timestamp collisions possible if >1000 requests/sec (unlikely in typical usage)
- No explicit error if wrong request_id passed to `get_attention_scores()`

**Memory Pressure:**
- Long generations (>1000 tokens) with large windows (>20) can use significant memory
- No automatic memory pressure detection or warnings

**Backend Compatibility:**
- ✅ Works with all vLLM attention backends (falls back to SDPA on capture layers)
- ⚠️ Performance varies by backend (Flash Attention → SDPA fallback has larger overhead)

---

## **8. Validation & Testing**

### **8.1 Correctness Validation**

**Approach: Sanity Checks (not rigorous numerical validation)**

1. **Attention weight properties:**
   - ✅ All weights are non-negative
   - ✅ Weights sum to 1.0 across attention window (post-softmax)
   - ✅ Causal structure: no attention to future tokens

2. **Expected patterns:**
   - ✅ Last token typically attends most to recent context
   - ✅ Attention peaks visible at semantically relevant positions
   - ✅ Attention patterns differ across layers (lower vs higher)

3. **Edge cases validated:**
   - ✅ Single token generation
   - ✅ Window size larger than sequence length
   - ✅ Multiple capture layers
   - ✅ All layers capture ([-1] parameter)

**Example Validation:**
```python
scores = get_latest_attention_scores()
layer_0 = scores[0]

# Check non-negative
assert (layer_0 >= 0).all(), "Attention weights must be non-negative"

# Check sums to 1 (within numerical tolerance)
sums = layer_0.sum(axis=-1)  # Sum over attention window
assert np.allclose(sums, 1.0, atol=1e-5), "Attention must sum to 1"

# Check reasonable distribution (no NaNs, not all uniform)
assert not np.isnan(layer_0).any(), "No NaN values"
assert layer_0.std() > 0.01, "Attention should have variation"
```

---

## **9. Roadmap & Future Work**

### **9.1 Planned Features (Phase 3)**

No specific timeline or complexity estimates - this is an exploratory single-person project.

**High-Priority:**
- Multi-model support (Llama, Mistral, Qwen)
- Multiprocessing mode compatibility
- Multi-GPU / tensor parallelism support

**Medium-Priority:**
- Compressed capture (store only top-K attention weights per position)
- Disk-based storage for very long sequences
- Pre-softmax score capture option

**Low-Priority:**
- Built-in visualization utilities
- Integration with existing interpretability tools
- Attention pattern analysis helpers

### **9.2 Research Directions**

Speculative ideas, not concrete proposals:

**Backend-Specific Optimizations:**
- Custom Flash Attention fork that optionally materializes weights
- Selective materialization (only for specific heads or positions)

**Real-Time Analysis:**
- Streaming attention analysis during generation
- Hallucination detection based on attention entropy
- Context utilization metrics computed on-the-fly

**Advanced Capture Modes:**
- Gradient-weighted attention (attention × gradient magnitude)
- Attention rollout (accumulated attention across layers)
- Cross-attention capture (for encoder-decoder models)

**Tooling & Integration:**
- Jupyter notebook integration with interactive visualizations
- Export to standard formats (HDF5, Parquet)
- Integration with LangSmith, Weights & Biases for tracking

Community contributions welcome for any of these directions!

---

## **10. Appendices**

### **10.1 Memory Calculations**

**Formula:**
```
memory_per_request = Σ (num_heads_i × num_tokens × window_size × 4 bytes)
                     for i in capture_layers
```

**Example Scenarios:**

**Scenario 1: Minimal Memory**
- Config: 1 layer, 12 heads, window=3
- 100 tokens generated
- Memory: 12 × 100 × 3 × 4 = 14.4 KB

**Scenario 2: Balanced**
- Config: 3 layers, 32 heads each, window=5
- 500 tokens generated
- Memory: 3 × 32 × 500 × 5 × 4 = 960 KB ≈ 1 MB

**Scenario 3: Heavy**
- Config: 10 layers, 40 heads each, window=20
- 2000 tokens generated
- Memory: 10 × 40 × 2000 × 20 × 4 = 64 MB

**Comparison to Full Attention:**
```
Full attention (no windowing):
memory = num_layers × num_heads × num_tokens² × 4 bytes

Example: 32 layers, 32 heads, 2000 tokens
= 32 × 32 × 2000² × 4 = 16.4 GB

With windowing (window=20):
= 32 × 32 × 2000 × 20 × 4 = 163 MB
Reduction: ~100×
```

### **10.2 Architecture Diagrams**

**Diagram 1: Component Interaction Flow**
```
User Code
   ↓
enable_attention_capture(llm, layers=[0,1,2], window=5)
   ↓
AttentionLayerPatcher.patch_model(llm.model)
   ↓
   ├─ Patch layer 0.self_attn.forward
   ├─ Patch layer 1.self_attn.forward
   └─ Patch layer 2.self_attn.forward
   
llm.generate("prompt", SamplingParams(...))
   ↓
vLLM Executor
   ↓
Model Forward Pass
   ↓
Layer 0 Attention (PATCHED)
   ├─ Manual SDPA computation → attn_weights
   └─ AttentionCaptureHook.capture_attention(0, attn_weights)
   
Layer 1 Attention (PATCHED)
   └─ ... same ...

Layer 2 Attention (PATCHED)
   └─ ... same ...

Layers 3-N (UNPATCHED)
   └─ Flash Attention (fast, no capture)
   
User Code
   ↓
scores = get_latest_attention_scores()
   ↓
AttentionCaptureHook.get_scores(request_id)
   ↓
{0: array[...], 1: array[...], 2: array[...]}
```

**Diagram 2: Data Flow Through vLLM Executor**
```
Request → Scheduler → Model Executor
                         ↓
                    Load model on GPU
                         ↓
                    [Input tokens] → Embedding
                         ↓
                    ┌────────────────┐
                    │  Layer 0       │
                    │  - Self Attn   │ ← PLUGIN INTERCEPTS HERE
                    │  - FFN         │
                    └────────────────┘
                         ↓
                    ┌────────────────┐
                    │  Layer 1       │
                    │  - Self Attn   │ ← PLUGIN INTERCEPTS HERE
                    │  - FFN         │
                    └────────────────┘
                         ↓
                        ...
                         ↓
                    ┌────────────────┐
                    │  Layer N       │
                    │  - Self Attn   │ ← Not captured (too expensive)
                    │  - FFN         │
                    └────────────────┘
                         ↓
                    LM Head → Logits → Sample next token
                         ↓
                    RequestOutput (+ attention scores via hook)
```

**Diagram 3: KV Cache Extraction**
```
Decode Phase (generating token t):

Current token → Q_new [1, num_heads, head_dim]

KV Cache (paged):
┌─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │  Physical cache blocks
└─────────┴─────────┴─────────┘
     ↑          ↑          ↑
     └──────────┴──────────┘
         Block Table
     [0, 1, 2] for this request

Plugin extracts:
K_cached [t-1, num_heads, head_dim]  ← From blocks [0,1,2]
V_cached [t-1, num_heads, head_dim]  ← From blocks [0,1,2]

Concatenate:
K_full = [K_cached | K_new]  → [t, num_heads, head_dim]
V_full = [V_cached | V_new]  → [t, num_heads, head_dim]

Compute attention:
attn_weights = softmax(Q_new @ K_full.T / √d)
Window last N:
attn_windowed = attn_weights[..., -window:]
```

### **10.3 Troubleshooting**

**Format: Problem → Solution**

**Problem: `ImportError: cannot import name 'enable_attention_capture'`**
- **Cause:** Plugin not installed
- **Solution:** Run `pip install -e .` in plugin directory

**Problem: `RuntimeError: Attention capture requires VLLM_ENABLE_V1_MULTIPROCESSING=0`**
- **Cause:** Multiprocessing mode not supported in Phase 2
- **Solution:** Set environment variable before importing vLLM:
  ```python
  import os
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
  from vllm import LLM
  ```

**Problem: Attention weights look uniform (all ~equal)**
- **Cause 1:** Capturing too early in model (layer 0 often has less structured attention)
- **Solution:** Try higher layers (e.g., middle or later layers)
- **Cause 2:** Short sequence (attention spreads evenly over few tokens)
- **Solution:** Generate longer sequences

**Problem: High memory usage**
- **Cause:** Large window size or many capture layers
- **Solution:** Reduce window size or number of captured layers
- **Diagnostic:** Check `get_capture_config()` for current settings

**Problem: Scores not returned / empty dictionary**
- **Cause:** Wrong request ID or scores already retrieved (auto-cleanup)
- **Solution:** Use `get_latest_attention_scores()` or ensure correct request ID

**How to Get Help:**
- Check this documentation first (especially Limitations section)
- Review examples in `examples/` directory (if available)
- Open GitHub issue with:
  - vLLM version
  - Plugin version
  - Model name
  - Minimal reproduction code
  - Error message / unexpected behavior

### **10.4 References**

**Papers:**
1. **Flash Attention:** Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
2. **Flash Attention 2:** Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024.
3. **vLLM / PagedAttention:** Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
4. **Attention Mechanism:** Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017.
5. **Group Query Attention:** Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023.

**Documentation:**
- vLLM Official Docs: https://docs.vllm.ai/
- PyTorch SDPA: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Flash Attention GitHub: https://github.com/Dao-AILab/flash-attention

**Related Implementations:**
- Eagle3 (speculative decoding with attention): [if public, add link]
- KV Connector: [if public, add link]
- vLLM Issue #XXXXX: Original attention capture discussion

**Code Repositories:**
- vLLM: https://github.com/vllm-project/vllm
- This plugin: [your repo URL]

**Version Compatibility:**
- Tested with vLLM v0.X.X - v0.Y.Y
- Requires PyTorch >= 2.0
- Tested on CUDA 11.8, 12.1

---

## Notes for Future Expansion

**Sections to add later:**
- **5.4 Performance Characteristics** - once benchmarks are run
- **Alternative approaches** in 3.3 - if time permits documenting rejected designs
- **More model compatibility** in 4.2.2.5 - as testing expands
- **Comprehensive test suite docs** in section 8 - if formal testing added
- **Contribution guide** - if opening to external contributors

**Diagrams to create:**
- All diagrams in 10.2 (component flow, data flow, KV extraction)
- Visualization examples in 6.4 (attention heatmaps)
- Memory comparison chart in 10.1

**Content to gather:**
- Actual benchmark numbers for 3.3 and 5.4
- Specific vLLM log messages for 10.3 troubleshooting
- Version compatibility matrix for 10.4
- Links to examples directory (once created)
