# Dynamic Actor Framework for vLLM Introspection

**Version:** 1.0 (Draft)  
**Date:** 2026-04-21  
**Status:** Design Document  

---

## Executive Summary

This document proposes a comprehensive framework for dynamically instrumenting vLLM models with **programmable actors** that can collect, process, aggregate, and react to internal model states during inference. Unlike static data collection approaches, actors are **runtime-reconfigurable agents** that can be controlled externally, chain together for complex analysis, and adapt their behavior based on observed patterns.

**Key Differentiators:**
- **Dynamic**: Configuration changes without restarting
- **Interactive**: External control via REST/WebSocket APIs
- **Composable**: Actors can be chained into pipelines
- **Efficient**: Process during collection, store only what matters
- **Production-Ready**: Real-time metrics for monitoring

---

## Table of Contents

1. [Motivation & Use Cases](#1-motivation--use-cases)
2. [System Architecture](#2-system-architecture)
3. [Core Components](#3-core-components)
4. [Actor Types & Examples](#4-actor-types--examples)
5. [Control Plane Design](#5-control-plane-design)
6. [Data Flow & Lifecycle](#6-data-flow--lifecycle)
7. [Performance Considerations](#7-performance-considerations)
8. [API Specifications](#8-api-specifications)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Open Questions & Trade-offs](#10-open-questions--trade-offs)
11. [Migration Path](#11-migration-path)
12. [Future Extensions](#12-future-extensions)

---

## 1. Motivation & Use Cases

### 1.1 Current Limitations

**Existing attention capture plugin:**
- ✅ Collects attention weights
- ❌ Fixed configuration (set at initialization)
- ❌ Passive collection only (no on-the-fly processing)
- ❌ Single data type (attention only)
- ❌ No external control during generation
- ❌ Manual cleanup required

### 1.2 Target Use Cases

#### Research & Debugging
- **Interactive Debugging**: Adjust instrumentation mid-generation based on observations
- **Pattern Detection**: Automatically flag repetition loops, context neglect, high uncertainty
- **Comparative Analysis**: Run multiple actors with different configs simultaneously
- **Hypothesis Testing**: Quickly reconfigure to test theories without restarting

#### Production Monitoring
- **Real-Time Dashboards**: Stream aggregated metrics (attention entropy, head specialization)
- **Anomaly Detection**: Alert when model exhibits unusual behavior
- **Performance Profiling**: Track computational bottlenecks dynamically
- **A/B Testing**: Compare instrumentation strategies live

#### Advanced Analysis
- **Multi-Stage Processing**: Chain actors to build complex analysis pipelines
- **Adaptive Sampling**: Increase granularity when interesting patterns emerge
- **Selective Storage**: Only persist data that meets criteria (high entropy, outliers)
- **Cross-Request Analysis**: Aggregate patterns across multiple generations

---

## 2. System Architecture

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                      External Control                        │
│  (REST API / WebSocket / CLI / Jupyter Notebook)             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     Lens Manager                             │
│  - Session Management                                        │
│  - Actor Registry                                            │
│  - Lifecycle Orchestration                                   │
│  - Hook Coordination                                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Actor Pipeline                             │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ Actor 1  │───▶│ Actor 2  │───▶│ Actor 3  │             │
│  │ Collect  │    │ Process  │    │ Store    │             │
│  └──────────┘    └──────────┘    └──────────┘             │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   vLLM Model                                 │
│  - Attention Layers (hooks injected)                         │
│  - MLP Layers (hooks injected)                               │
│  - Embedding/Output Layers (hooks injected)                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Key Design Principles

1. **Separation of Concerns**: Actors handle one responsibility (collect, process, store)
2. **Open/Closed**: Easy to add new actors without modifying core framework
3. **Runtime Reconfiguration**: Configuration changes propagate immediately
4. **Thread Safety**: Actors handle concurrent access (config updates during generation)
5. **Graceful Degradation**: Actor failures don't crash generation
6. **Memory Bounded**: Automatic cleanup and eviction policies

---

## 3. Core Components

### 3.1 Lens Manager

**Responsibilities:**
- Manage actor lifecycle (add, remove, pause, resume)
- Coordinate hook registration across actors
- Handle session creation/cleanup
- Route data from hooks to appropriate actors
- Expose control API

**API Surface:**
```python
class LensManager:
    def __init__(self, llm: Any, config: LensConfig):
        """Initialize manager for a vLLM instance"""
        
    def create_session(
        self, 
        name: str, 
        ttl: int | None = None
    ) -> Session:
        """Create a new instrumentation session"""
        
    def add_actor(
        self, 
        actor: Actor, 
        session: Session | None = None
    ) -> str:
        """Register actor, returns actor_id"""
        
    def remove_actor(self, actor_id: str) -> None:
        """Unregister actor and cleanup resources"""
        
    def get_actor(self, actor_id: str) -> Actor:
        """Retrieve actor by ID"""
        
    def pipeline(self, actors: list[Actor]) -> Pipeline:
        """Create a processing pipeline"""
        
    def get_data(
        self, 
        request_id: str, 
        actor_id: str | None = None
    ) -> dict[str, Any]:
        """Retrieve collected data"""
```

### 3.2 Actor Base Class

**Core abstraction for all actors:**

```python
from abc import ABC, abstractmethod
from typing import Any, Optional
import threading

class Actor(ABC):
    """Base class for all actors in the framework"""
    
    def __init__(
        self, 
        name: str, 
        config: dict[str, Any] | None = None
    ):
        self.name = name
        self.actor_id: Optional[str] = None  # Set by manager
        self.config = config or {}
        self._lock = threading.RLock()
        self._enabled = True
        self._hooks = []
        
    @abstractmethod
    def setup(self, model: Any, layer_indices: list[int]) -> None:
        """
        Called once when actor is registered.
        Register hooks in the model.
        
        Args:
            model: The vLLM model instance
            layer_indices: Which layers this actor should instrument
        """
        pass
        
    @abstractmethod
    def process(
        self, 
        hook_point: str,
        layer_idx: int, 
        data: Any,
        context: dict[str, Any]
    ) -> Any:
        """
        Called during forward pass for each hook point.
        
        Args:
            hook_point: Name of the hook (e.g., "attention_weights", "mlp_output")
            layer_idx: Layer index
            data: The tensor/data from the hook
            context: Additional context (request_id, token_idx, is_decode, etc.)
            
        Returns:
            Processed data (can be None to discard)
        """
        pass
        
    def retrieve(
        self, 
        request_id: str,
        format: str = "dict"
    ) -> Any:
        """
        Retrieve collected/processed data for a request.
        
        Args:
            request_id: The request identifier
            format: Output format ("dict", "numpy", "json", etc.)
            
        Returns:
            The collected data in specified format
        """
        return {}
        
    def cleanup(self) -> None:
        """
        Called when actor is removed or session ends.
        Remove hooks, free memory, close files, etc.
        """
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        
    def update_config(self, key: str, value: Any) -> None:
        """
        Update actor configuration at runtime.
        Thread-safe.
        """
        with self._lock:
            old_value = self.config.get(key)
            self.config[key] = value
            self.on_config_change(key, old_value, value)
            
    def on_config_change(
        self, 
        key: str, 
        old_value: Any, 
        new_value: Any
    ) -> None:
        """
        Hook for actors to react to configuration changes.
        Override to implement custom behavior.
        """
        pass
        
    def enable(self) -> None:
        """Enable this actor"""
        with self._lock:
            self._enabled = True
            
    def disable(self) -> None:
        """Disable this actor (stops processing)"""
        with self._lock:
            self._enabled = False
            
    def is_enabled(self) -> bool:
        """Check if actor is currently enabled"""
        return self._enabled
        
    def get_stats(self) -> dict[str, Any]:
        """
        Return statistics about this actor.
        Override to provide custom metrics.
        """
        return {
            "name": self.name,
            "enabled": self._enabled,
            "config": self.config.copy()
        }
```

### 3.3 Session

**Manages a logical grouping of actors:**

```python
class Session:
    """A session groups actors and manages their lifecycle"""
    
    def __init__(
        self, 
        name: str, 
        ttl: int | None = None,
        auto_cleanup: bool = True
    ):
        self.name = name
        self.session_id = generate_session_id()
        self.ttl = ttl  # Seconds
        self.auto_cleanup = auto_cleanup
        self.created_at = time.time()
        self.actors: dict[str, Actor] = {}
        self.data_store: dict[str, Any] = {}
        
    def add_actor(self, actor: Actor) -> str:
        """Add actor to this session"""
        actor_id = f"{self.session_id}:{actor.name}:{uuid.uuid4()}"
        actor.actor_id = actor_id
        self.actors[actor_id] = actor
        return actor_id
        
    def is_expired(self) -> bool:
        """Check if session has exceeded TTL"""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
        
    def cleanup(self) -> None:
        """Cleanup all actors in this session"""
        for actor in self.actors.values():
            actor.cleanup()
        self.actors.clear()
        self.data_store.clear()
```

### 3.4 Pipeline

**Chains actors for multi-stage processing:**

```python
class Pipeline:
    """Chains actors into a processing pipeline"""
    
    def __init__(self, actors: list[Actor]):
        self.actors = actors
        self._validate_pipeline()
        
    def _validate_pipeline(self) -> None:
        """Ensure actors are compatible for chaining"""
        # Check that each actor's output matches next actor's input
        pass
        
    def process(
        self, 
        hook_point: str, 
        layer_idx: int, 
        data: Any,
        context: dict[str, Any]
    ) -> Any:
        """Process data through the pipeline"""
        result = data
        for actor in self.actors:
            if actor.is_enabled():
                result = actor.process(hook_point, layer_idx, result, context)
                if result is None:
                    break  # Actor discarded data
        return result
```

---

## 4. Actor Types & Examples

### 4.1 Collector Actors

**Purpose**: Extract raw data from model internals

#### AttentionCollector

```python
class AttentionCollector(Actor):
    """Collects attention weights with optional windowing"""
    
    def __init__(
        self, 
        layers: list[int],
        window: int | None = None,
        per_head: bool = True
    ):
        super().__init__("attention_collector", {
            "layers": layers,
            "window": window,
            "per_head": per_head
        })
        self.storage: dict[str, dict[int, list]] = {}
        
    def setup(self, model: Any, layer_indices: list[int]) -> None:
        # Register hooks on attention layers
        for idx in self.config["layers"]:
            hook = register_attention_hook(model, idx, self._hook_fn)
            self._hooks.append(hook)
            
    def _hook_fn(self, layer_idx: int, attn_weights: torch.Tensor, context: dict):
        """Called by vLLM during forward pass"""
        return self.process("attention_weights", layer_idx, attn_weights, context)
        
    def process(
        self, 
        hook_point: str,
        layer_idx: int, 
        attn_weights: torch.Tensor,
        context: dict[str, Any]
    ) -> torch.Tensor:
        if not self.is_enabled():
            return attn_weights
            
        request_id = context["request_id"]
        
        # Apply windowing if configured
        if self.config["window"] is not None:
            attn_weights = apply_window(attn_weights, self.config["window"])
            
        # Store
        if request_id not in self.storage:
            self.storage[request_id] = {}
        if layer_idx not in self.storage[request_id]:
            self.storage[request_id][layer_idx] = []
            
        self.storage[request_id][layer_idx].append(
            attn_weights.detach().cpu()
        )
        
        return attn_weights  # Pass through unchanged
        
    def retrieve(self, request_id: str, format: str = "dict") -> dict:
        if request_id not in self.storage:
            return {}
            
        result = {}
        for layer_idx, chunks in self.storage[request_id].items():
            # Concatenate chunks
            concatenated = torch.cat(chunks, dim=1).numpy()
            result[layer_idx] = concatenated
            
        # Cleanup after retrieval
        del self.storage[request_id]
        
        return result
```

#### ActivationCollector

```python
class ActivationCollector(Actor):
    """Collects intermediate activations from specified components"""
    
    def __init__(
        self,
        layers: list[int],
        components: list[str],  # ["mlp", "attention", "residual"]
        reduction: str = "none"  # "none", "mean", "max"
    ):
        super().__init__("activation_collector", {
            "layers": layers,
            "components": components,
            "reduction": reduction
        })
        self.storage: dict[str, dict[str, list]] = {}
        
    def setup(self, model: Any, layer_indices: list[int]) -> None:
        for idx in self.config["layers"]:
            for component in self.config["components"]:
                hook = register_component_hook(
                    model, idx, component, self._hook_fn
                )
                self._hooks.append(hook)
                
    def process(
        self,
        hook_point: str,
        layer_idx: int,
        activations: torch.Tensor,
        context: dict[str, Any]
    ) -> torch.Tensor:
        if not self.is_enabled():
            return activations
            
        request_id = context["request_id"]
        
        # Apply reduction if configured
        if self.config["reduction"] == "mean":
            stored = activations.mean(dim=-1)
        elif self.config["reduction"] == "max":
            stored = activations.max(dim=-1)[0]
        else:
            stored = activations
            
        # Store
        key = f"layer_{layer_idx}_{hook_point}"
        if request_id not in self.storage:
            self.storage[request_id] = {}
        if key not in self.storage[request_id]:
            self.storage[request_id][key] = []
            
        self.storage[request_id][key].append(stored.detach().cpu())
        
        return activations
        
    def retrieve(self, request_id: str, format: str = "dict") -> dict:
        return self.storage.get(request_id, {})
```

#### LogitCollector

```python
class LogitCollector(Actor):
    """Collects logits before sampling"""
    
    def __init__(
        self,
        top_k: int = 50,
        include_scores: bool = True,
        track_entropy: bool = True
    ):
        super().__init__("logit_collector", {
            "top_k": top_k,
            "include_scores": include_scores,
            "track_entropy": track_entropy
        })
        self.storage: dict[str, list] = {}
        
    def process(
        self,
        hook_point: str,
        layer_idx: int,
        logits: torch.Tensor,
        context: dict[str, Any]
    ) -> torch.Tensor:
        if not self.is_enabled():
            return logits
            
        request_id = context["request_id"]
        
        # Get top-k
        top_k = self.config["top_k"]
        values, indices = torch.topk(logits, top_k, dim=-1)
        
        result = {
            "token_idx": context.get("token_idx", 0),
            "top_k_indices": indices.cpu().numpy(),
        }
        
        if self.config["include_scores"]:
            result["top_k_scores"] = values.cpu().numpy()
            
        if self.config["track_entropy"]:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            result["entropy"] = entropy.cpu().item()
            
        if request_id not in self.storage:
            self.storage[request_id] = []
        self.storage[request_id].append(result)
        
        return logits
        
    def retrieve(self, request_id: str, format: str = "dict") -> list:
        return self.storage.get(request_id, [])
```

### 4.2 Processor Actors

**Purpose**: Transform or aggregate data on-the-fly

#### AttentionAnalyzer

```python
class AttentionAnalyzer(Actor):
    """Computes metrics from attention weights in real-time"""
    
    def __init__(
        self,
        metrics: list[str],  # ["entropy", "sparsity", "head_agreement"]
        layers: list[int]
    ):
        super().__init__("attention_analyzer", {
            "metrics": metrics,
            "layers": layers
        })
        self.results: dict[str, dict[int, list]] = {}
        
    def process(
        self,
        hook_point: str,
        layer_idx: int,
        attn_weights: torch.Tensor,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.is_enabled():
            return None
            
        request_id = context["request_id"]
        metrics = {}
        
        if "entropy" in self.config["metrics"]:
            # Compute entropy per head
            entropy = compute_attention_entropy(attn_weights)
            metrics["entropy"] = entropy.cpu().numpy()
            
        if "sparsity" in self.config["metrics"]:
            # Compute sparsity (% of weights below threshold)
            sparsity = compute_sparsity(attn_weights, threshold=0.01)
            metrics["sparsity"] = sparsity.cpu().numpy()
            
        if "head_agreement" in self.config["metrics"]:
            # Measure how similar different heads are
            agreement = compute_head_agreement(attn_weights)
            metrics["head_agreement"] = agreement.cpu().item()
            
        # Store
        if request_id not in self.results:
            self.results[request_id] = {}
        if layer_idx not in self.results[request_id]:
            self.results[request_id][layer_idx] = []
            
        self.results[request_id][layer_idx].append({
            "token_idx": context.get("token_idx", 0),
            "metrics": metrics
        })
        
        return metrics
        
    def retrieve(self, request_id: str, format: str = "dict") -> dict:
        return self.results.get(request_id, {})
```

#### AttentionSummarizer

```python
class AttentionSummarizer(Actor):
    """Aggregates attention with configurable strategies"""
    
    def __init__(
        self,
        strategy: str = "running_average",  # Or "peak_only", "downsample"
        window: int = 10,
        layers: list[int] = None
    ):
        super().__init__("attention_summarizer", {
            "strategy": strategy,
            "window": window,
            "layers": layers or []
        })
        self.buffer: dict[str, dict[int, list]] = {}
        
    def process(
        self,
        hook_point: str,
        layer_idx: int,
        attn_weights: torch.Tensor,
        context: dict[str, Any]
    ) -> torch.Tensor | None:
        if not self.is_enabled():
            return None
            
        request_id = context["request_id"]
        strategy = self.config["strategy"]
        
        if strategy == "running_average":
            # Keep sliding window
            if request_id not in self.buffer:
                self.buffer[request_id] = {}
            if layer_idx not in self.buffer[request_id]:
                self.buffer[request_id][layer_idx] = []
                
            buffer = self.buffer[request_id][layer_idx]
            buffer.append(attn_weights)
            
            if len(buffer) > self.config["window"]:
                buffer.pop(0)
                
            # Return average
            return torch.stack(buffer).mean(dim=0)
            
        elif strategy == "peak_only":
            # Only keep high-entropy attention
            entropy = compute_attention_entropy(attn_weights).mean()
            threshold = self.config.get("entropy_threshold", 2.0)
            
            if entropy > threshold:
                return attn_weights
            return None  # Discard
            
        elif strategy == "downsample":
            # Keep every Nth token
            token_idx = context.get("token_idx", 0)
            rate = self.config.get("downsample_rate", 10)
            
            if token_idx % rate == 0:
                return attn_weights
            return None
            
        return None
        
    def on_config_change(self, key: str, old_value: Any, new_value: Any):
        """React to strategy changes"""
        if key == "strategy":
            # Clear buffer when strategy changes
            self.buffer.clear()
```

### 4.3 Detector Actors

**Purpose**: Identify patterns and trigger alerts

#### PatternDetector

```python
class PatternDetector(Actor):
    """Detects specific patterns in attention and triggers callbacks"""
    
    def __init__(
        self,
        patterns: dict[str, callable],
        on_detect: callable,
        layers: list[int]
    ):
        """
        Args:
            patterns: Dict of pattern_name -> detection_function
            on_detect: Callback(pattern_name, layer_idx, context) when detected
            layers: Which layers to monitor
        """
        super().__init__("pattern_detector", {
            "patterns": patterns,
            "layers": layers
        })
        self.on_detect = on_detect
        self.detections: dict[str, list] = {}
        
    def process(
        self,
        hook_point: str,
        layer_idx: int,
        attn_weights: torch.Tensor,
        context: dict[str, Any]
    ) -> None:
        if not self.is_enabled():
            return None
            
        request_id = context["request_id"]
        
        # Check each pattern
        for pattern_name, detector_fn in self.config["patterns"].items():
            detected = detector_fn(attn_weights, context)
            
            if detected:
                # Log detection
                if request_id not in self.detections:
                    self.detections[request_id] = []
                    
                self.detections[request_id].append({
                    "pattern": pattern_name,
                    "layer": layer_idx,
                    "token_idx": context.get("token_idx", 0),
                    "timestamp": time.time()
                })
                
                # Trigger callback
                try:
                    self.on_detect(pattern_name, layer_idx, context)
                except Exception as e:
                    logger.error(f"Pattern detector callback failed: {e}")
                    
        return None  # Doesn't modify data
        
    def retrieve(self, request_id: str, format: str = "dict") -> list:
        return self.detections.get(request_id, [])


# Example pattern detectors
def detect_repetition_loop(attn_weights: torch.Tensor, context: dict) -> bool:
    """Detect if model is stuck in repetition (attending to same tokens)"""
    # Check if attention is concentrated on recent few tokens
    recent_attn = attn_weights[:, -1, -5:].sum()  # Last token's attention to last 5
    return recent_attn > 0.8  # >80% attention to recent 5 tokens

def detect_context_neglect(attn_weights: torch.Tensor, context: dict) -> bool:
    """Detect if model is ignoring the prompt/context"""
    prompt_len = context.get("prompt_length", 0)
    if prompt_len > 0:
        prompt_attn = attn_weights[:, -1, :prompt_len].sum()
        return prompt_attn < 0.1  # <10% attention to prompt

def detect_high_uncertainty(attn_weights: torch.Tensor, context: dict) -> bool:
    """Detect high uncertainty (uniform attention distribution)"""
    entropy = compute_attention_entropy(attn_weights).mean()
    return entropy > 3.0  # High entropy threshold
```

### 4.4 Storage Actors

**Purpose**: Persist data with custom strategies

#### SmartArchiver

```python
class SmartArchiver(Actor):
    """Stores data based on configurable rules"""
    
    def __init__(
        self,
        path: str,
        format: str = "hdf5",  # Or "numpy", "json", "parquet"
        compression: bool = True,
        storage_rule: str = "all",  # Or "conditional"
        condition: callable | None = None
    ):
        """
        Args:
            path: Where to store files
            format: Storage format
            compression: Whether to compress
            storage_rule: "all" or "conditional"
            condition: Function(data, context) -> bool for conditional storage
        """
        super().__init__("smart_archiver", {
            "path": path,
            "format": format,
            "compression": compression,
            "storage_rule": storage_rule,
            "condition": condition
        })
        self.buffer: dict[str, list] = {}
        
    def process(
        self,
        hook_point: str,
        layer_idx: int,
        data: Any,
        context: dict[str, Any]
    ) -> None:
        if not self.is_enabled():
            return None
            
        request_id = context["request_id"]
        
        # Check storage rule
        should_store = True
        if self.config["storage_rule"] == "conditional":
            condition = self.config["condition"]
            if condition is not None:
                should_store = condition(data, context)
                
        if not should_store:
            return None
            
        # Buffer for batch writing
        if request_id not in self.buffer:
            self.buffer[request_id] = []
            
        self.buffer[request_id].append({
            "hook_point": hook_point,
            "layer_idx": layer_idx,
            "data": data,
            "context": context
        })
        
        # Flush if buffer is large
        if len(self.buffer[request_id]) > 100:
            self._flush(request_id)
            
        return None
        
    def _flush(self, request_id: str) -> None:
        """Write buffered data to disk"""
        if request_id not in self.buffer:
            return
            
        format = self.config["format"]
        path = Path(self.config["path"]) / f"{request_id}.{format}"
        
        if format == "hdf5":
            self._write_hdf5(path, self.buffer[request_id])
        elif format == "numpy":
            self._write_numpy(path, self.buffer[request_id])
        elif format == "json":
            self._write_json(path, self.buffer[request_id])
            
        self.buffer[request_id].clear()
        
    def cleanup(self) -> None:
        """Flush all remaining buffers"""
        for request_id in list(self.buffer.keys()):
            self._flush(request_id)
        super().cleanup()
```

---

## 5. Control Plane Design

### 5.1 REST API

**Endpoints for external control:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# === Session Management ===

class SessionCreate(BaseModel):
    name: str
    ttl: int | None = None
    
@app.post("/sessions")
def create_session(req: SessionCreate):
    """Create a new instrumentation session"""
    session = lens.create_session(req.name, req.ttl)
    return {"session_id": session.session_id, "name": session.name}

@app.get("/sessions")
def list_sessions():
    """List all active sessions"""
    return {"sessions": lens.list_sessions()}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Terminate a session and cleanup actors"""
    lens.delete_session(session_id)
    return {"status": "deleted"}


# === Actor Management ===

class ActorCreate(BaseModel):
    type: str  # "attention_collector", "attention_analyzer", etc.
    config: dict
    session_id: str | None = None
    
@app.post("/actors")
def create_actor(req: ActorCreate):
    """Create and register a new actor"""
    actor = actor_factory(req.type, req.config)
    actor_id = lens.add_actor(actor, session_id=req.session_id)
    return {"actor_id": actor_id}

@app.get("/actors")
def list_actors():
    """List all active actors"""
    return {"actors": lens.list_actors()}

@app.get("/actors/{actor_id}")
def get_actor_info(actor_id: str):
    """Get actor status and configuration"""
    actor = lens.get_actor(actor_id)
    return actor.get_stats()

@app.delete("/actors/{actor_id}")
def delete_actor(actor_id: str):
    """Remove actor and cleanup"""
    lens.remove_actor(actor_id)
    return {"status": "deleted"}


# === Actor Control ===

class ActorConfig(BaseModel):
    config: dict[str, Any]
    
@app.patch("/actors/{actor_id}/config")
def update_actor_config(actor_id: str, req: ActorConfig):
    """Update actor configuration at runtime"""
    actor = lens.get_actor(actor_id)
    for key, value in req.config.items():
        actor.update_config(key, value)
    return {"status": "updated", "config": actor.config}

@app.post("/actors/{actor_id}/enable")
def enable_actor(actor_id: str):
    """Enable a disabled actor"""
    actor = lens.get_actor(actor_id)
    actor.enable()
    return {"status": "enabled"}

@app.post("/actors/{actor_id}/disable")
def disable_actor(actor_id: str):
    """Disable an actor (stops processing)"""
    actor = lens.get_actor(actor_id)
    actor.disable()
    return {"status": "disabled"}


# === Data Retrieval ===

@app.get("/data/{request_id}")
def get_request_data(request_id: str, actor_id: str | None = None):
    """Retrieve collected data for a request"""
    data = lens.get_data(request_id, actor_id)
    return {"request_id": request_id, "data": data}

@app.get("/data/{request_id}/{actor_id}")
def get_actor_request_data(request_id: str, actor_id: str):
    """Retrieve data from specific actor for a request"""
    actor = lens.get_actor(actor_id)
    data = actor.retrieve(request_id)
    return {"request_id": request_id, "actor_id": actor_id, "data": data}


# === Pipeline Management ===

class PipelineCreate(BaseModel):
    name: str
    actor_ids: list[str]
    
@app.post("/pipelines")
def create_pipeline(req: PipelineCreate):
    """Create a processing pipeline from actors"""
    actors = [lens.get_actor(aid) for aid in req.actor_ids]
    pipeline = lens.pipeline(actors)
    return {"pipeline_id": pipeline.id, "actors": req.actor_ids}
```

### 5.2 WebSocket API (Real-Time Streaming)

```python
from fastapi import WebSocket

@app.websocket("/ws/actors/{actor_id}/stream")
async def stream_actor_output(websocket: WebSocket, actor_id: str):
    """Stream actor output in real-time"""
    await websocket.accept()
    
    actor = lens.get_actor(actor_id)
    
    try:
        # Subscribe to actor's output stream
        async for update in actor.stream():
            await websocket.send_json({
                "timestamp": time.time(),
                "actor_id": actor_id,
                "data": update
            })
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()


@app.websocket("/ws/sessions/{session_id}/stream")
async def stream_session(websocket: WebSocket, session_id: str):
    """Stream all actor outputs from a session"""
    await websocket.accept()
    
    session = lens.get_session(session_id)
    
    try:
        # Multiplex outputs from all actors in session
        async for actor_id, update in session.stream():
            await websocket.send_json({
                "timestamp": time.time(),
                "session_id": session_id,
                "actor_id": actor_id,
                "data": update
            })
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()
```

### 5.3 Python API (Programmatic Control)

```python
# Example: Interactive notebook usage
from vllm_lens import LensManager, AttentionCollector, AttentionAnalyzer

# Initialize
llm = LLM(model="gpt2")
lens = LensManager(llm)

# Create session
session = lens.create_session("my-research", ttl=3600)

# Add actors
collector = AttentionCollector(layers=[0, 1, 2], window=10)
analyzer = AttentionAnalyzer(metrics=["entropy", "sparsity"], layers=[0, 1, 2])

lens.add_actor(collector, session)
lens.add_actor(analyzer, session)

# Generate
outputs = llm.generate("Hello world", max_tokens=50)
request_id = outputs[0].request_id

# Retrieve data
attention = collector.retrieve(request_id)
metrics = analyzer.retrieve(request_id)

# Mid-generation control
lens.get_actor(collector.actor_id).update_config("window", 20)

# Cleanup
session.cleanup()
```

---

## 6. Data Flow & Lifecycle

### 6.1 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Forward Pass                         │
│                                                              │
│  Token Generation Loop:                                      │
│    for token in range(max_tokens):                          │
│      1. Attention Computation                                │
│      2. MLP Forward                                          │
│      3. Logits Computation                                   │
│      4. Sampling                                             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ Hooks Triggered
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Lens Manager Router                        │
│  - Identifies hook point                                     │
│  - Gathers context (request_id, token_idx, layer_idx)        │
│  - Routes to relevant actors                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Actor Processing                           │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Collector  │  │  Analyzer   │  │  Detector   │        │
│  │  - Store    │  │  - Compute  │  │  - Check    │        │
│  │    raw data │  │    metrics  │  │    patterns │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                 │
│         │                │                │                 │
│         ▼                ▼                ▼                 │
│    ┌─────────────────────────────────────────┐             │
│    │       Per-Request Storage               │             │
│    │  {request_id: {actor_id: data}}         │             │
│    └─────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
                  │
                  │ On Request Complete
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Retrieval                            │
│  - User calls lens.get_data(request_id)                      │
│  - Or actor.retrieve(request_id)                             │
│  - Data returned in requested format                         │
│  - Optional: Auto-cleanup after retrieval                    │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Actor Lifecycle

```
┌──────────────┐
│  Creation    │  actor = AttentionCollector(...)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Registration │  lens.add_actor(actor)
│              │  - Assigns actor_id
│              │  - Calls actor.setup(model)
│              │  - Registers hooks
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Active     │  ◄───┐ Config updates
│              │      │ Enable/disable
│  Processing  │ ─────┘ Data collection
└──────┬───────┘
       │
       │ On session end / explicit removal / TTL expired
       ▼
┌──────────────┐
│   Cleanup    │  actor.cleanup()
│              │  - Remove hooks
│              │  - Free memory
│              │  - Close files
└──────────────┘
```

### 6.3 Session Lifecycle

```
┌──────────────┐
│   Created    │  session = lens.create_session("name", ttl=3600)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Active     │  ◄───┐ Add/remove actors
│              │      │ Process data
│  Collecting  │ ─────┘ Configuration
└──────┬───────┘
       │
       │ TTL expired / explicit cleanup / context manager exit
       ▼
┌──────────────┐
│  Expiration  │  Check session.is_expired()
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Cleanup    │  session.cleanup()
│              │  - Cleanup all actors
│              │  - Clear data store
│              │  - Remove from manager
└──────────────┘
```

---

## 7. Performance Considerations

### 7.1 Overhead Analysis

**Sources of Overhead:**

1. **Hook Invocation**: Function call overhead per hook per token
2. **Data Copying**: Moving tensors from GPU to CPU
3. **Processing Logic**: Actor computation (metrics, pattern detection)
4. **Storage**: Writing to memory/disk
5. **Synchronization**: Locks for thread-safe config updates

**Mitigation Strategies:**

```python
# 1. Conditional Processing
class Actor:
    def process(self, hook_point, layer_idx, data, context):
        if not self.is_enabled():
            return data  # Fast path: no-op when disabled
        
        # Only process if this actor cares about this hook point
        if hook_point not in self.interested_hooks:
            return data
            
        # Main processing logic
        ...

# 2. Lazy Copying
class AttentionCollector:
    def process(self, hook_point, layer_idx, attn_weights, context):
        # Don't copy to CPU immediately
        # Store GPU reference and defer copying
        self.gpu_buffer.append((attn_weights, context))
        
        # Batch copy to CPU in background thread
        if len(self.gpu_buffer) > batch_size:
            self.async_copy_to_cpu()

# 3. Sampling
class ActivationCollector:
    def __init__(self, sample_rate: float = 0.1):
        self.sample_rate = sample_rate
        
    def process(self, hook_point, layer_idx, data, context):
        # Only collect 10% of tokens
        if random.random() > self.sample_rate:
            return data
        ...

# 4. In-Place Computation
class AttentionAnalyzer:
    def process(self, hook_point, layer_idx, attn_weights, context):
        # Compute metrics without allocating new tensors
        with torch.no_grad():
            entropy = self._compute_entropy_inplace(attn_weights)
        return entropy  # Return scalar, not full tensor

# 5. Lock-Free Config Updates
class Actor:
    def __init__(self):
        # Use atomic types for frequently-read config
        self._enabled = AtomicBool(True)
        self._sample_rate = AtomicFloat(1.0)
        
    def is_enabled(self):
        return self._enabled.load()  # No lock needed
```

### 7.2 Memory Management

**Memory Budget System:**

```python
class LensManager:
    def __init__(
        self, 
        llm: Any, 
        max_memory_mb: int = 1024
    ):
        self.max_memory_mb = max_memory_mb
        self.memory_tracker = MemoryTracker()
        
    def add_actor(self, actor: Actor) -> str:
        # Estimate memory for this actor
        estimated_mb = actor.estimate_memory()
        
        current_usage = self.memory_tracker.current_usage_mb()
        if current_usage + estimated_mb > self.max_memory_mb:
            # Evict least-recently-used data
            self._evict_lru(estimated_mb)
            
        actor_id = self._register_actor(actor)
        self.memory_tracker.register(actor_id, estimated_mb)
        return actor_id
        
    def _evict_lru(self, needed_mb: int):
        """Evict least-recently-used data to free memory"""
        freed = 0
        for request_id in self.lru_requests:
            for actor_id in self.actors:
                actor = self.actors[actor_id]
                freed += actor.evict_request(request_id)
                
            if freed >= needed_mb:
                break
```

**Per-Actor Memory Estimation:**

```python
class AttentionCollector(Actor):
    def estimate_memory(self) -> int:
        """Estimate memory usage in MB"""
        # Attention: [num_heads, num_tokens, window]
        num_heads = 12  # Model-specific
        max_tokens = 1024
        window = self.config.get("window") or max_tokens
        bytes_per_element = 4  # float32
        
        # Per token per layer
        per_token_mb = (num_heads * window * bytes_per_element) / (1024**2)
        
        # Total for expected generation length
        num_layers = len(self.config["layers"])
        estimated_mb = per_token_mb * max_tokens * num_layers
        
        # Add buffer for overhead
        return estimated_mb * 1.2
```

### 7.3 Benchmarking Framework

```python
class PerformanceBenchmark:
    """Measure actor overhead"""
    
    @staticmethod
    def benchmark_actor(
        actor: Actor, 
        model: Any,
        num_iterations: int = 100
    ) -> dict:
        """Measure actor overhead"""
        
        # Baseline: generation without actor
        start = time.time()
        outputs_baseline = generate_without_actor(model, num_iterations)
        baseline_time = time.time() - start
        
        # With actor
        lens.add_actor(actor)
        start = time.time()
        outputs_with_actor = generate_with_actor(model, num_iterations)
        actor_time = time.time() - start
        lens.remove_actor(actor.actor_id)
        
        overhead_pct = ((actor_time - baseline_time) / baseline_time) * 100
        
        return {
            "baseline_time": baseline_time,
            "actor_time": actor_time,
            "overhead_pct": overhead_pct,
            "overhead_per_token_ms": ((actor_time - baseline_time) / (num_iterations * 50)) * 1000
        }
```

**Expected Overhead (Rough Estimates):**

| Actor Type | Per-Token Overhead | Notes |
|-----------|-------------------|-------|
| AttentionCollector (no window) | 5-10ms | GPU→CPU copy dominates |
| AttentionCollector (window=10) | 0.5-1ms | Much smaller copy |
| AttentionAnalyzer | 1-2ms | Compute metrics only |
| LogitCollector (top-k=50) | 0.2-0.5ms | Minimal overhead |
| PatternDetector | 0.5-1ms | Depends on pattern complexity |
| SmartArchiver (buffered) | 0.1-0.3ms | Async writes |

**Target**: Keep total overhead <10% for typical configurations

---

## 8. API Specifications

### 8.1 Actor Configuration Schema

```python
from pydantic import BaseModel, Field

class AttentionCollectorConfig(BaseModel):
    layers: list[int] = Field(..., description="Layer indices to collect")
    window: int | None = Field(None, description="Attention window size (None = full)")
    per_head: bool = Field(True, description="Store per-head or averaged")
    
class AttentionAnalyzerConfig(BaseModel):
    layers: list[int]
    metrics: list[str] = Field(
        ["entropy", "sparsity"], 
        description="Metrics to compute"
    )
    
class PatternDetectorConfig(BaseModel):
    layers: list[int]
    patterns: list[str] = Field(
        ["repetition", "context_neglect"],
        description="Which patterns to detect"
    )
    alert_webhook: str | None = Field(None, description="Webhook URL for alerts")
```

### 8.2 Data Format Specifications

**Attention Data Format:**

```json
{
  "request_id": "0-abc123",
  "actor_id": "session-1:attention_collector:uuid",
  "layers": {
    "0": {
      "shape": [12, 50, 10],
      "dtype": "float32",
      "data": "<base64 or numpy array>",
      "metadata": {
        "num_heads": 12,
        "num_tokens": 50,
        "window_size": 10
      }
    }
  }
}
```

**Metrics Data Format:**

```json
{
  "request_id": "0-abc123",
  "actor_id": "session-1:attention_analyzer:uuid",
  "metrics": {
    "layer_0": [
      {
        "token_idx": 0,
        "entropy": 2.34,
        "sparsity": 0.67,
        "head_agreement": 0.82
      },
      {
        "token_idx": 1,
        "entropy": 2.41,
        "sparsity": 0.64,
        "head_agreement": 0.85
      }
    ]
  }
}
```

**Pattern Detection Format:**

```json
{
  "request_id": "0-abc123",
  "actor_id": "session-1:pattern_detector:uuid",
  "detections": [
    {
      "pattern": "repetition_loop",
      "layer": 3,
      "token_idx": 45,
      "timestamp": 1745250000.123,
      "metadata": {
        "self_attention_pct": 0.85
      }
    }
  ]
}
```

---

## 9. Implementation Roadmap

### Phase 1: Foundation (4-6 weeks)

**Goal**: Core infrastructure + migrate existing attention capture

**Tasks:**
1. ✅ Design Actor base class and interface
2. ✅ Implement LensManager
3. ✅ Implement Session management
4. ✅ Build hook coordination system
5. ✅ Refactor AttentionCollector to use new framework
6. ✅ Add backward compatibility layer
7. ✅ Write comprehensive tests
8. ✅ Performance benchmarking

**Deliverables:**
- `vllm_lens` core package
- `AttentionCollector` as first actor
- Basic session management
- Test suite with >80% coverage
- Performance baseline

### Phase 2: Control Plane (3-4 weeks)

**Goal**: Enable external control and real-time updates

**Tasks:**
1. ✅ Implement REST API (FastAPI)
2. ✅ Implement WebSocket streaming
3. ✅ Add runtime configuration updates
4. ✅ Build simple web dashboard (React)
5. ✅ Add authentication/authorization
6. ✅ Write API documentation (OpenAPI)

**Deliverables:**
- REST API server
- WebSocket real-time streaming
- Basic web UI for control
- API documentation

### Phase 3: Additional Actors (4-6 weeks)

**Goal**: Expand beyond attention to other model internals

**Tasks:**
1. ✅ Implement ActivationCollector
2. ✅ Implement LogitCollector
3. ✅ Implement AttentionAnalyzer
4. ✅ Implement AttentionSummarizer
5. ✅ Implement PatternDetector
6. ✅ Add pre-built pattern detectors
7. ✅ Write actor cookbook/examples

**Deliverables:**
- 5+ production-ready actors
- Pattern detector library
- Cookbook with examples

### Phase 4: Advanced Features (4-6 weeks)

**Goal**: Production-grade features and optimizations

**Tasks:**
1. ✅ Pipeline/chaining support
2. ✅ Memory management with LRU eviction
3. ✅ Distributed collection (multi-GPU)
4. ✅ Advanced storage backends (HDF5, Parquet)
5. ✅ Integration with observability tools (Prometheus)
6. ✅ Performance optimizations (async, batching)
7. ✅ Comprehensive documentation

**Deliverables:**
- Pipeline API
- Memory-bounded operation
- Multi-GPU support
- Integration guides
- Full documentation site

### Phase 5: Ecosystem (Ongoing)

**Goal**: Community contributions and extensions

**Tasks:**
1. ✅ Plugin system for custom actors
2. ✅ Actor marketplace/registry
3. ✅ Jupyter integration
4. ✅ VSCode extension (visualize live)
5. ✅ Tutorials and workshops
6. ✅ Community contributions

**Deliverables:**
- Plugin SDK
- VSCode extension
- Tutorial series
- Active community

---

## 10. Open Questions & Trade-offs

### 10.1 Design Decisions Requiring Input

#### Q1: Hook Registration Strategy

**Option A: Layer-level hooks**
- Pro: Simple, one hook per layer
- Con: Coarse-grained, can't instrument sub-components separately

**Option B: Component-level hooks**
- Pro: Fine-grained (attention, MLP, residual separately)
- Con: More complex registration, higher overhead

**Recommendation**: Start with Option A, add Option B later if needed

---

#### Q2: Data Storage Location

**Option A: In-memory only**
- Pro: Fast, no I/O overhead
- Con: Memory pressure, data lost on crash

**Option B: Hybrid (memory + disk)**
- Pro: Durable, can handle large datasets
- Con: I/O overhead, complexity

**Option C: Configurable per-actor**
- Pro: Flexibility
- Con: Complexity

**Recommendation**: Option C - let actors choose their storage strategy

---

#### Q3: Thread Safety Model

**Option A: Coarse-grained locking** (one lock per actor)
- Pro: Simple, safe
- Con: Contention on config updates during generation

**Option B: Fine-grained locking** (per-field locks)
- Pro: Better concurrency
- Con: Complex, risk of deadlocks

**Option C: Lock-free** (atomic types, copy-on-write)
- Pro: Best performance
- Con: Most complex, harder to debug

**Recommendation**: Start with Option A, profile, optimize to B/C if needed

---

#### Q4: Actor Discovery

**Option A: Manual registration** (user explicitly creates actors)
- Pro: Explicit, clear
- Con: Verbose

**Option B: Auto-discovery** (scan for actor classes, register automatically)
- Pro: Convenient
- Con: Implicit, harder to understand

**Option C: Plugin system** (actors in separate packages)
- Pro: Extensible, decoupled
- Con: Complexity

**Recommendation**: Option A for core, Option C for extensions

---

### 10.2 Performance Trade-offs

| Approach | Latency | Memory | Flexibility |
|----------|---------|---------|-------------|
| Collect everything, process later | Low | High | High |
| Process during collection | Medium | Low | Medium |
| Adaptive (switch based on load) | Variable | Medium | High |

**Recommendation**: Support multiple modes, let users choose

---

### 10.3 API Design Choices

#### Return Values from `process()`

**Option A: Return processed data**
```python
def process(self, hook_point, layer_idx, data, context) -> Any:
    return transformed_data  # Next actor receives this
```
- Pro: Enables pipelines
- Con: Must always return something

**Option B: Store internally, return None**
```python
def process(self, hook_point, layer_idx, data, context) -> None:
    self.storage[request_id].append(data)
    return None
```
- Pro: Simpler for actors that just collect
- Con: Can't chain

**Recommendation**: Option A, with convention that returning None breaks pipeline

---

### 10.4 vLLM Version Compatibility

**Challenge**: vLLM API changes frequently

**Mitigation**:
1. Abstract hook registration behind version-agnostic interface
2. Detect vLLM version at runtime
3. Use adapter pattern for version-specific code
4. Maintain compatibility matrix

```python
class HookAdapter:
    @staticmethod
    def register_attention_hook(model, layer_idx, callback):
        vllm_version = detect_vllm_version()
        
        if vllm_version.startswith("0."):
            # v0 API
            return register_v0_hook(model, layer_idx, callback)
        elif vllm_version.startswith("1."):
            # v1 API
            return register_v1_hook(model, layer_idx, callback)
        else:
            raise UnsupportedVersionError(vllm_version)
```

---

## 11. Migration Path

### 11.1 Backward Compatibility

**Existing Code:**
```python
from vllm_attention_capture_plugin import (
    enable_attention_capture,
    get_attention_scores
)

enable_attention_capture(llm, capture_layers=[0, 1, 2])
outputs = llm.generate("Hello")
scores = get_attention_scores(outputs[0].request_id)
```

**Strategy**: Keep existing API working via adapter

```python
# vllm_attention_capture_plugin/__init__.py

# Import new framework
from vllm_lens import LensManager, AttentionCollector

# Global state (for backward compat)
_legacy_managers = {}

def enable_attention_capture(
    llm, 
    capture_layers=None, 
    attention_window=None,
    auto_clear=True
):
    """Legacy API - wraps new framework"""
    
    # Create manager if doesn't exist
    llm_id = id(llm)
    if llm_id not in _legacy_managers:
        _legacy_managers[llm_id] = LensManager(llm)
    
    manager = _legacy_managers[llm_id]
    
    # Create session
    session = manager.create_session("legacy", ttl=None)
    
    # Create collector actor
    actor = AttentionCollector(
        layers=capture_layers or [],
        window=attention_window
    )
    
    manager.add_actor(actor, session)
    
    # Store for retrieval
    _legacy_managers[llm_id]["actor"] = actor

def get_attention_scores(request_id: str):
    """Legacy API - wraps new framework"""
    for manager_data in _legacy_managers.values():
        actor = manager_data.get("actor")
        if actor:
            data = actor.retrieve(request_id)
            if data:
                return data
    return None
```

### 11.2 Gradual Migration Guide

**Step 1**: Use existing API (no changes)
```python
enable_attention_capture(llm, capture_layers=[0, 1, 2])
```

**Step 2**: Opt-in to new API for new features
```python
from vllm_lens import LensManager, AttentionCollector, AttentionAnalyzer

lens = LensManager(llm)
lens.add_actor(AttentionCollector(layers=[0, 1, 2]))
lens.add_actor(AttentionAnalyzer(metrics=["entropy"]))  # New!
```

**Step 3**: Full migration to new API
```python
# Remove old imports
# from vllm_attention_capture_plugin import enable_attention_capture

# Use new framework exclusively
from vllm_lens import LensManager
```

---

## 12. Future Extensions

### 12.1 Distributed Collection

**Use Case**: Multi-GPU inference with data parallel

**Design**:
```python
class DistributedActor(Actor):
    """Actor that aggregates data across GPU ranks"""
    
    def __init__(self, base_actor: Actor, aggregation: str = "concat"):
        self.base_actor = base_actor
        self.aggregation = aggregation  # "concat", "mean", "first"
        self.rank = get_rank()
        
    def process(self, hook_point, layer_idx, data, context):
        # Process on this rank
        local_result = self.base_actor.process(hook_point, layer_idx, data, context)
        
        # Gather from all ranks
        if self.rank == 0:
            all_results = gather_from_all_ranks(local_result)
            
            if self.aggregation == "concat":
                return torch.cat(all_results, dim=0)
            elif self.aggregation == "mean":
                return torch.stack(all_results).mean(dim=0)
                
        return None
```

### 12.2 Causal Intervention Actors

**Use Case**: Modify model behavior by intervening on internal states

```python
class InterventionActor(Actor):
    """Modifies internal states during forward pass"""
    
    def __init__(
        self, 
        layer_idx: int,
        intervention_fn: callable
    ):
        self.layer_idx = layer_idx
        self.intervention_fn = intervention_fn
        
    def process(self, hook_point, layer_idx, data, context):
        if layer_idx == self.layer_idx:
            # Modify attention weights
            modified = self.intervention_fn(data, context)
            return modified  # vLLM will use modified weights
        return data

# Example: Force model to attend to first token
def force_first_token_attention(attn_weights, context):
    # Set all attention to first token
    modified = torch.zeros_like(attn_weights)
    modified[:, :, 0] = 1.0
    return modified

actor = InterventionActor(layer_idx=5, intervention_fn=force_first_token_attention)
```

### 12.3 Comparative Analysis Actors

**Use Case**: Compare two generations side-by-side

```python
class ComparisonActor(Actor):
    """Compares metrics across two requests"""
    
    def __init__(self, base_actor: Actor):
        self.base_actor = base_actor
        self.requests = {}
        
    def compare(self, request_id_1: str, request_id_2: str):
        data_1 = self.base_actor.retrieve(request_id_1)
        data_2 = self.base_actor.retrieve(request_id_2)
        
        return {
            "differences": compute_differences(data_1, data_2),
            "similarity": compute_similarity(data_1, data_2),
            "divergence_point": find_divergence(data_1, data_2)
        }
```

### 12.4 Auto-Tuning Actors

**Use Case**: Automatically adjust model behavior based on patterns

```python
class AutoTuneActor(Actor):
    """Adjusts generation parameters based on observed patterns"""
    
    def __init__(self, detector: PatternDetector):
        self.detector = detector
        
    def process(self, hook_point, layer_idx, data, context):
        # Check for patterns
        detections = self.detector.process(hook_point, layer_idx, data, context)
        
        # Adjust generation parameters
        if "repetition_loop" in detections:
            # Increase temperature to break loop
            context["sampling_params"].temperature *= 1.2
            
        if "high_uncertainty" in detections:
            # Increase top_k to consider more options
            context["sampling_params"].top_k += 10
            
        return data
```

### 12.5 Visualization Actors

**Use Case**: Generate visualizations on-the-fly

```python
class VisualizationActor(Actor):
    """Generates visualizations during generation"""
    
    def __init__(self, viz_type: str = "heatmap"):
        self.viz_type = viz_type
        self.output_dir = Path("visualizations")
        
    def process(self, hook_point, layer_idx, attn_weights, context):
        request_id = context["request_id"]
        token_idx = context["token_idx"]
        
        # Generate viz
        if self.viz_type == "heatmap":
            fig = plot_attention_heatmap(attn_weights)
            fig.savefig(self.output_dir / f"{request_id}_layer{layer_idx}_token{token_idx}.png")
            
        return attn_weights
```

---

## 13. Security & Privacy Considerations

### 13.1 Authentication & Authorization

**Requirements**:
- Secure API endpoints (API keys or OAuth)
- Role-based access control (RBAC)
- Audit logging of actor operations

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

def verify_token(credentials = Depends(security)):
    token = credentials.credentials
    if not is_valid_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return get_user_from_token(token)

@app.post("/actors")
def create_actor(req: ActorCreate, user = Depends(verify_token)):
    # Check user has permission to create actors
    if not user.has_permission("actor:create"):
        raise HTTPException(status_code=403, detail="Forbidden")
    ...
```

### 13.2 Data Privacy

**Concerns**:
- Collected data may contain sensitive information
- Need secure storage and transmission
- Compliance with data regulations (GDPR, etc.)

**Mitigations**:
1. **Encryption at rest**: Encrypt stored data
2. **Encryption in transit**: Use TLS for API
3. **Data anonymization**: Strip PII from collected data
4. **Automatic expiration**: Delete data after TTL
5. **Access controls**: Restrict who can retrieve data

```python
class PrivacyAwareActor(Actor):
    """Actor with built-in privacy protections"""
    
    def __init__(self, anonymize: bool = True, encryption: bool = True):
        self.anonymize = anonymize
        self.encryption = encryption
        
    def process(self, hook_point, layer_idx, data, context):
        # Collect data
        collected = self._collect(data, context)
        
        # Anonymize if enabled
        if self.anonymize:
            collected = self._anonymize(collected)
            
        # Encrypt if enabled
        if self.encryption:
            collected = self._encrypt(collected)
            
        return collected
```

---

## 14. Testing Strategy

### 14.1 Unit Tests

```python
def test_actor_lifecycle():
    """Test actor creation, registration, cleanup"""
    actor = AttentionCollector(layers=[0])
    assert actor.is_enabled()
    
    actor.disable()
    assert not actor.is_enabled()
    
    actor.cleanup()
    assert len(actor._hooks) == 0

def test_config_updates():
    """Test runtime configuration changes"""
    actor = AttentionCollector(layers=[0], window=10)
    assert actor.config["window"] == 10
    
    actor.update_config("window", 20)
    assert actor.config["window"] == 20

def test_data_retrieval():
    """Test data collection and retrieval"""
    actor = AttentionCollector(layers=[0])
    
    # Simulate processing
    mock_data = torch.randn(12, 5, 10)
    actor.process("attention_weights", 0, mock_data, {"request_id": "test"})
    
    # Retrieve
    data = actor.retrieve("test")
    assert "0" in data
    assert data["0"].shape == (12, 5, 10)
```

### 14.2 Integration Tests

```python
def test_end_to_end():
    """Test full pipeline with vLLM"""
    llm = LLM(model="gpt2")
    lens = LensManager(llm)
    
    # Create session
    session = lens.create_session("test")
    
    # Add actors
    collector = AttentionCollector(layers=[0])
    analyzer = AttentionAnalyzer(metrics=["entropy"], layers=[0])
    
    lens.add_actor(collector, session)
    lens.add_actor(analyzer, session)
    
    # Generate
    outputs = llm.generate("Hello world", max_tokens=10)
    request_id = outputs[0].request_id
    
    # Verify data collected
    attention = collector.retrieve(request_id)
    metrics = analyzer.retrieve(request_id)
    
    assert len(attention) > 0
    assert len(metrics) > 0
    assert "entropy" in metrics[0][0]["metrics"]
    
    # Cleanup
    session.cleanup()

def test_concurrent_requests():
    """Test actor handles concurrent requests correctly"""
    llm = LLM(model="gpt2")
    lens = LensManager(llm)
    
    actor = AttentionCollector(layers=[0])
    lens.add_actor(actor)
    
    # Generate two requests concurrently
    outputs = llm.generate(["Hello", "World"], max_tokens=10)
    
    # Verify separate data
    data_0 = actor.retrieve(outputs[0].request_id)
    data_1 = actor.retrieve(outputs[1].request_id)
    
    assert data_0 != data_1
```

### 14.3 Performance Tests

```python
def test_overhead():
    """Measure actor overhead"""
    llm = LLM(model="gpt2")
    
    # Baseline
    start = time.time()
    _ = llm.generate("Hello", max_tokens=50)
    baseline = time.time() - start
    
    # With actor
    lens = LensManager(llm)
    actor = AttentionCollector(layers=[0], window=10)
    lens.add_actor(actor)
    
    start = time.time()
    _ = llm.generate("Hello", max_tokens=50)
    with_actor = time.time() - start
    
    overhead_pct = ((with_actor - baseline) / baseline) * 100
    
    # Assert overhead is acceptable
    assert overhead_pct < 15.0  # <15% overhead

def test_memory_bounded():
    """Test memory management"""
    lens = LensManager(llm, max_memory_mb=100)
    
    actor = AttentionCollector(layers=list(range(12)))  # All layers
    lens.add_actor(actor)
    
    # Generate many requests
    for i in range(100):
        _ = llm.generate(f"Prompt {i}", max_tokens=50)
        
    # Verify memory stayed within budget
    assert lens.memory_tracker.current_usage_mb() <= 100
```

---

## 15. Documentation Plan

### 15.1 Documentation Structure

```
docs/
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── concepts.md
├── guides/
│   ├── creating-actors.md
│   ├── pipelines.md
│   ├── runtime-control.md
│   └── performance-tuning.md
├── api-reference/
│   ├── lens-manager.md
│   ├── actor-base.md
│   ├── rest-api.md
│   └── websocket-api.md
├── actors/
│   ├── attention-collector.md
│   ├── attention-analyzer.md
│   ├── logit-collector.md
│   └── pattern-detector.md
├── examples/
│   ├── basic-collection.md
│   ├── real-time-dashboard.md
│   ├── pattern-detection.md
│   └── custom-actor.md
└── advanced/
    ├── distributed.md
    ├── memory-management.md
    └── security.md
```

### 15.2 Code Examples

Each actor should have comprehensive examples:

```python
# Example 1: Basic usage
from vllm_lens import LensManager, AttentionCollector

llm = LLM(model="gpt2")
lens = LensManager(llm)

collector = AttentionCollector(layers=[0, 1, 2], window=10)
lens.add_actor(collector)

outputs = llm.generate("Hello world", max_tokens=50)
attention = collector.retrieve(outputs[0].request_id)

# Example 2: Runtime control
lens.get_actor(collector.actor_id).update_config("window", 20)

# Example 3: Pipeline
analyzer = AttentionAnalyzer(metrics=["entropy"], layers=[0, 1, 2])
pipeline = lens.pipeline([collector, analyzer])

# Example 4: Custom actor
class MyActor(Actor):
    def process(self, hook_point, layer_idx, data, context):
        # Your logic here
        return data
```

---

## 16. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| vLLM API breakage | High | High | Version adapter layer, maintain compatibility matrix |
| Performance regression | High | Medium | Comprehensive benchmarking, performance budgets |
| Memory leaks | High | Medium | Rigorous cleanup testing, memory profiling |
| Thread safety bugs | Medium | Medium | Lock-free where possible, extensive concurrency tests |
| Actor API too complex | Medium | High | Iterative design, user feedback, good examples |
| Storage overhead | Medium | Low | Compression, smart eviction, user controls |
| Security vulnerabilities | High | Low | Security audit, authentication, encryption |

---

## 17. Success Metrics

### 17.1 Technical Metrics

- **Performance Overhead**: <10% for typical configurations
- **Memory Usage**: Stay within configured budget 99% of time
- **Test Coverage**: >80% for core framework, >70% for actors
- **API Stability**: Zero breaking changes within minor versions

### 17.2 User Adoption Metrics

- **Active Users**: Track unique users/installations
- **Actor Usage**: Which actors are most popular
- **Session Duration**: How long users run instrumentation
- **Custom Actors**: Number of community-contributed actors

### 17.3 Qualitative Metrics

- **Ease of Use**: User surveys, onboarding time
- **Documentation Quality**: Completion rate of tutorials
- **Community Health**: GitHub stars, issues, PRs

---

## 18. Conclusion

The **Dynamic Actor Framework** represents a significant evolution from passive data collection to **programmable, interactive model introspection**. By treating instrumentation as runtime-reconfigurable middleware, we enable:

1. **Research Efficiency**: Interactive debugging and hypothesis testing
2. **Production Observability**: Real-time monitoring and alerting
3. **Extensibility**: Community-contributed actors and patterns
4. **Performance**: Collect only what's needed, when it's needed

**Next Steps**:
1. Review this document with stakeholders
2. Prototype Phase 1 (Foundation) with AttentionCollector migration
3. Validate API ergonomics with real users
4. Iterate based on feedback
5. Execute roadmap

**Questions to Answer Before Implementation**:
- Which actors are highest priority for Phase 3?
- What performance overhead is acceptable in production?
- Should we prioritize local-first or cloud-native deployment?
- How much backward compatibility do we need to maintain?

---

## Appendix A: Glossary

- **Actor**: A component that collects, processes, or reacts to model internals
- **Session**: A logical grouping of actors with shared lifecycle
- **Hook**: An interception point in model forward pass
- **Pipeline**: A chain of actors processing data sequentially
- **Lens**: The overall framework for model introspection
- **Control Plane**: APIs for external control of actors

## Appendix B: References

- vLLM Documentation: https://docs.vllm.ai
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- PyTorch Hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
- eBPF (inspiration for dynamic instrumentation): https://ebpf.io

---

**End of Document**
