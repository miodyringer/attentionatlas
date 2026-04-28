import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # Required for Phase 2 plugin

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from vllm_attention_capture_plugin import (
    enable_attention_capture,
    get_latest_attention_scores,
    get_capture_config,
)
import numpy as np

# Load environment variables
load_dotenv()

# Model Initialization
model_name = os.getenv("MODEL_NAME")
compare_model_name = os.getenv("COMPARE_MODEL")

if not model_name:
    raise ValueError("MODEL_NAME environment variable must be set in .env file")

# Only Mistral has Gated Access. HF_TOKEN is required. Get it on https://huggingface.co/settings/tokens
if "mistral" in model_name.lower() and os.getenv("HF_TOKEN") is None:
    raise ValueError("Mistral models require HF_TOKEN environment variable to be set in .env file")

print(f"Loading primary model: {model_name}")
model = LLM(
    model=model_name,
    enforce_eager=True,  # Required for CPU
)

# Enable attention capture with full attention (no windowing)
enable_attention_capture(
    model,
    capture_layers=None,  # First 3 layers
    attention_window=None,  # Capture full attention matrix
    auto_clear=False
)

print("Attention capture enabled for primary model (full attention)")
capture_config = get_capture_config(model)
print(f"Capture config: {capture_config}")

# Load comparison model only if COMPARE_MODEL is set and not empty
compare_model = None
if compare_model_name and compare_model_name.strip():
    print(f"Loading comparison model: {compare_model_name}")
    if "mistral" in compare_model_name.lower() and os.getenv("HF_TOKEN") is None:
        raise ValueError("Mistral models require HF_TOKEN environment variable to be set in .env file")
    compare_model = LLM(
        model=compare_model_name,
        device="cpu",
        enforce_eager=True,
    )

    # Enable capture for comparison model
    enable_attention_capture(
        compare_model,
        capture_layers=None,
        attention_window=None,  # Full attention
        auto_clear=True
    )
    print("Attention capture enabled for comparison model")
else:
    print("No comparison model specified. Model comparison feature will be disabled.")


app = FastAPI(title="Attention Capture API (vLLM)", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Store attention weights from generation
# Format: {full_text: {"scores": attention_dict, "tokens": token_list}}
attention_cache = {}


class GenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 100
    document_context: Optional[str] = None  # Deprecated - use documents instead
    documents: Optional[List[Dict[str, str]]] = None  # [{"id": "doc1", "text": "..."}]


class GenerateResponse(BaseModel):
    answer: str
    model: str
    metadata: Dict[str, Any]


class AnalyzeRequest(BaseModel):
    answer: str
    attn_layer: Optional[int] = -1
    token_boundaries: Optional[Dict[str, int]] = None  # Optional boundaries


class AnalyzeResponse(BaseModel):
    attention_pattern: list
    shape: list[int]
    num_tokens: int
    tokens: list[str]
    prefill_tokens: Optional[int] = None  # Number of prefill tokens


def count_tokens(text: str, llm_instance: LLM) -> int:
    """Count tokens in text using the model's tokenizer"""
    tokens = llm_instance.get_tokenizer().encode(text)
    return len(tokens)


def build_prompt(
    user_question: str,
    documents: List[Dict[str, str]] = None
) -> str:
    """
    Build a structured prompt that instructs the model to use documents.

    Args:
        user_question: The user's question
        documents: List of dicts with 'id' and 'text' keys

    Returns:
        Formatted prompt string
    """
    if not documents or len(documents) == 0:
        # No documents - just return question
        return user_question

    # Build document section
    doc_text = "\n\n".join([
        f"Document {i+1} ({doc['id']}):\n{doc['text']}"
        for i, doc in enumerate(documents)
    ])

    # Structured prompt with explicit instruction
    prompt = f"""Answer the following question using ONLY information from the provided documents below. If the answer cannot be found in the documents, respond with "I cannot answer this based on the provided documents."

{doc_text}

Question: {user_question}

Answer:"""

    return prompt


@app.get("/")
async def root():
    return {"message": "LLM API is running (vLLM backend)"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    try:
        # Build full prompt with structured documents or legacy document_context
        token_boundaries = None
        context_token_count = 0  # Initialize for backward compat metadata
        prompt_token_count = 0  # Initialize for backward compat metadata

        if request.documents:
            # New structured documents approach
            full_content = build_prompt(request.prompt, request.documents)

            # Track each document's token boundaries
            doc_boundaries = []
            current_pos = 0

            # Account for instruction text before documents
            instruction_text = "Answer the following question using ONLY information from the provided documents below. If the answer cannot be found in the documents, respond with \"I cannot answer this based on the provided documents.\"\n\n"
            current_pos = count_tokens(instruction_text, model)

            for i, doc in enumerate(request.documents):
                doc_header = f"Document {i+1} ({doc['id']}):\n"
                header_tokens = count_tokens(doc_header, model)
                doc_text_tokens = count_tokens(doc['text'], model)

                doc_boundaries.append({
                    "doc_id": doc['id'],
                    "start": current_pos + header_tokens,
                    "end": current_pos + header_tokens + doc_text_tokens
                })

                # Move position: header + doc_text + separator "\n\n"
                current_pos += header_tokens + doc_text_tokens + count_tokens("\n\n", model)

            # After all documents, track question position
            question_prefix = "Question: "
            question_start = current_pos + count_tokens(question_prefix, model)
            question_tokens = count_tokens(request.prompt, model)
            question_end = question_start + question_tokens

            # After question, there's "\n\nAnswer:"
            answer_prefix_tokens = count_tokens("\n\nAnswer:", model)

            # For backward compat, set context_token_count to end of documents
            context_token_count = current_pos
            prompt_token_count = question_tokens

            token_boundaries = {
                "documents": doc_boundaries,
                "question_start": question_start,
                "question_end": question_end,
                "has_template_tokens": True
            }

        elif request.document_context:
            # Legacy document_context approach (backward compatibility)
            full_content = request.document_context + "\n\n" + request.prompt
            context_token_count = count_tokens(request.document_context, model)
            prompt_token_count = count_tokens(request.prompt, model)

            token_boundaries = {
                "document_start": 0,
                "document_end": context_token_count,
                "prompt_start": context_token_count,
                "prompt_end": context_token_count + prompt_token_count,
                "has_template_tokens": False
            }

        else:
            # No documents - just user question
            full_content = request.prompt
            prompt_token_count = count_tokens(request.prompt, model)
            context_token_count = 0

            token_boundaries = {
                "document_start": 0,
                "document_end": 0,
                "prompt_start": 0,
                "prompt_end": prompt_token_count,
                "question_start": 0,
                "question_end": prompt_token_count,
                "has_template_tokens": False
            }

        # Count prompt tokens
        prompt_length = count_tokens(full_content, model)

        # DEBUG: Check hook state before generation
        from vllm_attention_capture_plugin import get_capture_hook
        hook = get_capture_hook(model)
        if hook:
            print(f"🔍 BEFORE generate: hook has {len(hook.captured_scores)} requests")
            for req_id, layers in hook.captured_scores.items():
                print(f"   Request {req_id}: {len(layers)} layers")

        # Generate tokens with vLLM
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            skip_special_tokens=True,
        )

        outputs = model.generate([full_content], sampling_params)
        output = outputs[0]

        # DEBUG: Check hook state after generation
        if hook:
            print(f"🔍 AFTER generate: hook has {len(hook.captured_scores)} requests")
            for req_id, layers in hook.captured_scores.items():
                print(f"   Request {req_id}: {len(layers)} layers")
                for layer_id, chunks in layers.items():
                    print(f"      Layer {layer_id}: {len(chunks)} chunks")

        # Small delay to ensure all decode attention has been captured
        import time
        time.sleep(0.1)

        # Extract generated text (full output includes prompt)
        full_answer = output.outputs[0].text
        new_answer = full_answer  # vLLM returns only new tokens by default

        # Get actual token counts
        actual_new_tokens = len(output.outputs[0].token_ids)
        total_length = prompt_length + actual_new_tokens

        # CAPTURE ATTENTION WEIGHTS DURING GENERATION
        # Get the attention scores that were just captured
        attention_scores = get_latest_attention_scores()

        if attention_scores:
            print(f"✅ Captured attention for layers: {sorted(attention_scores.keys())}")
            for layer_id, layer_attn in attention_scores.items():
                print(f"   Layer {layer_id}: shape {layer_attn.shape}")
                print(f"   Layer {layer_id}: type {type(layer_attn)}")
        else:
            print("⚠️  No attention scores captured!")

        # Tokenize the full text to get token strings
        tokenizer = model.get_tokenizer()
        full_text_for_cache = full_content + new_answer
        token_ids = tokenizer.encode(full_text_for_cache)
        token_strings = [tokenizer.decode([tid]) for tid in token_ids]

        # Complete token boundaries with response positions
        if token_boundaries is not None:
            token_boundaries["response_start"] = prompt_length
            token_boundaries["response_end"] = total_length

        # Store in cache with the full text as key
        attention_cache[full_text_for_cache] = {
            "scores": attention_scores,
            "tokens": token_strings,
            "num_tokens": len(token_strings),
            "prefill_tokens": int(prompt_length),  # Store prefill count
            "token_boundaries": token_boundaries  # Store boundaries for analysis
        }

        print(f"✅ Cached attention weights for {len(token_strings)} tokens")

        return GenerateResponse(
            answer=full_content + new_answer,  # Return full text like transformer_lens did
            model=model_name,
            metadata={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "prompt_tokens": int(prompt_length),
                "prefill_tokens": int(prompt_length),  # Add explicit prefill token count
                "generated_tokens": int(actual_new_tokens),
                "new_text": new_answer,
                "total_tokens": int(total_length),
                "context_token_count": int(context_token_count),
                "user_prompt_token_count": int(prompt_token_count),
                "has_document_context": request.document_context is not None,
                "token_boundaries": token_boundaries
            }
        )
    except Exception as e:
        import traceback
        print(f"Error in generate: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_answer(request: AnalyzeRequest):
    """
    Analyze attention weights that were captured during generation.
    No need to run through the model again - just retrieve from cache.
    """
    try:
        text = request.answer

        # Check if we have cached attention for this text
        if text not in attention_cache:
            raise HTTPException(
                status_code=400,
                detail="No attention weights found for this text. Please generate the text first."
            )

        cached_data = attention_cache[text]
        scores = cached_data["scores"]
        token_strings = cached_data["tokens"]
        num_tokens = cached_data["num_tokens"]
        # Calculate prefill tokens from the cache metadata if available
        prefill_tokens = cached_data.get("prefill_tokens", None)

        if scores is None:
            raise HTTPException(status_code=500, detail="No attention scores in cache")

        print(f"🔍 Available layers in cache: {sorted(scores.keys())}")
        for layer_id, layer_attn in scores.items():
            print(f"   Layer {layer_id}: shape {layer_attn.shape}")

        # Determine which layers to return based on request
        if request.attn_layer == -1:
            # All captured layers
            layer_ids = sorted(scores.keys())
        else:
            # Layers 0 through attn_layer
            layer_ids = [i for i in sorted(scores.keys()) if i <= request.attn_layer]

        if not layer_ids:
            raise HTTPException(status_code=400, detail="No layers available for requested layer range")

        # Stack attention patterns from available layers
        attention_patterns = []
        for layer_id in layer_ids:
            layer_attn = scores[layer_id]  # [num_heads, num_tokens, context_len]
            attention_patterns.append(layer_attn)

        # Stack into tensor-like structure: [num_layers, num_heads, num_tokens, context_len]
        attn = np.stack(attention_patterns)

        # Convert to list for JSON serialization
        attn_list = attn.tolist()
        shape = list(attn.shape)

        print(f"✅ Retrieved attention from cache: shape {shape}, {num_tokens} tokens")
        print(f"   Shape breakdown: [layers={shape[0]}, heads={shape[1]}, num_tokens={shape[2]}, context_len={shape[3]}]")
        print(f"   Prefill tokens: {prefill_tokens}")
        print(f"   Decode tokens: {num_tokens - prefill_tokens if prefill_tokens else 'unknown'}")

        return AnalyzeResponse(
            attention_pattern=attn_list,
            shape=shape,
            num_tokens=num_tokens,
            tokens=token_strings,
            prefill_tokens=prefill_tokens  # Include prefill count
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in analyze: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=AnalyzeResponse)
async def compare_analyze(request: AnalyzeRequest):
    """Analyze text using the comparison model"""
    if not compare_model:
        raise HTTPException(status_code=400, detail="No comparison model configured. Set COMPARE_MODEL in .env file")

    try:
        # Tokenize the text with comparison model
        tokenizer = compare_model.get_tokenizer()
        token_ids = tokenizer.encode(request.answer)
        num_tokens = len(token_ids)

        # Decode individual tokens
        token_strings = [tokenizer.decode([tid]) for tid in token_ids]

        # Generate with attention capture
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            skip_special_tokens=True,
        )

        _ = compare_model.generate([request.answer], sampling_params)

        # Get captured attention scores
        scores = get_latest_attention_scores()

        if scores is None:
            raise HTTPException(status_code=500, detail="No attention scores captured")

        # Convert format (same logic as analyze endpoint)
        if request.attn_layer == -1:
            layer_ids = sorted(scores.keys())
        else:
            layer_ids = [i for i in sorted(scores.keys()) if i <= request.attn_layer]

        if not layer_ids:
            raise HTTPException(status_code=400, detail="No layers available for requested layer range")

        attention_patterns = []
        for layer_id in layer_ids:
            layer_attn = scores[layer_id]  # [num_heads, num_tokens, seq_len]

            # The plugin now returns full attention matrix
            # Just use it directly
            attention_patterns.append(layer_attn)

        attn = np.stack(attention_patterns)
        attn_list = attn.tolist()
        shape = list(attn.shape)

        return AnalyzeResponse(
            attention_pattern=attn_list,
            shape=shape,
            num_tokens=num_tokens,
            tokens=token_strings
        )
    except Exception as e:
        import traceback
        print(f"Error in compare: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "backend": "vLLM"}


@app.get("/models")
async def get_models():
    """Return the primary and comparison model names"""
    return {
        "primary_model": model_name,
        "compare_model": compare_model_name,
        "backend": "vLLM"
    }


@app.post("/extract_pdf")
async def extract_pdf(file: UploadFile):
    """Extract text from PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    try:
        import PyPDF2
        import io

        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        return {"text": text, "pages": len(pdf_reader.pages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")


# ========== NEW ENDPOINTS: Hallucination Detection & RAG Scoring ==========

class HallucinationRequest(BaseModel):
    answer: str  # Full text (context + prompt + response)
    context_token_count: int  # How many tokens are context
    attn_layer: Optional[int] = -1  # Which layer to use (-1 = last)
    threshold: Optional[float] = 0.7


class HallucinationResponse(BaseModel):
    flagged_tokens: list  # Tokens above threshold
    all_scores: list  # All hallucination scores
    overall_confidence: float  # Overall answer confidence
    has_hallucinations: bool


@app.post("/detect_hallucination", response_model=HallucinationResponse)
async def detect_hallucination_endpoint(request: HallucinationRequest):
    """
    Detect hallucinations in generated text using attention patterns.

    Uses three signals:
    1. Low attention to source tokens (not grounded in context)
    2. High attention entropy (uncertainty)
    3. Self-referential attention (attending only to recent tokens)
    """
    try:
        from analysis.hallucination_detector import detect_hallucinations, get_flagged_tokens, compute_overall_confidence

        # Look up cached attention
        if request.answer not in attention_cache:
            # Debug: print what's in cache vs what we're looking for
            print(f"❌ Cache miss for text of length {len(request.answer)}")
            print(f"   Looking for text starting with: {request.answer[:100]}")
            print(f"   Cache has {len(attention_cache)} entries:")
            for cache_key in list(attention_cache.keys())[:3]:  # Show first 3
                print(f"   - Key length: {len(cache_key)}, starts with: {cache_key[:100]}")
                print(f"     Tokens: {attention_cache[cache_key]['num_tokens']}")

            raise HTTPException(
                status_code=404,
                detail="Attention data not found. Generate text first using /generate endpoint."
            )

        cached_data = attention_cache[request.answer]
        attn_scores = cached_data["scores"]
        tokens = cached_data["tokens"]

        if not attn_scores:
            raise HTTPException(status_code=500, detail="No attention scores in cache")

        # Use last layer if -1, otherwise use specified layer
        layer_idx = request.attn_layer if request.attn_layer >= 0 else max(attn_scores.keys())

        if layer_idx not in attn_scores:
            available_layers = sorted(attn_scores.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Layer {layer_idx} not found. Available layers: {available_layers}"
            )

        layer_attn = attn_scores[layer_idx]  # [num_heads, num_tokens, seq_len]

        # Context is [0:context_token_count], rest is generated
        source_range = (0, request.context_token_count)

        # Run hallucination detection
        scores = detect_hallucinations(
            attn_weights=layer_attn,
            tokens=tokens,
            source_token_range=source_range,
            layer_idx=layer_idx,
            threshold=request.threshold
        )

        # Filter flagged tokens
        flagged = get_flagged_tokens(scores, threshold=request.threshold)

        # Overall confidence (average of all tokens)
        overall_confidence = compute_overall_confidence(scores)

        return HallucinationResponse(
            flagged_tokens=[s.to_dict() for s in flagged],
            all_scores=[s.to_dict() for s in scores],
            overall_confidence=overall_confidence,
            has_hallucinations=len(flagged) > 0
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in detect_hallucination: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


class RAGScoreRequest(BaseModel):
    answer: str
    document_boundaries: list  # [{"start": int, "end": int, "doc_id": str}]
    generation_start: int  # Token index where generation begins


class RAGScoreResponse(BaseModel):
    document_scores: list  # Scored documents
    most_relevant_doc: str  # ID of most relevant document
    unused_docs: list  # IDs of ignored documents


@app.post("/score_rag_documents", response_model=RAGScoreResponse)
async def score_rag_endpoint(request: RAGScoreRequest):
    """
    Score RAG documents by attention attribution.

    Measures which retrieved documents the model actually uses during generation.
    """
    try:
        from analysis.rag_scorer import score_rag_documents, get_most_relevant_document, get_unused_documents

        # Look up cached attention
        if request.answer not in attention_cache:
            raise HTTPException(
                status_code=404,
                detail="Attention data not found. Generate text first using /generate endpoint."
            )

        cached_data = attention_cache[request.answer]
        attn_scores = cached_data["scores"]
        tokens = cached_data["tokens"]

        if not attn_scores:
            raise HTTPException(status_code=500, detail="No attention scores in cache")

        # Use last layer for attribution
        layer_idx = max(attn_scores.keys())
        layer_attn = attn_scores[layer_idx]  # [num_heads, num_tokens, seq_len]

        # Convert document boundaries from dict format to tuple format
        doc_ranges = [
            (d["start"], d["end"], d["doc_id"])
            for d in request.document_boundaries
        ]

        # Run RAG scoring
        doc_scores = score_rag_documents(
            attn_weights=layer_attn,
            tokens=tokens,
            document_ranges=doc_ranges,
            generation_start=request.generation_start
        )

        # Get most relevant and unused documents
        most_relevant = get_most_relevant_document(doc_scores)
        unused = get_unused_documents(doc_scores)

        return RAGScoreResponse(
            document_scores=[d.to_dict() for d in doc_scores],
            most_relevant_doc=most_relevant.doc_id if most_relevant else "",
            unused_docs=[d.doc_id for d in unused]
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in score_rag_documents: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def aggregate_attention_layers(
    layer_scores: Dict[int, np.ndarray],
    mode: str
) -> np.ndarray:
    """
    Aggregate attention across layers.

    Args:
        layer_scores: Dict mapping layer_idx -> attention array
        mode: "last", "last_3", "last_5", "all", or specific int like "5"

    Returns:
        Aggregated attention: [num_heads, num_tokens, seq_len]
    """
    layer_indices = sorted(layer_scores.keys())

    if mode == "last":
        return layer_scores[layer_indices[-1]]

    elif mode.startswith("last_"):
        n = int(mode.split("_")[1])
        selected_layers = layer_indices[-n:]
        # Weighted average: later layers get more weight
        weights = np.array([0.1, 0.2, 0.3, 0.4])[-n:]  # Last is 0.4
        weights = weights / weights.sum()  # Normalize

        aggregated = None
        for i, layer_idx in enumerate(selected_layers):
            layer_attn = layer_scores[layer_idx]
            if aggregated is None:
                aggregated = layer_attn * weights[i]
            else:
                aggregated += layer_attn * weights[i]
        return aggregated

    elif mode == "all":
        # Simple average
        aggregated = None
        for layer_attn in layer_scores.values():
            if aggregated is None:
                aggregated = layer_attn.copy()
            else:
                aggregated += layer_attn
        return aggregated / len(layer_scores)

    else:
        # Specific layer by index
        layer_idx = int(mode)
        if layer_idx not in layer_scores:
            raise ValueError(f"Layer {layer_idx} not in cache")
        return layer_scores[layer_idx]


class AnswerGroundingRequest(BaseModel):
    answer: str  # Full text
    layer_mode: str = "last"  # "last", "last_3", "last_5", "all", or specific int


class AnswerGroundingResponse(BaseModel):
    per_token: List[dict]
    document_usage: Dict[str, float]
    avg_document_attention: float
    avg_question_attention: float
    avg_self_attention: float
    well_grounded: bool
    hallucination_risk: float
    unused_documents: List[str]


@app.post("/analyze_answer_grounding", response_model=AnswerGroundingResponse)
async def analyze_answer_grounding_endpoint(request: AnswerGroundingRequest):
    """
    Analyze how the generated answer is grounded in documents, question, and itself.

    This endpoint provides comprehensive analysis of token-level attribution:
    - Which documents were used
    - How much the answer relies on the question vs self-reference
    - Per-token grounding details
    """
    try:
        from analysis.answer_grounding import analyze_answer_grounding

        # Look up cached attention and metadata
        if request.answer not in attention_cache:
            raise HTTPException(
                status_code=404,
                detail="Attention data not found. Generate text first using /generate endpoint."
            )

        cached_data = attention_cache[request.answer]
        attn_scores = cached_data["scores"]  # Dict[layer_idx, ndarray]
        tokens = cached_data["tokens"]
        token_boundaries = cached_data.get("token_boundaries")

        if not token_boundaries:
            raise HTTPException(
                status_code=400,
                detail="No token boundaries in cache. Make sure to use structured documents in /generate."
            )

        # Extract regions
        document_ranges = [
            (d["start"], d["end"], d["doc_id"])
            for d in token_boundaries.get("documents", [])
        ]

        question_range = (
            token_boundaries.get("question_start", 0),
            token_boundaries.get("question_end", 0)
        )

        answer_range = (
            token_boundaries.get("response_start"),
            token_boundaries.get("response_end")
        )

        # Validate ranges
        if answer_range[0] is None or answer_range[1] is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid answer range in token boundaries"
            )

        # Aggregate layers based on mode
        attn_weights = aggregate_attention_layers(
            attn_scores,
            mode=request.layer_mode
        )

        # Run analysis
        report = analyze_answer_grounding(
            attn_weights=attn_weights,
            tokens=tokens,
            document_ranges=document_ranges,
            question_range=question_range,
            answer_range=answer_range
        )

        return AnswerGroundingResponse(
            per_token=[t.to_dict() for t in report.per_token],
            document_usage=report.document_usage,
            avg_document_attention=report.avg_document_attention,
            avg_question_attention=report.avg_question_attention,
            avg_self_attention=report.avg_self_attention,
            well_grounded=report.well_grounded,
            hallucination_risk=report.hallucination_risk,
            unused_documents=report.unused_documents
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in analyze_answer_grounding: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT not set

    # Generate config.js for frontend
    config_content = f"""// Auto-generated configuration file
// This file is generated by api.py on startup
const API_CONFIG = {{
    port: {port},
    baseUrl: 'http://localhost:{port}'
}};
"""

    config_path = os.path.join(os.path.dirname(__file__), 'frontend', 'js', 'config.js')
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"Generated frontend config at {config_path}")
    print(f"Starting server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
