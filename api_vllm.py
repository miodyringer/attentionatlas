import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # Required for Phase 2 plugin
if sys.platform == "darwin":
    os.environ.setdefault(
        "TORCH_COMPILE_DISABLE", "1"
    )  # Avoid C++ compile issues on macOS

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
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
    raise ValueError(
        "Mistral models require HF_TOKEN environment variable to be set in .env file"
    )

print(f"Loading primary model: {model_name}")
model = LLM(
    model=model_name,
    enforce_eager=True,  # Required for CPU
    max_model_len=4096,  # Limit context length for CPU compatibility
)

# Enable attention capture with full attention (no windowing)
enable_attention_capture(
    model,
    capture_layers=[0, 1, 2],  # First 3 layers
    attention_window=None,  # Capture full attention matrix
    auto_clear=True,
)

print("Attention capture enabled for primary model (full attention)")
capture_config = get_capture_config(model)
print(f"Capture config: {capture_config}")

# Load comparison model only if COMPARE_MODEL is set and not empty
compare_model = None
if compare_model_name and compare_model_name.strip():
    print(f"Loading comparison model: {compare_model_name}")
    if "mistral" in compare_model_name.lower() and os.getenv("HF_TOKEN") is None:
        raise ValueError(
            "Mistral models require HF_TOKEN environment variable to be set in .env file"
        )
    compare_model = LLM(
        model=compare_model_name,
        enforce_eager=True,
    )

    # Enable capture for comparison model
    enable_attention_capture(
        compare_model,
        capture_layers=[0, 1, 2],
        attention_window=None,  # Full attention
        auto_clear=True,
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


class GenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 100
    document_context: Optional[str] = None  # Separate field for document context


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


def count_tokens(text: str, llm_instance: LLM) -> int:
    """Count tokens in text using the model's tokenizer"""
    tokens = llm_instance.get_tokenizer().encode(text)
    return len(tokens)


@app.get("/")
async def root():
    return {"message": "LLM API is running (vLLM backend)"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    try:
        # Build full prompt with document context if provided
        if request.document_context:
            full_content = request.document_context + "\n\n" + request.prompt
        else:
            full_content = request.prompt

        # Token boundary tracking
        context_token_count = 0
        prompt_token_count = 0
        token_boundaries = None

        # Count tokens for boundaries
        if request.document_context:
            context_token_count = count_tokens(request.document_context, model)
            prompt_token_count = count_tokens(request.prompt, model)

            # For vLLM, we concatenate directly (no special template handling in this version)
            token_boundaries = {
                "document_start": 0,
                "document_end": context_token_count,
                "prompt_start": context_token_count,
                "prompt_end": context_token_count + prompt_token_count,
                "has_template_tokens": False,
            }

        # Count prompt tokens
        prompt_length = count_tokens(full_content, model)

        # Generate tokens with vLLM
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            skip_special_tokens=True,
        )

        outputs = model.generate([full_content], sampling_params)
        output = outputs[0]

        # Extract generated text (full output includes prompt)
        full_answer = output.outputs[0].text
        new_answer = full_answer  # vLLM returns only new tokens by default

        # Get actual token counts
        actual_new_tokens = len(output.outputs[0].token_ids)
        total_length = prompt_length + actual_new_tokens

        # Complete token boundaries with response positions
        if token_boundaries is not None:
            token_boundaries["response_start"] = prompt_length
            token_boundaries["response_end"] = total_length

        return GenerateResponse(
            answer=full_content
            + new_answer,  # Return full text like transformer_lens did
            model=model_name,
            metadata={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "prompt_tokens": int(prompt_length),
                "generated_tokens": int(actual_new_tokens),
                "new_text": new_answer,
                "total_tokens": int(total_length),
                "context_token_count": int(context_token_count),
                "user_prompt_token_count": int(prompt_token_count),
                "has_document_context": request.document_context is not None,
                "token_boundaries": token_boundaries,
            },
        )
    except Exception as e:
        import traceback

        print(f"Error in generate: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_answer(request: AnalyzeRequest):
    try:
        # Tokenize the text with vLLM
        tokenizer = model.get_tokenizer()
        token_ids = tokenizer.encode(request.answer)
        num_tokens = len(token_ids)

        # Decode individual tokens
        token_strings = [tokenizer.decode([tid]) for tid in token_ids]

        # Generate with attention capture enabled to get attention patterns
        # We need to run through the model to capture attention
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,  # Just need one token to trigger attention capture
            skip_special_tokens=True,
        )

        # Run generation (this will populate attention cache)
        _ = model.generate([request.answer], sampling_params)

        # Get captured attention scores
        scores = get_latest_attention_scores()

        if scores is None:
            raise HTTPException(status_code=500, detail="No attention scores captured")

        # Convert from plugin format to transformer_lens format
        # Plugin now returns full attention when attention_window=None
        # Plugin format: {layer_id: np.ndarray of shape [num_heads, num_tokens_generated, seq_len]}
        # TransformerLens format: [num_layers, num_heads, seq_len, seq_len]

        # Determine which layers to return based on request
        if request.attn_layer == -1:
            # All captured layers
            layer_ids = sorted(scores.keys())
        else:
            # Layers 0 through attn_layer
            layer_ids = [i for i in sorted(scores.keys()) if i <= request.attn_layer]

        if not layer_ids:
            raise HTTPException(
                status_code=400, detail="No layers available for requested layer range"
            )

        # Stack attention patterns from available layers
        attention_patterns = []
        for layer_id in layer_ids:
            layer_attn = scores[layer_id]  # [num_heads, num_tokens, seq_len]

            # The plugin now returns full attention matrix
            # Just use it directly
            attention_patterns.append(layer_attn)

        # Stack into tensor-like structure: [num_layers, num_heads, seq_len, seq_len]
        attn = np.stack(attention_patterns)

        # Convert to list for JSON serialization
        attn_list = attn.tolist()
        shape = list(attn.shape)

        return AnalyzeResponse(
            attention_pattern=attn_list,
            shape=shape,
            num_tokens=num_tokens,
            tokens=token_strings,
        )
    except Exception as e:
        import traceback

        print(f"Error in analyze: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=AnalyzeResponse)
async def compare_analyze(request: AnalyzeRequest):
    """Analyze text using the comparison model"""
    if not compare_model:
        raise HTTPException(
            status_code=400,
            detail="No comparison model configured. Set COMPARE_MODEL in .env file",
        )

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
            raise HTTPException(
                status_code=400, detail="No layers available for requested layer range"
            )

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
            tokens=token_strings,
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
        "backend": "vLLM",
    }


@app.post("/extract_pdf")
async def extract_pdf(file: UploadFile):
    """Extract text from PDF file"""
    if not file.filename.endswith(".pdf"):
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

    config_path = os.path.join(os.path.dirname(__file__), "frontend", "js", "config.js")
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"Generated frontend config at {config_path}")
    print(f"Starting server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
