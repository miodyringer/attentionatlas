"""
Compare attention patterns between vLLM plugin and HuggingFace Transformers.

This script generates the same text with both backends and compares the attention
patterns to verify the vLLM plugin is capturing attention correctly.
"""
import logging
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from vllm import LLM, SamplingParams
from vllm_attention_capture_plugin import (
    enable_attention_capture,
    get_latest_attention_scores,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Reduce plugin logging verbosity
logging.getLogger('vllm_attention_capture_plugin')


def get_vllm_attention(prompt: str, model_name: str = "gpt2", max_tokens: int = 10):
    """Generate text with vLLM and capture attention."""
    print("=" * 80)
    print("VLLM ATTENTION CAPTURE")
    print("=" * 80)

    # Initialize vLLM
    llm = LLM(model=model_name, enforce_eager=True)

    # Enable attention capture (full attention, no windowing)
    enable_attention_capture(
        llm,
        capture_layers=[0],  # Just first layer for comparison
        attention_window=None  # Full attention
    )

    # Generate
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=max_tokens,
    )

    outputs = llm.generate(prompt, sampling_params)
    generated_text = outputs[0].outputs[0].text
    full_text = prompt + generated_text

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
    print(f"Full text: {full_text}")

    # Get attention scores
    scores = get_latest_attention_scores()

    if scores is None or 0 not in scores:
        raise ValueError("No attention scores captured!")

    # Get layer 0 attention: [num_heads, num_tokens, seq_len]
    attn = scores[0]

    # Get tokens
    tokenizer = llm.get_tokenizer()
    token_ids = tokenizer.encode(full_text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    print(f"\nAttention shape: {attn.shape}")
    print(f"Tokens ({len(tokens)}): {tokens}")

    return attn, tokens, full_text


def get_transformers_attention(prompt: str, model_name: str = "gpt2", max_tokens: int = 10):
    """Generate text with HuggingFace Transformers and capture attention."""
    print("\n" + "=" * 80)
    print("TRANSFORMERS ATTENTION CAPTURE")
    print("=" * 80)

    # Load model and tokenizer with attention output enabled
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True  # Enable attention output
    )
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    num_prompt_tokens = input_ids.shape[1]

    print(f"\nPrompt: {prompt}")
    print(f"Prompt tokens: {num_prompt_tokens}")

    # Step 1: Get prefill attention by running forward pass
    with torch.no_grad():
        prefill_outputs = model(input_ids, output_attentions=True)

    # Check if attentions are returned
    if prefill_outputs.attentions is None:
        raise ValueError("Model did not return attentions. Make sure model is configured correctly.")

    # prefill_outputs.attentions is a tuple of layer attentions
    # Each layer: [batch, num_heads, seq_len, seq_len]
    prefill_attn = prefill_outputs.attentions[0][0].cpu().numpy()  # Layer 0, batch 0
    num_heads = prefill_attn.shape[0]

    print(f"Prefill attention: {prefill_attn.shape}")

    # Step 2: Generate tokens one by one and capture attention
    current_ids = input_ids
    all_tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]

    # Initialize full attention matrix
    max_seq_len = num_prompt_tokens + max_tokens
    full_attn = np.zeros((num_heads, max_seq_len, max_seq_len), dtype=np.float32)

    # Fill prefill attention
    full_attn[:, :num_prompt_tokens, :num_prompt_tokens] = prefill_attn

    # Generate tokens one by one
    generated_tokens = []
    for step in range(max_tokens):
        with torch.no_grad():
            outputs = model(current_ids, output_attentions=True)

        # Get next token (greedy)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append the token
        current_ids = torch.cat([current_ids, next_token], dim=-1)

        # Run forward again to get attention for the new token
        with torch.no_grad():
            outputs_with_new = model(current_ids, output_attentions=True)

        step_attn_with_new = outputs_with_new.attentions[0][0].cpu().numpy()
        seq_len_with_new = step_attn_with_new.shape[1]

        # Extract attention for the newly added token (last row)
        new_token_attn = step_attn_with_new[:, -1, :]  # [num_heads, seq_len_with_new]

        # Store in full matrix
        current_token_idx = num_prompt_tokens + step
        full_attn[:, current_token_idx, :seq_len_with_new] = new_token_attn

        # Decode token
        token_str = tokenizer.decode([next_token[0].item()])
        all_tokens.append(token_str)
        generated_tokens.append(token_str)

        print(f"Decode step {step+1}: generated '{token_str}', seq_len={seq_len_with_new}")

    generated_text = "".join(generated_tokens)
    full_text = "".join(all_tokens)

    print(f"\nGenerated: {generated_text}")
    print(f"Full text: {full_text}")

    # Trim to actual size
    actual_len = len(all_tokens)
    full_attn = full_attn[:, :actual_len, :actual_len]

    print(f"\nFull attention shape: {full_attn.shape}")
    print(f"Tokens ({len(all_tokens)}): {all_tokens}")

    return full_attn, all_tokens, full_text


def plot_comparison(vllm_attn, vllm_tokens, transformers_attn, transformers_tokens,
                    head_idx=0, save_path="attention_comparison.png"):
    """Plot side-by-side heatmaps comparing vLLM and Transformers attention."""

    # Get head 0 from both
    vllm_head = vllm_attn[head_idx]  # [num_tokens, seq_len]
    transformers_head = transformers_attn[head_idx]  # [num_tokens, seq_len]

    # Compute difference
    # Need to make sure shapes match
    min_tokens = min(vllm_head.shape[0], transformers_head.shape[0])
    min_seq_len = min(vllm_head.shape[1], transformers_head.shape[1])

    vllm_head_trimmed = vllm_head[:min_tokens, :min_seq_len]
    transformers_head_trimmed = transformers_head[:min_tokens, :min_seq_len]

    diff = np.abs(vllm_head_trimmed - transformers_head_trimmed)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot vLLM attention
    im1 = axes[0].imshow(vllm_head, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title(f'vLLM Plugin - Head {head_idx}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Key Position (attending to)')
    axes[0].set_ylabel('Query Position (token)')
    axes[0].set_xticks(range(len(vllm_tokens)))
    axes[0].set_yticks(range(len(vllm_tokens)))
    axes[0].set_xticklabels([t[:8] for t in vllm_tokens], rotation=90, fontsize=8)
    axes[0].set_yticklabels([t[:8] for t in vllm_tokens], fontsize=8)
    plt.colorbar(im1, ax=axes[0], label='Attention Weight')

    # Plot Transformers attention
    im2 = axes[1].imshow(transformers_head, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title(f'HuggingFace Transformers - Head {head_idx}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Key Position (attending to)')
    axes[1].set_ylabel('Query Position (token)')
    axes[1].set_xticks(range(len(transformers_tokens)))
    axes[1].set_yticks(range(len(transformers_tokens)))
    axes[1].set_xticklabels([t[:8] for t in transformers_tokens], rotation=90, fontsize=8)
    axes[1].set_yticklabels([t[:8] for t in transformers_tokens], fontsize=8)
    plt.colorbar(im2, ax=axes[1], label='Attention Weight')

    # Plot difference
    im3 = axes[2].imshow(diff, cmap='Reds', aspect='auto', vmin=0, vmax=0.1)
    axes[2].set_title(f'Absolute Difference', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    axes[2].set_xticks(range(min_tokens))
    axes[2].set_yticks(range(min_tokens))
    axes[2].set_xticklabels([t[:8] for t in vllm_tokens[:min_tokens]], rotation=90, fontsize=8)
    axes[2].set_yticklabels([t[:8] for t in vllm_tokens[:min_tokens]], fontsize=8)
    plt.colorbar(im3, ax=axes[2], label='|vLLM - Transformers|')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot to: {save_path}")

    # Print statistics
    print("\n" + "=" * 80)
    print("COMPARISON STATISTICS")
    print("=" * 80)
    print(f"vLLM shape: {vllm_head.shape}")
    print(f"Transformers shape: {transformers_head.shape}")
    print(f"Comparison region: {diff.shape}")
    print(f"\nDifference statistics:")
    print(f"  Mean absolute difference: {diff.mean():.6f}")
    print(f"  Max absolute difference: {diff.max():.6f}")
    print(f"  Min absolute difference: {diff.min():.6f}")
    print(f"  % of values with diff > 0.01: {(diff > 0.01).sum() / diff.size * 100:.2f}%")
    print(f"  % of values with diff > 0.05: {(diff > 0.05).sum() / diff.size * 100:.2f}%")

    # Check if attention patterns are similar
    if diff.mean() < 0.01:
        print("\n✅ PASS: Attention patterns are very similar (mean diff < 0.01)")
    elif diff.mean() < 0.05:
        print("\n⚠️  WARNING: Attention patterns have small differences (mean diff < 0.05)")
    else:
        print("\n❌ FAIL: Attention patterns are significantly different (mean diff >= 0.05)")

    return diff


def print_attention_sample(vllm_attn, transformers_attn, tokens, num_tokens=5):
    """Print sample attention values for manual inspection."""
    print("\n" + "=" * 80)
    print("SAMPLE ATTENTION VALUES (First 5x5 tokens, Head 0)")
    print("=" * 80)

    n = min(num_tokens, len(tokens))

    print("\nvLLM Plugin:")
    print("     ", end="")
    for j in range(n):
        print(f"{tokens[j][:8]:>8}", end=" ")
    print()
    for i in range(n):
        print(f"{tokens[i][:8]:>8}", end=" ")
        for j in range(n):
            val = vllm_attn[0, i, j]
            print(f"{val:8.4f}", end=" ")
        print(f"  (sum: {vllm_attn[0, i, :].sum():.4f})")

    print("\nHuggingFace Transformers:")
    print("     ", end="")
    for j in range(n):
        print(f"{tokens[j][:8]:>8}", end=" ")
    print()
    for i in range(n):
        print(f"{tokens[i][:8]:>8}", end=" ")
        for j in range(n):
            val = transformers_attn[0, i, j]
            print(f"{val:8.4f}", end=" ")
        print(f"  (sum: {transformers_attn[0, i, :].sum():.4f})")

    print("\nDifference (|vLLM - Transformers|):")
    print("     ", end="")
    for j in range(n):
        print(f"{tokens[j][:8]:>8}", end=" ")
    print()
    for i in range(n):
        print(f"{tokens[i][:8]:>8}", end=" ")
        for j in range(n):
            diff = abs(vllm_attn[0, i, j] - transformers_attn[0, i, j])
            symbol = "✓" if diff < 0.01 else "⚠" if diff < 0.05 else "✗"
            print(f"{diff:7.4f}{symbol}", end=" ")
        print()


def main():
    # Test parameters
    prompt = "The quick brown fox"
    model_name = "gpt2"
    max_tokens = 10

    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")

    # Get attention from vLLM
    vllm_attn, vllm_tokens, vllm_text = get_vllm_attention(prompt, model_name, max_tokens)

    # Get attention from Transformers
    transformers_attn, transformers_tokens, transformers_text = get_transformers_attention(prompt, model_name, max_tokens)

    # Check if generated text matches
    print("\n" + "=" * 80)
    print("TEXT COMPARISON")
    print("=" * 80)
    print(f"vLLM text:         {vllm_text}")
    print(f"Transformers text: {transformers_text}")
    if vllm_text == transformers_text:
        print("✅ Generated text matches!")
    else:
        print("⚠️  Generated text differs (this is OK if using temperature > 0)")

    # Print sample values
    print_attention_sample(vllm_attn, transformers_attn, vllm_tokens)

    # Plot comparison
    diff = plot_comparison(vllm_attn, vllm_tokens, transformers_attn, transformers_tokens)

    print("\n" + "=" * 80)
    print("DONE! Check attention_comparison.png for visual comparison.")
    print("=" * 80)


if __name__ == "__main__":
    main()
