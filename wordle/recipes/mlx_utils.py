"""MLX model management: loading, generation with logprobs, checkpointing."""
from __future__ import annotations

import json
import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.utils import linear_to_lora_layers


def load_model_with_lora(
    model_path: str,
    lora_rank: int = 16,
    lora_layers: int = 16,
) -> tuple:
    """Load model via mlx_lm, apply LoRA, freeze base weights.

    linear_to_lora_layers handles freezing base params and making LoRA
    params trainable automatically.

    Returns (model, tokenizer).
    """
    model, tokenizer = load(model_path)

    # Apply LoRA to the last N layers
    lora_config = {"rank": lora_rank, "alpha": float(lora_rank), "dropout": 0.0, "scale": 1.0}
    linear_to_lora_layers(model, num_layers=lora_layers, config=lora_config)

    # Freeze all, then unfreeze only LoRA params
    model.freeze()
    for _, module in model.named_modules():
        if hasattr(module, "lora_a"):
            module.unfreeze(keys=["lora_a", "lora_b"])
    n_total = sum(p.size for _, p in tree_flatten(model.parameters()))
    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(
        f"[mlx] Model loaded: {model_path}\n"
        f"[mlx] Total params: {n_total:,}  Trainable: {n_trainable:,} "
        f"({100 * n_trainable / n_total:.2f}%)"
    )
    return model, tokenizer


def generate_with_logprobs(
    model,
    tokenizer,
    prompt_tokens: list[int],
    max_tokens: int = 512,
    temperature: float = 1.0,
) -> tuple[list[int], list[float], str]:
    """Generation loop with KV caching that collects per-token log-probabilities.

    Uses KV cache for fast autoregressive decoding. Cache is local to this
    call and discarded after — no cross-call memory leaks.

    Returns (generated_tokens, logprobs, decoded_text).
    """
    generated: list[int] = []
    logprobs: list[float] = []

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    # Create fresh KV cache for this generation
    cache = make_prompt_cache(model)

    # Prefill: process all prompt tokens at once
    tokens = mx.array([prompt_tokens])
    logits = model(tokens, cache=cache)  # (1, prompt_len, vocab_size)
    next_logits = logits[:, -1, :]  # (1, vocab_size)

    for _ in range(max_tokens):
        log_probs = nn.log_softmax(next_logits, axis=-1)

        # Sample
        if temperature <= 0:
            next_token_id = mx.argmax(next_logits, axis=-1).item()
        else:
            scaled = next_logits / temperature
            next_token_id = mx.random.categorical(scaled).item()

        token_logprob = log_probs[0, next_token_id].item()

        generated.append(next_token_id)
        logprobs.append(token_logprob)

        # Check for EOS
        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        # Decode next token using cache
        tokens = mx.array([[next_token_id]])
        logits = model(tokens, cache=cache)  # (1, 1, vocab_size)
        next_logits = logits[:, -1, :]

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated, logprobs, text


def generate_batch_with_logprobs(
    model,
    tokenizer,
    prompt_tokens_list: list[list[int]],
    max_tokens: int = 512,
    temperature: float = 1.0,
) -> list[tuple[list[int], list[float], str]]:
    """Batched generation with logprobs using MLX BatchKVCache.

    Generates completions for N prompts simultaneously using left-padded
    batching and BatchKVCache. Completed sequences are filtered out early
    to avoid wasting compute.

    Returns list of (generated_tokens, logprobs, decoded_text) per prompt.
    """
    from mlx_lm.generate import _left_pad_prompts, _make_cache

    N = len(prompt_tokens_list)
    if N == 0:
        return []
    if N == 1:
        # Fall back to single-sequence path (no batch overhead)
        tokens, lps, text = generate_with_logprobs(
            model, tokenizer, prompt_tokens_list[0], max_tokens, temperature,
        )
        return [(tokens, lps, text)]

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    # Left-pad prompts and create batch cache
    padded = _left_pad_prompts(prompt_tokens_list)  # (N, max_len)
    left_padding = [padded.shape[1] - len(p) for p in prompt_tokens_list]
    batch_cache = _make_cache(model, left_padding)

    # Prefill
    logits = model(padded, cache=batch_cache)  # (N, max_len, vocab)
    next_logits = logits[:, -1, :]  # (N, vocab)

    # Per-sequence accumulators
    all_generated: list[list[int]] = [[] for _ in range(N)]
    all_logprobs: list[list[float]] = [[] for _ in range(N)]
    active_indices = list(range(N))  # maps batch-pos → original index

    for _ in range(max_tokens):
        log_probs = nn.log_softmax(next_logits, axis=-1)  # (B, vocab)
        B = next_logits.shape[0]

        # Sample
        if temperature <= 0:
            next_ids = mx.argmax(next_logits, axis=-1)  # (B,)
        else:
            next_ids = mx.random.categorical(next_logits / temperature)  # (B,)

        # Gather per-token logprobs
        token_lps = log_probs[mx.arange(B), next_ids]  # (B,)

        # Force eval so we can read values
        mx.eval(next_ids, token_lps)

        # Distribute results and check for completion
        keep_mask = []
        for batch_pos in range(B):
            orig_idx = active_indices[batch_pos]
            tid = next_ids[batch_pos].item()
            lp = token_lps[batch_pos].item()
            all_generated[orig_idx].append(tid)
            all_logprobs[orig_idx].append(lp)

            done = (eos_token_id is not None and tid == eos_token_id)
            keep_mask.append(not done)

        # Filter completed sequences
        if not any(keep_mask):
            break

        new_active = []
        keep_indices = []
        kept_ids = []
        for batch_pos in range(B):
            if keep_mask[batch_pos]:
                keep_indices.append(batch_pos)
                new_active.append(active_indices[batch_pos])
                kept_ids.append(next_ids[batch_pos])

        if len(keep_indices) < B:
            keep_arr = mx.array(keep_indices)
            for c in batch_cache:
                if hasattr(c, "filter"):
                    c.filter(keep_arr)
            active_indices = new_active
            next_input = mx.array([[tid.item()] for tid in kept_ids])
        else:
            next_input = next_ids.reshape(-1, 1)

        # Decode next step
        logits = model(next_input, cache=batch_cache)
        next_logits = logits[:, -1, :]

    # Decode texts
    results = []
    for i in range(N):
        text = tokenizer.decode(all_generated[i], skip_special_tokens=True)
        results.append((all_generated[i], all_logprobs[i], text))
    return results


def compute_logprobs_for_sequence(
    model,
    full_tokens: list[int],
    prompt_len: int,
) -> mx.array:
    """Forward pass on [prompt + completion], return log-probs for completion tokens.

    Returns mx.array of shape (n_completion_tokens,) with log-probs under current policy.
    """
    tokens = mx.array([full_tokens])
    logits = model(tokens)  # (1, seq_len, vocab_size)
    log_probs = nn.log_softmax(logits, axis=-1)  # (1, seq_len, vocab_size)

    # For completion token at position i, its log-prob comes from logit at position i-1
    n_completion = len(full_tokens) - prompt_len
    if n_completion <= 0:
        return mx.array([])

    # Gather log-probs for completion tokens
    indices = list(range(prompt_len, len(full_tokens)))
    completion_logprobs = []
    for i in indices:
        token_id = full_tokens[i]
        lp = log_probs[0, i - 1, token_id]
        completion_logprobs.append(lp)

    return mx.stack(completion_logprobs)


def save_checkpoint(
    model,
    optimizer,
    step: int,
    path: str,
    metrics: dict | None = None,
) -> None:
    """Save LoRA adapters + optimizer state + step number."""
    os.makedirs(path, exist_ok=True)

    # Save trainable weights (LoRA + biases/norms)
    trainable = dict(tree_flatten(model.trainable_parameters()))
    mx.savez(os.path.join(path, "trainable_weights.npz"), **trainable)

    # Save optimizer state as a flat dict
    opt_state = optimizer.state
    flat_opt = {}
    for key, val in tree_flatten(opt_state):
        # Replace dots/brackets with underscores for npz key safety
        safe_key = key.replace(".", "__DOT__").replace("[", "__LB__").replace("]", "__RB__")
        flat_opt[safe_key] = val
    if flat_opt:
        mx.savez(os.path.join(path, "optimizer_state.npz"), **flat_opt)

    # Save metadata
    meta = {"step": step}
    if metrics:
        meta["metrics"] = metrics
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[mlx] Checkpoint saved: {path} (step {step})")


def load_checkpoint(
    model,
    optimizer,
    path: str,
) -> int:
    """Restore LoRA adapters + optimizer state. Returns step number."""
    # Load trainable weights
    weights_path = os.path.join(path, "trainable_weights.npz")
    if os.path.exists(weights_path):
        weights = dict(mx.load(weights_path))
        # tree_unflatten expects list of (key, value) pairs
        weight_tree = tree_unflatten(list(weights.items()))
        model.update(weight_tree)
        mx.eval(model.parameters())
        print(f"[mlx] Loaded {len(weights)} trainable weight tensors")

    # Load optimizer state
    opt_path = os.path.join(path, "optimizer_state.npz")
    if os.path.exists(opt_path):
        flat_opt = dict(mx.load(opt_path))
        # Restore key names
        restored = {}
        for key, val in flat_opt.items():
            orig_key = key.replace("__DOT__", ".").replace("__LB__", "[").replace("__RB__", "]")
            restored[orig_key] = val
        opt_tree = tree_unflatten(list(restored.items()))
        optimizer.state = opt_tree
        print(f"[mlx] Loaded optimizer state")

    # Load metadata
    meta_path = os.path.join(path, "metadata.json")
    step = 0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        step = meta.get("step", 0)

    print(f"[mlx] Checkpoint restored from {path} (step {step})")
    return step


def clear_cache() -> None:
    """Clear MLX Metal cache for memory management."""
    # Use non-deprecated API if available
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    else:
        mx.metal.clear_cache()


def get_memory_stats() -> dict[str, float]:
    """Get current Metal memory usage in GB."""
    try:
        if hasattr(mx, "get_active_memory"):
            active = mx.get_active_memory() / (1024**3)
            peak = mx.get_peak_memory() / (1024**3)
        else:
            active = mx.metal.get_active_memory() / (1024**3)
            peak = mx.metal.get_peak_memory() / (1024**3)
        return {"active_gb": round(active, 3), "peak_gb": round(peak, 3)}
    except Exception:
        return {"active_gb": 0.0, "peak_gb": 0.0}
