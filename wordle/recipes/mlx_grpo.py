"""GRPO algorithm for MLX: loss, advantage computation, gradient step."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from wordle.recipes.mlx_utils import compute_logprobs_for_sequence, clear_cache


def compute_advantages(
    rewards: np.ndarray,
    group_size: int,
) -> tuple[np.ndarray, int]:
    """Mean-centered, std-normalized advantages within each group.

    Mirrors existing train.py lines 464-476. Groups with zero variance
    get advantages of 0.0 (skipped).

    Args:
        rewards: shape (n_episodes,) — total reward per episode
        group_size: number of episodes per target word

    Returns:
        (advantages, groups_skipped)
    """
    advantages = np.zeros_like(rewards)
    n_groups = len(rewards) // group_size

    groups_skipped = 0
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group = rewards[start:end]
        mean = group.mean()
        std = group.std()

        if std < 1e-8:
            advantages[start:end] = 0.0
            groups_skipped += 1
        else:
            advantages[start:end] = (group - mean) / (std + 1e-8)

    return advantages, groups_skipped


def _grpo_loss_fn(
    model,
    episodes_batch: list[dict],
    old_logprobs_batch: list[list[mx.array]],
    advantages_batch: list[float],
    clip_low: float,
    clip_high: float,
    beta: float,
) -> mx.array:
    """Core differentiable GRPO loss (CISPO variant).

    For each episode's each turn:
        1. Forward pass to get current-policy logprobs
        2. Importance ratio: ratio = exp(new_lp - old_lp)
        3. CISPO clipping: clipped = clip(ratio, clip_low, clip_high)
        4. Per-token: min(ratio * adv, clipped * adv) for positive adv,
                      max(ratio * adv, clipped * adv) for negative adv
        5. KL penalty: beta * (new_lp - old_lp)

    Returns scalar mean loss (negated for gradient ascent → minimization).
    """
    total_loss = mx.array(0.0)
    total_tokens = 0

    for ep, old_lps, advantage in zip(
        episodes_batch, old_logprobs_batch, advantages_batch
    ):
        if abs(advantage) < 1e-8:
            continue

        adv = mx.array(advantage)

        for turn_idx in range(ep["total_turns"]):
            prompt_tokens = ep["prompt_tokens_per_turn"][turn_idx]
            completion_tokens = ep["completion_tokens_per_turn"][turn_idx]
            n_completion = len(completion_tokens)
            if n_completion == 0:
                continue

            full_tokens = prompt_tokens + completion_tokens
            prompt_len = len(prompt_tokens)

            # Current policy logprobs (differentiable)
            new_lp = compute_logprobs_for_sequence(model, full_tokens, prompt_len)

            # Old logprobs for this turn (detached)
            old_lp = mx.stop_gradient(old_lps[turn_idx])
            # Ensure lengths match
            min_len = min(len(new_lp), len(old_lp))
            if min_len == 0:
                continue
            new_lp = new_lp[:min_len]
            old_lp = old_lp[:min_len]

            # Importance ratio
            ratio = mx.exp(new_lp - old_lp)

            # CISPO clipping
            clipped_ratio = mx.clip(ratio, clip_low, clip_high)

            # Surrogate loss (negated: we minimize, GRPO maximizes)
            if advantage > 0:
                surrogate = -mx.minimum(ratio * adv, clipped_ratio * adv)
            else:
                surrogate = -mx.maximum(ratio * adv, clipped_ratio * adv)

            # KL penalty
            kl_penalty = beta * (new_lp - old_lp)

            per_token_loss = surrogate + kl_penalty
            total_loss = total_loss + mx.sum(per_token_loss)
            total_tokens += min_len

    if total_tokens == 0:
        return mx.array(0.0)

    return total_loss / total_tokens


def grpo_step(
    model,
    optimizer,
    episodes_batch: list[dict],
    old_logprobs_batch: list[list[mx.array]],
    advantages_batch: list[float],
    clip_low: float = 0.8,
    clip_high: float = 1.2,
    beta: float = 0.04,
    grad_clip_norm: float = 1.0,
    micro_batch_size: int = 1,
) -> dict:
    """Run GRPO gradient step with micro-batching.

    Accumulates gradients across micro-batches, applies gradient clipping,
    then optimizer step.

    Returns dict with loss and gradient norm.
    """
    n_episodes = len(episodes_batch)

    accumulated_grads = None
    total_loss = 0.0
    n_micro_batches = 0

    for mb_start in range(0, n_episodes, micro_batch_size):
        mb_end = min(mb_start + micro_batch_size, n_episodes)
        mb_episodes = episodes_batch[mb_start:mb_end]
        mb_old_lps = old_logprobs_batch[mb_start:mb_end]
        mb_advs = advantages_batch[mb_start:mb_end]

        # Skip micro-batch if all advantages are zero
        if all(abs(a) < 1e-8 for a in mb_advs):
            continue

        # nn.value_and_grad computes grads w.r.t. model.trainable_parameters()
        # The function should take no arguments (model accessed via closure)
        loss_and_grad_fn = nn.value_and_grad(
            model,
            lambda: _grpo_loss_fn(
                model, mb_episodes, mb_old_lps, mb_advs,
                clip_low, clip_high, beta,
            ),
        )
        loss_val, grads = loss_and_grad_fn()
        mx.eval(loss_val, grads)

        total_loss += loss_val.item()
        n_micro_batches += 1

        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = _add_grads(accumulated_grads, grads)

        clear_cache()

    if accumulated_grads is None or n_micro_batches == 0:
        return {"loss": 0.0, "grad_norm": 0.0}

    # Average gradients over micro-batches
    if n_micro_batches > 1:
        scale = 1.0 / n_micro_batches
        accumulated_grads = _scale_grads(accumulated_grads, scale)

    # Gradient clipping
    grad_norm = _compute_grad_norm(accumulated_grads)
    if grad_clip_norm > 0 and grad_norm > grad_clip_norm:
        scale = grad_clip_norm / (grad_norm + 1e-8)
        accumulated_grads = _scale_grads(accumulated_grads, scale)

    # Apply gradients: optimizer returns updated params, model.update applies them
    model.update(optimizer.apply_gradients(accumulated_grads, model))
    mx.eval(model.parameters(), optimizer.state)

    avg_loss = total_loss / n_micro_batches
    return {"loss": avg_loss, "grad_norm": grad_norm}


def _add_grads(grads_a, grads_b):
    """Element-wise add two gradient trees."""
    if isinstance(grads_a, dict):
        return {k: _add_grads(grads_a[k], grads_b[k]) for k in grads_a}
    elif isinstance(grads_a, list):
        return [_add_grads(a, b) for a, b in zip(grads_a, grads_b)]
    elif isinstance(grads_a, mx.array):
        return grads_a + grads_b
    return grads_a


def _scale_grads(grads, scale: float):
    """Scale gradient tree by a scalar."""
    if isinstance(grads, dict):
        return {k: _scale_grads(v, scale) for k, v in grads.items()}
    elif isinstance(grads, list):
        return [_scale_grads(g, scale) for g in grads]
    elif isinstance(grads, mx.array):
        return grads * scale
    return grads


def _compute_grad_norm(grads) -> float:
    """Compute L2 norm of gradient tree."""
    leaves = [g for g in tree_flatten(grads) if isinstance(g, tuple)]
    arrays = [g[1] for g in leaves] if leaves else _flatten_arrays(grads)
    if not arrays:
        return 0.0
    sum_sq = mx.array(0.0)
    for g in arrays:
        sum_sq = sum_sq + mx.sum(g * g)
    mx.eval(sum_sq)
    return float(mx.sqrt(sum_sq).item())


def _flatten_arrays(grads) -> list[mx.array]:
    """Flatten gradient tree to list of mx.arrays."""
    if isinstance(grads, mx.array):
        return [grads]
    elif isinstance(grads, dict):
        result = []
        for v in grads.values():
            result.extend(_flatten_arrays(v))
        return result
    elif isinstance(grads, list):
        result = []
        for g in grads:
            result.extend(_flatten_arrays(g))
        return result
    return []
