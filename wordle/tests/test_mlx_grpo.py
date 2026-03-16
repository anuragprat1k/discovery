"""Unit tests for MLX GRPO algorithm components."""
from __future__ import annotations

import numpy as np
import pytest

from wordle.recipes.mlx_grpo import compute_advantages


# ---------------------------------------------------------------------------
# TestComputeAdvantages (pure numpy, no mlx needed)
# ---------------------------------------------------------------------------

class TestComputeAdvantages:
    def test_basic_normalization(self):
        """Advantages within a group should be mean-centered and std-normalized."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        advantages, groups_skipped = compute_advantages(rewards, group_size=4)
        assert groups_skipped == 0
        assert abs(advantages.sum()) < 1e-6
        assert abs(advantages.std() - 1.0) < 0.1

    def test_zero_variance_group_skipped(self):
        """Groups where all rewards are equal get advantages=0 and are counted as skipped."""
        rewards = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        advantages, groups_skipped = compute_advantages(rewards, group_size=4)
        assert groups_skipped == 1
        assert np.all(advantages == 0.0)

    def test_multiple_groups(self):
        """Each group normalized independently."""
        rewards = np.array(
            [1.0, 3.0, 5.0, 7.0,  # group 0: non-zero variance
             10.0, 10.0, 10.0, 10.0],  # group 1: zero variance
            dtype=np.float32,
        )
        advantages, groups_skipped = compute_advantages(rewards, group_size=4)
        assert groups_skipped == 1
        assert not np.all(advantages[:4] == 0.0)
        assert np.all(advantages[4:8] == 0.0)

    def test_group_mean_centered(self):
        """Each group's advantages should sum to ~0."""
        rewards = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        advantages, _ = compute_advantages(rewards, group_size=4)
        assert abs(advantages[:4].mean()) < 1e-6

    def test_signs_correct(self):
        """Above-mean rewards get positive advantage, below-mean get negative."""
        rewards = np.array([1.0, 5.0], dtype=np.float32)
        advantages, _ = compute_advantages(rewards, group_size=2)
        assert advantages[0] < 0
        assert advantages[1] > 0

    def test_single_element_groups(self):
        """Group size 1 should always be zero variance -> skipped."""
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        advantages, groups_skipped = compute_advantages(rewards, group_size=1)
        assert groups_skipped == 3
        assert np.all(advantages == 0.0)

    def test_empty_rewards(self):
        """Empty rewards array should return empty advantages."""
        rewards = np.array([], dtype=np.float32)
        advantages, groups_skipped = compute_advantages(rewards, group_size=4)
        assert len(advantages) == 0
        assert groups_skipped == 0


# ---------------------------------------------------------------------------
# TestGRPO MLX-specific (requires mlx)
# ---------------------------------------------------------------------------

class TestGRPOMLX:
    """Tests that require mlx to be installed."""

    @pytest.fixture(autouse=True)
    def check_mlx(self):
        pytest.importorskip("mlx")

    def test_grpo_loss_imports(self):
        """Verify mlx_grpo module can be imported."""
        from wordle.recipes.mlx_grpo import _grpo_loss_fn, grpo_step
        assert callable(_grpo_loss_fn)
        assert callable(grpo_step)

    def test_compute_logprobs_for_sequence_shape(self):
        """Test that compute_logprobs_for_sequence returns correct shape."""
        import mlx.core as mx
        import mlx.nn as nn
        from wordle.recipes.mlx_utils import compute_logprobs_for_sequence

        class TinyModel(nn.Module):
            def __init__(self, vocab_size=100, hidden=32):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden)
                self.proj = nn.Linear(hidden, vocab_size)

            def __call__(self, x, cache=None, input_embeddings=None):
                h = self.embed(x)
                return self.proj(h)

        model = TinyModel()
        mx.eval(model.parameters())

        full_tokens = [1, 2, 3, 4, 5]
        prompt_len = 2
        logprobs = compute_logprobs_for_sequence(model, full_tokens, prompt_len)
        mx.eval(logprobs)

        assert logprobs.shape == (3,)
        assert all(lp < 0 for lp in logprobs.tolist())

    def test_grpo_loss_gradient_flows(self):
        """Test that GRPO loss produces finite loss and gradients."""
        import mlx.core as mx
        import mlx.nn as nn
        from wordle.recipes.mlx_grpo import _grpo_loss_fn

        class TinyModel(nn.Module):
            def __init__(self, vocab_size=100, hidden=32):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden)
                self.proj = nn.Linear(hidden, vocab_size)

            def __call__(self, x, cache=None, input_embeddings=None):
                h = self.embed(x)
                return self.proj(h)

        model = TinyModel()
        mx.eval(model.parameters())

        episode = {
            "prompt_tokens_per_turn": [[1, 2, 3]],
            "completion_tokens_per_turn": [[4, 5]],
            "total_turns": 1,
        }
        old_lps = [mx.array([-1.0, -2.0])]

        loss_and_grad_fn = nn.value_and_grad(
            model,
            lambda: _grpo_loss_fn(
                model, [episode], [old_lps], [1.0],
                clip_low=0.8, clip_high=1.2, beta=0.04,
            ),
        )
        loss_val, grads = loss_and_grad_fn()
        mx.eval(loss_val, grads)

        assert not mx.isnan(loss_val).item()
        assert not mx.isinf(loss_val).item()

    def test_memory_stats(self):
        """Test memory stats helper returns valid dict."""
        from wordle.recipes.mlx_utils import get_memory_stats
        stats = get_memory_stats()
        assert "active_gb" in stats
        assert "peak_gb" in stats
        assert stats["active_gb"] >= 0.0

    def test_clear_cache_no_error(self):
        """Test that clear_cache runs without error."""
        from wordle.recipes.mlx_utils import clear_cache
        clear_cache()
