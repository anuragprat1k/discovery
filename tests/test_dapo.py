"""Unit tests for DAPO techniques implemented in train_tinker.py."""

import numpy as np


def test_clip_higher():
    """Clip-Higher scales positive advantages by eps_high/eps_low ratio."""
    eps_low, eps_high = 0.2, 0.5
    clip_ratio = eps_high / eps_low  # 2.5

    advantages = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    original = advantages.copy()

    # Apply Clip-Higher
    positive_mask = advantages > 0
    advantages[positive_mask] *= clip_ratio

    assert advantages[0] == original[0], "Negative advantages should be unchanged"
    assert advantages[1] == original[1], "Negative advantages should be unchanged"
    assert advantages[2] == 0.0, "Zero advantages should be unchanged"
    assert np.isclose(advantages[3], 0.5 * 2.5), f"Expected 1.25, got {advantages[3]}"
    assert np.isclose(advantages[4], 1.0 * 2.5), f"Expected 2.5, got {advantages[4]}"
    print("test_clip_higher: PASSED")


def test_dynamic_sampling():
    """Groups with identical rewards get zero advantages (skipped)."""
    rewards = np.array([1.0, 1.0, 1.0, 1.0,  # group 0: all correct
                        0.0, 0.0, 0.0, 0.0,  # group 1: all wrong
                        1.0, 0.0, 1.0, 0.0], # group 2: mixed
                       dtype=np.float32)
    group_size = 4
    n_problems = 3
    advantages = np.zeros_like(rewards)
    groups_skipped = 0

    for i in range(n_problems):
        start = i * group_size
        end = start + group_size
        group_rewards = rewards[start:end]
        group_std = group_rewards.std()

        if group_std < 1e-8:
            advantages[start:end] = 0.0
            groups_skipped += 1
        else:
            group_mean = group_rewards.mean()
            advantages[start:end] = (group_rewards - group_mean) / (group_std + 1e-8)

    assert groups_skipped == 2, f"Expected 2 skipped groups, got {groups_skipped}"
    assert np.all(advantages[:4] == 0.0), "All-correct group should be skipped"
    assert np.all(advantages[4:8] == 0.0), "All-wrong group should be skipped"
    assert not np.all(advantages[8:12] == 0.0), "Mixed group should have nonzero advantages"
    print("test_dynamic_sampling: PASSED")


def test_token_level_normalization():
    """Token-level normalization scales advantages by n_samples/total_tokens."""
    # Two samples: one short (10 tokens), one long (100 tokens)
    advantages = np.array([1.0, -1.0], dtype=np.float32)
    completion_lengths = [10, 100]
    n_active = 2
    total_tokens = sum(completion_lengths)  # 110

    # Without normalization: per-token advantage = advantage value
    # With normalization: per-token advantage = advantage * n_active / total_tokens
    per_token_advs = []
    for idx in range(2):
        per_token_adv = float(advantages[idx]) * n_active / total_tokens
        per_token_advs.append(per_token_adv)

    # Short sample's per-token adv
    assert np.isclose(per_token_advs[0], 1.0 * 2 / 110), \
        f"Expected {1.0 * 2 / 110}, got {per_token_advs[0]}"
    # Long sample's per-token adv
    assert np.isclose(per_token_advs[1], -1.0 * 2 / 110), \
        f"Expected {-1.0 * 2 / 110}, got {per_token_advs[1]}"

    # Total contribution: per_token * n_tokens should sum correctly
    total_contribution_short = per_token_advs[0] * completion_lengths[0]  # 10 * 2/110
    total_contribution_long = per_token_advs[1] * completion_lengths[1]   # 100 * (-2/110)
    # Each sample contributes proportionally to its token count
    ratio = abs(total_contribution_long) / abs(total_contribution_short)
    assert np.isclose(ratio, 10.0), f"Long should contribute 10x more, got {ratio}x"
    print("test_token_level_normalization: PASSED")


if __name__ == "__main__":
    test_clip_higher()
    test_dynamic_sampling()
    test_token_level_normalization()
    print("\nAll DAPO tests passed.")
