# Reward Design: Path-Independent vs Path-Dependent

## Run 1: Path-Independent (Outcome-Only)

```
Per-turn reward: 0
Terminal reward: tests_passed / N + (1.0 if all pass)
Format penalty:  -0.1 if no code block
```

Only the final state matters. No per-turn shaping. Two rollouts that
reach the same final state get identical reward regardless of path.

## Run 2: Path-Dependent (Targeted Fix)

```
Per-turn reward:
  +0.5 × (newly_passing / N)                    [progress credit]
  +0.3 × (targeted_fixes / n_feedback_tests)    [targeted fix bonus]
Terminal:
  +1.0 if all pass
  +0.2 × (turns_remaining / max_turns) if solved [speed bonus]
Format: -0.1 if no code block
```

### Why Path-Dependent is Non-Potential-Based

A reward is potential-based if R(s,a,s') = Φ(s') - Φ(s) for some function Φ
that depends only on the state. Potential-based rewards telescope:
Σ R_t = Φ(s_final) - Φ(s_0), making the total reward path-independent.

**The targeted fix bonus is NOT potential-based because:**

The `targeted_fixes` count depends on which tests were **shown in feedback**
on the previous turn, which depends on the previous state's failing tests.

Consider two paths to the same final state (15/23 tests passing):

**Path A:** Turn 1 fails tests {1,2,3,...,13}. Feedback shows tests {1,2,3,...,10}.
Turn 2 fixes tests {1,2,3,4,5} → targeted_fixes = 5 (all from feedback).

**Path B:** Turn 1 fails tests {1,2,3,...,23} (all fail). Feedback shows tests {1,2,...,10}.
Turn 2 fixes tests {14,15,16,17,18} → targeted_fixes = 0 (none from feedback).

Same final state (15/23), but Path A gets +0.3 × 5/10 = 0.15 bonus and
Path B gets +0.3 × 0/10 = 0 bonus. **Same state, different reward.**

This violates the potential-based condition: there is no function Φ(s) such
that the targeted fix bonus equals Φ(s') - Φ(s), because the bonus depends
on the transition history (which tests were shown), not just the states.

### Why This Matters

Path-dependent reward teaches the model a **multi-turn skill**: read test
feedback, identify which specific tests failed, and target those tests
in the repair. This is analogous to Wordle's yellow tiles — information
about specific positions that the model should use on subsequent turns.

Path-independent reward only teaches the model to produce code that passes
tests. It provides no incentive to read or act on intermediate feedback.
The model learns the same thing whether it gets feedback or not.

### Gradient Signal Comparison

Within a GRPO group of 8 rollouts for the same problem:

**Path-independent:** All failures get reward 0. All solves get reward ~2.0.
Binary variance — either all solve or all fail (no gradient).

**Path-dependent:** Failures get different rewards based on:
- How many tests each rollout newly fixed (0.5 × newly_passing/N varies)
- Whether fixes targeted the feedback tests (0.3 × targeted/shown varies)
This creates within-group variance even among failures, giving GRPO gradient.
