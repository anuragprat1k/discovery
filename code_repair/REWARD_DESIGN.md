# Reward Design: Path-Independent vs Path-Dependent

## Run 1: Path-Independent (Outcome-Only)

```
Non-terminal:  0
Terminal:      tests_passed / N + (1.0 if all_pass) + (0.2 × turns_remaining / max_turns if all_pass)
No code:       -0.1
```

Only the final state matters. No per-turn shaping. Two rollouts that
reach the same final state get identical reward regardless of path.
No incentive to read or act on intermediate feedback.

### Example rewards (N=10, max_turns=4)

| Scenario             | Reward |
|----------------------|--------|
| Turn 1 solve (10/10) | 2.150  |
| Turn 2 solve (10/10) | 2.100  |
| Turn 4 solve (10/10) | 2.000  |
| Turn 4 fail (5/10)   | 0.500  |
| Turn 4 fail (0/10)   | 0.000  |
| Non-terminal (any)   | 0.000  |
| No code block        | -0.100 |

## Run 2: Path-Dependent (Per-Turn Shaping)

```
Non-terminal:  0.5 × (newly_passing / N) + 1.0 × (targeted_fixes / n_feedback_tests)
Terminal:      tests_passed / N + (1.0 if all_pass) + (0.2 × turns_remaining / max_turns if all_pass)
No code:       -0.1
```

Terminal reward is **identical** to path-independent. The difference is
on non-terminal turns, where the model gets two shaping signals:

1. **Progress credit** (0.5 weight): tests that now pass but didn't before
2. **Targeted fix bonus** (1.0 weight): tests that now pass AND were
   shown in the previous turn's error feedback

### Why Path-Dependent is Non-Potential-Based

A reward is potential-based if R(s,a,s') = Φ(s') - Φ(s) for some
function Φ that depends only on the state.

The targeted fix bonus violates this: it depends on which tests were
**shown in feedback** on the previous turn, which depends on the
previous state's failing tests — not just the current state.

Two paths to the same final state (e.g., 15/23 tests passing):

**Path A:** Turn 1 fails tests {1,...,13}. Feedback shows {1,...,10}.
Turn 2 fixes {1,...,5} → targeted_fixes = 5 (all from feedback).

**Path B:** Turn 1 fails tests {1,...,23}. Feedback shows {1,...,10}.
Turn 2 fixes {14,...,18} → targeted_fixes = 0 (none from feedback).

Same final state, different reward. This teaches the model a multi-turn
skill: read test feedback, identify which tests failed, target those.

### Example rewards (N=10, max_turns=4)

| Scenario                          | Reward |
|-----------------------------------|--------|
| Turn 1 solve (10/10)              | 2.150  |
| Turn 2 solve (10/10)              | 2.100  |
| Turn 4 solve (10/10)              | 2.000  |
| Turn 4 fail (5/10)                | 0.500  |
| Non-terminal, new=3, tgt=2/8      | 0.400  |
| Non-terminal, new=5, tgt=0        | 0.250  |
| Non-terminal, new=8, tgt=5/8      | 1.025  |
| Non-terminal, new=0, tgt=0        | 0.000  |
| No code block                     | -0.100 |

## Gradient Signal Comparison

Within a GRPO group of 8 rollouts for the same problem:

**Path-independent:** All failures get reward 0 (non-terminal turns).
Terminal failures get tests_passed/N. Binary variance — either all
solve or all fail → often zero advantage → skipped group.

**Path-dependent:** Failures get different rewards based on:
- How many tests each rollout newly fixed (0.5 × newly_passing/N varies)
- Whether fixes targeted the feedback tests (1.0 × targeted/shown varies)
This creates within-group variance even among failures, giving GRPO gradient.
