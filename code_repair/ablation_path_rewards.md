# Ablation: Path-Independent vs Path-Dependent Rewards for Multi-Turn Code Repair

## Overview

We ablate the reward structure for multi-turn GRPO training on competitive programming problems from LiveCodeBench (lcbv5). The model generates code, receives test failure feedback, and iterates for up to 4 turns. We compare two reward conditions:

- **Path-independent (outcome-only):** reward depends only on the final test-pass state, not on the trajectory of intermediate turns.
- **Path-dependent (per-turn shaping):** non-terminal turns receive reward for newly-passing tests and for fixing the *specific* tests highlighted in the previous turn's feedback.

The central question: does per-turn reward shaping improve multi-turn code repair, or does the model learn the same behavior from outcome-only signal?

## Reward Definitions

Both reward functions share the same terminal structure (test fraction + solve bonus + speed bonus) but differ on non-terminal turns.

### Path-Independent (`reward_path_indep`)

Non-terminal turns receive zero reward. Only the final turn is scored:

$$r = \begin{cases}
-0.1 & \text{if no code block generated} \\
0 & \text{if non-terminal turn} \\
\frac{\text{tests\_passed}}{\text{tests\_total}} + \mathbb{1}[\text{all\_passed}] \cdot (1.0 + 0.2 \cdot \frac{\text{max\_turns} - \text{turn}}{\text{max\_turns}}) & \text{if terminal}
\end{cases}$$

The speed bonus (0.0-0.2) rewards solving in fewer turns. Maximum possible reward for a single episode: ~2.2 (all tests pass on turn 1).

### Path-Dependent (`reward_path_dep`)

Non-terminal turns receive per-turn shaping with two components:

$$r_{\text{non-terminal}} = 0.5 \cdot \frac{\text{newly\_passing}}{n} + 1.0 \cdot \frac{\text{targeted\_fixes}}{n_{\text{shown}}}$$

where:
- `newly_passing`: tests that pass on this turn but did not pass on the previous turn
- `targeted_fixes`: tests that were *explicitly shown in the feedback message* on the previous turn and now pass -- a direct measure of whether the model used the feedback
- `n_shown`: number of failing tests included in the feedback (capped at 10)

Terminal turns use the same formula as path-independent. This reward is *non-potential* (path-dependent) because `targeted_fixes` depends on which tests were shown in prior feedback, not just the current state.

## Training Setup

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3-4B-Instruct-2507 |
| Training API | Tinker (remote GPU, LoRA rank 32) |
| Algorithm | GRPO with importance-sampling loss |
| Batch size | 8 problems x 8 rollouts = 64 episodes/step |
| Max turns | 4 |
| Max tokens | 8192 (with thinking mode enabled) |
| Temperature | 1.0 (train), 0.6 (eval) |
| Learning rate | 4e-5, AdamW, grad clip 1.0 |
| Dataset | LiveCodeBench v5 (lcbv5), >=5 test cases per problem, 599 training tasks |
| Eval set | 20 held-out lcbv5 problems, pass@1 with 4-turn multi-turn |
| Sandbox | Subprocess with 2 GiB per-process memory cap, 4s per-test timeout |
| Seed | 42 (both runs see identical problem batches) |

Both runs use the same seed, so they see the same problems in the same order. The only difference is the reward function applied to intermediate turns.

### Feedback Format

After each non-terminal turn where the solution fails, the model receives structured feedback:

```
Your solution failed 12/38 tests (26 passed).
- Input: [-5, 0] -> Expected: 0, Got: 1
- Runtime Error: IndexError: list index out of range @ line 42  (input: [1, 2, 3])
- Input: 101\n4 -> Expected: -1, Got: 5
[up to 5 failing test details]

Fix your solution. Output corrected code in a ```python``` code block.
```

Feedback includes exception type + line number for runtime errors, and input/expected/output triples for wrong answers.

## Results

### Training Metrics (150 steps indep, 138 steps dep)

| Window | indep solve | dep solve | indep reward | dep reward | indep disc | dep disc | indep mast | dep mast |
|--------|------------|----------|-------------|-----------|-----------|---------|-----------|---------|
| 1-10   | 0.64 | 0.64 | 1.50 | 1.66 | 51 | 57 | 1 | 1 |
| 11-20  | 0.59 | 0.59 | 1.42 | 1.55 | 97 | 104 | 1 | 1 |
| 21-30  | 0.62 | 0.64 | 1.47 | 1.62 | 137 | 146 | 4 | 4 |
| 31-40  | 0.61 | 0.62 | 1.47 | 1.63 | 179 | 184 | 10 | 10 |
| 41-50  | 0.64 | 0.66 | 1.52 | 1.67 | 215 | 220 | 15 | 14 |
| 51-60  | 0.58 | 0.58 | 1.40 | 1.54 | 242 | 247 | 21 | 21 |
| 61-70  | 0.68 | 0.67 | 1.56 | 1.67 | 280 | 283 | 27 | 28 |
| 71-80  | 0.57 | 0.60 | 1.41 | 1.56 | 304 | 308 | 36 | 37 |
| 81-90  | 0.65 | 0.67 | 1.52 | 1.67 | 328 | 324 | 45 | 44 |
| 91-100 | 0.70 | 0.72 | 1.65 | 1.77 | 342 | 346 | 50 | 52 |
| 101-110| 0.60 | 0.63 | 1.44 | 1.60 | 362 | 361 | 62 | 59 |
| 111-120| 0.64 | 0.63 | 1.50 | 1.64 | 382 | 389 | 76 | 76 |
| 121-130| 0.63 | 0.62 | 1.49 | 1.63 | 397 | 402 | 87 | 90 |
| 131-140| 0.63 | 0.64 | 1.51 | 1.66 | 413 | 416 | 103 | 112 |
| 141-150| 0.64 | -- | 1.51 | -- | 426 | -- | 128 | -- |

**Final state at termination:** indep step 150 (disc=426, mast=128), dep step 138 (disc=416, mast=112).

### Eval Pass@1

| Step | indep | dep |
|------|-------|-----|
| 10   | 11/20 | 10/20 |
| 20   | 10/20 | 11/20 |
| 30   | 10/20 | 10/20 |
| 40   | 9/20  | 11/20 |
| 50   | 11/20 | 10/20 |
| 60   | 10/20 | 10/20 |
| 70   | 10/20 | 11/20 |
| 80   | 10/20 | 10/20 |
| 90   | 10/20 | 10/20 |
| 100  | 9/20  | 11/20 |
| 110  | 10/20 | 10/20 |
| 120  | 10/20 | 10/20 |
| 130  | 10/20 | 9/20 |
| 140  | 11/20 | -- |

**Eval is completely flat at 10-11/20 for both runs across all checkpoints.**

## Key Findings

### 1. Solve rate is flat; mastery is the real signal

Per-step solve rate oscillates 0.57-0.72 for both runs, driven by batch difficulty (same seed = same batches). Neither run shows an upward trend. However, **mastery** (problems solved >=3 times) grows from 1 to 128/112 -- the model is becoming more *reliable* on problems it has already solved, not learning to solve new categories.

### 2. Eval does not improve -- training gains do not generalize

Eval pass@1 is locked at 10-11/20 from step 10 to step 150. The solved set is almost entirely frozen:
- **Always solved** (both runs): problems 0, 1, 4, 5, 7, 9, 10, 16 (8 problems)
- **Never solved** (both runs): problems 3, 6, 11, 12, 14, 18, 19 (7 problems)
- **Variable**: problems 2, 8, 13, 15, 17 flip unpredictably with no learning trend

No hard problem ever transitions from "never solved" to "always solved." Training mastery is not transferring to held-out problems.

### 3. Multi-turn repair barely works

Turn distribution is bimodal: episodes either solve on turn 1 or exhaust all 4 turns.

| Metric | indep | dep |
|--------|-------|-----|
| Solved on turn 1 | 95.8% of solves | 97.0% of solves |
| Solved on turn 2-4 | 4.2% | 3.0% |
| Code changed on turn 2+ | 70.7% | 64.2% |
| Multi-turn improvement rate | 25.7% | 23.7% |
| Regression rate (turn 4 < turn 1) | <12% | <12% |

The model *does* change code between turns (~65-70% of the time) and acknowledges feedback in its reasoning ("let me fix the issue..."). But rewrites rarely lead to solve -- the dominant failure mode is **"small progress, plateau"** where the model passes some tests on turn 1 and then cannot improve further across 3 more attempts.

### 4. Path-dependent shows higher reward but no better outcomes

dep consistently achieves higher mean_reward (1.63 vs 1.47 averaged across all steps) because non-terminal turns contribute positive reward from `newly_passing` and `targeted_fixes`. But this higher reward does not translate into:
- Higher solve rate (0.64 vs 0.62, within noise)
- Better eval pass@1 (both 10-11/20)
- More multi-turn solves (3.0% vs 4.2%, slightly worse)

The per-turn shaping signal provides denser reward but does not teach genuinely better repair behavior.

### 5. Token efficiency diverges

| Metric | indep early | indep late | dep early | dep late |
|--------|------------|-----------|----------|---------|
| comp_tokens_mean | 1256 | 1010 (-20%) | 1268 | 1227 (-3%) |
| comp_tokens_p95 | 3420 | 2349 (-31%) | 3678 | 3244 (-12%) |

indep learns to write shorter, more targeted completions. dep stays verbose -- possibly because per-turn shaping rewards elaborate multi-turn attempts, teaching the model that longer outputs across multiple turns earn more reward.

### 6. Targeted fix rate: dep improves more but from a low base

| | Early | Late | Change |
|--|-------|------|--------|
| indep targeted_fix_avg | 0.41 | 0.45 | +10% |
| dep targeted_fix_avg | 0.42 | 0.54 | +29% |

dep shows stronger improvement in fixing the specific tests highlighted in feedback. The path-dependent reward is successfully teaching more *precise* edits -- but this does not translate to higher solve rate because the model cannot bridge the gap from "fixed 2 failing tests" to "all tests pass."

### 7. dep suffered a verbosity spike at step 80

At step 80, dep exhibited: solve=0.27, avg_turns=3.23, comp_tokens_p95=8192 (max), step_time=3421s. The model was hitting the token limit and using all 4 turns on nearly every episode. This recovered by step 81 (solve=0.58, p95=3050) -- likely a hard batch combined with mode-collapse tendencies from per-turn shaping. indep never showed this behavior.

## Infrastructure Notes

This ablation required resolving several sandbox issues that initially masked the true training dynamics:

1. **stdin test execution bug:** The sandbox runner embedded user code at Python's top level, causing stdin-reading solutions to crash with EOFError before any test executed. ~55% of problems are stdin-style. Fix: only include user code at top level for functional (class method) tests.

2. **Argument length overflow:** Long model-generated code passed via `python -c "CODE"` exceeded OS argument limits (Errno 7). Fix: write runner scripts and user code to temp files.

3. **Uninformative feedback:** The feedback builder silently dropped error details due to fragile `eval()` parsing. Fix: read pre-parsed error dicts directly from the sandbox return value.

4. **Unbounded subprocess memory:** User code with `@lru_cache(maxsize=None)` on multi-argument DP functions consumed 8+ GB per sandbox subprocess, causing container OOM kills. Fix: set `RLIMIT_AS = 2 GiB` per subprocess.

These fixes collectively increased the training solve rate from ~30% to ~70% and the sandbox pass count from ~100 to ~180 per step.

## Conclusion

**The ablation result is a null finding on the metric that matters (eval pass@1).** Both reward conditions produce identical generalization performance (10-11/20) despite 150 steps of training. The path-dependent reward creates denser signal and marginally better targeted-fix behavior, but the bottleneck is the 4B model's ability to generalize code repair strategies to unseen problems -- not the reward structure.

The training mastery gains (1 -> 128) without eval improvement indicate the model is *memorizing* training problems rather than learning transferable debugging skills. Multi-turn repair is structurally limited by the model's capacity to change its approach based on test feedback: 96% of solves happen on turn 1, and turn 2-4 rewrites rarely bridge the gap.

For multi-turn code repair to work, the intervention likely needs to be at the model scale (larger base model), data diversity (more varied problems), or inference strategy (tool use, retrieval) level -- not the reward function.
