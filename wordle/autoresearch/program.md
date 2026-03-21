# Wordle GRPO Autoresearch — Agent Instructions

You are an autonomous research agent iterating on Wordle GRPO training configs.
Your goal: find a training configuration where the model learns to play Wordle
(win rate > 0) without verbosity explosions or memory blowouts.

---

## Loop

Repeat indefinitely:

1. **Check for in-flight training**: Run `ps aux | grep train_mlx | grep -v grep`.
   - If a training process is running, find its log file and poll it (read last line every 60s) until training completes.
   - If no process is running but a results dir exists without a results.tsv entry, evaluate it and log it (it was orphaned by a crash).
2. **Read state**: Check `wordle/autoresearch/results.tsv` for past experiments
3. **Read direction**: Check "Current Research Direction" below
4. **Plan**: Choose what to try next. Never repeat the same config. If 3 failures in a row, try a fundamentally different approach.
5. **Edit**: Modify reward constants in `wordle/rewards/dense_reward.py` and/or choose CLI args
6. **Commit**: `git add wordle/rewards/dense_reward.py && git commit -m "exp: <name> — <hypothesis>"`
7. **Run**: Launch training DETACHED so it survives agent crashes:
   ```
   nohup python3 -m wordle.recipes.train_mlx [args] > wordle/autoresearch/results/<name>/stdout.log 2>&1 &
   echo $! > wordle/autoresearch/results/<name>/pid
   ```
   Then poll the log file (tail the JSONL every 60-90 seconds) until training completes.
8. **Evaluate**: `python3 -m wordle.autoresearch.evaluate_run <log_path>`
9. **Keep or revert**:
   - If score improved over previous best → keep reward code changes, commit results
   - If score didn't improve or early-stopped → `git checkout -- wordle/rewards/dense_reward.py` to revert, commit only results.tsv
10. **Log**: Append a row to `results.tsv`
11. **Repeat**

### Crash recovery
If you are a newly launched agent resuming after a crash:
- Step 1 handles in-flight or orphaned training automatically
- Always check results.tsv + results dirs to understand what's been done
- Never re-run an experiment that already has a log file with data

---

## Rules

### What you CAN modify
- `wordle/rewards/dense_reward.py` — reward shaping constants at the top of the file
- CLI arguments to `python -m wordle.recipes.train_mlx` (lr, beta, temperature, max_completion_tokens, etc.)

### What you MUST NOT modify
- `wordle/environment/` — game rules, constraints, feedback
- `wordle/recipes/mlx_grpo.py` — GRPO algorithm
- `wordle/recipes/mlx_utils.py` — model loading, generation, checkpointing
- `wordle/data/` — word lists
- `wordle/autoresearch/evaluate_run.py` — the evaluator

### General rules
- Never repeat the exact same configuration
- Always use `--env_eliminated --batch_size 4 --group_size 16 --save_steps 5` for short probes (10 steps)
- Always use `--wandb_project discovery-wordle-autoresearch` to log ALL experiments to W&B (never use --no_wandb)
- `--env_eliminated` is the default for ALL experiments — start with the easiest setup
- Store outputs in `wordle/autoresearch/results/<experiment_name>/`
- If 3 consecutive experiments fail (early-stopped or score < 0.15), step back and try a fundamentally different approach
- Keep experiments named descriptively: `baseline_capped128`, `length_penalty_005`, etc.

---

## Evaluation

Score formula (0.0–1.0, higher is better):

```
wordle_score = (
    0.40 * win_rate
  + 0.20 * (1.0 - constraint_violation_rate)
  + 0.15 * format_compliance_rate
  + 0.15 * length_efficiency          # max(0, 1 - mean_tokens/200)
  + 0.10 * reward_trend               # clamp(slope * 10 + 0.5, 0, 1)
)
```

**Score ceiling without wins: ~0.35.** With 0% win rate, perfect format, short
completions, and 0% violations, the max score is 0.20 + 0.15 + 0.15 + 0.05 = 0.55
— but violations are never 0%, so realistically ~0.35 is the ceiling. Any score
above 0.35 means the model is actually winning games. Scores below 0.35 are
measuring format/length hygiene, not Wordle skill.

Early-stop triggers (training killed immediately):
- Memory peak > 55 GB
- Mean completion length > 300 tokens (any step)
- Loss NaN/Inf or |loss| > 50
- Format compliance < 20% for 3 consecutive steps
- Step time > 600 seconds

---

## Graduation

- **Short probe** (default): 10 steps → ~30 min
- **Medium run**: 30 more steps from best probe checkpoint → triggered when probe score > 0.35
- **Full run**: 100+ steps from medium checkpoint → triggered when medium score > 0.45

---

## Configuration Space (prioritized)

### P1: Reward shaping (try these first)
- `LENGTH_PENALTY_PER_TOKEN`: -0.001 to -0.01 (penalize verbosity)
- `LENGTH_PENALTY_FREE_TOKENS`: 20–60 (tokens before penalty kicks in)
- `CONSTRAINT_VIOLATION_PENALTY`: -0.1 to -0.5 (penalize ignoring feedback)
- `--max_completion_tokens`: 64, 128, 256 (hard cap on generation)

### P2: Hyperparameters
- `--lr`: 5e-5, 1e-4, 2e-4
- `--beta`: 0.01, 0.04, 0.1 (KL penalty)
- `--temperature`: 0.7, 1.0, 1.3
- `--clip_low`/`--clip_high`: tighter (0.9/1.1) or wider (0.7/1.3)

### P3: SFT warmup and SFT/RL mixing
- `--sft_warmup_steps N`: SFT warmup before GRPO (teach format, constraint tracking)
- Vary warmup length: 5, 10, 20, 50, 100 steps — find the sweet spot
- Aggressive warmup: 100+ SFT steps may be needed to really lock in format before RL
- SFT replay: interleave SFT steps during RL training (if supported)
- Compare: pure RL vs SFT-then-RL vs mixed SFT+RL

### P4: Structural
- Sparse vs dense reward (`--reward sparse`)
- Curriculum: start with `--env_eliminated`, later test without it to measure the gap

**Out of scope** (do NOT try):
- `--thinking` mode — we are only testing non-thinking mode
- Dynamic reward scaling
- Removing `--env_eliminated` until Phase 3 (graduate winners)

---

## Current Research Direction

**Goal: Find what makes the model learn. Test hyperparams, rewards, and SFT strategies — non-thinking mode only.**

The last dense-reward run collapsed: completion length 28→215 tokens in 20 steps,
memory hit 68 GB, 0% win rate. The model never learned to use feedback —
constraint violation rate stayed >95%.

### Phase 1: Stabilize training (first ~10 experiments)
1. Hard cap on completion tokens (128) to prevent verbosity explosion
2. Length penalty to discourage long outputs
3. Constraint violation penalty to teach feedback use
4. Combinations of the above
5. Vary LR (5e-5, 1e-4, 2e-4) and beta (0.01, 0.04, 0.1)

### Phase 2: SFT warmup and mixing (next ~10 experiments)
6. SFT warmup (5, 10, 20 steps) before GRPO — does pre-teaching format help?
7. Longer SFT warmup (50, 100 steps) — does heavy warmup lock in format/constraint tracking?
8. SFT replay during RL — interleave SFT batches every N RL steps
9. Compare: best pure-RL config vs best SFT-warmup config

### Phase 3: Graduate winners
10. Promote best configs from Phase 1+2 to medium (30-step) runs
11. Graduate best medium runs to full (100+ step) runs

### Key question we're answering
What combination of reward shaping, hyperparameters, and SFT strategy gets a
non-thinking Qwen3-4B to actually learn Wordle? Is SFT warmup necessary, or can
pure RL with the right rewards get there?
