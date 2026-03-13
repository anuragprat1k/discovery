# Iterative Countdown — Session Logbook

## 2026-03-12 — Eval Pipeline Implementation & Config Updates

### What was done

**Goal**: Implement the multi-turn Tinker eval sampling in `eval_pass_at_k.py` (was a stub), generate the dataset, and update training configs for the 3 Countdown experiments.

### Step 1: Ran existing tests
- `pytest iterative_countdown/tests/ -v` → **106 passed** in 7.55s
- No skips (the torch-dependent tests that were expected to skip are apparently absent — all 106 are standalone tests for env, parser, rewards, dataset)

### Step 2: Generated dataset
```
python -m iterative_countdown.data.generate_dataset \
    --n_train 500 --n_eval 200 --seed 42 \
    --output_dir iterative_countdown/data/
```
- Train: 500 problems (150 easy / 200 medium / 150 hard), 338 have exact solutions (68%)
- Eval: 200 problems (60 easy / 80 medium / 60 hard), 132 have exact solutions (66%)
- Files: `train_problems.json`, `eval_problems.json`, `dataset_metadata.json`

### Step 3: Implemented `eval_pass_at_k.py` main()
Replaced the stub with functional multi-turn Tinker eval. Key design:

- **Multi-turn episode simulation**: Each sample creates a fresh `CountdownMessageEnv`, loops up to `max_turns` calling the model via Tinker, stepping the env, and checking termination.
- **Two modes**:
  - `--model Qwen/Qwen3-4B-Instruct-2507` → base model eval (step 0). Creates a LoRA training client then gets a sampling client.
  - `--tinker_path tinker://run-id/weights/step_N` → checkpoint eval. Restores from state then gets sampling client.
- **Frugal sampling**: `n_samples` defaults to `max(k_values)`. For pass@1-only runs, this means just 1 sample per problem (not 24). Only generates more when higher k values are requested.
- **Resume support**: JSONL sidecar file (`step_NNNN.partial.jsonl`) stores per-problem results. If interrupted, re-running resumes from where it left off.
- **Output format**: Includes `per_problem` list for compatibility with `eval_discovery.py`, plus `by_difficulty` breakdown, overall pass@k, and `mean_best_distance`.

**CLI flags**:
```
--problems_path, --output_dir, --n_samples (default: max(k_values)),
--k_values (default: [1]), --step, --max_turns (default: 5),
--model, --tinker_path, --temperature (default: 1.0), --max_tokens (default: 256)
```

**Not implemented** (intentional):
- Batching across episodes within a turn (would require significant refactoring of the async loop for marginal speedup at n_samples=1). Sequential per-episode is fine for the frugal approach.

### Step 4: Updated training configs

**`recipes/train.py`**:
- `n_eval: int = 100` → `n_eval: int = 200`
- Added `max_steps: int = 500` to CLIConfig
- Passes `max_steps` through `build_config()` to tinker-cookbook `Config`

**`recipes/e1_binary.py`, `e2_dense.py`, `e3_prime.py`**:
- Added `train_problems_path="iterative_countdown/data/train_problems.json"`
- Added `eval_problems_path="iterative_countdown/data/eval_problems.json"`
- Added `max_steps=500`

### Step 5: Smoke tested environment
- Loaded eval problems, created `CountdownMessageEnv` for 3 problems
- Verified prompt format, step mechanics, rewards
- Easy problem (target=6, numbers=[4,2]): `4+2=6`, reward=1.0, done=True
- Medium problem (target=177, numbers=[100,2,7,25]): `100+2=102`, reward=0.35, done=False

### Step 6: Ran tests after all changes
- `pytest iterative_countdown/tests/ -v` → **106 passed** in 10.60s
- Additional manual tests:
  - eval module import and helpers: all OK
  - pass_at_k estimator: correct
  - evaluate_countdown_episode: correct (multi-turn)
  - score_problem + aggregate_results: correct
  - Sidecar write/read round-trip: correct
  - Frugal n_samples logic: k=[1] → n=1, k=[1,4,16] → n=16

### Files modified
| File | Change |
|------|--------|
| `iterative_countdown/evaluation/eval_pass_at_k.py` | Full rewrite: stub → functional Tinker multi-turn eval |
| `iterative_countdown/recipes/train.py` | n_eval=200, added max_steps=500, passed to Config |
| `iterative_countdown/recipes/e1_binary.py` | Added problem paths, max_steps=500 |
| `iterative_countdown/recipes/e2_dense.py` | Added problem paths, max_steps=500 |
| `iterative_countdown/recipes/e3_prime.py` | Added problem paths, max_steps=500 |

### Files created
| File | Purpose |
|------|---------|
| `iterative_countdown/data/train_problems.json` | 500 training problems |
| `iterative_countdown/data/eval_problems.json` | 200 eval problems |
| `iterative_countdown/data/dataset_metadata.json` | Generation metadata |

### What's left to do (requires Tinker API)
1. Run baseline eval (step 0) with `--model Qwen/Qwen3-4B-Instruct-2507 --k_values 1`
2. Run 3 training experiments: `e1_binary`, `e2_dense`, `e3_prime`
3. Checkpoint eval at steps 50, 100, 150, 200, 300, 500
4. Discovery metrics + plots

### Notes
- tinker-cookbook is not pip-installable (it's a local/custom package). Training configs import it at runtime. The eval code only imports tinker (not tinker-cookbook).
- The `max_steps` field was added to `Config()` call — this assumes tinker-cookbook's Config supports it. If not, it'll be a clear error at training time and easy to fix.

## 2026-03-13 — V2 Countdown Training (50 steps) with Per-Turn + Formatting Fixes

### Context
Re-ran 50-step binary and dense training after two key fixes landed on `main`:
1. **Per-turn training** — train on all turns per episode, not just the last (`fix(train): train on all turns per episode`)
2. **Formatting reward** — +0.1 reward for correct `Expression: <expr>` format in both binary and dense reward modules (`feat(rewards): add formatting reward`)

Rebased `exp/full-countdown-runs` onto `origin/main` — cherry-picked commits were auto-skipped.

### Training Config
- Model: Qwen/Qwen3-4B-Instruct-2507
- LoRA rank: 32, LR: 3e-05
- Batch: 16 problems × 16 rollouts, max 5 turns
- 50 steps, eval every 10 steps (20 problems sampled from eval_50)
- W&B group: `v2-50step`

### Training Metrics Summary

| Metric | Binary V1 | Binary V2 | Dense V1 | Dense V2 |
|--------|-----------|-----------|----------|----------|
| Avg reward (first 10) | 0.016 | 0.303 | 0.008 | 0.225 |
| Avg reward (last 10) | 0.017 | 0.301 | -0.003 | 0.222 |
| Wall-clock time | 66.2m | 52.6m | 80.6m | 63.7m |

Key observation: **V2 rewards are ~15-20x higher than V1** due to the formatting reward (+0.1 per correctly formatted turn). This means the formatting signal is dominating — most of the reward comes from learning to output `Expression: ...` rather than from solving the problem.

### Eval Results (pass@1 on 20-problem sample)

| Step | Binary V1 | Binary V2 | Dense V1 | Dense V2 |
|------|-----------|-----------|----------|----------|
| 10 | 0.02 | 0.05 | 0.00 | 0.05 |
| 20 | 0.00 | 0.00 | 0.04 | 0.00 |
| 30 | 0.04 | 0.00 | 0.02 | 0.05 |
| 40 | 0.00 | 0.00 | 0.02 | 0.00 |
| 50 | 0.00 | 0.00 | 0.00 | 0.00 |

### Eval Results (mean_best_distance on 20-problem sample)

| Step | Binary V1 | Binary V2 | Dense V1 | Dense V2 |
|------|-----------|-----------|----------|----------|
| 10 | 200.6 | 206.7 | 196.5 | 177.7 |
| 20 | 178.6 | 230.8 | 212.2 | 204.7 |
| 30 | 216.0 | 186.8 | 173.6 | 243.3 |
| 40 | 219.6 | 210.8 | 225.2 | 247.0 |
| 50 | 212.4 | 255.2 | 164.7 | 171.4 |

### W&B Links
- Binary V2: https://wandb.ai/anuragprateek/discovery-countdown/runs/faetzkyl
- Dense V2: https://wandb.ai/anuragprateek/discovery-countdown/runs/842qpxuu

### Key Takeaways

1. **Formatting reward dominates**: V2 mean rewards (~0.22-0.30) are vastly higher than V1 (~0.01-0.02), but this is almost entirely from the +0.1 formatting bonus on ~3-5 turns per episode. The actual solve rate barely changed.

2. **No meaningful improvement in pass@1**: Both V1 and V2 hover around 0-5% pass@1 across all steps. The per-turn training fix and formatting reward don't meaningfully improve problem-solving ability at 50 steps.

3. **Distance metrics noisy**: mean_best_distance fluctuates wildly (120-255) across steps for all runs, with no clear trend. The 20-problem eval sample is too small for stable distance estimates.

4. **Binary vs Dense still inconclusive at 50 steps**: Neither reward type shows a clear advantage. The hypothesis about binary "sharpening" vs dense "discovery" cannot be evaluated at this training scale.

5. **Dense V2 has more reward variance**: Dense reward_std (~0.28-0.47) is much higher than binary (~0.08-0.22), expected since dense rewards are continuous while binary rewards cluster near 0/1 + formatting bonus.

### Issues
- Dense training process hit Tinker API connection errors after step 50 completed, causing the step 50 eval to only finish 18/20 problems before the session died. All other evals completed normally.
