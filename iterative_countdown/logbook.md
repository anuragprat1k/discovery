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
