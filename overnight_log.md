# Overnight Training Monitor — Qwen3-4B Dense Reward

**Started**: 2026-03-16 ~23:30
**Run**: `wordle_mlx_dense` (W&B: https://wandb.ai/anuragprateek/discovery-wordle/runs/hhjtqk1s)
**Config**: Qwen3-4B-bf16, dense reward, batch=4, group=8, 200 steps, lr=1e-4, beta=0.04
**Memory at load**: 7.49 GB

---

## Log

### 23:30 — Training launched
- Model loaded successfully (4B params, 14.7M trainable via LoRA)
- Step 0 probe trajectories: all 5 probe words failed (expected — untrained)
- W&B connected, logging to `discovery-wordle/wordle_mlx_dense`
- Monitoring loop set up: checking every 10 minutes

### 23:40 — Check #1 (step 2)
- **Status**: HEALTHY, process running
- loss=7.9e-5 | win_rate=0% | format_compliance=97.9% | constraint_violations=99.4%
- memory: 7.66 GB active, 19.79 GB peak (44 GB headroom)
- ~5 min/step pace, on track for ~17h total
- No issues. Model already knows `<guess>` format but not using feedback yet (expected early).

### 23:50 — Check #2 (step 5)
- **Status**: HEALTHY, process running
- loss=4.3e-3 | reward_mean=1.672 (↑ from 0.875) | win_rate=0% | format=98.4% | violations=97.5%
- memory: 7.66 GB active, 34.0 GB peak (peak jumped — likely trajectory save at step 5)
- Step 5 checkpoint + trajectories saved. All 5 probe words still failing.
- Reward trending up (+0.8 over 5 steps) — dense per-turn shaping providing signal even without wins.
- 6 GB warning in console is spurious (threshold set for smaller models, irrelevant for 4B).

### 00:00 — Check #3 (step 19)
- **Status**: HEALTHY, process running
- loss=3.4e-3 | mean_reward=~1.2-1.7 range | win_rate=0% | format=96.4% | violations=100%
- memory: 7.66 GB active, **67.3 GB peak** (above 64 GB physical — hitting swap on trajectory saves)
- Step pace: ~3.5 min/step for normal steps, ~6 min on checkpoint steps (every 5). On track.
- Reward oscillating 1.0-1.7 — dense shaping providing signal but no wins yet. Constraint violations stuck at ~98-100%.
- All probe words still failing at step 15.
- Fixed monitor script field names (`mean_reward` not `reward_mean`, `time_total` not `time_step`).
- **Note**: Peak memory 67 GB is a concern — macOS swap handles it but may slow checkpoint steps. Active memory stable at 7.7 GB so not a sustained leak. Will continue watching.

### 00:10 — Check #4 (still step 19)
- **Status**: HEALTHY, process running, mid-step (generation phase)
- Same step as last check — step 19 taking longer (time_gen=387s so far). Normal variance.
- Last 5 avg: reward=1.47, loss=0.0025, win_rate=0%
- Estimated ~17h remaining (~1020 min). Pace ~5.6 min/step.
- Fixed monitor script field names in previous check (now showing correct reward values).

### 00:20 — Check #5 (step 20)
- **Status**: HEALTHY, process running. Checkpoint + trajectory save in progress.
- loss=2.0e-3 | mean_reward=1.16 | win_rate=0% | format=94.8% | violations=96.9%
- memory: 7.66 GB active, **68.1 GB peak** (crept up slightly)
- **Step times increasing**: step 16=225s, 17=570s, 18=689s, 19=716s, **20=1131s** (18.8 min!)
- Early steps were ~200s. Likely swap pressure from 68 GB peak or model generating longer completions.
- Last 5 avg: reward=1.45, last 20 avg: reward=1.39
- Revised ETA: ~1293 min (~21.5h) at current pace, up from original 17h estimate.
- No collapse detected — reward stable, loss stable. But slowdown is notable.

### 00:30 — Training stopped by user (step 20)
- User identified collapse. Process killed (SIGTERM, exit 144).
- Monitoring cron cancelled.

---

## Post-Mortem Analysis

### What happened
The model never learned to play Wordle (0% win rate across 20 steps). Two failure modes:

**1. Completion length explosion (primary failure)**
- Completions grew from ~28 tokens (step 1) to **215 tokens** (step 20)
- At step 0: clean `<eliminated>C, A, N</eliminated>\n<guess>STARE</guess>` (16-35 tokens)
- At step 20: paragraphs of "simulated reasoning" disclaimers like *"Since I cannot simulate the game process without feedback..."* before the `<guess>` tag
- This caused step times to explode (200s → 1131s) and memory to hit 68 GB
- Root cause: dense reward gives +0.1 for format compliance and +0.4/+0.2 for tiles but **no penalty for verbosity**

**2. Constraint violations stuck at ~100%**
- Model never learned to use feedback from previous turns
- At step 0 it at least tracked eliminated letters; by step 20 it stopped entirely
- Root cause: constraint violations tracked as metric only, **no negative reward signal**

### Fixes needed for next run
1. **Cap `max_completion_tokens` to ~80** (from 512) — hard constraint on verbosity
2. **Add length penalty** to dense reward (-0.005 per token above 40)
3. **Add constraint violation penalty** (-0.3 per turn)
4. Verify token truncation is enforced in generation code

### 00:30 — Check #6 (still step 20, trajectory save just finished)
- **Status**: HEALTHY but severely slowing. Step 20 trajectory save took 63s (was 13s at step 5).
- Step times: 200s (early) → 570s (step 17) → 1131s (step 20). 5.6x slowdown.
- Likely cause: completion length growing (model learning verbose CoT) + swap pressure at 68 GB peak.
- User requested stop — concerned about collapse. Killing training.

---

# Ablation: Env-Provided Eliminated Letters (no SFT warmup)

**Started**: 2026-03-17 ~10:08
**Run**: `wordle_mlx_dense` (W&B: https://wandb.ai/anuragprateek/discovery-wordle/runs/ty2p2c5q)
**Config**: Qwen3-4B-bf16, dense reward, batch=4, group=4, 200 steps, lr=1e-4, beta=0.04, `--env_eliminated`
**Change**: Environment passes eliminated letters in feedback. Model only outputs `<guess>WORD</guess>`. No SFT warmup.

---

## Log

### 10:08 — Training launched
- Model loaded (4B params, 14.7M trainable LoRA)
- Step-0 probes: all 5 words failed (expected)
- Completions are ~9 tokens (just `<guess>WORD</guess>`) — no length explosion risk

### 10:20 — Check #1 (step 8)
- **Status**: HEALTHY
- reward: 1.1→2.2 (oscillating), win_rate=0%, format=100%, violations=97-100%
- memory: 7.7 GB active, 17.2 GB peak (stable)
- ~2 min/step → ETA ~6.5h for 200 steps
- Key difference from prev run: completions stay at 9 tokens (vs 28→215 explosion before)

