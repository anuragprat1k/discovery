# Overnight Monitoring Log — 2026-04-07

## Config
- **Runs**: path_indep vs path_dep, max_tokens=8192, max_turns=4, thinking=enabled
- **Model**: Qwen/Qwen3-4B-Instruct-2507, batch=8×8, lr=4e-5, lora_rank=32
- **Wandb**: path_indep=ubutuor9, path_dep=TBD

---


### 22:55 — Both runs started, step 1 complete

**Wandb links:**
- path_indep: https://wandb.ai/anuragprateek/discovery-code-repair/runs/ubutuor9
- path_dep: https://wandb.ai/anuragprateek/discovery-code-repair/runs/xjq4mozp

**Step 1 results:**

| Metric | path_indep | path_dep |
|--------|-----------|---------|
| solve_rate | 0.39 | 0.38 |
| mean_reward | 0.94±1.01 | 0.98±1.02 |
| loss | 0.0027 | -0.0018 |
| avg_turns | 2.9 | 2.9 |
| discovery | 4 | 3 |
| groups_skipped | 7/8 | 5/8 |
| step_time | 584.7s | 537.0s |
| no_code | **0** | **0** |

**Key observations:**
- Zero truncation (no_code=0) — 8192 tokens + thinking mode resolved the 10% truncation from prior runs
- ~10 min per step with 4 turns (was ~1.5 min with 2 turns and no thinking)
- path_dep has fewer skipped groups (5/8 vs 7/8) — per-turn shaping creates within-group variance even when all fail, as expected
- Both starting step 2

### 23:35 — Steps 1-4 progress

**path_indep** (3 steps done):

| Step | solve | reward | loss | turns | disc | skip | time |
|------|-------|--------|------|-------|------|------|------|
| 1 | 0.39 | 0.94±1.01 | 0.0027 | 2.9 | 4 | 7/8 | 585s |
| 2 | 0.39 | 0.91±1.01 | 0.0016 | 2.9 | 8 | 6/8 | 582s |
| 3 | 0.47 | 1.04±1.06 | 0.0001 | 2.6 | 11 | 6/8 | 318s |

**path_dep** (4 steps done):

| Step | solve | reward | loss | turns | disc | skip | time |
|------|-------|--------|------|-------|------|------|------|
| 1 | 0.38 | 0.98±1.02 | -0.0018 | 2.9 | 3 | 5/8 | 537s |
| 2 | 0.44 | 1.04±1.08 | -0.0005 | 2.8 | 7 | 6/8 | 488s |
| 3 | 0.47 | 1.11±1.13 | -0.0011 | 2.7 | 10 | 5/8 | 441s |
| 4 | 0.03 | 0.30±0.58 | 0.0021 | 3.9 | 12 | 6/8 | 898s |

**Observations:**
- no_code ≈ 0 across both runs (truncation solved)
- Some sandbox calls taking 55-133s (problems with expensive test suites)
- Step time varies 318s-898s depending on problem difficulty
- path_dep consistently skipping fewer groups (more within-group variance from per-turn shaping)
- Discovery rate similar so far (11 vs 12)
- Step 4 of path_dep hit a hard batch (3% solve, nearly 4 turns avg)
- Estimated completion: ~200 steps × ~8 min/step ≈ 27 hours

### 00:15 — Steps 4-5

Both processes alive. ~10-17 min per step depending on difficulty.

**path_indep** (4 steps): step 4 hit hard batch (5% solve, 1060s). disc=12.
**path_dep** (5 steps): step 5 back to 30% solve after hard step 4. disc=14.

At ~10 min/step average, ETA for 200 steps is ~33 hours (tomorrow evening).

No errors, no crashes, no truncation. Monitoring continues.

### 01:00 — Step 9 (1hr check)

Both runs at step 9, perfectly in sync. Both processes alive.

**Summary steps 1-9:**

| Metric | path_indep | path_dep |
|--------|-----------|---------|
| Discovery | 17 | 17 |
| Avg solve_rate | 0.23 | 0.24 |
| Avg step time | 618s (~10 min) | 604s (~10 min) |
| Zero-solve steps | 1 (step 9) | 1 (step 9) |
| Groups skipped (avg) | 6.7/8 | 6.3/8 |

**Concerning**: Both runs have very high skip rates (6-8/8 groups skipped per step). This means most groups have zero within-group variance → zero gradient → wasted compute. path_dep is slightly better (6.3 vs 6.7 avg skipped) as expected from per-turn shaping.

Step 9 was brutal: 0% solve, all groups skipped, 1300s for both. The hard LCB problems dominate when randomly sampled.

**Step 10 will trigger first eval** — trajectories will be saved.

**ETA**: At ~10 min/step, 200 steps ≈ 33 hours total. Currently 1.5 hrs in → ~31.5 hrs remaining (Wednesday ~8:30 AM).

### 02:30 — Step 12, first eval done

Both runs at step 12. Both processes alive.

**First eval (step 10):**
- path_indep: **4/20 = 0.20** pass@1
- path_dep: **4/20 = 0.20** pass@1
- Trajectories saved to `checkpoints/lcb_path_{indep,dep}_8k_s42/eval_trajectories_step_0010.jsonl`

**Steps 1-12 summary:**

| Metric | path_indep | path_dep |
|--------|-----------|---------|
| Discovery | 21 | 22 |
| Mastery | 0 | 0 |
| Eval pass@1 (step 10) | 0.20 | 0.20 |
| Avg step time | ~600s | ~600s |

No divergence yet between conditions — eval pass@1 identical, discovery similar. Still very early (6% of training). The high skip rate (6-8/8) means effective training steps are sparse.

### 05:30 — Step 21, second eval done

Both runs at step 21, both alive.

**Eval results:**

| Step | path_indep | path_dep |
|------|-----------|---------|
| 10 | 4/20 = 0.20 | 4/20 = 0.20 |
| 20 | 3/20 = 0.15 | 3/20 = 0.15 |

**Steps 11-21 summary:**

| Metric | path_indep | path_dep |
|--------|-----------|---------|
| Discovery | 31 | 32 |
| Mastery | 0 | 0 |
| Avg solve_rate | 0.15 | 0.16 |
| Avg skip rate | 7.0/8 | 6.9/8 |

**Observations:**
- Eval pass@1 dropped 0.20 → 0.15 for both — mild regression, possibly noise with only 20 eval problems
- Discovery growing steadily (30-32 unique problems solved at least once across training)
- Still zero mastery (no problem solved 3+ times)
- Skip rates remain very high (7/8) — most groups have zero within-group variance
- Step 16 was extremely slow for both (~1590s = 26 min) — must be a problem with very expensive tests
- No divergence between conditions yet. Both tracking identically on eval
- path_dep has 1 more discovery (32 vs 31) and slightly fewer skips, but difference is marginal

**Pace**: 21 steps in ~7 hours = ~3 steps/hr. ETA for 200 steps: ~60 more hours (Thursday morning). Slower than initial estimate due to hard problem batches.

### 08:30 — Step 34, third eval done

Both at step 34, both alive.

**Eval results over time:**

| Step | path_indep | path_dep |
|------|-----------|---------|
| 10 | 4/20 = 0.20 | 4/20 = 0.20 |
| 20 | 3/20 = 0.15 | 3/20 = 0.15 |
| 30 | 3/20 = 0.15 | 3/20 = 0.15 |

**Steps 21-34 summary:**

| Metric | path_indep | path_dep |
|--------|-----------|---------|
| Discovery | 48 | 48 |
| Mastery | 2 | 2 |

**Observations:**
- Eval flatlined at 0.15 for both conditions across steps 20-30
- Discovery growing steadily: 48 unique problems solved for both (out of 599)
- First mastery at step 24 for both — same problem likely
- Conditions remain indistinguishable on all metrics
- path_dep skip rate slightly lower on step 24 (4/8 vs 5/8) — the one clear structural advantage of per-turn shaping
- Step 24 was a good batch for both: 48% solve rate
- Pace: ~34 steps in ~10 hours ≈ 3.4 steps/hr. ETA 200 steps: ~49 more hours (Thursday ~10 AM)

### 11:50 — Killed batch_size=8 runs, restarted with batch_size=32

**Why**: After 47 steps, eval pass@1 flatlined at 0.15-0.20. Analysis showed:
- Zero multi-turn improvement: model never improved between turns on unsolved problems
- Same 3 problems solved at every eval checkpoint (problems 5, 7, 9), always on turn 1
- 6-8/8 groups skipped per step (zero within-group variance → no gradient)
- Batch_size=8 (64 samples/step) was 16x smaller than the tinker-cookbook recipe (1024 samples/step)

**Changes for new runs:**
- batch_size: 8 → 32 (256 samples/step, 4x more training signal)
- Added asyncio.Semaphore(16) to cap concurrent sandbox subprocesses
- Output dirs: `checkpoints/lcb_path_{indep,dep}_b32_s42`

**New wandb:**
- path_indep: https://wandb.ai/anuragprateek/discovery-code-repair/runs/7e8v98ld
- path_dep: https://wandb.ai/anuragprateek/discovery-code-repair/runs/wgn2dpjk

### 12:10 — batch_size=32 step 1 results

**path_indep step 1:**
- solve=0.32, reward=0.78±0.97
- **242 training datums** (vs ~40 with batch=8, 6x more)
- 11/32 groups trained, 21/32 skipped (vs 1-2/8 trained before)
- 12 discoveries in one step
- Step time: 919s (~15 min)
- no_code: 0

**path_dep**: still completing step 1 (sandbox ~135s per turn batch)

This is dramatically better training signal — 6x more datums per step, 5-6x more non-skipped groups. The skip rate is still ~65% but in absolute terms we're getting 11 useful groups per step vs 1-2 before.

### 13:10 — batch_size=32 step 3

Both at step 3, alive. ~20 min/step.

| Step | path_indep solve | path_dep solve | indep skip | dep skip | indep disc | dep disc |
|------|-----------------|---------------|-----------|---------|-----------|---------|
| 1 | 0.32 | 0.31 | 21/32 | 24/32 | 12 | 12 |
| 2 | 0.16 | 0.18 | 26/32 | 26/32 | 17 | 18 |
| 3 | 0.16 | 0.16 | 28/32 | 29/32 | 21 | 23 |

Skip rates 80-90% — still high. Only 3-6 groups per step get trained. But in absolute terms that's still 3-6x more than the old batch_size=8 runs (1-2 groups/step).

### 15:30 — Step 10, eval running (slow)

Both at step 10, running eval. Eval is sequential (20 problems × 4 turns) and some problems hit long sandbox timeouts.

**Steps 1-10 (batch_size=32):**

| Metric | path_indep | path_dep |
|--------|-----------|---------|
| Discovery | 51 | 52 |
| Mastery | 1 | 1 |
| Avg solve_rate | 0.18 | 0.18 |
| Avg skip rate | 27/32 | 27/32 |

Compare to batch_size=8: 19 discoveries in 10 steps vs 51-52 now (2.7x more, because we see 4x more problems per step). Skip rate ~85% is similar percentage but 4-6 trained groups per step vs 1-2 before.

**Concern**: Eval is very slow — some LCB problems have 100+ tests with 6s timeout each = 14 min per problem worst case. May need to reduce eval problem count or add a timeout cap.

## Overnight Monitor — 2026-04-11 23:41

Config: path_indep vs path_dep, batch=32×8, max_turns=4, thinking=enabled
Fixes deployed: informative feedback, stdin runner fix, 2GiB sandbox mem cap

### 23:41 — procs=2 mem=28403MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 14 | 0.32 | 0.76 | 139 | 8 | 22/32 | 81 | 626s |
| dep | 13 | 0.38 | 0.89 | 133 | 7 | 26/32 | 98 | 804s |

### 23:56 — procs=2 mem=27700MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 15 | 0.42 | 0.92 | 143 | 12 | 25/32 | 107 | 821s |
| dep | 15 | 0.41 | 0.92 | 143 | 13 | 26/32 | 104 | 653s |

### 00:11 — procs=2 mem=27787MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 16 | 0.43 | 0.98 | 152 | 14 | 26/32 | 111 | 568s |
| dep | 16 | 0.43 | 1.03 | 152 | 15 | 26/32 | 110 | 795s |

### 00:26 — procs=2 mem=28177MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 17 | 0.50 | 1.08 | 164 | 14 | 26/32 | 128 | 937s |
| dep | 17 | 0.49 | 1.08 | 164 | 15 | 25/32 | 125 | 629s |

### 00:41 — procs=2 mem=27671MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 19 | 0.41 | 0.89 | 178 | 21 | 28/32 | 106 | 300s |
| dep | 19 | 0.42 | 0.92 | 178 | 20 | 30/32 | 108 | 311s |

### 00:56 — procs=2 mem=27576MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 20 | 0.25 | 0.60 | 180 | 22 | 24/32 | 65 | 1055s |
| dep | 20 | 0.26 | 0.64 | 180 | 21 | 25/32 | 66 | 972s |

### 01:11 — procs=2 mem=27806MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 20 | 0.25 | 0.60 | 180 | 22 | 24/32 | 65 | 1055s |
| dep | 20 | 0.26 | 0.64 | 180 | 21 | 25/32 | 66 | 972s |

### 01:27 — procs=2 mem=27817MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 21 | 0.45 | 0.99 | 185 | 25 | 25/32 | 116 | 730s |
| dep | 21 | 0.44 | 1.01 | 184 | 24 | 25/32 | 113 | 768s |

### 01:42 — procs=2 mem=27716MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 22 | 0.41 | 0.91 | 189 | 28 | 27/32 | 104 | 882s |
| dep | 23 | 0.42 | 1.00 | 194 | 28 | 24/32 | 108 | 568s |

### 01:57 — procs=2 mem=28037MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 23 | 0.43 | 0.96 | 195 | 30 | 26/32 | 110 | 656s |
| dep | 24 | 0.40 | 0.97 | 203 | 29 | 27/32 | 102 | 772s |

### 02:12 — procs=2 mem=28326MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 24 | 0.41 | 0.93 | 204 | 30 | 26/32 | 104 | 809s |
| dep | 25 | 0.56 | 1.32 | 214 | 33 | 22/32 | 144 | 603s |

### 02:27 — procs=2 mem=28215MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 25 | 0.56 | 1.26 | 215 | 33 | 22/32 | 143 | 780s |
| dep | 26 | 0.47 | 1.07 | 221 | 39 | 25/32 | 121 | 902s |

### 02:42 — procs=2 mem=27896MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 27 | 0.29 | 0.62 | 226 | 42 | 29/32 | 73 | 588s |
| dep | 27 | 0.28 | 0.64 | 225 | 42 | 29/32 | 71 | 633s |

### 02:57 — procs=2 mem=27823MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 28 | 0.36 | 0.84 | 232 | 43 | 24/32 | 92 | 623s |
| dep | 28 | 0.36 | 0.94 | 230 | 43 | 26/32 | 91 | 1428s |

### 03:12 — procs=2 mem=27766MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 30 | 0.41 | 0.95 | 236 | 49 | 26/32 | 106 | 582s |
| dep | 30 | 0.42 | 1.03 | 234 | 49 | 22/32 | 108 | 604s |

### 03:28 — procs=2 mem=27690MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 30 | 0.41 | 0.95 | 236 | 49 | 26/32 | 106 | 582s |
| dep | 30 | 0.42 | 1.03 | 234 | 49 | 22/32 | 108 | 604s |

### 03:43 — procs=2 mem=27754MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 31 | 0.44 | 0.99 | 242 | 50 | 26/32 | 113 | 795s |
| dep | 30 | 0.42 | 1.03 | 234 | 49 | 22/32 | 108 | 604s |

### 03:58 — procs=2 mem=27897MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 33 | 0.38 | 0.87 | 247 | 61 | 27/32 | 96 | 576s |
| dep | 32 | 0.47 | 1.08 | 242 | 55 | 28/32 | 120 | 460s |

### 04:13 — procs=2 mem=27890MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 34 | 0.35 | 0.76 | 249 | 69 | 29/32 | 89 | 661s |
| dep | 33 | 0.37 | 0.91 | 244 | 61 | 27/32 | 94 | 616s |

### 04:28 — procs=2 mem=27836MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 35 | 0.22 | 0.54 | 250 | 73 | 25/32 | 57 | 1237s |
| dep | 34 | 0.33 | 0.74 | 246 | 68 | 28/32 | 85 | 796s |

### 04:43 — procs=2 mem=27831MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 35 | 0.22 | 0.54 | 250 | 73 | 25/32 | 57 | 1237s |
| dep | 35 | 0.25 | 0.63 | 248 | 72 | 24/32 | 63 | 1132s |

### 04:58 — procs=2 mem=27782MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 37 | 0.38 | 0.88 | 258 | 78 | 27/32 | 98 | 421s |
| dep | 35 | 0.25 | 0.63 | 248 | 72 | 24/32 | 63 | 1132s |

### 05:14 — procs=2 mem=27706MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 39 | 0.34 | 0.76 | 260 | 87 | 28/32 | 88 | 394s |
| dep | 37 | 0.39 | 0.93 | 255 | 76 | 26/32 | 99 | 538s |

### 05:29 — procs=2 mem=27681MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 40 | 0.43 | 0.94 | 265 | 90 | 26/32 | 109 | 422s |
| dep | 39 | 0.34 | 0.78 | 257 | 85 | 27/32 | 87 | 454s |

### 05:44 — procs=2 mem=27667MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 40 | 0.43 | 0.94 | 265 | 90 | 26/32 | 109 | 422s |
| dep | 40 | 0.44 | 1.02 | 263 | 88 | 23/32 | 113 | 568s |

### 05:59 — procs=2 mem=27739MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 41 | 0.37 | 0.83 | 265 | 92 | 25/32 | 94 | 780s |
| dep | 40 | 0.44 | 1.02 | 263 | 88 | 23/32 | 113 | 568s |

### 06:14 — procs=2 mem=27760MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 42 | 0.42 | 0.94 | 267 | 93 | 23/32 | 108 | 922s |
| dep | 40 | 0.44 | 1.02 | 263 | 88 | 23/32 | 113 | 568s |

### 06:29 — procs=2 mem=27758MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 43 | 0.54 | 1.16 | 269 | 96 | 24/32 | 137 | 1430s |
| dep | 40 | 0.44 | 1.02 | 263 | 88 | 23/32 | 113 | 568s |

### 06:44 — procs=2 mem=27790MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 43 | 0.54 | 1.16 | 269 | 96 | 24/32 | 137 | 1430s |
| dep | 41 | 0.36 | 0.89 | 263 | 90 | 22/32 | 92 | 2239s |

### 06:59 — procs=2 mem=27754MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 44 | 0.37 | 0.82 | 272 | 99 | 28/32 | 94 | 1472s |
| dep | 42 | 0.41 | 0.95 | 266 | 91 | 23/32 | 104 | 766s |

### 07:15 — procs=2 mem=27783MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 45 | 0.58 | 1.25 | 273 | 105 | 26/32 | 148 | 1046s |
| dep | 43 | 0.56 | 1.21 | 268 | 94 | 25/32 | 143 | 1199s |

### 07:30 — procs=2 mem=27667MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 45 | 0.58 | 1.25 | 273 | 105 | 26/32 | 148 | 1046s |
| dep | 44 | 0.36 | 0.89 | 272 | 97 | 26/32 | 92 | 1158s |

### 07:45 — procs=2 mem=27837MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 46 | 0.37 | 0.83 | 275 | 108 | 25/32 | 95 | 1426s |
| dep | 44 | 0.36 | 0.89 | 272 | 97 | 26/32 | 92 | 1158s |

### 08:00 — procs=2 mem=29471MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 47 | 0.39 | 0.87 | 279 | 109 | 25/32 | 100 | 1371s |
| dep | 45 | 0.59 | 1.27 | 273 | 103 | 28/32 | 150 | 1158s |

### 08:15 — procs=2 mem=27722MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 48 | 0.39 | 0.87 | 281 | 116 | 26/32 | 100 | 534s |
| dep | 46 | 0.38 | 0.87 | 274 | 106 | 25/32 | 96 | 1331s |

### 08:30 — procs=2 mem=28553MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 49 | 0.45 | 1.01 | 281 | 123 | 27/32 | 114 | 899s |
| dep | 47 | 0.41 | 0.93 | 278 | 107 | 25/32 | 106 | 1142s |

### 08:45 — procs=2 mem=27804MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 50 | 0.49 | 1.10 | 282 | 129 | 22/32 | 126 | 873s |
| dep | 48 | 0.38 | 0.88 | 280 | 113 | 25/32 | 96 | 511s |

### 09:01 — procs=2 mem=27727MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 50 | 0.49 | 1.10 | 282 | 129 | 22/32 | 126 | 873s |
| dep | 49 | 0.48 | 1.10 | 281 | 121 | 24/32 | 124 | 960s |

### 09:16 — procs=2 mem=27729MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 50 | 0.49 | 1.10 | 282 | 129 | 22/32 | 126 | 873s |
| dep | 50 | 0.49 | 1.15 | 282 | 127 | 22/32 | 125 | 803s |

### 09:31 — procs=2 mem=27806MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 51 | 0.36 | 0.81 | 282 | 132 | 28/32 | 91 | 1447s |
| dep | 50 | 0.49 | 1.15 | 282 | 127 | 22/32 | 125 | 803s |

### 09:46 — procs=2 mem=27807MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 52 | 0.63 | 1.39 | 286 | 139 | 23/32 | 162 | 1060s |
| dep | 50 | 0.49 | 1.15 | 282 | 127 | 22/32 | 125 | 803s |

### 10:01 — procs=2 mem=29824MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 52 | 0.63 | 1.39 | 286 | 139 | 23/32 | 162 | 1060s |
| dep | 51 | 0.37 | 0.87 | 284 | 130 | 26/32 | 95 | 1654s |

### 10:16 — procs=2 mem=27909MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 53 | 0.52 | 1.20 | 286 | 145 | 21/32 | 132 | 2277s |
| dep | 52 | 0.63 | 1.42 | 288 | 137 | 25/32 | 161 | 1105s |

### 10:31 — procs=2 mem=27777MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 55 | 0.53 | 1.14 | 288 | 157 | 30/32 | 136 | 355s |
| dep | 52 | 0.63 | 1.42 | 288 | 137 | 25/32 | 161 | 1105s |

### 10:46 — procs=2 mem=28343MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
parse error: Expecting value: line 1 column 1 (char 0)

### 11:02 — procs=2 mem=27758MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
parse error: Expecting value: line 1 column 1 (char 0)

### 11:17 — procs=2 mem=28256MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 1 | 0.70 | 1.64 | 26 | 0 | 14/32 | 179 | 1288s |
| dep | 1 | 0.71 | 1.78 | 24 | 0 | 12/32 | 181 | 1534s |

### 11:32 — procs=2 mem=28969MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 1 | 0.70 | 1.64 | 26 | 0 | 14/32 | 179 | 1288s |
| dep | 1 | 0.71 | 1.78 | 24 | 0 | 12/32 | 181 | 1534s |

### 11:47 — procs=2 mem=28302MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 1 | 0.70 | 1.64 | 26 | 0 | 14/32 | 179 | 1288s |
| dep | 1 | 0.71 | 1.78 | 24 | 0 | 12/32 | 181 | 1534s |

### 12:02 — procs=2 mem=28718MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 2 | 0.63 | 1.44 | 49 | 0 | 15/32 | 161 | 3428s |
| dep | 2 | 0.63 | 1.55 | 45 | 0 | 12/32 | 162 | 3406s |

### 12:17 — procs=2 mem=29485MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 2 | 0.63 | 1.44 | 49 | 0 | 15/32 | 161 | 3428s |
| dep | 2 | 0.63 | 1.55 | 45 | 0 | 12/32 | 162 | 3406s |

### 12:32 — procs=2 mem=29190MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 2 | 0.63 | 1.44 | 49 | 0 | 15/32 | 161 | 3428s |
| dep | 2 | 0.63 | 1.55 | 45 | 0 | 12/32 | 162 | 3406s |

### 12:48 — procs=2 mem=28780MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 2 | 0.63 | 1.44 | 49 | 0 | 15/32 | 161 | 3428s |
| dep | 2 | 0.63 | 1.55 | 45 | 0 | 12/32 | 162 | 3406s |

### 13:03 — procs=2 mem=29411MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
parse error: Expecting value: line 1 column 1 (char 0)

### 13:18 — procs=2 mem=28632MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 1 | 0.73 | 1.74 | 6 | 0 | 3/32 | 47 | 467s |
| dep | 2 | 0.62 | 1.63 | 13 | 0 | 4/32 | 40 | 651s |

### 13:33 — procs=2 mem=28508MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 3 | 0.86 | 1.94 | 18 | 0 | 3/32 | 55 | 415s |
| dep | 3 | 0.77 | 1.97 | 19 | 0 | 4/32 | 49 | 355s |

### 13:50 — procs=2 mem=28142MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 4 | 0.52 | 1.32 | 22 | 0 | 3/32 | 33 | 935s |
| dep | 4 | 0.59 | 1.61 | 24 | 0 | 4/32 | 38 | 838s |

### 14:05 — procs=2 mem=28126MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 5 | 0.44 | 1.05 | 26 | 0 | 4/32 | 28 | 1286s |
| dep | 5 | 0.50 | 1.28 | 30 | 0 | 4/32 | 32 | 1253s |

### 14:20 — procs=2 mem=31646MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 6 | 0.75 | 1.69 | 33 | 0 | 4/32 | 48 | 579s |
| dep | 6 | 0.75 | 1.75 | 38 | 0 | 3/32 | 48 | 657s |

### 14:36 — procs=2 mem=28164MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 7 | 0.42 | 1.06 | 37 | 0 | 3/32 | 27 | 1261s |
| dep | 8 | 0.78 | 1.85 | 47 | 1 | 5/32 | 50 | 395s |

### 14:51 — procs=2 mem=29708MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 8 | 0.78 | 1.76 | 42 | 1 | 3/32 | 50 | 392s |
| dep | 9 | 0.52 | 1.62 | 51 | 1 | 1/32 | 33 | 948s |

### 15:06 — procs=2 mem=28195MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 9 | 0.53 | 1.36 | 46 | 1 | 2/32 | 34 | 998s |
| dep | 10 | 0.73 | 1.79 | 57 | 1 | 4/32 | 47 | 577s |

### 15:21 — procs=2 mem=28212MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 10 | 0.72 | 1.59 | 51 | 1 | 5/32 | 46 | 618s |
| dep | 10 | 0.73 | 1.79 | 57 | 1 | 4/32 | 47 | 577s |

### 15:36 — procs=2 mem=28308MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 10 | 0.72 | 1.59 | 51 | 1 | 5/32 | 46 | 618s |
| dep | 10 | 0.73 | 1.79 | 57 | 1 | 4/32 | 47 | 577s |

### 16:05 — procs=2 mem=28139MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 10 | 0.72 | 1.59 | 51 | 1 | 5/32 | 46 | 618s |
| dep | 10 | 0.73 | 1.79 | 57 | 1 | 4/32 | 47 | 577s |

### 16:20 — procs=2 mem=28921MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 11 | 0.78 | 1.71 | 58 | 1 | 4/32 | 50 | 354s |
| dep | 11 | 0.77 | 1.71 | 64 | 1 | 5/32 | 49 | 366s |

### 16:36 — procs=2 mem=28961MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 11 | 0.78 | 1.71 | 58 | 1 | 4/32 | 50 | 354s |
| dep | 12 | 0.52 | 1.27 | 68 | 1 | 5/32 | 33 | 1426s |

### 16:51 — procs=2 mem=30572MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 13 | 0.48 | 1.33 | 63 | 1 | 2/32 | 31 | 638s |
| dep | 13 | 0.56 | 1.66 | 72 | 1 | 2/32 | 36 | 577s |

### 17:06 — procs=2 mem=28142MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 14 | 0.52 | 1.21 | 67 | 1 | 3/32 | 33 | 1082s |
| dep | 15 | 0.62 | 1.55 | 82 | 1 | 3/32 | 40 | 320s |

### 17:21 — procs=2 mem=28955MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 15 | 0.62 | 1.47 | 74 | 1 | 4/32 | 40 | 363s |
| dep | 15 | 0.62 | 1.55 | 82 | 1 | 3/32 | 40 | 320s |

### 17:36 — procs=2 mem=28143MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 17 | 0.81 | 1.85 | 83 | 1 | 4/32 | 52 | 537s |
| dep | 17 | 0.81 | 1.95 | 90 | 1 | 5/32 | 52 | 788s |

### 17:51 — procs=2 mem=28815MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 18 | 0.53 | 1.34 | 88 | 1 | 3/32 | 34 | 660s |
| dep | 18 | 0.56 | 1.54 | 95 | 1 | 3/32 | 36 | 564s |

### 18:06 — procs=2 mem=28357MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 19 | 0.70 | 1.59 | 94 | 1 | 2/32 | 45 | 468s |
| dep | 19 | 0.64 | 1.69 | 101 | 1 | 2/32 | 41 | 520s |

### 18:22 — procs=2 mem=28301MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 20 | 0.41 | 1.06 | 97 | 1 | 3/32 | 26 | 885s |
| dep | 20 | 0.44 | 1.24 | 104 | 1 | 3/32 | 28 | 856s |

### 18:37 — procs=2 mem=28343MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 20 | 0.41 | 1.06 | 97 | 1 | 3/32 | 26 | 885s |
| dep | 20 | 0.44 | 1.24 | 104 | 1 | 3/32 | 28 | 856s |

### 18:52 — procs=2 mem=28089MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 20 | 0.41 | 1.06 | 97 | 1 | 3/32 | 26 | 885s |
| dep | 20 | 0.44 | 1.24 | 104 | 1 | 3/32 | 28 | 856s |

### 19:07 — procs=2 mem=28116MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 20 | 0.41 | 1.06 | 97 | 1 | 3/32 | 26 | 885s |
| dep | 20 | 0.44 | 1.24 | 104 | 1 | 3/32 | 28 | 856s |

### 19:22 — procs=2 mem=29982MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 20 | 0.41 | 1.06 | 97 | 1 | 3/32 | 26 | 885s |
| dep | 20 | 0.44 | 1.24 | 104 | 1 | 3/32 | 28 | 856s |

### 19:37 — procs=2 mem=28508MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 21 | 0.41 | 1.04 | 100 | 1 | 3/32 | 26 | 1306s |
| dep | 21 | 0.41 | 1.31 | 107 | 1 | 3/32 | 26 | 1318s |

### 19:52 — procs=2 mem=28296MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 22 | 0.58 | 1.45 | 103 | 1 | 2/32 | 37 | 493s |
| dep | 22 | 0.62 | 1.75 | 110 | 1 | 2/32 | 40 | 668s |

### 20:08 — procs=2 mem=28295MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 23 | 0.75 | 1.70 | 108 | 1 | 6/32 | 48 | 1259s |
| dep | 23 | 0.75 | 1.85 | 115 | 1 | 6/32 | 48 | 1253s |

### 20:23 — procs=2 mem=30571MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 24 | 0.69 | 1.56 | 112 | 2 | 3/32 | 44 | 584s |
| dep | 24 | 0.69 | 1.72 | 119 | 2 | 2/32 | 44 | 854s |

### 20:38 — procs=2 mem=28637MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 26 | 0.78 | 1.80 | 123 | 3 | 6/32 | 50 | 292s |
| dep | 25 | 0.75 | 1.80 | 124 | 3 | 2/32 | 48 | 668s |

### 20:53 — procs=2 mem=28368MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 28 | 0.61 | 1.44 | 132 | 3 | 4/32 | 39 | 422s |
| dep | 26 | 0.77 | 1.84 | 131 | 3 | 5/32 | 49 | 387s |

### 21:08 — procs=2 mem=28163MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 29 | 0.56 | 1.26 | 133 | 4 | 4/32 | 36 | 726s |
| dep | 28 | 0.69 | 1.69 | 141 | 3 | 4/32 | 44 | 501s |

### 21:33 — procs=2 mem=28220MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 30 | 0.61 | 1.46 | 137 | 4 | 3/32 | 39 | 345s |
| dep | 30 | 0.58 | 1.55 | 146 | 4 | 3/32 | 37 | 655s |

### 21:48 — procs=2 mem=27977MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 30 | 0.61 | 1.46 | 137 | 4 | 3/32 | 39 | 345s |
| dep | 30 | 0.58 | 1.55 | 146 | 4 | 3/32 | 37 | 655s |

### 22:07 — procs=2 mem=28091MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 30 | 0.61 | 1.46 | 137 | 4 | 3/32 | 39 | 345s |
| dep | 30 | 0.58 | 1.55 | 146 | 4 | 3/32 | 37 | 655s |

### 22:22 — procs=2 mem=28236MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 30 | 0.61 | 1.46 | 137 | 4 | 3/32 | 39 | 345s |
| dep | 30 | 0.58 | 1.55 | 146 | 4 | 3/32 | 37 | 655s |

### 22:37 — procs=2 mem=28227MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 31 | 0.77 | 1.76 | 141 | 4 | 3/32 | 49 | 1202s |
| dep | 30 | 0.58 | 1.55 | 146 | 4 | 3/32 | 37 | 655s |

### 22:52 — procs=2 mem=30064MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 33 | 0.84 | 1.85 | 153 | 6 | 3/32 | 54 | 410s |
| dep | 30 | 0.58 | 1.55 | 146 | 4 | 3/32 | 37 | 655s |

### 23:07 — procs=2 mem=30856MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 35 | 0.70 | 1.61 | 158 | 6 | 5/32 | 45 | 298s |
| dep | 32 | 0.53 | 1.44 | 154 | 5 | 2/32 | 34 | 407s |

### 23:22 — procs=2 mem=31251MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 35 | 0.70 | 1.61 | 158 | 6 | 5/32 | 45 | 298s |
| dep | 33 | 0.81 | 1.87 | 160 | 6 | 4/32 | 52 | 504s |

### 23:37 — procs=2 mem=32081MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 36 | 0.44 | 1.10 | 163 | 6 | 4/32 | 28 | 1163s |
| dep | 35 | 0.72 | 1.73 | 165 | 6 | 6/32 | 46 | 505s |

### 23:53 — procs=2 mem=29134MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 37 | 0.59 | 1.44 | 168 | 6 | 2/32 | 38 | 1191s |
| dep | 35 | 0.72 | 1.73 | 165 | 6 | 6/32 | 46 | 505s |

### 00:08 — procs=2 mem=34756MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 39 | 0.52 | 1.32 | 177 | 7 | 4/32 | 33 | 466s |
| dep | 36 | 0.44 | 1.17 | 170 | 6 | 4/32 | 28 | 1239s |

### 00:23 — procs=2 mem=28335MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 40 | 0.77 | 1.80 | 179 | 10 | 4/32 | 49 | 329s |
| dep | 37 | 0.56 | 1.62 | 174 | 6 | 2/32 | 36 | 1276s |

### 00:38 — procs=2 mem=28079MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 40 | 0.77 | 1.80 | 179 | 10 | 4/32 | 49 | 329s |
| dep | 39 | 0.53 | 1.52 | 181 | 7 | 2/32 | 34 | 464s |

### 00:53 — procs=2 mem=28194MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 40 | 0.77 | 1.80 | 179 | 10 | 4/32 | 49 | 329s |
| dep | 40 | 0.80 | 1.96 | 184 | 10 | 4/32 | 51 | 283s |

### 01:08 — procs=2 mem=28230MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 40 | 0.77 | 1.80 | 179 | 10 | 4/32 | 49 | 329s |
| dep | 40 | 0.80 | 1.96 | 184 | 10 | 4/32 | 51 | 283s |

### 01:23 — procs=2 mem=28551MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 41 | 0.62 | 1.47 | 182 | 10 | 4/32 | 40 | 413s |
| dep | 40 | 0.80 | 1.96 | 184 | 10 | 4/32 | 51 | 283s |

### 01:39 — procs=2 mem=28441MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 42 | 0.61 | 1.38 | 186 | 10 | 3/32 | 39 | 965s |
| dep | 40 | 0.80 | 1.96 | 184 | 10 | 4/32 | 51 | 283s |

### 01:54 — procs=2 mem=28494MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 44 | 0.44 | 1.18 | 194 | 11 | 3/32 | 28 | 400s |
| dep | 41 | 0.62 | 1.69 | 188 | 10 | 3/32 | 40 | 626s |

### 02:09 — procs=2 mem=28106MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 45 | 0.73 | 1.62 | 197 | 12 | 4/32 | 47 | 1131s |
| dep | 43 | 0.77 | 1.71 | 196 | 10 | 4/32 | 49 | 275s |

### 02:24 — procs=2 mem=37251MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 46 | 0.59 | 1.44 | 200 | 13 | 2/32 | 38 | 708s |
| dep | 44 | 0.52 | 1.54 | 200 | 10 | 3/32 | 33 | 457s |

### 02:39 — procs=2 mem=28379MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 47 | 0.69 | 1.57 | 204 | 13 | 4/32 | 44 | 445s |
| dep | 45 | 0.75 | 1.69 | 203 | 11 | 3/32 | 48 | 874s |

### 02:54 — procs=2 mem=28441MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 48 | 0.61 | 1.40 | 205 | 14 | 4/32 | 39 | 1320s |
| dep | 46 | 0.66 | 1.73 | 206 | 12 | 1/32 | 42 | 623s |

### 03:09 — procs=2 mem=29628MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 49 | 0.75 | 1.67 | 209 | 15 | 4/32 | 48 | 1056s |
| dep | 47 | 0.69 | 1.73 | 210 | 12 | 4/32 | 44 | 984s |

### 03:25 — procs=2 mem=28171MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 50 | 0.56 | 1.34 | 215 | 15 | 2/32 | 36 | 609s |
| dep | 48 | 0.69 | 1.70 | 211 | 13 | 4/32 | 44 | 1186s |

### 03:40 — procs=2 mem=28091MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 50 | 0.56 | 1.34 | 215 | 15 | 2/32 | 36 | 609s |
| dep | 49 | 0.75 | 1.76 | 215 | 14 | 3/32 | 48 | 1135s |

### 03:55 — procs=2 mem=28165MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 50 | 0.56 | 1.34 | 215 | 15 | 2/32 | 36 | 609s |
| dep | 50 | 0.55 | 1.47 | 220 | 14 | 2/32 | 35 | 578s |

### 04:10 — procs=2 mem=28150MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 50 | 0.56 | 1.34 | 215 | 15 | 2/32 | 36 | 609s |
| dep | 50 | 0.55 | 1.47 | 220 | 14 | 2/32 | 35 | 578s |

### 04:25 — procs=2 mem=29739MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 50 | 0.56 | 1.34 | 215 | 15 | 2/32 | 36 | 609s |
| dep | 50 | 0.55 | 1.47 | 220 | 14 | 2/32 | 35 | 578s |

### 04:40 — procs=2 mem=28811MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 51 | 0.38 | 0.98 | 218 | 16 | 2/32 | 24 | 1213s |
| dep | 50 | 0.55 | 1.47 | 220 | 14 | 2/32 | 35 | 578s |

### 04:55 — procs=2 mem=28547MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 52 | 0.59 | 1.42 | 221 | 16 | 5/32 | 38 | 911s |
| dep | 50 | 0.55 | 1.47 | 220 | 14 | 2/32 | 35 | 578s |

### 05:11 — procs=2 mem=28209MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 53 | 0.55 | 1.32 | 222 | 16 | 4/32 | 35 | 958s |
| dep | 50 | 0.55 | 1.47 | 220 | 14 | 2/32 | 35 | 578s |

### 05:26 — procs=2 mem=28737MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 54 | 0.38 | 1.02 | 225 | 16 | 3/32 | 24 | 854s |
| dep | 51 | 0.39 | 1.20 | 222 | 16 | 3/32 | 25 | 1073s |

### 05:41 — procs=2 mem=31757MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 55 | 0.66 | 1.57 | 229 | 16 | 3/32 | 42 | 540s |
| dep | 52 | 0.62 | 1.55 | 225 | 16 | 4/32 | 40 | 922s |

### 05:56 — procs=2 mem=30572MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 56 | 0.42 | 1.22 | 229 | 17 | 3/32 | 27 | 918s |
| dep | 53 | 0.48 | 1.38 | 226 | 16 | 4/32 | 31 | 1106s |

### 06:11 — procs=2 mem=31773MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 58 | 0.67 | 1.51 | 237 | 19 | 5/32 | 43 | 646s |
| dep | 55 | 0.67 | 1.69 | 233 | 16 | 3/32 | 43 | 652s |

### 06:26 — procs=2 mem=31557MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 59 | 0.77 | 1.69 | 239 | 20 | 4/32 | 49 | 889s |
| dep | 56 | 0.44 | 1.63 | 234 | 17 | 3/32 | 28 | 946s |

### 06:41 — procs=2 mem=28328MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 60 | 0.73 | 1.76 | 242 | 21 | 5/32 | 47 | 419s |
| dep | 57 | 0.72 | 1.83 | 237 | 19 | 5/32 | 46 | 441s |

### 06:56 — procs=2 mem=28277MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 60 | 0.73 | 1.76 | 242 | 21 | 5/32 | 47 | 419s |
| dep | 59 | 0.81 | 1.80 | 244 | 20 | 4/32 | 52 | 734s |

### 07:12 — procs=2 mem=28279MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 60 | 0.73 | 1.76 | 242 | 21 | 5/32 | 47 | 419s |
| dep | 60 | 0.64 | 1.68 | 247 | 21 | 3/32 | 41 | 761s |

### 07:27 — procs=2 mem=28294MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 60 | 0.73 | 1.76 | 242 | 21 | 5/32 | 47 | 419s |
| dep | 60 | 0.64 | 1.68 | 247 | 21 | 3/32 | 41 | 761s |

### 07:42 — procs=2 mem=28282MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 60 | 0.73 | 1.76 | 242 | 21 | 5/32 | 47 | 419s |
| dep | 60 | 0.64 | 1.68 | 247 | 21 | 3/32 | 41 | 761s |

### 07:57 — procs=2 mem=30265MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 62 | 0.50 | 1.33 | 249 | 23 | 2/32 | 32 | 460s |
| dep | 60 | 0.64 | 1.68 | 247 | 21 | 3/32 | 41 | 761s |

### 08:12 — procs=2 mem=28280MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 63 | 0.52 | 1.25 | 251 | 24 | 2/32 | 33 | 738s |
| dep | 60 | 0.64 | 1.68 | 247 | 21 | 3/32 | 41 | 761s |

### 08:27 — procs=2 mem=29258MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 65 | 0.81 | 1.80 | 260 | 25 | 5/32 | 52 | 450s |
| dep | 60 | 0.64 | 1.68 | 247 | 21 | 3/32 | 41 | 761s |

### 08:42 — procs=2 mem=28324MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 67 | 0.64 | 1.45 | 267 | 25 | 5/32 | 41 | 638s |
| dep | 62 | 0.50 | 1.45 | 254 | 23 | 2/32 | 32 | 496s |

### 08:58 — procs=2 mem=30054MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 68 | 0.62 | 1.43 | 270 | 25 | 3/32 | 40 | 631s |
| dep | 62 | 0.50 | 1.45 | 254 | 23 | 2/32 | 32 | 496s |

### 09:13 — procs=2 mem=28029MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 70 | 0.80 | 1.83 | 280 | 27 | 3/32 | 51 | 624s |
| dep | 64 | 0.70 | 1.61 | 260 | 24 | 6/32 | 45 | 625s |

### 09:28 — procs=2 mem=29550MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 70 | 0.80 | 1.83 | 280 | 27 | 3/32 | 51 | 624s |
| dep | 66 | 0.75 | 1.78 | 268 | 25 | 6/32 | 48 | 435s |

### 09:43 — procs=2 mem=28664MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 70 | 0.80 | 1.83 | 280 | 27 | 3/32 | 51 | 624s |
| dep | 67 | 0.66 | 1.58 | 272 | 26 | 5/32 | 42 | 695s |

### 09:58 — procs=2 mem=28415MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 70 | 0.80 | 1.83 | 280 | 27 | 3/32 | 51 | 624s |
| dep | 68 | 0.64 | 1.52 | 275 | 26 | 4/32 | 41 | 783s |

### 10:13 — procs=2 mem=28114MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 70 | 0.80 | 1.83 | 280 | 27 | 3/32 | 51 | 624s |
| dep | 70 | 0.72 | 1.85 | 283 | 28 | 5/32 | 46 | 425s |

### 10:28 — procs=2 mem=29471MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 70 | 0.80 | 1.83 | 280 | 27 | 3/32 | 51 | 624s |
| dep | 70 | 0.72 | 1.85 | 283 | 28 | 5/32 | 46 | 425s |

### 10:44 — procs=2 mem=28283MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 70 | 0.80 | 1.83 | 280 | 27 | 3/32 | 51 | 624s |
| dep | 70 | 0.72 | 1.85 | 283 | 28 | 5/32 | 46 | 425s |

### 11:58 — procs=2 mem=28297MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 76 | 0.44 | 1.13 | 293 | 35 | 3/32 | 28 | 1045s |
| dep | 71 | 0.66 | 1.90 | 284 | 30 | 2/32 | 42 | 1502s |

### 12:13 — procs=2 mem=27991MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 77 | 0.55 | 1.36 | 297 | 35 | 2/32 | 35 | 1039s |
| dep | 72 | 0.69 | 1.67 | 287 | 33 | 5/32 | 44 | 598s |

### 12:28 — procs=2 mem=30327MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 77 | 0.55 | 1.36 | 297 | 35 | 2/32 | 35 | 1039s |
| dep | 73 | 0.64 | 1.68 | 288 | 34 | 4/32 | 41 | 1318s |

### 12:44 — procs=2 mem=29043MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 79 | 0.44 | 1.26 | 302 | 36 | 1/32 | 28 | 471s |
| dep | 75 | 0.78 | 1.76 | 295 | 36 | 5/32 | 50 | 538s |

### 12:59 — procs=2 mem=28317MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 80 | 0.16 | 0.68 | 304 | 36 | 1/32 | 10 | 1020s |
| dep | 76 | 0.53 | 1.35 | 297 | 36 | 2/32 | 34 | 909s |

### 13:14 — procs=2 mem=28269MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 80 | 0.16 | 0.68 | 304 | 36 | 1/32 | 10 | 1020s |
| dep | 76 | 0.53 | 1.35 | 297 | 36 | 2/32 | 34 | 909s |

### 13:29 — procs=2 mem=28326MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 80 | 0.16 | 0.68 | 304 | 36 | 1/32 | 10 | 1020s |
| dep | 77 | 0.55 | 1.55 | 302 | 36 | 1/32 | 35 | 1709s |

### 13:44 — procs=2 mem=28383MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 80 | 0.16 | 0.68 | 304 | 36 | 1/32 | 10 | 1020s |
| dep | 78 | 0.61 | 1.56 | 303 | 37 | 4/32 | 39 | 1069s |

### 13:59 — procs=2 mem=28068MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 80 | 0.16 | 0.68 | 304 | 36 | 1/32 | 10 | 1020s |
| dep | 79 | 0.53 | 1.60 | 306 | 37 | 3/32 | 34 | 859s |

### 14:14 — procs=2 mem=28101MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 80 | 0.16 | 0.68 | 304 | 36 | 1/32 | 10 | 1020s |
| dep | 79 | 0.53 | 1.60 | 306 | 37 | 3/32 | 34 | 859s |

### 14:29 — procs=2 mem=28105MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 80 | 0.16 | 0.68 | 304 | 36 | 1/32 | 10 | 1020s |
| dep | 79 | 0.53 | 1.60 | 306 | 37 | 3/32 | 34 | 859s |

### 14:45 — procs=2 mem=28080MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 80 | 0.16 | 0.68 | 304 | 36 | 1/32 | 10 | 1020s |
| dep | 79 | 0.53 | 1.60 | 306 | 37 | 3/32 | 34 | 859s |

### 16:07 — procs=2 mem=28181MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 82 | 0.45 | 1.21 | 308 | 39 | 3/32 | 29 | 1953s |
| dep | 80 | 0.27 | 1.15 | 308 | 37 | 3/32 | 17 | 3421s |

### 16:39 — procs=2 mem=28186MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 83 | 0.67 | 1.49 | 310 | 40 | 5/32 | 43 | 2012s |
| dep | 80 | 0.27 | 1.15 | 308 | 37 | 3/32 | 17 | 3421s |

### 17:02 — procs=2 mem=28172MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 83 | 0.67 | 1.49 | 310 | 40 | 5/32 | 43 | 2012s |
| dep | 80 | 0.27 | 1.15 | 308 | 37 | 3/32 | 17 | 3421s |

### 17:17 — procs=2 mem=28087MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 84 | 0.75 | 1.72 | 313 | 41 | 4/32 | 48 | 1983s |
| dep | 80 | 0.27 | 1.15 | 308 | 37 | 3/32 | 17 | 3421s |

### 17:40 — procs=2 mem=28114MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 85 | 0.70 | 1.61 | 316 | 42 | 5/32 | 45 | 1506s |
| dep | 80 | 0.27 | 1.15 | 308 | 37 | 3/32 | 17 | 3421s |

### 17:56 — procs=2 mem=28262MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 86 | 0.77 | 1.76 | 316 | 43 | 4/32 | 49 | 1158s |
| dep | 80 | 0.27 | 1.15 | 308 | 37 | 3/32 | 17 | 3421s |

### 18:11 — procs=2 mem=34115MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 86 | 0.77 | 1.76 | 316 | 43 | 4/32 | 49 | 1158s |
| dep | 80 | 0.27 | 1.15 | 308 | 37 | 3/32 | 17 | 3421s |

### 19:05 — procs=2 mem=28117MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 87 | 0.61 | 1.49 | 319 | 43 | 3/32 | 39 | 2795s |
| dep | 80 | 0.27 | 1.15 | 308 | 37 | 3/32 | 17 | 3421s |

### 19:46 — procs=2 mem=28304MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 88 | 0.64 | 1.53 | 322 | 44 | 3/32 | 41 | 2782s |
| dep | 80 | 0.27 | 1.15 | 308 | 37 | 3/32 | 17 | 3421s |

### 20:01 — procs=2 mem=28869MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 88 | 0.64 | 1.53 | 322 | 44 | 3/32 | 41 | 2782s |
| dep | 80 | 0.27 | 1.15 | 308 | 37 | 3/32 | 17 | 3421s |

### 20:17 — procs=2 mem=28085MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 88 | 0.64 | 1.53 | 322 | 44 | 3/32 | 41 | 2782s |
| dep | 81 | 0.64 | 1.57 | 310 | 39 | 2/32 | 41 | 2947s |

### 20:32 — procs=2 mem=31212MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 89 | 0.50 | 1.18 | 326 | 44 | 3/32 | 32 | 3307s |
| dep | 81 | 0.64 | 1.57 | 310 | 39 | 2/32 | 41 | 2947s |

### 20:47 — procs=2 mem=28245MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 89 | 0.50 | 1.18 | 326 | 44 | 3/32 | 32 | 3307s |
| dep | 82 | 0.45 | 1.45 | 312 | 40 | 2/32 | 29 | 2312s |

### 21:02 — procs=2 mem=28244MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 90 | 0.73 | 1.65 | 328 | 45 | 5/32 | 47 | 1812s |
| dep | 82 | 0.45 | 1.45 | 312 | 40 | 2/32 | 29 | 2312s |

### 21:24 — procs=2 mem=28295MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 90 | 0.73 | 1.65 | 328 | 45 | 5/32 | 47 | 1812s |
| dep | 82 | 0.45 | 1.45 | 312 | 40 | 2/32 | 29 | 2312s |

### 21:39 — procs=2 mem=28294MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 90 | 0.73 | 1.65 | 328 | 45 | 5/32 | 47 | 1812s |
| dep | 82 | 0.45 | 1.45 | 312 | 40 | 2/32 | 29 | 2312s |

### 22:25 — procs=2 mem=28102MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 90 | 0.73 | 1.65 | 328 | 45 | 5/32 | 47 | 1812s |
| dep | 85 | 0.70 | 1.64 | 319 | 43 | 5/32 | 45 | 757s |

### 22:59 — procs=2 mem=33707MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 90 | 0.73 | 1.65 | 328 | 45 | 5/32 | 47 | 1812s |
| dep | 86 | 0.81 | 1.95 | 319 | 44 | 4/32 | 52 | 686s |

### 23:14 — procs=2 mem=29455MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 90 | 0.73 | 1.65 | 328 | 45 | 5/32 | 47 | 1812s |
| dep | 87 | 0.69 | 1.80 | 324 | 44 | 3/32 | 44 | 1966s |

### 23:29 — procs=2 mem=28284MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 90 | 0.73 | 1.65 | 328 | 45 | 5/32 | 47 | 1812s |
| dep | 87 | 0.69 | 1.80 | 324 | 44 | 3/32 | 44 | 1966s |

### 00:02 — procs=2 mem=28270MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 92 | 0.88 | 1.98 | 331 | 47 | 5/32 | 56 | 536s |
| dep | 88 | 0.62 | 1.64 | 327 | 45 | 4/32 | 40 | 2046s |

### 00:49 — procs=2 mem=28340MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 94 | 0.69 | 1.60 | 336 | 48 | 4/32 | 44 | 2386s |
| dep | 90 | 0.69 | 1.65 | 332 | 46 | 4/32 | 44 | 1239s |

### 01:04 — procs=2 mem=28518MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 94 | 0.69 | 1.60 | 336 | 48 | 4/32 | 44 | 2386s |
| dep | 90 | 0.69 | 1.65 | 332 | 46 | 4/32 | 44 | 1239s |

### 01:19 — procs=2 mem=28357MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 95 | 0.66 | 1.63 | 338 | 49 | 2/32 | 42 | 1542s |
| dep | 90 | 0.69 | 1.65 | 332 | 46 | 4/32 | 44 | 1239s |

### 01:35 — procs=2 mem=28228MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 95 | 0.66 | 1.63 | 338 | 49 | 2/32 | 42 | 1542s |
| dep | 90 | 0.69 | 1.65 | 332 | 46 | 4/32 | 44 | 1239s |

### 01:50 — procs=2 mem=28179MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 97 | 0.69 | 1.60 | 347 | 51 | 4/32 | 44 | 534s |
| dep | 90 | 0.69 | 1.65 | 332 | 46 | 4/32 | 44 | 1239s |

### 02:05 — procs=2 mem=28209MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 98 | 0.67 | 1.58 | 350 | 53 | 3/32 | 43 | 983s |
| dep | 90 | 0.69 | 1.65 | 332 | 46 | 4/32 | 44 | 1239s |

### 02:20 — procs=2 mem=28183MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 99 | 0.77 | 1.70 | 353 | 54 | 3/32 | 49 | 816s |
| dep | 90 | 0.69 | 1.65 | 332 | 46 | 4/32 | 44 | 1239s |

### 02:35 — procs=2 mem=28032MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 100 | 0.62 | 1.49 | 355 | 55 | 4/32 | 40 | 789s |
| dep | 90 | 0.69 | 1.65 | 332 | 46 | 4/32 | 44 | 1239s |

### 02:50 — procs=2 mem=29078MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 100 | 0.62 | 1.49 | 355 | 55 | 4/32 | 40 | 789s |
| dep | 90 | 0.69 | 1.65 | 332 | 46 | 4/32 | 44 | 1239s |

### 03:05 — procs=2 mem=28641MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 100 | 0.62 | 1.49 | 355 | 55 | 4/32 | 40 | 789s |
| dep | 90 | 0.69 | 1.65 | 332 | 46 | 4/32 | 44 | 1239s |

### 03:21 — procs=2 mem=28106MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 100 | 0.62 | 1.49 | 355 | 55 | 4/32 | 40 | 789s |
| dep | 90 | 0.69 | 1.65 | 332 | 46 | 4/32 | 44 | 1239s |

### 03:36 — procs=2 mem=28203MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 100 | 0.62 | 1.49 | 355 | 55 | 4/32 | 40 | 789s |
| dep | 91 | 0.75 | 1.74 | 333 | 49 | 5/32 | 48 | 2770s |

### 03:51 — procs=2 mem=28277MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 100 | 0.62 | 1.49 | 355 | 55 | 4/32 | 40 | 789s |
| dep | 93 | 0.66 | 1.62 | 338 | 50 | 3/32 | 42 | 620s |

### 04:06 — procs=2 mem=28231MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 100 | 0.62 | 1.49 | 355 | 55 | 4/32 | 40 | 789s |
| dep | 93 | 0.66 | 1.62 | 338 | 50 | 3/32 | 42 | 620s |

### 04:21 — procs=2 mem=28318MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 100 | 0.62 | 1.49 | 355 | 55 | 4/32 | 40 | 789s |
| dep | 94 | 0.72 | 1.79 | 340 | 50 | 4/32 | 46 | 1956s |

### 04:36 — procs=2 mem=29106MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 101 | 0.89 | 1.94 | 357 | 57 | 6/32 | 57 | 488s |
| dep | 94 | 0.72 | 1.79 | 340 | 50 | 4/32 | 46 | 1956s |

### 04:51 — procs=2 mem=28256MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 102 | 0.72 | 1.55 | 360 | 58 | 6/32 | 46 | 1246s |
| dep | 95 | 0.78 | 1.93 | 342 | 51 | 3/32 | 50 | 1681s |

### 05:06 — procs=2 mem=28332MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 102 | 0.72 | 1.55 | 360 | 58 | 6/32 | 46 | 1246s |
| dep | 95 | 0.78 | 1.93 | 342 | 51 | 3/32 | 50 | 1681s |

### 05:22 — procs=2 mem=28307MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 103 | 0.45 | 1.13 | 363 | 60 | 4/32 | 29 | 1711s |
| dep | 96 | 0.62 | 1.78 | 346 | 51 | 4/32 | 40 | 1703s |

### 05:37 — procs=2 mem=28299MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 103 | 0.45 | 1.13 | 363 | 60 | 4/32 | 29 | 1711s |
| dep | 97 | 0.73 | 1.77 | 351 | 53 | 4/32 | 47 | 1157s |

### 05:52 — procs=2 mem=28102MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 104 | 0.72 | 1.66 | 364 | 62 | 4/32 | 46 | 1654s |
| dep | 97 | 0.73 | 1.77 | 351 | 53 | 4/32 | 47 | 1157s |

### 06:07 — procs=2 mem=31096MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 104 | 0.72 | 1.66 | 364 | 62 | 4/32 | 46 | 1654s |
| dep | 99 | 0.81 | 1.81 | 357 | 56 | 4/32 | 52 | 646s |

### 06:22 — procs=2 mem=28473MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 105 | 0.47 | 1.21 | 365 | 65 | 1/32 | 30 | 1439s |
| dep | 100 | 0.62 | 1.62 | 359 | 57 | 5/32 | 40 | 925s |

### 06:37 — procs=2 mem=28455MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 106 | 0.28 | 0.89 | 367 | 66 | 3/32 | 18 | 1545s |
| dep | 100 | 0.62 | 1.62 | 359 | 57 | 5/32 | 40 | 925s |

### 06:52 — procs=2 mem=28412MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 106 | 0.28 | 0.89 | 367 | 66 | 3/32 | 18 | 1545s |
| dep | 100 | 0.62 | 1.62 | 359 | 57 | 5/32 | 40 | 925s |

### 07:08 — procs=2 mem=28435MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 106 | 0.28 | 0.89 | 367 | 66 | 3/32 | 18 | 1545s |
| dep | 100 | 0.62 | 1.62 | 359 | 57 | 5/32 | 40 | 925s |

### 07:23 — procs=2 mem=28162MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 107 | 0.33 | 0.94 | 368 | 66 | 2/32 | 21 | 2031s |
| dep | 100 | 0.62 | 1.62 | 359 | 57 | 5/32 | 40 | 925s |

### 07:38 — procs=2 mem=28360MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 108 | 0.66 | 1.53 | 370 | 67 | 5/32 | 42 | 1452s |
| dep | 100 | 0.62 | 1.62 | 359 | 57 | 5/32 | 40 | 925s |

### 07:53 — procs=2 mem=28333MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 109 | 0.58 | 1.42 | 374 | 68 | 6/32 | 37 | 912s |
| dep | 100 | 0.62 | 1.62 | 359 | 57 | 5/32 | 40 | 925s |

### 08:08 — procs=2 mem=28395MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 109 | 0.58 | 1.42 | 374 | 68 | 6/32 | 37 | 912s |
| dep | 100 | 0.62 | 1.62 | 359 | 57 | 5/32 | 40 | 925s |

### 08:23 — procs=2 mem=28320MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 110 | 0.56 | 1.37 | 376 | 70 | 2/32 | 36 | 1444s |
| dep | 100 | 0.62 | 1.62 | 359 | 57 | 5/32 | 40 | 925s |

### 08:38 — procs=2 mem=28406MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 110 | 0.56 | 1.37 | 376 | 70 | 2/32 | 36 | 1444s |
| dep | 101 | 0.84 | 1.90 | 361 | 59 | 6/32 | 54 | 857s |

### 08:54 — procs=2 mem=28199MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 110 | 0.56 | 1.37 | 376 | 70 | 2/32 | 36 | 1444s |
| dep | 101 | 0.84 | 1.90 | 361 | 59 | 6/32 | 54 | 857s |

### 09:09 — procs=2 mem=27749MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 110 | 0.56 | 1.37 | 376 | 70 | 2/32 | 36 | 1444s |
| dep | 102 | 0.75 | 1.62 | 364 | 60 | 7/32 | 48 | 1719s |

### 09:38 — SSH FAILED

### 10:15 — procs=2 mem=27856MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 110 | 0.56 | 1.37 | 376 | 70 | 2/32 | 36 | 1444s |
| dep | 104 | 0.62 | 1.70 | 369 | 65 | 2/32 | 40 | 1872s |

### 10:31 — procs=2 mem=27614MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 111 | 0.39 | 1.05 | 376 | 70 | 1/32 | 25 | 1692s |
| dep | 105 | 0.61 | 1.65 | 370 | 68 | 3/32 | 39 | 1932s |

### 10:46 — procs=2 mem=27988MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 112 | 0.91 | 1.99 | 379 | 72 | 5/32 | 58 | 664s |
| dep | 106 | 0.41 | 1.27 | 371 | 69 | 1/32 | 26 | 1026s |

### 11:01 — procs=2 mem=28260MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 113 | 0.70 | 1.66 | 379 | 73 | 3/32 | 45 | 1033s |
| dep | 107 | 0.31 | 1.09 | 372 | 69 | 2/32 | 20 | 1216s |

### 11:16 — procs=2 mem=28268MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 115 | 0.91 | 2.00 | 381 | 75 | 4/32 | 58 | 377s |
| dep | 108 | 0.66 | 1.65 | 374 | 70 | 5/32 | 42 | 748s |

### 11:31 — procs=2 mem=29261MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 116 | 0.53 | 1.25 | 383 | 75 | 4/32 | 34 | 1110s |
| dep | 109 | 0.70 | 1.75 | 379 | 71 | 5/32 | 45 | 737s |

### 11:46 — procs=2 mem=27999MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 117 | 0.50 | 1.32 | 385 | 76 | 3/32 | 32 | 917s |
| dep | 110 | 0.53 | 1.65 | 382 | 73 | 2/32 | 34 | 1279s |

### 12:19 — procs=2 mem=27953MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 120 | 0.50 | 1.18 | 388 | 82 | 1/32 | 32 | 611s |
| dep | 110 | 0.53 | 1.65 | 382 | 73 | 2/32 | 34 | 1279s |

### 13:24 — procs=2 mem=27917MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 120 | 0.50 | 1.18 | 388 | 82 | 1/32 | 32 | 611s |
| dep | 111 | 0.42 | 1.25 | 382 | 73 | 2/32 | 27 | 896s |

### 13:39 — procs=2 mem=27804MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 121 | 0.77 | 1.73 | 391 | 82 | 3/32 | 49 | 483s |
| dep | 112 | 0.98 | 2.16 | 385 | 75 | 5/32 | 63 | 565s |

### 13:54 — procs=2 mem=27888MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 121 | 0.77 | 1.73 | 391 | 82 | 3/32 | 49 | 483s |
| dep | 113 | 0.73 | 1.77 | 385 | 76 | 5/32 | 47 | 1103s |

### 14:09 — procs=2 mem=29252MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 123 | 0.89 | 1.96 | 394 | 85 | 7/32 | 57 | 499s |
| dep | 115 | 0.98 | 2.16 | 388 | 78 | 5/32 | 63 | 211s |

### 15:19 — procs=2 mem=28189MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 128 | 0.86 | 1.92 | 402 | 91 | 2/32 | 55 | 459s |
| dep | 119 | 0.80 | 1.90 | 392 | 84 | 4/32 | 51 | 463s |

### 15:34 — procs=2 mem=27890MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 129 | 0.38 | 0.95 | 403 | 92 | 4/32 | 24 | 972s |
| dep | 119 | 0.80 | 1.90 | 392 | 84 | 4/32 | 51 | 463s |

### 15:49 — procs=2 mem=27942MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 129 | 0.38 | 0.95 | 403 | 92 | 4/32 | 24 | 972s |
| dep | 120 | 0.45 | 1.29 | 394 | 85 | 2/32 | 29 | 1391s |

### 16:04 — procs=2 mem=27892MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 130 | 0.45 | 1.26 | 404 | 92 | 3/32 | 29 | 1241s |
| dep | 120 | 0.45 | 1.29 | 394 | 85 | 2/32 | 29 | 1391s |

### 18:13 — SSH FAILED

### 18:31 — SSH FAILED

### 23:00 — procs=2 mem=26552MB oom_kill=8

| run | step | solve | reward | disc | mast | skip | sandbox_pass | step_time |
|-----|------|-------|--------|------|------|------|-------------|-----------|
| indep | 150 | 0.61 | 1.48 | 426 | 128 | 3/32 | 39 | 618s |
| dep | 138 | 0.73 | 1.77 | 416 | 112 | 4/32 | 47 | 639s |

