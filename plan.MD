# A Recipe for Discovery: Configuring RL Beyond Sharpening
## Experiment Plan (v5)

## Thesis

Given any LLM checkpoint, RL post-training defaults to sharpening. We identify which RL-time knobs — reward design, exploration pressure, KL budget — shift the outcome toward discovery, and provide a practical recipe for practitioners.

**Why sharpening is the default**: GRPO with binary rewards can only learn from solutions the model already produces. Easy problems sharpen (concentrating mass on already-dominant correct paths). Hard problems get zero gradient (0/16 correct = no signal). The model learns nothing at the frontier. Pass@1 rises (concentration), pass@256 falls (diversity dies).

**Three mechanisms break this**:
1. **Dense rewards** → provide gradient signal on hard problems (partial credit for partial progress)
2. **Exploration pressure** → prevent concentration death spiral on medium problems
3. **Low KL** → let the policy actually move to new regions when gradient points there

Predicted effect size ranking: reward design > exploration > KL. Dense reward attacks the root cause (zero gradient at frontier). The other two only help if some signal already exists.

---

## 1. Diagnostic Framework

### Primary metric: pass@k at {1, 16, 64, 256}
Standard, cheap, computed from 256 samples per problem.

Plot trajectories in **(Δpass@1, Δpass@256)** space over training. Each run produces one curve:
- **(+, +) = Discovery** — greedy improves AND new solutions appear
- **(+, −) = Sharpening** — greedy improves, diversity collapses

The core question: **do different RL configs produce visually different curves?** Specifically, does dense reward keep the trajectory in the (+, +) quadrant longer than binary reward?

### Secondary metrics (all free or near-free)

| Metric | What it measures | Cost |
|--------|-----------------|------|
| **Unique correct answers** | Distinct final answers among correct samples | Free (string comparison) |
| **Base-model perplexity of RL solutions** | PPL of RL's correct solutions under starting checkpoint | 1 forward pass per solution |
| **Token-level top-1 concentration** | Probability mass on greedy token at high-entropy positions | Free (logged at generation) |
| **Answer entropy** | Shannon entropy over answer distribution in 256 samples | Free |

No LLM-as-judge. No CoT-Pass@K. Everything is free at generation time or a single forward pass.

### Eval schedule
- Checkpoints every 50 steps for first 500, every 200 to 1500. ~15 checkpoints per run.
- **Eval sets**: MATH500 (primary), AIME 2025 (hard, low contamination).
- Bucket MATH500 by starting-checkpoint pass@64: easy (>50%), medium (10-50%), hard (<10%).

---

## 2. Starting Checkpoints (off the shelf)

| Checkpoint | Type | Why |
|-----------|------|-----|
| Qwen3.5-9B-Base | Pretrained | Cleanest test. No prior RL. |
| Qwen3.5-9B (instruct) | SFT + RL'd | Can further RL still discover? |
| Qwen3.5-4B-Base | Pretrained, smaller | Scale check / fallback. |

---

## 3. RL-Time Levers

All runs: GRPO via **verl**, 1500 steps, 16 rollouts/prompt, MATH training set. Vary one lever at a time from default.

**Default**: binary 0/1 reward, vanilla GRPO, β_KL = 0.04.

### Lever 1: Reward Design (the key lever)

| Level | Description | Implementation | Extra cost |
|-------|-------------|---------------|------------|
| **Binary 0/1** | Correct final answer = 1, else 0 | verl native | None |
| **Off-the-shelf PRM** | Per-step score from Qwen2.5-Math-PRM-7B | Custom reward fn in verl (see §4) | +30-40% training time |
| **Partial credit heuristic** | Score = fraction of intermediate numerical values matching reference solution steps | Custom reward fn, no model needed | Negligible |

#### The PRM path (fastest credible dense reward)

**Model**: `Qwen2.5-Math-PRM-7B` (on HuggingFace)
- Trained on PRM800K + Math-Shepherd by Axolotl team
- Based on Qwen2.5-Math-7B-Instruct
- Scores each reasoning step via next-token prediction (+/- classification)
- Working vLLM inference code available in axolotl-cookbook

**How it works in GRPO**: For each of the 16 rollouts per problem:
1. Split the rollout into reasoning steps (by `\n\n` or step markers)
2. Feed each step to the PRM, get per-step correctness probability
3. Aggregate: reward = product of step scores (harshest) or min of step scores (identifies first error) or mean (smoothest)
4. Use this as the reward signal in GRPO instead of binary 0/1

**Why this matters mechanistically**: On a hard problem where 0/16 rollouts get the right final answer, binary reward gives zero gradient. But some rollouts may get steps 1-4 right before failing at step 5. The PRM scores those rollouts ~0.6 instead of 0. Now the model has gradient signal: "the approach that got steps 1-4 right is better than the approach that failed at step 2." Over training, the model reliably nails steps 1-4, raising the probability of eventually getting step 5 right in a future rollout → the ratchet mechanism.

**Potential issue**: The PRM was trained on Qwen2.5 outputs. Qwen3.5 uses a different tokenizer and `<think>` format. The PRM may not score Qwen3.5 outputs well.

**Mitigation / fallback — partial credit heuristic (Tier 1)**:
- Parse the model's reasoning for numerical expressions (regex for numbers, fractions, equations)
- Compare against intermediate values in the reference solution
- Score = (matched intermediates) / (total intermediates in reference)
- Zero extra inference cost. Noisy but directional.
- Can also combine: `reward = 0.5 * PRM_score + 0.5 * binary` as a hedge

**For the pilot**: Start with the off-the-shelf PRM. If scores look uncorrelated with Qwen3.5 output quality (sanity check: does the PRM rank correct solutions above incorrect ones?), fall back to partial credit.

### Lever 2: Exploration Pressure

| Level | Description | verl config |
|-------|-------------|-------------|
| Vanilla GRPO | No exploration mechanism | `entropy_coeff=0` |
| Entropy bonus | α=0.01 additive entropy term | `entropy_coeff=0.01` |
| DAPO | Generous clipping (clip_ratio=0.28) | `recipe/dapo` |
| KL-Cov | Clip high-covariance tokens | Native in verl |
| DrGRPO | Length-normalized (fixes length bias) | Native in verl |

### Lever 3: KL Budget

| Level | β_KL | verl config |
|-------|------|-------------|
| Unconstrained | 0.0 | `kl_loss_coef=0` |
| Loose | 0.01 | `kl_loss_coef=0.01` |
| Default | 0.04 | `kl_loss_coef=0.04` |
| Tight | 0.1 | `kl_loss_coef=0.1` |

### Run Count

Per starting checkpoint: 3 (reward) + 5 (exploration) + 4 (KL) − 1 (shared default) = **~11 unique configs**.

| | 9B-Base | 9B-Instruct | 4B-Base |
|---|---|---|---|
| Runs | 11 | 11 | 3 |
| **Total** | | | **~25 runs** |

---

## 4. Framework: verl

### Why verl
- GRPO, DAPO, DrGRPO, KL-Cov, entropy bonus all native config flags
- Official Qwen docs point to verl with working scripts
- Qwen3.5 + GRPO validation actively underway (GitHub issue #5441)
- vLLM/SGLang integration for rollout generation
- 19.7k GitHub stars, EuroSys 2025, active community

### PRM reward function integration

verl supports custom reward functions. The integration looks like:

```python
# In verl's reward function interface
class PRMReward:
    def __init__(self):
        self.prm = load_prm("Qwen/Qwen2.5-Math-PRM-7B")  # ~7B model

    def __call__(self, prompts, responses, ground_truths):
        rewards = []
        for prompt, response, gt in zip(prompts, responses, ground_truths):
            # Binary component (still check final answer)
            binary = 1.0 if extract_answer(response) == gt else 0.0

            # PRM component (step-level scores)
            steps = split_into_steps(response)
            step_scores = [self.prm.score(prompt, steps[:i+1]) for i in range(len(steps))]
            prm_score = min(step_scores)  # first-error aggregation

            # Combined reward
            reward = 0.5 * binary + 0.5 * prm_score
            rewards.append(reward)
        return rewards
```

The PRM runs on a separate GPU (or shares with the reference model during non-overlapping phases). For 9B actor + 7B PRM on 4× A100 80GB, memory is comfortable.

**Fallback partial credit (no extra model)**:

```python
class PartialCreditReward:
    def __call__(self, prompts, responses, ground_truths):
        rewards = []
        for response, gt_solution in zip(responses, ground_truths):
            # Extract all numbers from model output and reference
            model_nums = extract_numbers(response)
            ref_nums = extract_numbers(gt_solution)
            if not ref_nums:
                rewards.append(float(extract_answer(response) == gt))
                continue
            matched = len(set(model_nums) & set(ref_nums))
            rewards.append(matched / len(ref_nums))
        return rewards
```

---

## 5. Compute Budget

### Per-run cost (Qwen3.5-9B, 1500 GRPO steps, 16 rollouts/prompt)

| Component | A100-hours | Notes |
|-----------|-----------|-------|
| Training (binary reward) | 80 | Generation-dominated |
| Training (PRM reward) | 110 | +30-40% for PRM inference |
| Eval (15 ckpts × 256 samples × 500 problems) | 250 | SGLang batched |
| Base-model PPL scoring | 20 | Forward pass only |
| **Total per run (binary)** | **~350** | |
| **Total per run (PRM)** | **~380** | |

### Full experiment (~25 runs)
| Component | A100-hours |
|-----------|-----------|
| 22 binary-reward runs × 350 | 7,700 |
| 3 PRM-reward runs × 380 | 1,140 |
| 3 runs on 4B (scale check) × 200 | 600 |
| **Total** | **~9,400** |

**Cost**: A100 spot ~$1.5/hr → **~$14K**. H100 spot ~$2.5/hr → **~$24K**.

**Reduced scope** (one starting checkpoint, 11 runs): ~4,000 hrs → **$6-10K**.

---

## 6. Pilot: 2 Weeks, ~$2-3K

The pilot tests the core prediction: **dense reward produces a different trajectory shape than binary reward.** If the curves overlap, the thesis needs rethinking. If they separate, the full experiment is worth running.

### Day 0: Setup
- [ ] Verify Qwen3.5-9B-Base on HuggingFace
- [ ] Install verl, run GRPO example on Qwen3-8B to verify setup
- [ ] Swap model to Qwen3.5-9B, verify single training step completes
- [ ] Download `Qwen2.5-Math-PRM-7B`, sanity check: does it rank correct Qwen3.5 solutions above incorrect ones on 50 MATH problems?
- [ ] Set up SGLang eval pipeline: checkpoint → 256 samples → pass@k
- [ ] Reserve 4× A100 80GB

### Days 1-5: Run 1 — Binary Reward (expect sharpening)
- [ ] Config: 9B-Base → vanilla GRPO, binary 0/1, β=0.04
- [ ] Save checkpoints every 50 steps (first 500), every 200 to 1500
- [ ] Eval each checkpoint: pass@{1,16,64,256} on MATH500
- [ ] **Sanity check**: pass@1 goes up, pass@256 goes down. If not, debug.

### Days 5-9: Run 2 — PRM Reward (expect discovery or delayed sharpening)
- [ ] Config: 9B-Base → vanilla GRPO, PRM reward (Qwen2.5-Math-PRM-7B), β=0.04
- [ ] Same eval schedule
- [ ] If PRM sanity check failed on Day 0: use partial credit heuristic instead
- [ ] **Key question**: does the trajectory stay in (+, +) longer? Does pass@256 hold up or even improve?

### Days 9-12: Run 3 — Combined (expect strongest discovery signal)
- [ ] Config: 9B-Base → DAPO + PRM reward + β=0.01
- [ ] All three mechanisms active: dense reward + exploration pressure + low KL
- [ ] Same eval schedule
- [ ] **Key question**: does combining all three levers produce a trajectory that stays firmly in the discovery quadrant?

### Days 12-14: Analysis & Go/No-Go
- [ ] Plot 3 trajectories in Δpass@1 × Δpass@256 space
- [ ] Stratify by difficulty: easy/medium/hard
- [ ] Compute base-model PPL for final checkpoints (do PRM-trained solutions look novel to the base model?)
- [ ] Compute unique answer count per problem across checkpoints

### Decision gate:
- **Trajectories visibly separate** → full experiment is go. Write up pilot as workshop paper or section 1 of the full paper.
- **PRM trajectory slightly different** → promising but need more levers. Proceed with caution, possibly narrow scope.
- **All trajectories overlap** → dense reward doesn't change the dynamics. Fundamental rethink needed. Could still publish as negative result if analysis is thorough.

### Pilot compute
- Run 1 (binary): 350 A100-hours
- Run 2 (PRM): 380 A100-hours
- Run 3 (combined): 380 A100-hours
- **Total: ~1,100 A100-hours ≈ $1.5-2.5K**

---

## 7. Deliverables

### Figure 1: Trajectory Atlas
All runs overlaid in Δpass@1 × Δpass@256 space, color-coded by lever. The signature visualization.

### Figure 2: Per-Lever Panels
For each lever, trajectories of its levels + difficulty-stratified pass@k at final checkpoint.

### Figure 3: Mechanistic View
Token-level concentration, base-model PPL, answer entropy over training per config.

### Table 1: The Recipe
| Goal | Reward | Exploration | β_KL |
|---|---|---|---|
| Maximize pass@1 (deployment) | Binary | Vanilla GRPO | 0.04 |
| Maximize frontier (pass@256) | PRM | DAPO | 0.01 |
| Balance both | [from experiments] | [from experiments] | [from experiments] |

### Table 2: Effect Size Ranking
Rank levers by Δpass@256. Prediction: reward design >> exploration > KL.

---

## 8. Paper Outline

1. **Introduction** (1p): Sharpening is the default. Why (bootstrapping trap). What to do about it.
2. **Framework** (1p): Δpass@1 × Δpass@k trajectories. Three mechanisms that break the trap.
3. **Setup** (1p): Starting checkpoints, RL levers, verl, PRM integration.
4. **Results** (3p): Per-lever results. Trajectory atlas. Difficulty stratification. Mechanistic analysis.
5. **The Recipe** (0.5p): Practitioner-facing summary.
6. **Discussion** (1p): Connection to CMU three-regime framework. Limitations. Open questions.
7. **Related Work** (0.5p)

**Target**: NeurIPS 2026 (May) or EMNLP 2026.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Qwen3.5-9B-Base not released | Use 4B-Base (confirmed) or Qwen3-8B. |
| verl doesn't support Qwen3.5's Gated Delta Nets | Test Day 0. Fallback: OpenRLHF or Qwen3-8B (standard transformer). |
| Off-the-shelf PRM doesn't generalize to Qwen3.5 format | Day 0 sanity check. Fallback: partial credit heuristic (zero model cost). |
| PRM adds too much training time | Use PRM scoring only every N steps, or score a random subset of rollouts. Or use partial credit. |
| Dense reward doesn't change trajectory shape | Publishable negative result if analysis is thorough. Implies the bottleneck isn't reward signal but something else (e.g., optimization landscape). |
| NeurIPS deadline too tight | EMNLP 2026 or workshop paper from pilot alone. |
