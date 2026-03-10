# Experiment Logbook

**Created**: 2026-03-10
**Purpose**: Quick validation — do dense vs binary GRPO rewards produce visibly different trajectory shapes in (Δpass@1, Δpass@64) space?

---

## Rationale

### Research Question

The central question is whether the choice of reward granularity leaves a detectable fingerprint in the joint (Δpass@1, Δpass@64) trajectory. The hypothesis, grounded in the information-theoretic role of rewards in policy gradient methods, is as follows:

- **Binary rewards** collapse the reward signal to {0, 1}. For hard problems where the model almost never reaches a correct final answer, the gradient variance is near zero — all rollouts look alike from the reward's perspective. The policy has no signal to explore. What signal does exist pushes the model to exploit whichever reasoning patterns already correlate with correct final answers. This manifests as "sharpening": pass@1 rises because the mode of the distribution improves, but pass@64 falls because the policy concentrates probability mass, suppressing diversity.

- **Dense rewards** (partial credit) give nonzero gradient even when the final answer is wrong, provided intermediate steps partially match reference reasoning. This keeps the gradient flowing on hard problems and should sustain or expand diversity — "discovery" — so that pass@64 falls more slowly or rises alongside pass@1.

If the effect is real and visible at small scale (4B model, 128 training problems, 200 steps), it saves $14K in cloud budget before committing to a full-scale run.

### Model Choice

**Qwen/Qwen3-4B-Instruct** was chosen over alternatives for two independent reasons.

First, the Instruct variant already knows the `\boxed{}` answer format from RLHF/SFT. Training a Base model would spend early gradient steps teaching format compliance rather than mathematical reasoning — wasted signal that could obscure the reward comparison we care about. Starting from a model with higher baseline pass@k also means richer GRPO training from step 1, because a model that occasionally solves problems generates more informative advantage estimates than one that never does.

Second, the 4B scale fits on a single 32GB RTX 5090 with headroom. The training configuration uses bf16 for the actor and 4-bit quantization for the frozen reference model, leaving approximately 4GB free. A 9B model does not fit this configuration without multi-GPU setup, which introduces complexity orthogonal to the reward comparison.

### Dataset Choice

**MATH** (DigitalLearningGmbH/MATH-lighteval) satisfies every requirement: verified ground truth with exact-match answer checking, difficulty stratification (levels 1–5) enabling principled problem selection, and broad adoption in GRPO-adjacent work (DAPO, DeepSeek-R1, Dr.GRPO) making results interpretable against prior baselines. The held-out MATH500 test set provides uncontaminated final evaluation.

The **128-problem training set** is stratified 42 easy / 64 medium / 22 hard, approximating a 33%/50%/17% split. This weighting is deliberate: medium-difficulty problems are the primary test case, because binary reward gives *some* signal there but partial credit should diverge — medium is where the two reward functions make different predictions. Hard problems are the decisive test case: binary reward yields near-zero gradient (the model almost never produces a correct final answer), while partial credit should yield nonzero gradient from intermediate matching. Easy problems are included to prevent gradient starvation but downweighted to avoid swamping the signal with trivially-solved examples.

The **200-problem eval set** enables reliable pass@k estimation. At 64 samples per problem, the total evaluation budget is 200 × 64 = 12,800 forward passes per checkpoint — tractable in a few hours and sufficient statistical power to detect 3pp differences in pass@64.

### Reward Function Design

**Binary**: `1.0 if boxed_answer_correct else 0.0`. This is the standard formulation used in GRPO literature. Its virtue is interpretability; its flaw, from a gradient-flow perspective, is that it provides no learning signal on problems the model cannot yet solve.

**Partial credit**: `0.5 × binary + 0.5 × (matched_intermediates / reference_intermediates)`. The formula was designed with two constraints. First, the binary component must remain to ensure correct final answers are still rewarded unambiguously — pure intermediate matching could reward verbose but ultimately wrong reasoning chains. Second, the intermediate component must be zero-cost to compute (no auxiliary model required). The 0.5/0.5 split is a neutral prior; there is no strong empirical basis at this scale for weighting one over the other, and asymmetric weights would add a second free parameter to interpret.

**Why not a Process Reward Model**: Qwen2.5-Math-PRM-7B is a strong candidate but requires loading a 7B model alongside the 4B actor — GPU memory rules this out on a single card. Partial credit is a useful directional proxy that requires zero additional memory.

### Training Hyperparameters

**200 steps**: Prior GRPO work (DAPO, Dr.GRPO) shows sharpening becoming visible between steps 50–100 on similar model scales. 200 steps provides a complete trajectory with margin to observe late-stage behavior, while remaining feasible within a day of GPU time.

**8 rollouts**: The minimum for GRPO advantage estimates to be statistically meaningful. Advantage is estimated from within-rollout variance; with fewer than 8 samples, the estimate is too noisy for reliable gradient direction.

**KL β = 0.04**: The value from the original GRPO paper. Critically, it is held *identical* between the binary and dense runs — KL regularization is intentionally not a variable in this experiment. Any observed trajectory difference can be attributed solely to the reward function.

**LR = 1e-6**: Conservative for GRPO on an instruction-tuned model. Higher rates (e.g., 1e-5) risk policy collapse in the first 20 steps, particularly when the reward signal is sparse (as it is for binary on hard problems). Lower rates would show no movement over 200 steps.

**Temperature = 1.0**: Standard for rollout diversity; lower temperatures reduce the variance needed to form meaningful advantages.

### Framework Choice

TRL GRPOTrainer was chosen over verl because verl is designed for multi-node distributed training and its single-GPU path is less maintained. TRL handles the Instruct chat format natively via tokenizer chat templates, reducing the risk of prompt formatting errors that could confound the reward comparison. TRL also provides a simpler debugging surface — fewer abstraction layers between reward function output and gradient computation.

### Evaluation Design

**64 samples per problem** balances statistical reliability against eval cost. 256 samples would require roughly 16 hours per checkpoint; at 4 checkpoints that is 64 hours of eval time — longer than training itself. 64 samples cuts this to approximately 4 hours per checkpoint while still providing reliable estimates at the pass@k levels relevant to this experiment (k ∈ {1, 4, 16, 64}).

The **unbiased pass@k estimator** from the Codex paper (Chen et al., 2021) is used rather than a naive proportion. The naive estimator consistently overestimates pass@k when k < number_of_samples, which would introduce a systematic bias that grows with k — problematic when comparing k=1 to k=64 on the same trajectory plot.

---

## Implementation Notes

### Data Pipeline

`data/prepare_math.py` downloads MATH-lighteval, stratifies by difficulty level into the 42/64/22 easy/medium/hard split, and formats each problem as a chat prompt using the model's tokenizer chat template. The eval split (200 problems) is drawn separately and held out from all training decisions.

### Reward Functions

`rewards/reward_fns.py` implements both reward functions with a shared interface: `reward_fn(completions, reference_answer, reference_steps) -> List[float]`. The binary function extracts the final `\boxed{}` content and checks exact match after normalization (stripping whitespace, converting equivalent forms). The partial credit function additionally tokenizes intermediate steps and computes overlap against reference steps using token-level F1 to handle paraphrase.

### Training Setup

`train.py` wraps TRL GRPOTrainer. The `--reward` flag selects between `binary` and `dense`. The reference model is loaded in 4-bit and frozen for the duration of training. Checkpoints are saved at steps 50, 100, 150, 200.

### Eval Pipeline

`eval/pass_at_k.py` implements the unbiased estimator and accepts a checkpoint directory. `eval/eval_all_checkpoints.sh` loops over the four checkpoint steps for both reward conditions, producing a JSON results file per checkpoint. Eval is run after training completes, not interleaved, to avoid GPU contention.

### Analysis

`analysis/plot_trajectories.py` reads the per-checkpoint JSON files and plots two panels: (step, Δpass@1) and (step, Δpass@64), with binary and dense as separate series. A secondary scatter plot shows the (Δpass@1, Δpass@64) joint trajectory for each condition, which directly tests the "sharpening vs discovery" framing.

---

## Go/No-Go Criteria

| Outcome | Criterion | Decision |
|---|---|---|
| Clear Go | Δpass@64 gap ≥ 3pp (dense > binary) at step 200; curves visually separate by step 100 | Proceed to full-scale run |
| Weak Go | Gap < 3pp or separation only visible after step 150 | Extend 200 more steps before deciding |
| No-Go | Both conditions show same magnitude sharpening throughout | Hypothesis does not hold at this scale; halt, saving $14K cloud budget |

The 3pp threshold is chosen because it exceeds expected noise from the unbiased estimator at n=200, k=64 with reasonable model variance.

---

## Ambiguity Debugging Sequence

If results are ambiguous (curves cross, or gap is 1–2pp), check in order:

1. **Reward function bug**: Verify partial credit scores on 10 held-out problems by hand. Confirm intermediate matching is not accidentally giving full credit on wrong answers.
2. **Format leakage**: Check whether the model is producing `\boxed{}` on a different fraction of rollouts under each reward condition — a format difference, not a reasoning difference, could drive apparent pass@1 divergence.
3. **Difficulty confound**: Stratify pass@k results by difficulty level. If the gap is present on hard problems but not medium, the effect is real but scale-limited.
4. **KL saturation**: Plot KL divergence from reference over steps. If KL hits the penalty ceiling, the KL term is dominating and reward differences are masked.

---

## Step-by-Step Execution Log

### 2026-03-10 — Phase 0: Bug Fixes Applied

- **Bug 1**: Created `rewards/__init__.py` (empty) — fixes ImportError when `train.py` imports `rewards.reward_fns`
- **Bug 2**: Fixed `train.py` line ~243: `score_fn(completion, ...)` → `score_fn([completion], ...)[0]` — fixes TypeError (scalars vs lists)
- **Bug 3**: Fixed `eval/pass_at_k.py` lines 250/252/259: `"answer"` → `"ground_truth"` column name — fixes KeyError
- **Bug 4**: Added explicit OOM guard in `eval/pass_at_k.py`: `if current_batch < 1: raise RuntimeError("OOM at batch_size=1")`
- **Bug 5**: Plan to call `pass_at_k.py` directly with `--checkpoint_dir` + `--step` instead of using `eval_all_checkpoints.sh`
- **Added**: `--use_vllm` flag to `eval/pass_at_k.py` using `vllm.LLM` with `SamplingParams(n=64, ...)` for ~5-10× eval speedup

### 2026-03-10 — Phase 0: Dependencies Installed

```
pip install vllm trl bitsandbytes datasets pandas accelerate matplotlib tqdm hf_transfer
```
Verified: `import vllm, trl, bitsandbytes, datasets` → OK

### 2026-03-10 — Phase 1: Data Prepared

- `python data/prepare_math.py --output_dir data/` completed successfully
- `data/train_128.parquet`: 128 rows — level distribution: 1→8, 2→34, 3→64, 4→7, 5→15
- `data/eval_200.parquet`: 200 rows — level distribution: 1→17, 2→36, 3→45, 4→49, 5→53
- Columns: `['prompt', 'ground_truth', 'solution', 'level', 'problem_type', 'problem']`

### 2026-03-10 — Phase 1: Unit Tests Passed

- `python rewards/reward_fns.py` → All tests passed
- Binary reward: correct=1.0, incorrect=0.0 ✓
- Partial credit: correct=0.875, incorrect=0.25 ✓
- Helper functions: extract_boxed_answer, answers_match, extract_numbers all OK

### 2026-03-10 — Architecture Change: LoRA Fine-tuning

**Decision**: Switched from full fine-tuning to LoRA (PEFT) for memory reasons.

**Why**: Full fine-tuning on 32 GB GPU requires:
- Actor (bf16): ~8 GB
- Ref model (bf16, auto-created by TRL): ~8 GB
- Optimizer state (AdamW on 4B params): ~8 GB
- Activations + rollout buffers: ~8 GB
- Total: >32 GB → OOM

**LoRA configuration**:
- Base model: `Qwen/Qwen3-4B-Instruct-2507` in bf16, fully frozen
- LoRA rank=64, alpha=128, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Trainable params: 132M / 4155M = **3.18%** (~200 MB optimizer state)
- TRL detects `is_peft_model()` → skips loading a ref model entirely; disables adapters in-place for KL computation
- LR raised from 1e-6 to 1e-4 (standard for LoRA adapters)

**Scientific validity**: LoRA does not affect the binary vs dense reward comparison. Both runs use identical LoRA config; only the reward function varies. The sharpening vs discovery trajectory question is still answerable.

### 2026-03-10 — Phase 2: Smoke Test Passed

- `python train.py --reward binary --max_steps 2 --output_dir checkpoints/smoke_binary --no_wandb`
- Step 1: reward=1.0 (easy problem, all 8 rollouts correct)
- Step 2: reward=0.0 (harder problem, completions clipped at 1024 tokens)
- No OOM, no import errors, exit 0 ✓
- Step time: ~86s/step on RTX 5090 → estimated 200 steps ≈ **4.8 hours**

**Additional fixes applied during smoke test**:
- `model_name` updated: `Qwen/Qwen3-4B-Instruct` (404) → `Qwen/Qwen3-4B-Instruct-2507`
- `GRPOConfig`: `max_new_tokens` → `max_completion_length` (TRL API change)
- `GRPOTrainer`: removed `ref_model` arg (no longer accepted in TRL >= 0.13)
- Reward wrapper: extract `.content` from conversational completions (TRL passes `[{"role":"assistant","content":"..."}]`)

### 2026-03-10 — vllm Fix: Option B (Build from Source) — SUCCESS

**[ERROR]** vllm 0.17.0 prebuilt wheel compiled with CUDA 12.9 PTX; runtime only supports CUDA 12.8 → `cudaErrorUnsupportedPtxVersion` on RTX 5090 (sm_120).

**Attempted Options A (VLLM_USE_V1=0) and VLLM_ATTENTION_BACKEND variants**: All failed — vllm 0.17.0 dropped the `VLLM_ATTENTION_BACKEND` env var and always selects FLASH_ATTN via internal priority.

**Remediation**: Built vllm from source (v0.17.0 tag) with `TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 pip install -e . --no-build-isolation` using nvcc 12.8 on host. Build confirmed compiling `_vllm_fa2_C` for `arch=compute_120a,code=sm_120a`. Build took ~75 minutes.

**Outcome**: vllm smoke test passed (generated text, no PTX error). `Successfully installed vllm-0.17.0+cu128`. vllm now usable for all evals.

- [ ] **Step 1** — Run `data/prepare_math.py`; verify split sizes and prompt format on 3 random examples. Record any normalization edge cases.
- [ ] **Step 2** — Smoke test `rewards/reward_fns.py` on 5 hand-crafted examples with known answers; log partial credit scores.
- [ ] **Step 3** — Launch binary reward training run (`--reward binary`). Record wall-clock time per step.
- [ ] **Step 4** — Confirm checkpoints saved at steps 50, 100, 150, 200.
- [ ] **Step 5** — Launch dense reward training run (`--reward dense`). Confirm comparable per-step time (reward overhead should be negligible).
- [ ] **Step 6** — Confirm checkpoints saved at steps 50, 100, 150, 200.
- [ ] **Step 7** — Run `eval/eval_all_checkpoints.sh` for all 8 checkpoint directories. Record eval wall-clock time.
- [ ] **Step 8** — Run `analysis/plot_trajectories.py`; record gap at step 200 and first step where curves visually separate (or note non-separation).
- [ ] **Step 9** — Apply Go/No-Go criteria. Record decision and reasoning here.
- [ ] **Step 10** — If No-Go, document what would need to change (scale, dataset, reward formula) before retrying.
