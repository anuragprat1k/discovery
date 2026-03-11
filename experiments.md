# Experiments & Tasks

Status: `TODO` | `IN PROGRESS` | `DONE` | `BLOCKED`

## Experiments

| ID | Name | Status | Branch | Depends On | Description |
|----|------|--------|--------|------------|-------------|
| E1 | Binary reward training | DONE | tinker-training | — | Trained via Tinker with binary reward. Results in results/tinker_binary/ at steps 0/50/100/150/200. |
| E2 | LLM rubric reward training | TODO | — | — | **Build**: Implement LLM-based rubric reward using Tinker's SamplingClient as judge. (1) Create a separate `judge_client = service.create_sampling_client(base_model="Qwen/Qwen3-8B")` alongside the training client. (2) New reward fn in `rewards/reward_fns.py`: builds a rubric-evaluation prompt containing the problem, reference solution, and model completion, then asks the judge to score on criteria (correct setup, valid reasoning steps, correct intermediate values, correct final answer) returning a 0-1 score. (3) Parse judge output for numeric score. (4) Add `--reward rubric --judge_model Qwen/Qwen3-8B` flags to `train_tinker.py`. Judge calls are async (Tinker futures) so all 64 completions per step can be scored in parallel. Write unit tests for rubric prompt construction and score parsing. **Run**: `python train_tinker.py --reward rubric --judge_model Qwen/Qwen3-8B --max_steps 200 --batch_size 8 --group_size 8`. Eval all checkpoints. Compare trajectory to E1 binary. Key papers: "Rubrics as Rewards" (2507.17746), "RM-R1: Chain-of-Rubrics" (2505.02387). |
| E3 | DAPO techniques | TODO | — | — | **Build**: Add three DAPO improvements to `train_tinker.py`: (1) Clip-Higher — asymmetric PPO clipping with ε_low=0.2, ε_high=0.5; (2) Dynamic sampling — skip training on batches where all rollouts agree (0% or 100% correct), since these produce zero advantage variance; (3) Token-level loss normalization — normalize loss across all tokens in batch, not per-sample. Add `--dapo` flag to enable all three. **Run**: `python train_tinker.py --reward binary --dapo --max_steps 200 --batch_size 8 --group_size 8`. Compare trajectory to vanilla binary (E1) to isolate algorithmic improvements from reward density. |
| E4 | PRIME implicit PRM | IN PROGRESS | exp/E4-prime-implicit-prm | — | **Build**: Implement PRIME (Process Reinforcement through Implicit Rewards) in `train_tinker.py`. For each token in a completion, compute implicit reward as `β * log(π_policy(token) / π_ref(token))`. This turns the reference model (already used for KL) into a free process reward model — tokens where the policy diverges from the reference get stronger signal. Modify the per-token advantage array in `grpo_step()` to use these token-level rewards instead of uniform completion-level advantages. Add `--reward_type prime` flag. No extra model needed. **Run**: `python train_tinker.py --reward binary --reward_type prime --max_steps 200 --batch_size 8 --group_size 8`. Compare trajectory to E1 (same binary outcome reward, but with dense token-level credit assignment). Paper: arxiv 2502.01456. |
| E5 | Eval & plot all dense runs | TODO | — | E2, E3, E4 | Eval any remaining checkpoints. Run `python analysis/plot_trajectories.py --results_dir results` across all completed runs. Record Δpass@1 vs Δpass@64 gap at step 200 for each reward type. Apply go/no-go criteria. |

## Tasks

| ID | Name | Status | Branch | Description |
|----|------|--------|--------|-------------|
