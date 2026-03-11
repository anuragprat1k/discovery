# Discovery: Dense vs Binary GRPO Rewards

ML research project testing whether dense (partial-credit) vs binary (all-or-nothing) GRPO rewards produce different training trajectories — specifically whether binary rewards cause "sharpening" (exploiting known problems) while dense rewards enable "discovery" (learning to solve new ones). Uses Qwen models on MATH dataset problems, with TRL/Tinker for training and vLLM for evaluation.

## Key Files

| Path | Purpose |
|---|---|
| `experiments.md` | **Task tracker.** Source of truth for what to work on. |
| `logbook.md` | **Results journal.** Record findings after completing work. |
| `plan.MD` | Full experiment plan (v5). Background, hypotheses, configs. |
| `train.py` | TRL/LoRA GRPO training (local GPU). |
| `train_tinker.py` | Tinker API GRPO training (remote GPU). |
| `eval/pass_at_k.py` | Evaluation harness (vLLM or Tinker). |
| `rewards/reward_fns.py` | Binary and partial-credit reward implementations. |
| `data/train_128.parquet` | Training split (128 problems). DO NOT MODIFY. |
| `data/eval_200.parquet` | Eval split (200 problems). DO NOT MODIFY. |
| `analysis/plot_trajectories.py` | Trajectory visualization. |
| `results/` | Per-run eval JSON results. |

## Agent Workflow

**You are an autonomous agent. The human may be away. Never block on human input. Never ask for clarification. Make reasonable decisions. Never stop before completion.**

### Step 1: Read experiments.md
Find an experiment or task with status `TODO`. Check it's not claimed (no branch exists on remote).

### Step 2: Claim it
- Branch from main: `git checkout -b exp/<id>-<slug> main` (experiments) or `task/<slug>` (tasks)
- Push immediately: `git push -u origin <branch>`
- Update experiments.md: change status from `TODO` to `IN PROGRESS`
- Commit + push that change

### Step 3: Execute to completion
- Follow the experiment/task spec exactly
- Training method (local vs Tinker) is specified per experiment
- Experiments may have two phases:
  1. **Build phase** (if specified): Implement required features, write tests, commit code changes
  2. **Run phase**: Execute training → eval → analysis using specified commands
- Handle both phases end-to-end on the same branch
- If something fails, debug and fix it — do not stop
- Run to completion — all implementation, training, eval, and analysis specified

### Step 4: Record results
- Add a dated entry to `logbook.md` with findings, metrics, wall-clock times
- Update experiments.md: change status to `DONE`
- Commit all results, code changes, logbook update

### Step 5: Create PR
- `/Users/aragun/bin/gh pr create` targeting `main`
- PR body: summary of findings, key metrics, issues encountered, go/no-go assessment if applicable
- Do NOT merge — leave for human review

## Coordination Protocol

- Branch existence on remote = distributed lock
- One agent per experiment/task
- `git fetch origin && git branch -r` before claiming
- Check dependency ordering: if experiment X depends on Y, verify Y is `DONE`
- If all items claimed, wait — do not invent new work

## Commit Conventions

Conventional commits matching repo style:
- `feat(component):` / `fix(component):` / `eval(step-N):` / `refactor:` / `docs:`
- Components: train, eval, rewards, data, analysis, tinker

## Common Commands

Training (local): `python train.py --reward {binary,dense} --max_steps 200 --output_dir checkpoints/<name>`
Training (Tinker): `python train_tinker.py --reward {binary,dense} --max_steps 200 --batch_size 8 --group_size 8`
Eval (vLLM): `python eval/pass_at_k.py --use_vllm --checkpoint_dir <path> --eval_parquet data/eval_200.parquet --output_dir results/<run> --n_samples 64 --step <N>`
Eval (Tinker): `python eval/pass_at_k.py --use_tinker --tinker_path "tinker://<id>/weights/step_<N>" --eval_parquet data/eval_200.parquet --output_dir results/<run> --n_samples 64 --step <N>`
Analysis: `python analysis/plot_trajectories.py --results_dir results`
Reward tests: `python rewards/reward_fns.py`

## Safety Rules

1. NEVER commit .env or files with API keys
2. NEVER force-push to main
3. NEVER modify data/*.parquet (fixed experimental splits)
4. NEVER change hyperparameters between comparison runs (only reward function varies)
5. Always use `/Users/aragun/bin/gh` for GitHub CLI
6. Always work on feature branches, never commit to main directly
7. Large files (.safetensors, .pt) are gitignored — do not add them

## Environment

- `.env` has WANDB_API_KEY and TINKER_API_KEY (loaded via python-dotenv)
- `gh` CLI: `/Users/aragun/bin/gh`
- Python deps: vllm trl bitsandbytes datasets pandas accelerate matplotlib tqdm hf_transfer peft python-dotenv tinker
