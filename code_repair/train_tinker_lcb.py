"""
Multi-turn GRPO training on LiveCodeBench via Tinker + SandboxFusion.

Model generates code, sees test results, iterates up to max_turns.
Three reward conditions: sparse (terminal-only), dense (per-turn pass
fraction via HWM), dense_full (dense + speed bonus).

Usage:
    export SANDBOX_URL=http://localhost:8080/run_code
    python3.11 -m code_repair.train_tinker_lcb --reward sparse --max_steps 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time

import numpy as np

try:
    import tinker
    from tinker import ServiceClient
except ImportError:
    print("ERROR: tinker not installed", file=sys.stderr)
    sys.exit(1)

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

from code_repair.deepcoder.code_env import load_deepcoder_tasks
from code_repair.deepcoder.code_grading import sandbox_check_correctness, extract_code_from_model
from code_repair.deepcoder.deepcoder_tool import DeepcoderTask


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--reward", required=True, choices=["sparse", "dense", "dense_full"])
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--max_turns", type=int, default=4)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4, help="Problems per step.")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--loss_fn", default="importance_sampling",
                   choices=["importance_sampling", "ppo", "cispo"])
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--eval_temperature", type=float, default=0.6)
    p.add_argument("--refresh_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--eval_n_problems", type=int, default=20)
    p.add_argument("--sandbox_timeout", type=int, default=6)
    p.add_argument("--min_tests", type=int, default=5,
                   help="Only use problems with >= this many test cases.")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="discovery-code-repair")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def reward_sparse(info: dict, is_terminal: bool) -> float:
    """Terminal only: +1 if solved, 0 otherwise. Format penalty."""
    if info.get("no_code"):
        return -0.1  # format penalty
    if is_terminal:
        return 1.0 if info["all_passed"] else 0.0
    return 0.0


def reward_dense(info: dict, is_terminal: bool) -> float:
    """Per-turn HWM of test fraction (potential-based) + terminal."""
    if info.get("no_code"):
        return -0.1
    n = max(info["tests_total"], 1)
    hwm_delta = info["hw_passed"] - info["old_hw"]
    r = 0.5 * (hwm_delta / n) if hwm_delta > 0 else 0.0
    if is_terminal:
        r += 1.0 if info["all_passed"] else 0.0
    return r


def reward_dense_full(info: dict, is_terminal: bool) -> float:
    """Dense + speed bonus for solving early."""
    if info.get("no_code"):
        return -0.1
    n = max(info["tests_total"], 1)
    hwm_delta = info["hw_passed"] - info["old_hw"]
    r = 0.5 * (hwm_delta / n) if hwm_delta > 0 else 0.0
    if is_terminal and info["all_passed"]:
        speed_bonus = 0.2 * (info["max_turns"] - info["turn"]) / info["max_turns"]
        r += 1.0 + speed_bonus
    elif is_terminal:
        r += 0.0
    return r


REWARD_FNS = {"sparse": reward_sparse, "dense": reward_dense, "dense_full": reward_dense_full}

FEEDBACK_TEMPLATE = """Your solution failed some tests.
- Tests passed: {passed}/{total}
Fix your solution. Output corrected code in a ```python``` code block."""


# ---------------------------------------------------------------------------
# GRPO step
# ---------------------------------------------------------------------------

def grpo_step(
    step: int,
    batch_tasks: list[DeepcoderTask],
    sampling_client,
    training_client,
    tokenizer,
    reward_fn,
    args,
) -> dict:
    loop = asyncio.new_event_loop()
    t0 = time.time()
    n_total = len(batch_tasks) * args.group_size

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_tokens, temperature=args.temperature,
    )

    # --- Phase A: Initialize episodes ---
    episodes = []
    group_indices = []
    for t_idx, task in enumerate(batch_tasks):
        start = len(episodes)
        for _ in range(args.group_size):
            episodes.append({
                "task": task,
                "messages": [{"role": "user", "content": task.problem}],
                "prompt_tokens_per_turn": [],
                "completion_tokens_per_turn": [],
                "logprobs_per_turn": [],
                "per_turn_rewards": [],
                "hw_passed": 0,
                "all_passed": False,
                "done": False,
            })
        group_indices.append((start, start + args.group_size))

    # --- Phase B: Turn-depth batched rollouts ---
    for turn in range(1, args.max_turns + 1):
        active = [i for i, ep in enumerate(episodes) if not ep["done"]]
        if not active:
            break

        # Fire Tinker futures
        futures = []
        for i in active:
            ep = episodes[i]
            pt = tokenizer.apply_chat_template(
                ep["messages"], add_generation_prompt=True,
                tokenize=False,
            )
            ptok = tokenizer.encode(pt, add_special_tokens=False)
            future = sampling_client.sample(
                prompt=tinker.ModelInput.from_ints(ptok),
                num_samples=1, sampling_params=sampling_params,
            )
            futures.append((i, future, ptok))

        print(f"  [step {step} t{turn}] fired {len(futures)}", end=" ", flush=True)

        # Collect generation results
        collected = []
        for i, future, ptok in futures:
            result = future.result()
            seq = result.sequences[0]
            ct = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            collected.append((i, ptok, list(seq.tokens),
                              [lp if lp is not None else 0.0 for lp in (seq.logprobs or [])],
                              ct))

        # Run sandbox tests (could parallelize with ThreadPool if needed)
        for i, ptok, comp_tokens, logprobs, ct in collected:
            ep = episodes[i]
            ep["prompt_tokens_per_turn"].append(ptok)
            ep["completion_tokens_per_turn"].append(comp_tokens)
            ep["logprobs_per_turn"].append(logprobs)

            code = extract_code_from_model(ct)
            if code is None:
                info = {"no_code": True, "turn": turn, "max_turns": args.max_turns,
                        "tests_total": len(ep["task"].tests), "hw_passed": ep["hw_passed"],
                        "old_hw": ep["hw_passed"], "all_passed": False}
                is_terminal = turn >= args.max_turns
                ep["per_turn_rewards"].append(reward_fn(info, is_terminal))
                ep["messages"].append({"role": "assistant", "content": ct})
                ep["messages"].append({"role": "user", "content": "No code block found. Use ```python```."})
                if is_terminal:
                    ep["done"] = True
                continue

            ap, details = loop.run_until_complete(
                sandbox_check_correctness(ep["task"].tests, code, timeout=args.sandbox_timeout)
            )
            tp = details.get("tests_passed", len(ep["task"].tests) if ap else 0)
            tt = details.get("tests_total", len(ep["task"].tests))

            old_hw = ep["hw_passed"]
            ep["hw_passed"] = max(ep["hw_passed"], tp)
            ep["all_passed"] = ap
            is_terminal = ap or turn >= args.max_turns

            info = {"turn": turn, "max_turns": args.max_turns,
                    "tests_total": tt, "tests_passed": tp,
                    "hw_passed": ep["hw_passed"], "old_hw": old_hw,
                    "all_passed": ap, "no_code": False}
            ep["per_turn_rewards"].append(reward_fn(info, is_terminal))

            if ap:
                ep["done"] = True
            elif turn < args.max_turns:
                ep["messages"].append({"role": "assistant", "content": ct})
                ep["messages"].append({"role": "user", "content":
                    FEEDBACK_TEMPLATE.format(passed=tp, total=tt)})
            else:
                ep["done"] = True

        n_done = sum(1 for ep in episodes if ep["done"])
        print(f"-> {n_done}/{n_total} done", flush=True)

    loop.close()
    t_gen = time.time() - t0

    # --- Phase C: Rewards + advantages ---
    episode_rewards = np.array([sum(ep["per_turn_rewards"]) for ep in episodes], dtype=np.float32)
    advantages = np.zeros_like(episode_rewards)
    groups_skipped = 0
    for start, end in group_indices:
        group = episode_rewards[start:end]
        std = group.std()
        if std < 1e-8:
            groups_skipped += 1
        else:
            advantages[start:end] = (group - group.mean()) / (std + 1e-8)

    # --- Phase D: Build Datum objects ---
    data = []
    total_comp_tokens = 0
    for ep_idx, ep in enumerate(episodes):
        if abs(advantages[ep_idx]) < 1e-8:
            continue
        adv = float(advantages[ep_idx])
        for t_idx in range(len(ep["per_turn_rewards"])):
            ptok = ep["prompt_tokens_per_turn"][t_idx]
            ctok = ep["completion_tokens_per_turn"][t_idx]
            lps = ep["logprobs_per_turn"][t_idx]
            n_p, n_c = len(ptok), len(ctok)
            full = ptok + ctok
            lp_list = [0.0] * n_p + lps[:n_c]
            if len(lp_list) < len(full):
                lp_list.extend([0.0] * (len(full) - len(lp_list)))
            lp_list = lp_list[:len(full)]
            data.append(tinker.Datum(
                model_input=tinker.ModelInput.from_ints(full),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(data=full, dtype="int64"),
                    "logprobs": tinker.TensorData(data=lp_list, dtype="float32"),
                    "advantages": tinker.TensorData(
                        data=[0.0]*n_p + [adv]*n_c, dtype="float32"),
                },
            ))
            total_comp_tokens += n_c

    # --- Phase E: Train ---
    loss_val = 0.0
    if data:
        cfg = {}
        if args.loss_fn == "cispo":
            cfg = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
        fb = training_client.forward_backward(
            data=data, loss_fn=args.loss_fn,
            loss_fn_config=cfg if cfg else None,
        )
        training_client.optim_step(
            tinker.AdamParams(learning_rate=args.lr, grad_clip_norm=args.grad_clip_norm)
        )
        fbr = fb.result()
        ls = fbr.metrics.get("loss:sum", fbr.metrics.get("loss", 0))
        loss_val = ls / total_comp_tokens if total_comp_tokens > 0 else 0.0

    t_total = time.time() - t0
    solve_rate = np.mean([ep["all_passed"] for ep in episodes])
    avg_turns = np.mean([len(ep["per_turn_rewards"]) for ep in episodes])

    batch_solved = {ep["task"].problem[:50] for ep in episodes if ep["all_passed"]}

    return {
        "mean_reward": float(episode_rewards.mean()),
        "reward_std": float(episode_rewards.std()),
        "loss": float(loss_val),
        "solve_rate": float(solve_rate),
        "avg_turns": float(avg_turns),
        "groups_skipped": groups_skipped,
        "n_data": len(data),
        "batch_solved": batch_solved,
        "time_gen": round(t_gen, 1),
        "time_total": round(t_total, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir or f"checkpoints/lcb_{args.reward}_s{args.seed}"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "metrics.jsonl")

    print(f"[lcb] Reward: {args.reward} | Model: {args.model}")
    print(f"[lcb] Batch: {args.batch_size}×{args.group_size} | Max turns: {args.max_turns}")
    print(f"[lcb] Steps: {args.max_steps} | Output: {output_dir}")

    service = ServiceClient()
    tc = service.create_lora_training_client(base_model=args.model, rank=args.lora_rank)
    sc = tc.save_weights_and_get_sampling_client(name="init")
    tok = tc.get_tokenizer()
    print("[lcb] Tinker connected.")

    # Load tasks
    all_tasks = load_deepcoder_tasks("train", seed=args.seed)
    tasks = [t for t in all_tasks if len(t.tests) >= args.min_tests]
    print(f"[lcb] {len(tasks)} train tasks (>= {args.min_tests} tests)")

    eval_tasks_all = load_deepcoder_tasks("test", seed=args.seed)
    eval_tasks = [t for t in eval_tasks_all if len(t.tests) >= args.min_tests][:args.eval_n_problems]
    print(f"[lcb] {len(eval_tasks)} eval tasks")

    reward_fn = REWARD_FNS[args.reward]

    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project,
                   name=f"lcb-{args.reward}-s{args.seed}",
                   config=vars(args))
        print(f"[lcb] wandb: {wandb.run.url}")

    rng = random.Random(args.seed)
    ever_solved: set[str] = set()
    solve_counts: dict[str, int] = {}
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        batch = [rng.choice(tasks) for _ in range(args.batch_size)]

        metrics = grpo_step(step, batch, sc, tc, tok, reward_fn, args)

        batch_solved = metrics.pop("batch_solved", set())
        new_disc = batch_solved - ever_solved
        ever_solved |= batch_solved
        for tid in batch_solved:
            solve_counts[tid] = solve_counts.get(tid, 0) + 1
        mastered = sum(1 for c in solve_counts.values() if c >= 3)

        print(f"[step {step}/{args.max_steps}] "
              f"solve={metrics['solve_rate']:.2f} reward={metrics['mean_reward']:.2f}±{metrics['reward_std']:.2f} "
              f"loss={metrics['loss']:.4f} turns={metrics['avg_turns']:.1f} "
              f"disc={len(ever_solved)} mast={mastered} "
              f"skip={metrics['groups_skipped']}/{args.batch_size} "
              f"t={metrics['time_total']}s", flush=True)

        metrics["discovery"] = len(ever_solved)
        metrics["mastery"] = mastered
        metrics["step"] = step

        with open(log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if use_wandb:
            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

        if step % args.refresh_steps == 0:
            sc = tc.save_weights_and_get_sampling_client(name=f"step_{step:04d}")

        if step % args.save_steps == 0:
            sr = tc.save_state(f"step_{step:04d}").result()
            print(f"[lcb] Checkpoint: {getattr(sr, 'path', None)}", flush=True)

        # Eval
        if eval_tasks and step % args.eval_steps == 0:
            print(f"[eval] Running on {len(eval_tasks)} problems...", flush=True)
            loop = asyncio.new_event_loop()
            eval_params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.eval_temperature)
            solved = 0
            for ei, et in enumerate(eval_tasks):
                msgs = [{"role": "user", "content": et.problem}]
                ep_solved = False
                for turn in range(1, args.max_turns + 1):
                    pt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                  tokenize=False, enable_thinking=False)
                    ptok = tok.encode(pt, add_special_tokens=False)
                    r = sc.sample(prompt=tinker.ModelInput.from_ints(ptok),
                                  num_samples=1, sampling_params=eval_params).result()
                    ct = tok.decode(r.sequences[0].tokens, skip_special_tokens=True)
                    code = extract_code_from_model(ct)
                    if code is None:
                        msgs.append({"role": "assistant", "content": ct})
                        msgs.append({"role": "user", "content": "No code block. Use ```python```."})
                        continue
                    ap, det = loop.run_until_complete(
                        sandbox_check_correctness(et.tests, code, timeout=args.sandbox_timeout))
                    if ap:
                        ep_solved = True
                        break
                    tp = det.get("tests_passed", 0)
                    tt = det.get("tests_total", len(et.tests))
                    msgs.append({"role": "assistant", "content": ct})
                    msgs.append({"role": "user", "content":
                        FEEDBACK_TEMPLATE.format(passed=tp, total=tt)})
                if ep_solved:
                    solved += 1
            loop.close()
            pass1 = solved / len(eval_tasks) if eval_tasks else 0
            print(f"[eval] step={step}: {solved}/{len(eval_tasks)} = {pass1:.2f}", flush=True)
            if use_wandb:
                wandb.log({"eval/pass_at_1": pass1, "eval/solved": solved}, step=step)

    elapsed = time.time() - t_start
    tc.save_state("final").result()
    print(f"\n[lcb] Done. {elapsed/60:.1f} min. Discovery={len(ever_solved)} Mastery={mastered}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
