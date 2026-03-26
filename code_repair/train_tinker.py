"""
Multi-turn GRPO training for code repair using the Tinker API.

Each episode: model sees buggy code + test results, outputs <repair>,
gets updated test results, repeats for up to max_turns. All active
episodes sample in parallel via Tinker futures (turn-depth batching).

Three reward conditions:
  - sparse:       terminal only (win/loss)
  - dense_passes: per-turn HWM delta (potential-based)
  - dense_full:   HWM + partial-correctness bonuses (non-potential)

Usage:
    python -m code_repair.train_tinker --reward {sparse,dense_passes,dense_full} [options]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    import tinker
    from tinker import ServiceClient
except ImportError:
    print("[code-repair] ERROR: tinker not installed.", file=sys.stderr)
    sys.exit(1)

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

from code_repair.env.sandbox import run_tests
from code_repair.env.code_repair_env import (
    SYSTEM_PROMPT,
    CodeRepairMessageEnv,
    MessageStepResult,
    extract_repair,
    format_initial_prompt,
    format_feedback,
)
from code_repair.env.rewards import (
    TERMINAL_WIN, SPEED_BONUS, TERMINAL_LOSS, FORMAT_PENALTY,
    PASS_REWARD_PER_TEST, NO_CRASH_BONUS, TYPE_MATCH_BONUS, SHAPE_MATCH_BONUS,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--reward", required=True, choices=["sparse", "dense_passes", "dense_full"])
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--max_turns", type=int, default=4)
    p.add_argument("--problems_path", default="code_repair/data/problems_combined/train.json")
    p.add_argument("--eval_problems_path", default="code_repair/data/problems_combined/eval.json")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4, help="Problems per step.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.04, help="KL coef (documentation; CISPO uses clip thresholds).")
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--loss_fn", default="cispo", choices=["importance_sampling", "ppo", "cispo"])
    p.add_argument("--max_completion_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--refresh_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--eval_samples", type=int, default=4)
    p.add_argument("--eval_temperature", type=float, default=0.6)
    p.add_argument("--sandbox_timeout", type=int, default=5)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="discovery-code-repair")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-turn reward functions (same as train.py)
# ---------------------------------------------------------------------------

def _reward_sparse(info: dict, is_terminal: bool) -> float:
    if info.get("format_violation"):
        return FORMAT_PENALTY
    if is_terminal:
        if info["all_passed"]:
            return TERMINAL_WIN + SPEED_BONUS * (info["max_turns"] - info["turn"])
        return TERMINAL_LOSS
    return 0.0


def _reward_dense_passes(info: dict, is_terminal: bool) -> float:
    r = 0.0
    if info.get("format_violation"):
        return FORMAT_PENALTY
    n_tests = max(info["num_tests"], 1)
    hwm_delta = info["hw_passing"] - info["old_hw"]
    if hwm_delta > 0:
        r += PASS_REWARD_PER_TEST * hwm_delta / n_tests
    if is_terminal:
        if info["all_passed"]:
            r += TERMINAL_WIN + SPEED_BONUS * (info["max_turns"] - info["turn"])
        else:
            r += TERMINAL_LOSS
    return r


def _reward_dense_full(info: dict, is_terminal: bool) -> float:
    r = 0.0
    if info.get("format_violation"):
        return FORMAT_PENALTY
    n_tests = max(info["num_tests"], 1)
    hwm_delta = info["hw_passing"] - info["old_hw"]
    if hwm_delta > 0:
        r += PASS_REWARD_PER_TEST * hwm_delta / n_tests
    r += NO_CRASH_BONUS * info.get("no_crash_failing", 0) / n_tests
    r += TYPE_MATCH_BONUS * info.get("type_match_failing", 0) / n_tests
    r += SHAPE_MATCH_BONUS * info.get("shape_match_failing", 0) / n_tests
    if is_terminal:
        if info["all_passed"]:
            r += TERMINAL_WIN + SPEED_BONUS * (info["max_turns"] - info["turn"])
        else:
            r += TERMINAL_LOSS
    return r


REWARD_FNS = {
    "sparse": _reward_sparse,
    "dense_passes": _reward_dense_passes,
    "dense_full": _reward_dense_full,
}


# ---------------------------------------------------------------------------
# GRPO step — turn-depth batched sampling via Tinker
# ---------------------------------------------------------------------------

def grpo_step(
    step: int,
    batch_problems: list[dict],
    sampling_client,
    training_client,
    tokenizer,
    reward_fn,
    args,
) -> dict:
    import asyncio
    loop = asyncio.new_event_loop()

    t0 = time.time()
    n_total = len(batch_problems) * args.group_size

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_completion_tokens,
        temperature=args.temperature,
    )

    # --- Phase A: Initialize episode states via MessageEnv ---
    episodes = []
    group_indices = []
    for p_idx, p in enumerate(batch_problems):
        start = len(episodes)
        for _ in range(args.group_size):
            env = CodeRepairMessageEnv(p, max_turns=args.max_turns, sandbox_timeout=args.sandbox_timeout)
            messages = loop.run_until_complete(env.initial_observation())
            episodes.append({
                "env": env,
                "problem": p,
                "messages": messages,
                "prompt_tokens_per_turn": [],
                "completion_tokens_per_turn": [],
                "logprobs_per_turn": [],
                "completion_texts": [],
                "per_turn_rewards": [],
                "all_passed": False,
                "done": False,
            })
        group_indices.append((start, start + args.group_size))

    # --- Phase B: Turn-depth batched sampling ---
    for turn in range(1, args.max_turns + 1):
        active = [i for i, ep in enumerate(episodes) if not ep["done"]]
        if not active:
            break

        # Fire all sample requests as Tinker futures
        futures = []
        for i in active:
            ep = episodes[i]
            prompt_text = tokenizer.apply_chat_template(
                ep["messages"], add_generation_prompt=True,
                tokenize=False, enable_thinking=False,
            )
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            future = sampling_client.sample(
                prompt=tinker.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=sampling_params,
            )
            futures.append((i, future, prompt_tokens))

        t_fired = time.time()
        print(f"  [step {step} turn {turn}] fired {len(futures)} requests "
              f"in {t_fired - t0:.1f}s, waiting...", flush=True)

        # Collect Tinker results (fast — already computed server-side)
        collected = []
        for i, future, prompt_tokens in futures:
            result = future.result()
            seq = result.sequences[0]
            completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            collected.append((i, prompt_tokens, seq, completion_text))

        t_collected = time.time()

        # Step environments — sandbox tests run in parallel via ProcessPool
        import concurrent.futures

        def _run_env_step(item):
            """Run env.step synchronously (sandbox is subprocess-based, not truly async)."""
            i, prompt_tokens, seq, completion_text = item
            ep = episodes[i]
            ep["prompt_tokens_per_turn"].append(prompt_tokens)
            ep["completion_tokens_per_turn"].append(list(seq.tokens))
            ep["logprobs_per_turn"].append(
                [lp if lp is not None else -100.0 for lp in (seq.logprobs or [])]
            )
            ep["completion_texts"].append(completion_text)

            # Call the sync version of step directly to avoid asyncio-in-thread issues
            env = ep["env"]
            env.turn += 1
            repair = extract_repair(completion_text)

            if repair is None:
                step_result = env._make_format_error_result()
            else:
                # Inline the sandbox call — this is the expensive part
                from code_repair.env.sandbox import run_tests as _run_tests
                test_results = _run_tests(
                    repair, env.problem["test_code"], env.problem["entry_point"],
                    timeout=env.sandbox_timeout, detailed=True,
                )
                env.repairs.append(repair)
                env.test_results_history.append(test_results)

                prev_results = env.test_results_history[-2]
                prev_passing = {r.name for r in prev_results if r.passed}
                curr_passing = {r.name for r in test_results if r.passed}
                num_tests = max(len(test_results), 1)
                old_hw = env.hw_passing
                env.hw_passing = max(env.hw_passing, len(curr_passing))
                all_passed = (
                    len(curr_passing) == len(test_results) and len(test_results) > 0
                    and not (len(test_results) == 1 and test_results[0].name in
                             ("timeout", "syntax_error", "runtime_error", "import_error"))
                )
                env.all_passed = all_passed
                no_crash = sum(1 for r in test_results if r.no_crash and not r.passed)
                type_match = sum(1 for r in test_results if not r.passed and r.no_crash and r.return_type)
                shape_match = sum(1 for r in test_results if not r.passed and r.return_shape is not None)
                episode_done = all_passed or env.turn >= env.max_turns

                metrics = {
                    "turn": float(env.turn), "max_turns": float(env.max_turns),
                    "num_tests": float(num_tests),
                    "curr_passing": float(len(curr_passing)),
                    "hw_passing": float(env.hw_passing), "old_hw": float(old_hw),
                    "all_passed": 1.0 if all_passed else 0.0,
                    "no_crash_failing": float(no_crash),
                    "type_match_failing": float(type_match),
                    "shape_match_failing": float(shape_match),
                }

                if all_passed:
                    feedback = f"All {num_tests} tests passing! Fixed in {env.turn} turn(s)."
                else:
                    feedback = format_feedback(
                        test_results, repair, turn=env.turn, max_turns=env.max_turns,
                        prev_passing=prev_passing,
                    )
                next_msgs = env._build_conversation_sync(feedback)
                step_result = MessageStepResult(
                    reward=0.0, episode_done=episode_done,
                    next_messages=next_msgs, metrics=metrics,
                )

            m = step_result.metrics
            info = {
                "turn": int(m.get("turn", env.turn)),
                "max_turns": int(m.get("max_turns", args.max_turns)),
                "num_tests": int(m.get("num_tests", 0)),
                "hw_passing": int(m.get("hw_passing", 0)),
                "old_hw": int(m.get("old_hw", 0)),
                "all_passed": m.get("all_passed", 0.0) > 0,
                "format_violation": m.get("format_violation", 0.0) > 0,
                "no_crash_failing": int(m.get("no_crash_failing", 0)),
                "type_match_failing": int(m.get("type_match_failing", 0)),
                "shape_match_failing": int(m.get("shape_match_failing", 0)),
            }
            ep["per_turn_rewards"].append(reward_fn(info, step_result.episode_done))
            ep["all_passed"] = info["all_passed"]

            if step_result.episode_done:
                ep["done"] = True
            else:
                ep["messages"] = step_result.next_messages

        # Run sandbox tests in parallel (subprocess-based, releases GIL)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(collected), 16)) as pool:
            list(pool.map(_run_env_step, collected))

        n_done = sum(1 for ep in episodes if ep["done"])
        print(f"  [step {step} turn {turn}] collected {len(futures)} results "
              f"({n_done}/{n_total} episodes done)", flush=True)

    loop.close()

    t_gen = time.time() - t0

    # --- Phase C: Compute rewards and advantages ---
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
    total_completion_tokens = 0
    for ep_idx, ep in enumerate(episodes):
        if abs(advantages[ep_idx]) < 1e-8:
            continue
        adv_value = float(advantages[ep_idx])
        n_turns = len(ep["per_turn_rewards"])

        for turn_idx in range(n_turns):
            prompt_tokens = ep["prompt_tokens_per_turn"][turn_idx]
            completion_tokens = ep["completion_tokens_per_turn"][turn_idx]
            sample_logprobs = ep["logprobs_per_turn"][turn_idx]

            n_prompt = len(prompt_tokens)
            n_comp = len(completion_tokens)
            full_tokens = prompt_tokens + completion_tokens

            # Pad logprobs for prompt positions
            lp_list = [-100.0] * n_prompt + sample_logprobs
            if len(lp_list) < len(full_tokens):
                lp_list.extend([-100.0] * (len(full_tokens) - len(lp_list)))
            lp_list = lp_list[:len(full_tokens)]

            datum = tinker.Datum(
                model_input=tinker.ModelInput.from_ints(full_tokens),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(data=full_tokens, dtype="int64"),
                    "logprobs": tinker.TensorData(data=lp_list, dtype="float32"),
                    "advantages": tinker.TensorData(
                        data=[0.0] * n_prompt + [adv_value] * n_comp,
                        dtype="float32",
                    ),
                },
            )
            data.append(datum)
            total_completion_tokens += n_comp

    # --- Phase E: Forward-backward + optimizer step ---
    if not data:
        loss_val = 0.0
    else:
        loss_fn_config = {}
        if args.loss_fn == "cispo":
            loss_fn_config = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
        fwdbwd_future = training_client.forward_backward(
            data=data, loss_fn=args.loss_fn,
            loss_fn_config=loss_fn_config if loss_fn_config else None,
        )
        optim_future = training_client.optim_step(
            tinker.AdamParams(learning_rate=args.lr, grad_clip_norm=args.grad_clip_norm)
        )
        fwdbwd_result = fwdbwd_future.result()
        optim_future.result()

        loss_sum = fwdbwd_result.metrics.get("loss:sum", fwdbwd_result.metrics.get("loss", 0))
        loss_val = loss_sum / total_completion_tokens if total_completion_tokens > 0 else 0.0

    t_total = time.time() - t0

    # Metrics
    solve_rate = np.mean([ep["all_passed"] for ep in episodes])
    avg_turns = np.mean([len(ep["per_turn_rewards"]) for ep in episodes])
    avg_hw = np.mean([ep["env"].hw_passing for ep in episodes])

    # Per-problem solve tracking (for discovery/mastery)
    batch_solved = set()
    for ep in episodes:
        if ep["all_passed"]:
            batch_solved.add(ep["problem"]["task_id"])

    return {
        "step": step,
        "mean_reward": float(episode_rewards.mean()),
        "reward_std": float(episode_rewards.std()),
        "loss": float(loss_val),
        "solve_rate": float(solve_rate),
        "avg_turns": float(avg_turns),
        "avg_hwm": float(avg_hw),
        "groups_skipped": groups_skipped,
        "n_train_samples": len(data),
        "n_completion_tokens": total_completion_tokens,
        "batch_solved": batch_solved,
        "time_gen": round(t_gen, 1),
        "time_total": round(t_total, 1),
    }


# ---------------------------------------------------------------------------
# Eval (Tinker-based multi-turn)
# ---------------------------------------------------------------------------

def run_eval(
    sampling_client, tokenizer, problems, args, step,
) -> dict:
    """Run multi-turn eval using Tinker sampling client."""
    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_completion_tokens,
        temperature=args.eval_temperature,
    )
    per_problem = {}
    total_solved = 0
    total_samples = 0
    unique_solved = set()
    traces = []

    for prob_idx, p in enumerate(problems):
        n_solved = 0
        initial_results = run_tests(
            p["buggy_code"], p["test_code"], p["entry_point"],
            timeout=args.sandbox_timeout, detailed=True,
        )
        user_msg = format_initial_prompt(p["buggy_code"], initial_results, args.max_turns)

        for sample_idx in range(args.eval_samples):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            hw_passing = 0
            all_passed = False
            test_history = [initial_results]
            turns_data = []

            for turn in range(1, args.max_turns + 1):
                prompt_text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True,
                    tokenize=False, enable_thinking=False,
                )
                prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                result = sampling_client.sample(
                    prompt=tinker.ModelInput.from_ints(prompt_tokens),
                    num_samples=1,
                    sampling_params=sampling_params,
                ).result()
                completion_text = tokenizer.decode(
                    result.sequences[0].tokens, skip_special_tokens=True,
                )

                repair = extract_repair(completion_text)
                if repair is None:
                    is_terminal = turn >= args.max_turns
                    feedback = (f"ERROR: No <repair> tags found. You MUST wrap your code in "
                               f"<repair>...</repair> tags. ({args.max_turns - turn} remaining)")
                    turns_data.append({
                        "turn": turn, "repair_found": False,
                        "curr_passing": 0, "num_tests": 0, "all_passed": False,
                    })
                else:
                    test_results = run_tests(
                        repair, p["test_code"], p["entry_point"],
                        timeout=args.sandbox_timeout, detailed=True,
                    )
                    test_history.append(test_results)
                    prev_passing = {r.name for r in test_history[-2] if r.passed}
                    curr_passing = {r.name for r in test_results if r.passed}
                    hw_passing = max(hw_passing, len(curr_passing))
                    all_passed = (
                        len(curr_passing) == len(test_results) and len(test_results) > 0
                        and not (len(test_results) == 1 and test_results[0].name in
                                 ("timeout", "syntax_error", "runtime_error"))
                    )
                    is_terminal = all_passed or turn >= args.max_turns
                    if all_passed:
                        feedback = f"All tests pass! Fixed in {turn} turn(s)."
                    else:
                        feedback = format_feedback(
                            test_results, repair, turn=turn, max_turns=args.max_turns,
                            prev_passing=prev_passing,
                        )
                    turns_data.append({
                        "turn": turn, "repair_found": True,
                        "curr_passing": len(curr_passing),
                        "num_tests": len(test_results),
                        "all_passed": all_passed,
                        "repair": (repair or "")[:500],
                    })

                messages.append({"role": "assistant", "content": completion_text})
                messages.append({"role": "user", "content": feedback})
                if is_terminal:
                    break

            if all_passed:
                n_solved += 1
                unique_solved.add(p["task_id"])

            # Collect trace for first sample of first 5 problems
            if sample_idx == 0 and prob_idx < 5:
                traces.append({
                    "task_id": p["task_id"],
                    "n_bugs": p.get("n_bugs", 1),
                    "solved": all_passed,
                    "total_turns": len(turns_data),
                    "hw_passing": hw_passing,
                    "turns": turns_data,
                })

        # pass@k
        n = args.eval_samples
        c = n_solved
        pass1 = 1.0 - float(np.prod(1.0 - 1 / np.arange(n - c + 1, n + 1))) if c > 0 else 0.0
        per_problem[p["task_id"]] = {"pass_at_1": pass1, "n_solved": n_solved, "n_samples": n}
        total_solved += n_solved
        total_samples += n
        print(f"  [{prob_idx+1}/{len(problems)}] {p['task_id']}: {n_solved}/{n} solved, pass@1={pass1:.3f}")

    overall_pass1 = np.mean([r["pass_at_1"] for r in per_problem.values()])
    raw_solve_rate = total_solved / total_samples if total_samples else 0

    print(f"[eval] step={step}: pass@1={overall_pass1:.3f} "
          f"unique={len(unique_solved)}/{len(problems)} "
          f"raw={total_solved}/{total_samples}", flush=True)

    return {
        "eval_pass_at_1": float(overall_pass1),
        "eval_unique_solved": len(unique_solved),
        "eval_n_problems": len(problems),
        "eval_raw_solve_rate": raw_solve_rate,
        "eval_unique_solved_set": sorted(unique_solved),
        "traces": traces,
        "per_problem": per_problem,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_trajectories(
    step: int,
    probe_problems: list[dict],
    sampling_client,
    tokenizer,
    args,
    output_dir: str,
):
    """Run one episode per probe problem and save full conversations to disk."""
    import asyncio
    loop = asyncio.new_event_loop()

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_completion_tokens,
        temperature=args.eval_temperature,
    )

    traces = []
    for p in probe_problems:
        env = CodeRepairMessageEnv(p, max_turns=args.max_turns, sandbox_timeout=args.sandbox_timeout)
        messages = loop.run_until_complete(env.initial_observation())
        turns = []

        for turn in range(1, args.max_turns + 1):
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=False, enable_thinking=False,
            )
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            result = sampling_client.sample(
                prompt=tinker.ModelInput.from_ints(prompt_tokens),
                num_samples=1, sampling_params=sampling_params,
            ).result()
            completion = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)

            step_result = loop.run_until_complete(
                env.step({"role": "assistant", "content": completion})
            )
            m = step_result.metrics
            turns.append({
                "turn": int(m.get("turn", turn)),
                "completion": completion[:3000],
                "curr_passing": int(m.get("curr_passing", 0)),
                "hw_passing": int(m.get("hw_passing", 0)),
                "num_tests": int(m.get("num_tests", 0)),
                "all_passed": m.get("all_passed", 0) > 0,
                "format_violation": m.get("format_violation", 0) > 0,
                "feedback": step_result.next_messages[-1]["content"][:500] if step_result.next_messages else "",
            })

            if step_result.episode_done:
                break
            messages = step_result.next_messages

        traces.append({
            "task_id": p["task_id"],
            "bug_types": p.get("bug_types", []),
            "n_bugs": p.get("n_bugs", 1),
            "solved": env.all_passed,
            "total_turns": env.turn,
            "hw_passing": env.hw_passing,
            "turns": turns,
        })

    loop.close()

    trace_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"step_{step:04d}.json")
    with open(trace_path, "w") as f:
        json.dump(traces, f, indent=2)

    n_solved = sum(1 for t in traces if t["solved"])
    print(f"[code-repair] Saved {len(traces)} trajectories to {trace_path} "
          f"({n_solved}/{len(traces)} solved)", flush=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir or f"checkpoints/code_repair_{args.reward}_s{args.seed}"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "metrics.jsonl")

    print(f"[code-repair] Reward      : {args.reward}")
    print(f"[code-repair] Model       : {args.model}")
    print(f"[code-repair] Max turns   : {args.max_turns}")
    print(f"[code-repair] LoRA rank   : {args.lora_rank}")
    print(f"[code-repair] Loss fn     : {args.loss_fn}")
    print(f"[code-repair] Batch       : {args.batch_size} problems × {args.group_size} episodes")
    print(f"[code-repair] Max steps   : {args.max_steps}")
    print(f"[code-repair] Output dir  : {output_dir}")

    # Initialize Tinker
    print("[code-repair] Connecting to Tinker API ...")
    service = ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=args.model, rank=args.lora_rank,
    )
    sampling_client = training_client.save_weights_and_get_sampling_client(name="init")
    tokenizer = training_client.get_tokenizer()
    print("[code-repair] Tinker clients initialized.")

    # Load problems
    with open(args.problems_path) as f:
        problems = json.load(f)
    print(f"[code-repair] Loaded {len(problems)} training problems")

    eval_problems = []
    if os.path.exists(args.eval_problems_path):
        with open(args.eval_problems_path) as f:
            eval_problems = json.load(f)
        print(f"[code-repair] Loaded {len(eval_problems)} eval problems")

    reward_fn = REWARD_FNS[args.reward]

    # Select probe problems for trajectory saving (mix of easy + hard)
    probe_rng = random.Random(args.seed + 1)
    probe_single = [p for p in eval_problems if p.get("n_bugs", 1) == 1][:3]
    probe_multi = [p for p in eval_problems if p.get("n_bugs", 1) >= 2][:3]
    probe_problems = probe_single + probe_multi
    if not probe_problems:
        probe_problems = eval_problems[:5]
    print(f"[code-repair] Probe problems: {[p['task_id'] for p in probe_problems]}")

    # Save step-0 trajectories (before any training)
    save_trajectories(
        step=0, probe_problems=probe_problems,
        sampling_client=sampling_client, tokenizer=tokenizer,
        args=args, output_dir=output_dir,
    )

    # W&B
    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"code-repair-{args.reward}-s{args.seed}",
            config=vars(args),
        )
        print(f"[code-repair] W&B run: {wandb.run.url}")

    # Training loop
    rng = random.Random(args.seed)
    t_start = time.time()

    # Discovery/mastery tracking (cumulative across all steps)
    ever_solved: set[str] = set()      # unique problems ever solved (discovery)
    solve_counts: dict[str, int] = {}  # task_id -> times solved (for mastery)

    for step in range(1, args.max_steps + 1):
        batch = [rng.choice(problems) for _ in range(args.batch_size)]

        metrics = grpo_step(
            step=step, batch_problems=batch,
            sampling_client=sampling_client,
            training_client=training_client,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            args=args,
        )

        # Update discovery/mastery tracking
        batch_solved = metrics.pop("batch_solved", set())
        new_discoveries = batch_solved - ever_solved
        ever_solved |= batch_solved
        for tid in batch_solved:
            solve_counts[tid] = solve_counts.get(tid, 0) + 1
        mastered = sum(1 for c in solve_counts.values() if c >= 3)

        metrics["discovery"] = len(ever_solved)
        metrics["mastery"] = mastered
        metrics["new_discoveries"] = len(new_discoveries)
        metrics["batch_unique_solved"] = len(batch_solved)

        print(
            f"[step {step:4d}/{args.max_steps}] "
            f"solve={metrics['solve_rate']:.3f}  reward={metrics['mean_reward']:.2f}±{metrics['reward_std']:.2f}  "
            f"turns={metrics['avg_turns']:.1f}  hwm={metrics['avg_hwm']:.1f}  "
            f"loss={metrics['loss']:.4f}  "
            f"discovery={len(ever_solved)}  mastery={mastered}  "
            f"time={metrics['time_total']}s"
        )

        with open(log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if use_wandb:
            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
            # Log discovery set as artifact periodically
            if step % 10 == 0:
                wandb.log({
                    "discovery/ever_solved": sorted(ever_solved),
                    "discovery/solve_counts": dict(solve_counts),
                }, step=step)

        # Refresh sampling client for near-on-policy rollouts
        if step % args.refresh_steps == 0:
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"step_{step:04d}"
            )

        # Checkpoint + save trajectories
        if step % args.save_steps == 0:
            ckpt_name = f"step_{step:04d}"
            print(f"[code-repair] Saving checkpoint: {ckpt_name} ...")
            save_result = training_client.save_state(ckpt_name).result()
            ckpt_path = getattr(save_result, "path", None)
            print(f"[code-repair] Checkpoint saved: {ckpt_path}")

            manifest = os.path.join(output_dir, "checkpoints.jsonl")
            with open(manifest, "a") as f:
                f.write(json.dumps({"step": step, "name": ckpt_name, "path": ckpt_path}) + "\n")

            # Save probe trajectories for offline inspection
            save_trajectories(
                step=step,
                probe_problems=probe_problems,
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                args=args,
                output_dir=output_dir,
            )

        # Eval
        if eval_problems and step % args.eval_steps == 0:
            eval_results = run_eval(sampling_client, tokenizer, eval_problems, args, step)
            if use_wandb:
                wandb.log({
                    "eval/pass_at_1": eval_results["eval_pass_at_1"],
                    "eval/unique_solved": eval_results["eval_unique_solved"],
                    "eval/raw_solve_rate": eval_results["eval_raw_solve_rate"],
                }, step=step)

                # Log trajectory table
                traces = eval_results.get("traces", [])
                if traces:
                    cols = ["step", "task_id", "n_bugs", "solved", "total_turns", "hw_passing"]
                    max_t = max(t["total_turns"] for t in traces)
                    for ti in range(max_t):
                        cols.extend([f"t{ti+1}_passing", f"t{ti+1}_repair"])
                    table = wandb.Table(columns=cols)
                    for t in traces:
                        row = [step, t["task_id"], t["n_bugs"], t["solved"],
                               t["total_turns"], t["hw_passing"]]
                        for ti in range(max_t):
                            if ti < len(t["turns"]):
                                td = t["turns"][ti]
                                row.append(f"{td['curr_passing']}/{td['num_tests']}")
                                row.append(td.get("repair", "(no tag)")[:300])
                            else:
                                row.extend(["", ""])
                        table.add_data(*row)
                    wandb.log({"eval/trajectories": table}, step=step)

    # Final save
    print("[code-repair] Saving final checkpoint ...")
    training_client.save_state("final").result()

    elapsed = time.time() - t_start
    print(f"\n[code-repair] === Training Complete ===")
    print(f"[code-repair] Total time: {elapsed / 60:.1f} min")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
