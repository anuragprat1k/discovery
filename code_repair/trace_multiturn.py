"""One-off: run 1 problem for 4 turns, dump full trace to file."""
from __future__ import annotations
import asyncio, json, sys, os
from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker import ServiceClient
from code_repair.deepcoder.code_env import load_deepcoder_tasks
from code_repair.deepcoder.code_grading import extract_code_from_model
from code_repair.subprocess_sandbox import check_correctness
from code_repair.train_tinker_lcb import _build_feedback, reward_path_dep, reward_path_indep

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MAX_TOKENS = 8192
MAX_TURNS = 4
TEMPERATURE = 0.6
SANDBOX_TIMEOUT = 6

def main():
    service = ServiceClient()
    tc = service.create_lora_training_client(base_model=MODEL, rank=32)
    sc = tc.save_weights_and_get_sampling_client(name="trace_init")
    tok = tc.get_tokenizer()
    print("Tinker connected.", flush=True)

    # Pick a medium-difficulty problem that the model is unlikely to solve on turn 1
    tasks = load_deepcoder_tasks("train", seed=42, sources=("lcbv5",))
    tasks = [t for t in tasks if len(t.tests) >= 5]
    # Pick one with many tests (harder)
    tasks.sort(key=lambda t: len(t.tests), reverse=True)
    task = tasks[5]  # skip the very hardest, pick something mid-range
    print(f"Problem: {len(task.tests)} tests", flush=True)
    print(f"---\n{task.problem[:500]}\n---", flush=True)

    loop = asyncio.new_event_loop()
    msgs = [{"role": "user", "content": task.problem}]
    trace = {"problem": task.problem[:500], "n_tests": len(task.tests), "turns": []}

    prev_passing_set = set()
    feedback_test_indices = set()
    n_feedback_tests = 0
    hw_passed = 0

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n{'='*60}\nTURN {turn}\n{'='*60}", flush=True)

        pt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        ptok = tok.encode(pt, add_special_tokens=False)
        avail = 32768 - len(ptok) - 64
        print(f"Prompt tokens: {len(ptok)}, available: {avail}", flush=True)

        if avail < 256:
            print("Context full, stopping.", flush=True)
            break

        sp = tinker.SamplingParams(max_tokens=min(MAX_TOKENS, avail), temperature=TEMPERATURE)
        result = sc.sample(prompt=tinker.ModelInput.from_ints(ptok), num_samples=1, sampling_params=sp).result()
        # Decode WITH thinking tokens preserved
        ct_full = tok.decode(result.sequences[0].tokens, skip_special_tokens=False)
        ct = tok.decode(result.sequences[0].tokens, skip_special_tokens=True)
        comp_tokens = len(result.sequences[0].tokens)
        print(f"Completion: {comp_tokens} tokens", flush=True)

        code = extract_code_from_model(ct)
        turn_data = {"turn": turn, "comp_tokens": comp_tokens, "code_found": code is not None,
                     "full_completion": ct_full}

        if code is None:
            print("NO CODE BLOCK FOUND (thinking truncated?)", flush=True)
            print(f"Last 200 chars: ...{ct_full[-200:]}", flush=True)
            turn_data["completion_tail"] = ct_full[-500:]
            turn_data["feedback"] = "No code block found. Use ```python```."
            msgs.append({"role": "assistant", "content": ct})
            msgs.append({"role": "user", "content": "No code block found. Use ```python```."})
            trace["turns"].append(turn_data)
            continue

        turn_data["code"] = code

        # Run sandbox
        ap, details = loop.run_until_complete(check_correctness(task.tests, code, timeout=SANDBOX_TIMEOUT))
        tp = details.get("tests_passed", 0)
        tt = details.get("tests_total", len(task.tests))
        per_test = details.get("per_test", [])

        curr_passing_set = set(i for i, p in enumerate(per_test) if p)
        newly_passing = len(curr_passing_set - prev_passing_set)
        targeted_fixes = len(curr_passing_set & feedback_test_indices)

        old_hw = hw_passed
        hw_passed = max(hw_passed, tp)
        is_terminal = ap or turn >= MAX_TURNS

        info = {"turn": turn, "max_turns": MAX_TURNS, "tests_total": tt, "tests_passed": tp,
                "hw_passed": hw_passed, "old_hw": old_hw, "all_passed": ap, "no_code": False,
                "newly_passing": newly_passing, "targeted_fixes": targeted_fixes,
                "n_feedback_tests": n_feedback_tests}

        r_indep = reward_path_indep(info, is_terminal)
        r_dep = reward_path_dep(info, is_terminal)

        turn_data.update({
            "tests_passed": tp, "tests_total": tt, "all_passed": ap,
            "newly_passing": newly_passing, "targeted_fixes": targeted_fixes,
            "n_feedback_tests": n_feedback_tests,
            "reward_indep": r_indep, "reward_dep": r_dep,
        })

        print(f"Tests: {tp}/{tt} {'SOLVED' if ap else 'FAILED'}", flush=True)
        print(f"Newly passing: {newly_passing}, Targeted fixes: {targeted_fixes}/{n_feedback_tests}", flush=True)
        print(f"Reward — indep: {r_indep:.3f}, dep: {r_dep:.3f}", flush=True)

        if ap:
            turn_data["feedback"] = f"All {tt} tests passed!"
            print("SOLVED!", flush=True)
            trace["turns"].append(turn_data)
            break

        if turn < MAX_TURNS:
            feedback = _build_feedback(details, tp, tt)
            turn_data["feedback"] = feedback
            print(f"\nFEEDBACK:\n{feedback}\n", flush=True)
            msgs.append({"role": "assistant", "content": ct})
            msgs.append({"role": "user", "content": feedback})

            # Update state for next turn
            prev_passing_set = curr_passing_set
            failing_indices = set(i for i, p in enumerate(per_test) if not p)
            feedback_test_indices = set(list(failing_indices)[:10])
            n_feedback_tests = len(feedback_test_indices)
        else:
            turn_data["feedback"] = "(terminal turn, no feedback)"
            print("Final turn, no more attempts.", flush=True)

        trace["turns"].append(turn_data)

    loop.close()

    out_path = "code_repair/multi_turn_trace_4t.json"
    with open(out_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"\nTrace written to {out_path}", flush=True)

if __name__ == "__main__":
    main()
