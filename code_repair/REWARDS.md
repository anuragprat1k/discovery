# Code Repair Reward Functions

Three reward conditions for the iterative code repair task, mirroring the Wordle experiment structure. Each condition shares the same terminal rewards and format penalty, differing only in per-turn shaping.

## Shared Components

| Component | Value | When |
|-----------|-------|------|
| Terminal win | **+3.0** | All tests pass (episode solved) |
| Speed bonus | **+0.1 × turns_remaining** | Added to terminal win |
| Terminal loss | **-1.0** | Max turns exhausted without solving |
| Format penalty | **-1.0** | Turn with no valid `<repair>` tag |

## 1. Sparse (terminal-only)

**Per-turn reward:** 0.0 (no shaping signal)

**Only receives reward at episode end:** +3.0 for win, -1.0 for loss.

The model gets no credit for partial progress. Fixing 9 out of 10 tests is the same as fixing 0 — both get -1.0 terminal loss.

**Wordle analog:** Outcome-only reward (+3 for guess, -1 for loss).

## 2. Dense Passes (potential-based)

**Per-turn reward:** `+0.4 × (HWM_delta / num_tests)`

where `HWM_delta = max(0, current_hw_passing - previous_hw_passing)` is the increase in the high-water mark of passing tests.

**Properties:**
- **Potential-based:** Reward = Φ(s') - Φ(s) where Φ(s) = HWM(s). Can never be negative.
- **Monotonic:** Once a test enters the HWM, it stays. Even if a repair breaks previously-passing tests, the HWM doesn't decrease — so no reward is lost.
- **Normalized:** Divided by `num_tests` so reward scale is comparable across problems with 3 tests vs 20 tests.

**Example (10 tests):**
```
Turn 1: 0 → 4 passing (HWM 0→4) → reward = 0.4 × 4/10 = +0.16
Turn 2: 4 → 2 passing (HWM stays 4) → reward = 0.0 (regression, but HWM didn't drop)
Turn 3: 2 → 7 passing (HWM 4→7) → reward = 0.4 × 3/10 = +0.12
Turn 4: 7 → 10 (solved) → reward = 0.4 × 3/10 + 3.0 = +3.12
```

**Wordle analog:** Green tiles (new correct-position letters). Potential-based — once a letter is green, it stays green.

## 3. Dense Full (non-potential)

**Per-turn reward:** Everything from Dense Passes, PLUS for each **failing** test:
- **+0.05 / num_tests** if the test ran without crashing (function returned something)
- **+0.05 / num_tests** if the return type matches the expected type
- **+0.05 / num_tests** if the return shape/length matches expected

**Properties:**
- **Non-potential:** These partial-correctness signals CAN decrease turn-over-turn. A repair that goes from "returns wrong value" (no_crash=True, type correct) to "crashes entirely" (no_crash=False) loses these bonuses.
- **Complementary:** The HWM component is still potential-based; only the partial-correctness bonuses are non-potential.

**Example (10 tests, 4 passing, 3 failing-but-no-crash with correct types):**
```
HWM reward:    0.4 × 4/10 = +0.16
No-crash:      0.05 × 3/10 = +0.015
Type match:    0.05 × 3/10 = +0.015
Shape match:   0 (no shape info)
Terminal:      -1.0 (not solved)
Total:         -0.81
vs sparse:     -1.0
vs dense_passes: -0.84
```

**Wordle analog:** Yellow tiles (letter in word but wrong position). Non-potential — a guess can go from having yellows to having none if the model stops using those letters.

## Key Hypothesis

Dense rewards provide gradient signal on partially-correct solutions that sparse ignores. This should enable:
- **Discovery:** The model learns to solve new problems it couldn't before (credit for partial progress guides exploration)
- **Without sharpening:** pass@k should increase alongside pass@1 (the model discovers, not just gets more consistent on already-solvable problems)

The potential-based (dense_passes) vs non-potential (dense_full) comparison tests whether the additional partial-correctness signal helps or hurts — can the model learn to preserve working components across repairs?
