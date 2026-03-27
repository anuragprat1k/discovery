# LiveCodeBench Reward Functions

Multi-turn code repair on competitive programming problems. Model generates code, sees test results, iterates up to 4 turns.

## Current Rewards

### Sparse (terminal-only)
```
Per-turn: 0 (no shaping)
Terminal: +1 if ALL tests pass, 0 otherwise
Format:  -0.1 if no code block
```
No credit for partial progress. Passing 99/100 tests = 0 reward = same as passing 0.

### Dense (potential-based, HWM)
```
Per-turn: +0.5 × (HWM_delta / num_tests)
Terminal: +1 if all tests pass
Format:  -0.1 if no code block
```
High-water mark tracks the max tests ever passed across turns. Only rewards *new* progress — if turn 2 regresses from 80/100 to 50/100, no penalty (HWM stays 80). Strictly potential-based: Φ(s) = HWM/total, reward = Φ(s') - Φ(s) ≥ 0.

### Dense Full (potential-based + speed bonus)
```
Per-turn: same as dense
Terminal: +1 + 0.2 × (turns_remaining / max_turns) if solved
Format:  -0.1 if no code block
```
Same as dense but adds a speed bonus for solving early. Still potential-based — the speed bonus can't go negative.

## What's Missing: Non-Potential Reward

**Problem**: On LCB problems, test results are binary at the problem level — either the algorithm is correct (all pass) or wrong (all fail). There's rarely partial test passage because competitive programming tests exercise the same logic with different inputs.

**However**, in multi-turn there ARE non-potential signals we could add:

### Dense Full (Green + Yellow)
```
Per-turn: +0.4 × (HWM_delta / N)       [GREEN: potential, locked progress]
        + 0.2 × (curr_passed / N)       [YELLOW: non-potential, can regress]
Terminal: +1 if all tests pass
Format:  -0.1 if no code block
```

**Green signal** (potential): HWM delta. Once you've proven you can pass a test, that's locked. Even if you regress on the next turn, the HWM stays.

**Yellow signal** (non-potential): Current pass fraction. If the model rewrites working code and breaks it, this drops — penalizing careless rewrites. If the model fixes one bug but introduces another, the yellow signal captures the regression.

**Wordle parallel:**
| Wordle | Code Repair |
|--------|-------------|
| Green tiles (position locked, can't lose) | HWM of tests passing (+0.4/N per new HWM test) |
| Yellow tiles (info about word, can lose by not using letter) | Current tests passing (+0.2/N, drops if model regresses) |

## Observation from Baseline Eval

On LCB train split problems, test results are mostly all-or-nothing (0/N or N/N). The dense reward's advantage comes from **temporal signal**: crediting turn-2 fixes that sparse ignores (both get 0 per-turn with sparse, but dense credits the eventual solve via HWM jump). The non-potential signal would add value when the model writes code that partially works but then overwrites it with worse code on the next turn.
