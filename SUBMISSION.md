# HW6 submission

Course CSS-4007, Narxoz University. Student: Artyom Ostanin (GitHub: Dssolved).

## AI-use declaration

I used Claude (Anthropic) as a coding and writing assistant.

- **Code and tests:** Claude wrote all the code in `gridworld/`, `run_part_b.py`, `run_part_c.py`, the tests in `tests/`, and the first drafts of the explanations in this report.
- **Part A:** Claude explained the calculations and showed me how to lay out the page. I wrote the calculations by hand on paper and took the photos myself.
- **Fixed policy (Part B):** Claude designed it from the map and gave me the reasons. I committed it before I ran any experiment and I did not change it afterwards.
- **Runs:** I ran every command on my own computer. All logs in `outputs/` and all numbers in the tables below come from my own runs. Claude also ran the code in its own environment to test it. The numbers were the same, but those runs are not used as results.
- The commits made with Claude's help carry a `Co-Authored-By: Claude` line.

## Setup and reproduction

My environment (from `outputs/part_b/environment.txt` and my terminal):

| Item | Version |
| --- | --- |
| OS | Windows 11 |
| Python | 3.13.5 |
| gymnasium | 1.4.0 |
| numpy | 2.5.5 |
| pytest | 9.2.0 |

All pinned versions are in `requirements.txt` (direct packages) and `requirements-lock.txt` (every package). No API key is needed.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q                      # 15 passed
mkdir outputs -Force
python run_part_b.py | Out-File -Encoding utf8 outputs/part_b_console.txt
python run_part_c.py | Out-File -Encoding utf8 outputs/part_c_console.txt
```

Repository layout:

| Path | Content |
| --- | --- |
| `part_a/` | photos of my paper work (`1` to `5`), `paper_values.json` |
| `part_b/` | `fixed_policy.json` and `fixed_policy.md` (written before the experiments) |
| `gridworld/` | `rules.py` (floor rules, read from `data/maps.json`), `mdp.py` (transition model), `solver.py` (value iteration), `simulate.py`, `checks.py` (rule checks), `render.py` |
| `run_part_b.py`, `run_part_c.py` | the two programs |
| `tests/` | 15 tests |
| `outputs/part_b/`, `outputs/part_c/` | all results |

The order of my commits: (1) Part A photos, (2) fixed policy, (3) code, (4) outputs, (5) this report. The photos were committed before any program file.

## Part A: Paper calculations

Photos of my handwritten work: `part_a/1.jpg`, `part_a/2.jpg`, `part_a/3.jpg`, `part_a/4.jpg`, `part_a/5.jpg` (they contain my own words about the problem, the V1 map, the V2 calculations, the arrows and the explanation).

**My understanding, in short.**
- States: the 9 tiles where the agent can stand and choose. G and H are terminal (value 0, no reward after them). `#` is a wall, not a state.
- Actions: up, right, down, left.
- Transitions: moves are certain, except on ice `I`: 0.8 the intended direction, 0.1 each side. A move into a wall or the border keeps the agent in place.
- Reward: the reward of the tile where the move ends (a blocked move gets the reward of the agent's own tile). `.`, `S`, `I` = -1, `M` = -3, `G` = +10, `H` = -10.
- Goal: maximize the expected discounted sum of rewards, discount 0.9.

**V1** (every nonterminal tile; G and H are 0):

```text
 -1    -1    10     0
 -1     #    -1     0
 -1    -1    -1    -1
```

At the ice tile (2,3) the four actions give up -1.9, right -8.2, down -1.9, left -1.0, so V1 = -1.0.

**V2** at the three requested tiles (all four actions, using V1):

| Tile | up | right | down | left | V2 | Best action |
| --- | --- | --- | --- | --- | --- | --- |
| (1,2) mud | -3.9 | **8.0** | -3.9 | -1.9 | 8.0 | right |
| (1,3) | 8.0 | **10.0** | -1.9 | -3.9 | 10.0 | right |
| (2,3) ice | **5.21** | -7.39 | -2.71 | -0.91 | 5.21 | up |

For the ice tile each action is a weighted sum. For example, `up = 0.8*8.0 + 0.1*(-1.9) + 0.1*(-10) = 5.21`.

**Ties:** there are no ties in these three tiles. The best value is clearly larger than the other three.

**Why two rounds are not enough:** two rounds look only two steps ahead and the values are still changing. When the neighbour values grow in later rounds, the best action in a tile can change. We can trust a policy only after the values stop changing. This is why Part C runs until the change is below 0.000001.

## Part B: FrozenLake

### Fixed policy, written before the experiments

File: `part_b/fixed_policy.json` and the reasons in `part_b/fixed_policy.md`. State numbers are `4 * row + col`. Actions: 0 left, 1 down, 2 right, 3 up.

| State | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 10 | 13 | 14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Action | down | right | down | left | down | down | right | right | down | right | right |

The main route is 0 -> 4 -> 8 -> 9 -> 10 -> 14 -> 15 (6 steps, no hole). Each action avoids stepping into a neighbouring hole. The policy looks only at the current state. It is designed for the map without slipping, so before the runs I predicted that it would be much worse with `is_slippery=True`.

### Results

Full table: `outputs/part_b/summary.md`. Episode rows (600 rows, one per episode): `outputs/part_b/episodes.csv`. Batch table: `outputs/part_b/summary.csv`.

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no | random | 11 | 0 | 50 | 0 | 0 | 7.98 |
| no | random | 22 | 0 | 50 | 0 | 0 | 6.42 |
| no | random | 33 | 1 | 49 | 0 | 2 | 6.66 |
| no | fixed | 11 | 50 | 0 | 0 | 100 | 6.00 |
| no | fixed | 22 | 50 | 0 | 0 | 100 | 6.00 |
| no | fixed | 33 | 50 | 0 | 0 | 100 | 6.00 |
| yes | random | 11 | 0 | 50 | 0 | 0 | 8.30 |
| yes | random | 22 | 2 | 48 | 0 | 4 | 7.76 |
| yes | random | 33 | 0 | 50 | 0 | 0 | 6.74 |
| yes | fixed | 11 | 3 | 47 | 0 | 6 | 4.52 |
| yes | fixed | 22 | 2 | 48 | 0 | 4 | 5.06 |
| yes | fixed | 33 | 0 | 50 | 0 | 0 | 4.06 |

Totals over the three batches (150 episodes per condition):

| Slippery? | Policy | Goals /150 | Holes /150 | Time limits /150 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- |
| no | random | 1 | 149 | 0 | 0.7 | 7.02 |
| no | fixed | 150 | 0 | 0 | 100.0 | 6.00 |
| yes | random | 2 | 148 | 0 | 1.3 | 7.60 |
| yes | fixed | 5 | 145 | 0 | 3.3 | 4.55 |

**How the runs were made.** Episode `i` of a batch is reset with `env.reset(seed=1000 * batch_seed + i)`. The random policy uses `numpy.random.default_rng(episode_seed)`, so its action generator is seeded too. The seed lists are the same in all four conditions (I checked this from the CSV file). The step limit is 100. Outcomes use the final state: state 15 is a goal, any other terminal state is a hole.

**Traces** (text rendering, `[x]` is the agent):
- Example episode (no slipping, fixed policy, seed 11000): `outputs/part_b/example_episode.txt`. It goes 0 -> 4 -> 8 -> 9 -> 10 -> 14 -> 15 in 6 steps.
- A successful slippery episode (fixed policy, seed 11012): `outputs/part_b/traces/slippery_True_fixed_goal.txt`. It needed 21 steps because the agent stayed on the start tile for several steps.
- A failed slippery episode (fixed policy, seed 11000): `outputs/part_b/traces/slippery_True_fixed_failure.txt`. The action "down" moved the agent right to state 1, then the action "right" moved it down into the hole at state 5. It ended after 2 steps.
- The other first-goal and first-failure traces are in `outputs/part_b/traces/`.
- **Not observed:** with `is_slippery=False` the fixed policy never failed, so there is no failed trace for it (the file says so). No episode in any condition ended by the time limit.

### Explanation

**What slipping changed.** The fixed policy went from 150 goals out of 150 to 5 out of 150. With slipping, the chosen action is only one of three possible moves, so the agent often goes sideways, and this route runs next to holes. The path also got shorter on average (6.00 to 4.55 steps) because many episodes now end early in a hole. The random policy was bad in both cases (1 and 2 goals out of 150), because a random walk already ends in a hole most of the time, so slipping adds little.

**Consistency of the batches.** Without slipping, the fixed policy gave the same result in every batch (50 of 50), because the environment is deterministic and the policy is fixed. With slipping the batches differ: the fixed policy had 3, 2 and 0 goals, the random policy 0, 2 and 0. These numbers are small, so a difference of one or two episodes changes the percentage a lot. Random-policy results without slipping were 0, 0 and 1 goals.

**Why one success is not enough.** In a slippery environment luck decides a lot. For example, one successful slippery episode (seed 11012) took 21 steps, and the policy in the batch with seed 33 never reached the goal. A single success does not show that a policy is good, and a single failure does not show that it is bad. We need many episodes with different seeds, and even 150 episodes give only a rough estimate when the success rate is this low.

**Why this is not training.** The policy never changed during the runs. The program only played the policy and counted the outcomes. Nothing was learned from rewards, no values were updated, and the results did not influence the next episode. This is evaluation (simulation) of a fixed policy. Training would change the policy, or learn values, from the experience.

## Part C: Custom solver

Program: `run_part_c.py`. Full printed report: `outputs/part_c/report.txt` (also `outputs/part_c_console.txt`). Values and policies as data: `outputs/part_c/values_and_policies.json`.

The floor rules (reward, slippery or not, terminal, wall) are stored in `data/maps.json` and read by `gridworld/rules.py`. The solver (`gridworld/solver.py`) only calls `transitions` and `q_value` of the model, so a new floor needs one new entry in `maps.json` and no change in the solver. A test (`test_new_floor_needs_only_a_rule_not_a_solver_change`) checks this.

### Compare code with paper

The program printed (computed from V0 = 0):

```text
V1:                          V2:
 -1.000 -1.000 10.000 0       -1.900  8.000 10.000 0
 -1.000   #    -1.000 0       -1.900   #     5.210 0
 -1.000 -1.000 -1.000 -1      -1.900 -1.900 -1.900 -1.900
```

V2 at the three requested tiles (the four actions are in `outputs/part_c/report.txt`):

| Tile | Program V2 | Paper V2 |
| --- | --- | --- |
| (1,2) | 8.0000 (right) | 8.0 |
| (1,3) | 10.0000 (right) | 10.0 |
| (2,3) | 5.2100 (up) | 5.21 |

`part_a/paper_values.json` holds the nine V1 values and three V2 values from my paper work. The program compared them automatically and reported: **all paper values match the program**. There was no discrepancy to correct.

### Values, policies and convergence

| Map | Iterations | Final largest change | Converged |
| --- | --- | --- | --- |
| Part A map | 12 | 2.165e-07 | yes |
| Extended map | 11 | 2.406e-07 | yes |

The stopping rule is a largest absolute change below 0.000001 in a full update, with a safeguard of 10,000 iterations. The safeguard was not reached. Each update uses the values of the previous round only.

**Tie-breaking rule.** Two actions are tied when their Q-values are within 1e-6 of the best one. Among tied actions the first in the order up, right, down, left is chosen. **No ties happened** on either map. The smallest gap between the best and the second-best action was 0.0338 (Part A map, tile (3,1)) and 0.966 (extended map, tile (3,1)).

Final values and greedy policy, Part A map (`.M.G / .#IH / S...`):

```text
Values V*                            Policy
  4.2000  8.0000 10.0000  0.0000      >  >  >  G
  2.7800   #      5.8242  0.0000      ^  #  ^  H
  1.5358  2.8176  4.2418  2.8176      >  >  ^  <
```

Extended map (`.M.G / .#IH / SD.W`):

```text
Values V*                            Policy
  4.2000  8.0000 10.0000  0.0000      >  >  >  G
  2.7800   #      5.8242  0.0000      ^  #  ^  H
  1.5020  2.8176  4.2418  2.8176      ^  >  ^  <
```

### Simulation results

The computed policy was played for 3 batches of 50 episodes with the Part B seed rule (`1000 * batch_seed + i`), a 100-step limit, and the discounted return `r0 + 0.9*r1 + 0.9^2*r2 + ...`. Rows: `outputs/part_c/episodes.csv` (300 rows). Batch table: `outputs/part_c/summary.csv`.

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| Part A | 11 | 47 | 3 | 0 | 5.04 | 2.2297 |
| Part A | 22 | 44 | 6 | 0 | 4.96 | 1.4424 |
| Part A | 33 | 42 | 8 | 0 | 4.88 | 0.9700 |
| Extended | 11 | 50 | 0 | 0 | 5.00 | 1.5020 |
| Extended | 22 | 50 | 0 | 0 | 5.00 | 1.5020 |
| Extended | 33 | 50 | 0 | 0 | 5.00 | 1.5020 |

Over all 150 episodes: Part A map has 133 goals and 17 holes, mean discounted return 1.5474, while V*(start) = 1.5358. The extended map has 150 goals, mean discounted return 1.5020, and V*(start) = 1.5020.

### Explain your results

**Which values and actions changed.** Only the start tile changed its action: on the Part A map the policy goes right from (3,1), on the extended map it goes up. With sand, going right costs -2 for entering `D`: `-2 + 0.9 * 2.8176 = 0.5358`. Going up costs -1: `-1 + 0.9 * 2.78 = 1.502`. So the route through the sand is worse, and the policy changes to the route up the left side and through the mud (cost -3 once). V*(3,1) fell from 1.5358 to 1.5020. All other values are the same on both maps, because the best actions in those tiles never enter sand or water. The computed policy never enters the water tile `W`, so its cost -5 did not change anything.

**Where cost and risk mattered.** On the Part A map the best route goes along the bottom and then up through the ice tile (2,3) to the goal. The other route goes up the left side and through the mud (-3). The expected values are close: 1.5358 against 1.5020, a gap of only 0.034. The ice route wins in expectation, but it is risky. At the ice tile the policy chooses "up", and there is a 0.1 chance of sliding right into the hole. If the agent slides left it stays on the tile (a wall) and tries again, so the chance to end in the hole is `0.1 / (0.1 + 0.8) = 0.111`, which I calculated myself. The simulation gave 17 holes in 150 episodes (0.113), which agrees. A goal episode on that route has a discounted return of about +3.12 and a hole episode about -10.0, so the batch means jump a lot (2.23, 1.44, 0.97 with 3, 6 and 8 holes). On the extended map the policy takes the deterministic route up through the mud, so it avoids the ice completely. All 150 episodes reached the goal and every episode has the same return 1.5020.

**How ice was handled.** The solver models ice with the three outcomes and uses the tile where the move starts. Moving onto ice from a normal tile is certain. Only leaving the ice tile is random. Blocked slip directions (the wall at (2,2)) keep the agent in place with the reward of the ice tile.

**Finite simulations and infinite-horizon values.** V* is the expected discounted sum over an infinite number of steps. A simulation has a limit of 100 steps and only 150 episodes. The effect of the step limit is tiny (`0.9^100` is about 0.00003), and no episode reached it. The main difference is random sampling: with only 50 episodes per batch, the batch means (2.23, 1.44, 0.97) are far from V*(start) = 1.5358, and the mean of all 150 episodes (1.5474) is close but not equal. On the extended map there is no randomness in the policy path, so the simulation equals V*(start) exactly. A simulation estimates the value with noise; it does not prove it. Also, V* is the value of the best policy only if value iteration has converged, and I stop at a change below 1e-6, so the values have a very small error.

### Rule checks

Before using the solver, `run_part_c.py` runs 40 checks (18 on the Part A map and 22 on the extended map, which has 4 extra checks for sand and water). All passed (0 failed). Details: `outputs/part_c/rule_checks.json` and the top of `outputs/part_c/report.txt`. Some examples with the printed numbers:

| Rule | Check | Result |
| --- | --- | --- |
| Border | (1,1) up stays at (1,1), reward -1 | pass |
| Wall | (1,2) down into `#` stays at (1,2), reward -3 (mud tile) | pass |
| Wall | (2,1) right into `#` stays at (2,1), reward -1 | pass |
| Wall is not a state | (2,2) is not in the state list | pass |
| Ice probabilities | all four actions on the ice tile and all other (state, action) pairs sum to 1 | pass |
| Ice with a blocked direction | (2,3) up: 0.8 to (1,3), 0.1 stay (wall on the left), 0.1 to H (reward -10) | pass |
| Slip depends on the start tile | (1,3) down onto the ice is certain (probability 1) | pass |
| Mud | entering `M` gives -3 | pass |
| Sand (extended map) | entering `D` gives -2 | pass |
| Water (extended map) | entering `W` gives -5 | pass |
| Goal | entering `G` gives +10 and nothing else | pass |
| Terminal tiles | G and H have no transitions, so no reward after termination | pass |
| Episode end | a walk to the goal gives exactly 5 rewards [-1, -1, -3, -1, +10] and stops at G | pass |
| Discounted return | the same walk gives `-1 - 0.9 - 3*0.81 - 0.729 + 10*0.6561` | pass |

In addition, the test suite (`python -m pytest -q`, 15 passed) checks the solver (fixed point of the update, safeguard, tie-breaking) and the Part B runner (a terminal step on the last allowed step is recorded as the terminal outcome, holes are recorded as holes, the same seed gives the same episode).

## Limits of this work

- The fixed policy in Part B ignores slipping, so it is weak in the slippery cases. I did not change it after I saw the results.
- The Part B success rates under slipping are based on 150 episodes and a handful of successes, so they are rough estimates.
- The simulation results of Part C are random samples; they are close to the computed values but not equal to them.
