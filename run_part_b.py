"""Part B: random vs fixed policy on Gymnasium FrozenLake-v1 (4x4, default rewards),
is_slippery False / True, 3 batches x 50 episodes.  Usage:  python run_part_b.py"""
from __future__ import annotations

import csv
import json
import platform
import re
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

OUT = Path("outputs/part_b")
POLICY_FILE = Path("part_b/fixed_policy.json")
BATCH_SEEDS = (11, 22, 33)
EPISODES = 50
MAX_STEPS = 100
ACTION_NAME = {0: "left", 1: "down", 2: "right", 3: "up"}
ANSI_HIGHLIGHT = re.compile(r"\x1b\[41m(.)\x1b\[0m")


def clean(frame: str) -> str:
    """Text rendering without terminal colour codes; the agent's tile is shown as [x]."""
    return ANSI_HIGHLIGHT.sub(r"[\1]", frame)


def load_fixed_policy(env) -> dict:
    data = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    policy = {int(k): int(v) for k, v in data["actions"].items()}
    desc = env.unwrapped.desc.astype(str)
    nonterminal = {i for i, ch in enumerate(desc.flat) if ch in "SF"}
    assert set(policy) == nonterminal, "fixed policy must give one action to every nonterminal state"
    assert all(a in ACTION_NAME for a in policy.values())
    return policy


def run_episode(env, choose_action, seed: int, max_steps: int = MAX_STEPS) -> dict:
    """reset(seed) then step() until terminated / truncated / max_steps."""
    obs, _ = env.reset(seed=seed)
    desc = env.unwrapped.desc.astype(str).flat
    frames = [clean(env.render())]
    steps_log = []
    total, steps, outcome = 0.0, 0, "time limit"
    for _ in range(max_steps):
        action = choose_action(obs)
        new_obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        total += reward
        steps_log.append({"step": steps, "state": int(obs), "action": ACTION_NAME[int(action)],
                          "next_state": int(new_obs), "reward": float(reward)})
        frames.append(clean(env.render()))
        obs = new_obs
        if terminated:                       # checked first: a terminal step on the last allowed step still counts
            outcome = "goal" if desc[obs] == "G" else "hole"
            break
        if truncated:
            outcome = "time limit"
            break
    return {"steps": steps, "total_reward": total, "outcome": outcome,
            "final_state": int(obs), "log": steps_log, "frames": frames}


def format_trace(title: str, ep: dict, seed: int) -> str:
    lines = [title, f"episode seed: {seed}", f"outcome: {ep['outcome']}, steps: {ep['steps']}, "
             f"total reward: {ep['total_reward']}", "", "initial state:", ep["frames"][0]]
    for entry, frame in zip(ep["log"], ep["frames"][1:]):
        lines.append(f"step {entry['step']}: state {entry['state']} -> action {entry['action']} "
                     f"-> state {entry['next_state']}, reward {entry['reward']}")
        lines.append(frame)
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(exist_ok=True)
    env_info = (f"python {platform.python_version()}\ngymnasium {gym.__version__}\nnumpy {np.__version__}\n"
                f"platform {platform.platform()}\n")
    (OUT / "environment.txt").write_text(env_info, encoding="utf-8")
    print(env_info)

    rows, summary, notes = [], [], []
    example_saved = False
    for slippery in (False, True):
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery, render_mode="ansi")
        assert env.spec.max_episode_steps == MAX_STEPS
        fixed = load_fixed_policy(env)
        first_goal, first_fail = {}, {}
        for policy_name in ("random", "fixed"):
            cond = f"slippery={slippery}, policy={policy_name}"
            for bs in BATCH_SEEDS:
                eps = []
                for i in range(EPISODES):
                    seed = 1000 * bs + i
                    if policy_name == "random":
                        rng = np.random.default_rng(seed)               # action generator seeded per episode
                        choose = lambda obs, rng=rng: int(rng.integers(4))
                    else:
                        choose = lambda obs: fixed[int(obs)]
                    ep = run_episode(env, choose, seed)
                    eps.append(ep)
                    rows.append({"condition": cond, "slippery": slippery, "policy": policy_name,
                                 "batch_seed": bs, "episode_index": i, "episode_seed": seed,
                                 "steps": ep["steps"], "total_reward": ep["total_reward"],
                                 "outcome": ep["outcome"], "final_state": ep["final_state"]})
                    key = (policy_name,)
                    if ep["outcome"] == "goal" and key not in first_goal:
                        first_goal[key] = (ep, seed)
                    if ep["outcome"] != "goal" and key not in first_fail:
                        first_fail[key] = (ep, seed)
                    if not example_saved and not slippery and policy_name == "fixed":
                        (OUT / "example_episode.txt").write_text(
                            format_trace("Example episode: is_slippery=False, fixed policy, batch seed 11, i=0",
                                         ep, seed), encoding="utf-8")
                        example_saved = True
                summary.append({
                    "slippery": slippery, "policy": policy_name, "batch_seed": bs,
                    "goals": sum(e["outcome"] == "goal" for e in eps),
                    "holes": sum(e["outcome"] == "hole" for e in eps),
                    "time_limits": sum(e["outcome"] == "time limit" for e in eps),
                    "success_pct": 100.0 * sum(e["outcome"] == "goal" for e in eps) / EPISODES,
                    "mean_steps": float(np.mean([e["steps"] for e in eps])),
                })
        for policy_name in ("random", "fixed"):
            tag = f"slippery_{slippery}_{policy_name}"
            for kind, store in (("goal", first_goal), ("failure", first_fail)):
                entry = store.get((policy_name,))
                path = OUT / "traces" / f"{tag}_{kind}.txt"
                if entry:
                    ep, seed = entry
                    path.write_text(format_trace(f"First {kind} trace: slippery={slippery}, policy={policy_name}",
                                                 ep, seed), encoding="utf-8")
                else:
                    msg = f"No {kind} episode occurred for slippery={slippery}, policy={policy_name}."
                    path.write_text(msg + "\n", encoding="utf-8")
                    notes.append(msg)
        env.close()

    with open(OUT / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    table = ["| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in summary:
        table.append(f"| {'yes' if r['slippery'] else 'no'} | {r['policy']} | {r['batch_seed']} | {r['goals']} | "
                     f"{r['holes']} | {r['time_limits']} | {r['success_pct']:.0f} | {r['mean_steps']:.2f} |")
    table += ["", "Totals over the 3 batches (150 episodes per condition):", "",
              "| Slippery? | Policy | Goals /150 | Holes /150 | Time limits /150 | Success % | Mean steps |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for slippery in (False, True):
        for policy_name in ("random", "fixed"):
            sel = [r for r in rows if r["slippery"] == slippery and r["policy"] == policy_name]
            g = sum(r["outcome"] == "goal" for r in sel)
            h = sum(r["outcome"] == "hole" for r in sel)
            t = sum(r["outcome"] == "time limit" for r in sel)
            table.append(f"| {'yes' if slippery else 'no'} | {policy_name} | {g} | {h} | {t} | "
                         f"{100 * g / len(sel):.1f} | {np.mean([r['steps'] for r in sel]):.2f} |")
    text = "\n".join(table) + "\n" + ("\n" + "\n".join(notes) + "\n" if notes else "")
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"saved to {OUT}/ : episodes.csv, summary.csv, summary.md, example_episode.txt, traces/, environment.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
