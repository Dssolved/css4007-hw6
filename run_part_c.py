"""Part C: value iteration on the Part A map and on the extended map, rule checks,
simulation of the computed policies. Usage:  python run_part_c.py"""
from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

from gridworld.checks import run_checks
from gridworld.mdp import GridMDP
from gridworld.render import policy_map, tile_map, value_map
from gridworld.rules import load_config
from gridworld.simulate import run_episode
from gridworld.solver import (TIE_TOLERANCE, extract_policy, smallest_gap,
                              value_iteration, action_values)

OUT = Path("outputs/part_c")
PAPER_VALUES = Path("part_a/paper_values.json")     # typed from my own Part A scan (optional)
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def key(s) -> str:
    return f"{s[0]},{s[1]}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    say(f"python {platform.python_version()} | numpy {np.__version__}")
    say(f"discount={cfg.discount} tolerance={cfg.tolerance} max_iterations={cfg.max_iterations} "
        f"tie_tolerance={TIE_TOLERANCE}")
    say("Tie-breaking: actions whose Q-value is within the tie tolerance of the best count as tied;"
        " the first one in the order up, right, down, left is chosen.")

    mdps = {n: GridMDP(rows, cfg.tiles, cfg.discount) for n, rows in cfg.maps.items()}

    # ---------------------------------------------------------------- rule checks
    say("\n=== RULE CHECKS (before trusting the solver) ===")
    all_checks = []
    for name, mdp in mdps.items():
        for c in run_checks(mdp, name, cfg.max_steps, cfg.discount):
            all_checks.append(c)
    shown = [c for c in all_checks if "probabilities sum to 1" not in c["check"] or c["check"].startswith("every")]
    for c in shown:
        say(f"[{'PASS' if c['passed'] else 'FAIL'}] ({c['map']}) {c['check']}\n        {c['detail']}")
    n_fail = sum(not c["passed"] for c in all_checks)
    say(f"rule checks: {len(all_checks)} run, {n_fail} failed "
        f"(individual ice-sum checks are in rule_checks.json)")
    (OUT / "rule_checks.json").write_text(json.dumps(all_checks, indent=2, ensure_ascii=False), encoding="utf-8")
    if n_fail:
        say("STOP: a rule check failed, results below are not trustworthy.")

    # ---------------------------------------------------------------- solve
    results, policies, sims = {}, {}, {}
    for name, mdp in mdps.items():
        say(f"\n=== MAP: {name} ===")
        say(tile_map(mdp))
        vi = value_iteration(mdp, cfg.tolerance, cfg.max_iterations)
        policy, ties = extract_policy(mdp, vi.V)
        results[name], policies[name] = vi, policy

        say(f"\nValue iteration: iterations = {vi.iterations}, final max change = {vi.final_change:.3e}, "
            f"converged = {vi.converged}" + ("" if vi.converged else "  <-- NOT CONVERGED (safeguard reached)"))
        say("\nV1:")
        say(value_map(mdp, vi.history[1]))
        say("\nV2:")
        say(value_map(mdp, vi.history[2]))
        say("\nV2 at the three requested positions (all four actions, computed from V1):")
        for s in cfg.paper_v2_positions:
            q = action_values(mdp, vi.history[1], s)
            body = ", ".join(f"{a}={v:.4f}" for a, v in q.items())
            say(f"  V2{s} = {max(q.values()):.4f}   [{body}]")
        say("\nFinal values V*:")
        say(value_map(mdp, vi.V, 4))
        say("\nGreedy policy from the final values (G/H/# are not states):")
        say(policy_map(mdp, policy))
        if ties:
            say("Tied actions (within tie tolerance): " + "; ".join(f"{s}: {t}" for s, t in ties.items()))
        else:
            say("No ties: in every state one action is strictly best by more than the tie tolerance.")
        s_gap, gap = smallest_gap(mdp, vi.V)
        say(f"Smallest gap between best and second-best action over all non-tied states: {gap:.6f} at {s_gap}")

    # ---------------------------------------------------------------- compare with paper
    say("\n=== COMPARISON WITH MY PAPER WORK (Part A, paper map) ===")
    vi = results["paper"]
    if PAPER_VALUES.exists():
        paper = json.loads(PAPER_VALUES.read_text(encoding="utf-8"))
        mism = []
        for k, v in paper["v1"].items():
            r, c = map(int, k.split(","))
            if abs(vi.history[1][(r, c)] - v) > 1e-9:
                mism.append(("V1", k, v, vi.history[1][(r, c)]))
        for k, v in paper["v2"].items():
            r, c = map(int, k.split(","))
            if abs(vi.history[2][(r, c)] - v) > 1e-9:
                mism.append(("V2", k, v, vi.history[2][(r, c)]))
        say(f"compared {len(paper['v1'])} V1 values and {len(paper['v2'])} V2 values with {PAPER_VALUES}")
        say("all paper values match the program" if not mism else
            "MISMATCHES (name, tile, paper, program): " + str(mism))
    else:
        say(f"{PAPER_VALUES} not found: compare the printed V1 / V2 above with the scan by hand.")

    # ---------------------------------------------------------------- simulation
    say("\n=== SIMULATION OF THE COMPUTED POLICY (3 batches x 50, seeds 1000*batch_seed+i) ===")
    rows_csv, summary = [], []
    for name, mdp in mdps.items():
        for bs in cfg.batch_seeds:
            eps = []
            for i in range(cfg.episodes_per_batch):
                seed = 1000 * bs + i
                ep = run_episode(mdp, policies[name], seed, cfg.max_steps, cfg.discount)
                eps.append(ep)
                rows_csv.append({
                    "map": name, "batch_seed": bs, "episode_index": i, "episode_seed": seed,
                    "steps": ep["steps"], "outcome": ep["outcome"],
                    "total_reward": ep["undiscounted_return"],
                    "discounted_return": round(ep["discounted_return"], 10),
                    "path": " ".join(f"{r}{c}" for r, c in ep["path"]),
                })
            n = len(eps)
            summary.append({
                "map": name, "batch_seed": bs,
                "goals": sum(e["outcome"] == "goal" for e in eps),
                "holes": sum(e["outcome"] == "hole" for e in eps),
                "time_limits": sum(e["outcome"] == "time limit" for e in eps),
                "mean_steps": float(np.mean([e["steps"] for e in eps])),
                "mean_discounted_return": float(np.mean([e["discounted_return"] for e in eps])),
                "episodes": n,
            })
    header = "| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |"
    say(header)
    say("| --- | --- | --- | --- | --- | --- | --- |")
    for r in summary:
        say(f"| {r['map']} | {r['batch_seed']} | {r['goals']} | {r['holes']} | {r['time_limits']} | "
            f"{r['mean_steps']:.2f} | {r['mean_discounted_return']:.4f} |")
    for name, mdp in mdps.items():
        part = [r for r in summary if r["map"] == name]
        mean_ret = float(np.mean([r["mean_discounted_return"] for r in part]))
        say(f"\n{name}: mean discounted return over all 150 episodes = {mean_ret:.4f}; "
            f"value-iteration V*(start {mdp.start}) = {results[name].V[mdp.start]:.4f}")

    # ---------------------------------------------------------------- compare maps
    say("\n=== COMPARISON OF THE TWO MAPS (tiles that exist in both) ===")
    p, e = mdps["paper"], mdps["extended"]
    say("tile | V* paper | V* extended | policy paper | policy extended")
    for s in p.states:
        if s in e.states and not p.is_terminal(s):
            vp, ve = results["paper"].V[s], results["extended"].V[s]
            pp, pe = policies["paper"][s], policies["extended"][s]
            flag = "  <- action changed" if pp != pe else ""
            say(f"{s} {p.letter(s)}/{e.letter(s)} | {vp:.4f} | {ve:.4f} | {pp} | {pe}{flag}")

    # ---------------------------------------------------------------- save
    with open(OUT / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    out_json = {}
    for name, mdp in mdps.items():
        vi = results[name]
        out_json[name] = {
            "map": mdp.rows, "iterations": vi.iterations, "final_change": vi.final_change,
            "converged": vi.converged,
            "V1": {key(s): v for s, v in vi.history[1].items()},
            "V2": {key(s): v for s, v in vi.history[2].items()},
            "V_final": {key(s): v for s, v in vi.V.items()},
            "policy": {key(s): a for s, a in policies[name].items()},
        }
    (OUT / "values_and_policies.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    say(f"\nsaved to {OUT}/ : report.txt, summary.csv, episodes.csv, values_and_policies.json, rule_checks.json")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 1 if n_fail or not all(r.converged for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
