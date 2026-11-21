"""Simulates a fixed policy on a GridMDP. This is evaluation, not training:
the policy never changes while episodes are played."""
from __future__ import annotations

import numpy as np

from .mdp import GridMDP


def run_episode(mdp: GridMDP, policy: dict, seed: int, max_steps: int, discount: float) -> dict:
    rng = np.random.default_rng(seed)
    s = mdp.start
    path, rewards = [s], []
    outcome = "time limit"
    for _ in range(max_steps):
        a = policy[s]
        outs = mdp.transitions(s, a)
        probs = [p for p, _, _ in outs]
        i = int(rng.choice(len(outs), p=probs))
        _, t, r = outs[i]
        rewards.append(r)
        path.append(t)
        s = t
        if mdp.is_terminal(s):
            outcome = "goal" if mdp.letter(s) == "G" else "hole"
            break
    ret = sum((discount ** k) * r for k, r in enumerate(rewards))
    return {"steps": len(rewards), "outcome": outcome, "discounted_return": ret,
            "undiscounted_return": sum(rewards), "path": path, "rewards": rewards}
