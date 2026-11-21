"""Rule checks run BEFORE the solver is trusted. Each check is a plain assertion
about the transition model, with the numbers printed as evidence."""
from __future__ import annotations

from .mdp import GridMDP
from .rules import ACTIONS
from .simulate import run_episode


def _fmt(outs):
    return "; ".join(f"p={p:g} -> {t} r={r:g}" for p, t, r in outs)


def run_checks(mdp: GridMDP, name: str, max_steps: int, discount: float) -> list[dict]:
    res = []

    def add(title, ok, detail):
        res.append({"map": name, "check": title, "passed": bool(ok), "detail": detail})

    # 1. wall and boundary: stay in place, reward of the tile where the move ends (= own tile)
    outs = mdp.transitions((1, 1), "up")            # top boundary
    add("boundary: (1,1) up stays at (1,1), reward of own tile S (-1)",
        outs == [(1.0, (1, 1), -1.0)], _fmt(outs))
    outs = mdp.transitions((1, 1), "left")          # left boundary
    add("boundary: (1,1) left stays at (1,1), reward -1",
        outs == [(1.0, (1, 1), -1.0)], _fmt(outs))
    outs = mdp.transitions((1, 2), "down")          # wall at (2,2); own tile is mud M (-3)
    add("wall: (1,2) down into '#' stays at (1,2), reward of own tile M (-3)",
        outs == [(1.0, (1, 2), -3.0)], _fmt(outs))
    outs = mdp.transitions((2, 1), "right")         # wall at (2,2); own tile is '.' (-1)
    add("wall: (2,1) right into '#' stays at (2,1), reward -1",
        outs == [(1.0, (2, 1), -1.0)], _fmt(outs))
    add("wall is not a state", (2, 2) not in mdp.states, "(2,2) is not in the state list")

    # 2. ice: probabilities sum to 1 for every action, blocked outcomes handled separately
    ice = [s for s in mdp.nonterminal_states() if mdp.rule(s).slippery]
    for s in ice:
        for a in ACTIONS:
            total = sum(p for p, _, _ in mdp.transitions(s, a))
            add(f"ice {s} action {a}: probabilities sum to 1", abs(total - 1.0) < 1e-12, f"sum = {total!r}")
    for s in mdp.nonterminal_states():
        for a in ACTIONS:
            total = sum(p for p, _, _ in mdp.transitions(s, a))
            if abs(total - 1.0) > 1e-12:
                add(f"state {s} action {a}: probabilities sum to 1", False, f"sum = {total!r}")
    add("every (nonterminal state, action) pair sums to 1", all(
        abs(sum(p for p, _, _ in mdp.transitions(s, a)) - 1.0) < 1e-12
        for s in mdp.nonterminal_states() for a in ACTIONS), "checked for all pairs")
    outs = mdp.transitions((2, 3), "up")            # ice: up 0.8, left is wall (stay), right is H
    expected = [(0.8, (1, 3), -1.0), (0.1, (2, 3), -1.0), (0.1, (2, 4), -10.0)]
    add("ice (2,3) up: 0.8 to (1,3), left = wall -> stay 0.1, right = H 0.1",
        all(abs(p - ep) < 1e-12 and t == et and r == er for (p, t, r), (ep, et, er) in zip(outs, expected))
        and len(outs) == 3, _fmt(outs))
    outs = mdp.transitions((1, 3), "down")          # moving INTO ice from '.' is deterministic
    add("slip depends on the START tile: (1,3) down onto ice is deterministic",
        outs == [(1.0, (2, 3), -1.0)], _fmt(outs))

    # 3. floor rewards
    outs = mdp.transitions((1, 1), "right")
    add("mud: entering M (1,2) gives -3", outs == [(1.0, (1, 2), -3.0)], _fmt(outs))
    outs = mdp.transitions((1, 3), "right")
    add("goal: entering G gives +10 and nothing else is added", outs == [(1.0, (1, 4), 10.0)], _fmt(outs))
    outs = mdp.transitions((1, 3), "down")
    add("ice reward: entering I gives -1", outs[0][2] == -1.0, _fmt(outs))
    if name == "extended":
        outs = mdp.transitions((3, 1), "right")
        add("sand: entering D (3,2) gives -2", outs == [(1.0, (3, 2), -2.0)], _fmt(outs))
        outs = mdp.transitions((3, 3), "right")
        add("water: entering W (3,4) gives -5", outs == [(1.0, (3, 4), -5.0)], _fmt(outs))
        outs = mdp.transitions((3, 2), "up")        # moving from sand: deterministic, blocked by wall above
        add("sand tile moves are deterministic ((3,2) up hits wall (2,2), stays, reward -2)",
            outs == [(1.0, (3, 2), -2.0)], _fmt(outs))
        outs = mdp.transitions((3, 4), "up")
        add("water tile moves are deterministic ((3,4) up to H (2,4) = -10)",
            outs == [(1.0, (2, 4), -10.0)], _fmt(outs))

    # 4. no rewards after termination
    terminals = [s for s in mdp.states if mdp.is_terminal(s)]
    add("terminal tiles have no transitions (no reward after termination)",
        all((s, a) not in mdp._table for s in terminals for a in ACTIONS),
        f"terminal tiles: {terminals}")
    # an episode that walks to G (all tiles on the way are deterministic) stops at G:
    # exactly 5 rewards, the last one is the goal reward, nothing is added afterwards
    walk = {(3, 1): "up", (2, 1): "up", (1, 1): "right", (1, 2): "right", (1, 3): "right"}
    ep = run_episode(mdp, walk, seed=1, max_steps=max_steps, discount=discount)
    add("episode to the goal stops at G: 5 rewards [-1, -1, -3, -1, +10], none after G",
        ep["outcome"] == "goal" and ep["steps"] == 5 and ep["rewards"] == [-1.0, -1.0, -3.0, -1.0, 10.0]
        and ep["path"][-1] == (1, 4) and all(not mdp.is_terminal(p) for p in ep["path"][:-1]),
        f"outcome={ep['outcome']}, steps={ep['steps']}, rewards={ep['rewards']}, path={ep['path']}")
    add("discounted return of that episode = -1 -0.9 -3*0.81 -0.729 +10*0.6561",
        abs(ep["discounted_return"] - (-1 - 0.9 - 3 * 0.81 - 0.729 + 10 * 0.6561)) < 1e-12,
        f"discounted_return={ep['discounted_return']!r}")
    return res
